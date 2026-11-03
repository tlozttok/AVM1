"""对话能力（caps）：模式、组、根表与工具集。

权限体系的机制层数据。本模块不依赖 Core 与 Conversation，供两者引用：

- 模式（mode）：kernel / user。内核态持有真实 `MEM` 根；用户态只持有自己的根表。
- 组（groups）：镜像层权限策略使用的标签，事件载荷会带上它。
- 根表（roots）：根名 -> `$MEM` 之后的真实段序列。解析引用时第一段按本表匹配。
- 工具集（tools）：本对话可调用的指令名集合，是 ALL_TOOL_NAMES 的子集。

`AddressSpace` 是一次解析的地址上下文；`Memory` 的公开方法接受它，None 表示
只允许 `MEM` 根（宿主与旧路径）。设计见：

    docs/private/Agents/implemented/design-permission-system-2026-10-27.md
    docs/private/Agents/implemented/design-event-layer-permission-2026-10-28.md
"""
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .exceptions import IllegalOperationError, VMMemoryError

MODE_KERNEL = "kernel"
# 内核控制面设备（挂载在 $MEM 下）：它们的路径是保留区，不能被镜像根注册表
# 或运行期授予根指向（设备本身也不可作为普通根授予）。
CONTROL_DEVICE_ROOTS = frozenset({"conversations", "listeners", "roots", "caps"})
MODE_USER = "user"
MODES = (MODE_KERNEL, MODE_USER)

MEM_ROOT_NAME = "MEM"

# 指令集的工具名。与 core._instruction_registry() 的键必须一致；
# tests/test_capabilities.py 钉住这条，避免两处清单漂移。
ALL_TOOL_NAMES = frozenset({
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
    "create_cmd",
    "create_sub",
    "send_instruction",
    "close_conversation",
    "listen_event",
})

# 根命名空间的保留名：MEM 是物理空间根，conv/batch 是 Core.unwrap 的伪根。
RESERVED_ROOT_NAMES = frozenset({MEM_ROOT_NAME, "conv", "batch", "caps"})
# 内部根表允许出现的保留根：MEM 由内核态持有，caps 由 Core 绑定到能力自述设备
INTERNAL_RESERVED_ROOTS = frozenset({MEM_ROOT_NAME, "caps"})


def check_mode(mode) -> None:
    if mode not in MODES:
        raise IllegalOperationError(f"模式必须是 {list(MODES)} 之一，得到 {mode!r}")


def check_root_name(name, allow_reserved: frozenset = frozenset()) -> None:
    """根名规则：字符串、非空、不含点号、不以 $ 开头、不占用保留名。

    `allow_reserved` 给内部根表用：根表可以包含 MEM（内建根），但新声明的根不行。
    """
    if not isinstance(name, str):
        raise IllegalOperationError(f"根名必须是字符串，得到 {type(name).__name__}")
    if not name:
        raise IllegalOperationError("根名不能为空字符串")
    if "." in name:
        raise IllegalOperationError(f"根名不能包含点号（点号是路径分隔符）：{name!r}")
    if name.startswith("$"):
        raise IllegalOperationError(f"根名不能以 $ 开头：{name!r}")
    if name in RESERVED_ROOT_NAMES and name not in allow_reserved:
        raise IllegalOperationError(f"根名不能占用保留名：{name!r}（保留：{sorted(RESERVED_ROOT_NAMES)}）")


def _normalize_roots(roots: Optional[Mapping[str, Iterable[str]]]) -> Dict[str, Tuple[str, ...]]:
    if roots is None:
        return {MEM_ROOT_NAME: ()}
    if not isinstance(roots, Mapping):
        raise IllegalOperationError(f"根表必须是映射（根名 -> 段序列），得到 {type(roots).__name__}")
    table: Dict[str, Tuple[str, ...]] = {}
    for name, segments in roots.items():
        check_root_name(name, allow_reserved=INTERNAL_RESERVED_ROOTS)
        if isinstance(segments, str) or not isinstance(segments, (tuple, list)):
            raise IllegalOperationError(
                f"根 {name!r} 的目标必须是段序列（字符串列表），得到 {type(segments).__name__}"
            )
        segs = tuple(str(s) for s in segments)
        if any(seg == "" for seg in segs):
            raise IllegalOperationError(f"根 {name!r} 的目标不能有空段：{segs!r}")
        table[name] = segs
    return table


def _normalize_tools(tools: Optional[Iterable[str]]) -> frozenset:
    if tools is None:
        return ALL_TOOL_NAMES
    toolset = frozenset(tools)
    unknown = toolset - ALL_TOOL_NAMES
    if unknown:
        raise IllegalOperationError(
            f"未知工具名：{sorted(unknown)}（允许：{sorted(ALL_TOOL_NAMES)}）"
        )
    return toolset


class AddressSpace:
    """一次引用解析的地址上下文：根名 -> 真实段序列。

    None 或只含 MEM 的空上下文表示"只允许 `$MEM` 根"；`Memory` 的 `space=None`
    即此形态。对象按只读使用。
    """

    __slots__ = ("_roots",)

    def __init__(self, roots: Optional[Mapping[str, Iterable[str]]] = None):
        self._roots = _normalize_roots(roots) if roots is not None else {MEM_ROOT_NAME: ()}

    def root(self, name: str) -> Optional[Tuple[str, ...]]:
        return self._roots.get(name)

    def names(self) -> Tuple[str, ...]:
        return tuple(self._roots)

    def has_mem(self) -> bool:
        return MEM_ROOT_NAME in self._roots

    def as_roots(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self._roots)

    def __eq__(self, other) -> bool:
        return isinstance(other, AddressSpace) and self._roots == other._roots

    def __repr__(self) -> str:
        return f"AddressSpace({self._roots!r})"


class Capability:
    """对话能力：模式、组、根表与工具集。

    对象创建后按只读使用；需要变体时用 `copy()` 重建。
    """

    __slots__ = ("mode", "groups", "roots", "tools")

    def __init__(
        self,
        mode: str = MODE_KERNEL,
        groups: Iterable[str] = (),
        roots: Optional[Mapping[str, Iterable[str]]] = None,
        tools: Optional[Iterable[str]] = None,
    ):
        check_mode(mode)
        self.mode = mode
        group_list = []
        for g in groups or ():
            if not isinstance(g, str) or not g:
                raise IllegalOperationError(f"组名必须是非空字符串，得到 {g!r}")
            group_list.append(g)
        self.groups: Tuple[str, ...] = tuple(group_list)
        self.roots: Dict[str, Tuple[str, ...]] = _normalize_roots(roots)
        if mode == MODE_USER and MEM_ROOT_NAME in self.roots:
            raise IllegalOperationError("用户态根表不能包含 MEM 根")
        self.tools: frozenset = _normalize_tools(tools)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def root_segments(self, name: str) -> Optional[Tuple[str, ...]]:
        return self.roots.get(name)

    def has_mem(self) -> bool:
        return MEM_ROOT_NAME in self.roots

    def tool_allowed(self, name: str) -> bool:
        return name in self.tools

    def space(self) -> AddressSpace:
        return AddressSpace(self.roots)

    def copy(
        self,
        *,
        mode: Optional[str] = None,
        groups: Optional[Iterable[str]] = None,
        roots: Optional[Mapping[str, Iterable[str]]] = None,
        tools: Optional[Iterable[str]] = None,
    ) -> "Capability":
        return Capability(
            mode=self.mode if mode is None else mode,
            groups=self.groups if groups is None else groups,
            roots=self.roots if roots is None else roots,
            tools=self.tools if tools is None else tools,
        )

    # ------------------------------------------------------------------
    # 比较与表示
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Capability)
            and self.mode == other.mode
            and self.groups == other.groups
            and self.roots == other.roots
            and self.tools == other.tools
        )

    def __repr__(self) -> str:
        return (
            f"Capability(mode={self.mode!r}, groups={self.groups!r}, "
            f"roots={self.roots!r}, tools={sorted(self.tools)!r})"
        )

    # ------------------------------------------------------------------
    # 便捷
    # ------------------------------------------------------------------

    @classmethod
    def kernel_full(cls) -> "Capability":
        """默认内核态全能力：MEM 根、全部工具、无组。"""
        return cls()


def default_kernel_caps() -> Capability:
    return Capability.kernel_full()


# ----------------------------------------------------------------------
# 序列化（检查点、内核转储、控制台）
# ----------------------------------------------------------------------

def caps_to_dict(cap: Capability) -> dict:
    """JSON 安全、字段顺序固定（检查点与转储用）。"""
    return {
        "mode": cap.mode,
        "groups": list(cap.groups),
        "roots": {name: list(segs) for name, segs in cap.roots.items()},
        "tools": sorted(cap.tools),
    }


def caps_from_dict(data) -> Capability:
    if not isinstance(data, dict):
        raise IllegalOperationError(f"caps 必须是对象，得到 {type(data).__name__}")
    mode = data.get("mode", MODE_KERNEL)
    groups = data.get("groups") or ()
    roots = data.get("roots")
    tools = data.get("tools")
    return Capability(mode=mode, groups=groups, roots=roots, tools=tools)

# ----------------------------------------------------------------------
# 路径前缀与授予根目标
# ----------------------------------------------------------------------

def path_is_prefix(prefix: tuple, path: tuple) -> bool:
    """段序列前缀关系：prefix 是 path 的祖先或相等。"""
    return len(prefix) <= len(path) and tuple(path[:len(prefix)]) == tuple(prefix)


def grant_target_segments(ref) -> tuple:
    """把授予根的目标 `$MEM...` 引用转成真实段序列（不含 MEM 根）。"""
    if not isinstance(ref, str) or not ref.startswith("$"):
        raise IllegalOperationError(f"授予根的目标必须是 $MEM 引用：{ref!r}")
    parts = ref[1:].split(".")
    if not parts or parts[0] != MEM_ROOT_NAME:
        raise IllegalOperationError(f"授予根的目标必须是 $MEM 引用：{ref!r}")
    segments = parts[1:]
    if not segments:
        raise VMMemoryError("授予根的目标不能是 $MEM 根本身（会暴露整棵树）")
    if any(seg == "" for seg in segments):
        raise IllegalOperationError(f"授予根的目标不能有空段：{ref!r}")
    if any(seg.startswith("-") and seg[1:].isdigit() for seg in segments):
        raise IllegalOperationError(f"授予根的目标不能含负下标：{ref!r}")
    return tuple(segments)


def targets_prefix_free(existing, new_target) -> bool:
    """`new_target` 与 `existing` 里的每个目标都不可前缀比较（相等也算冲突）。"""
    for target in existing:
        if path_is_prefix(tuple(target), tuple(new_target)) or path_is_prefix(tuple(new_target), tuple(target)):
            return False
    return True

"""内存节点本身：怎么读一格、写一格、判一次访问，以及键与地址的书写规则。

这里只认"一棵节点树"，不认识地址空间、根、设备——那些是 memory.py 的事。分开
的理由不是行数，是这两种东西的变化原因不同：节点规则跟着节点类型（str / dict /
list / link）变，寻址规则跟着调用方（指令、事件、控制面、面板）变。混在一个类里，
改一处总要确认另一处没被带上。

全部是无状态函数，参数就是它们要操作的那个节点，所以可以直接对着一个手搓的
dict 调，不需要造一棵树。

依赖方向：memory_nodes -> types / exceptions / capabilities / memory_device。
memory.py 反过来 import 它，所以这里不能出现对 Memory 的引用。
"""
from typing import Any, Optional

from .capabilities import MEM_ROOT_NAME
from .exceptions import (
    IllegalOperationError,
    MemoryIndexOutOfRangeError,
    MemoryKeyNotFoundError,
    MemoryTypeError,
    VMMemoryError,
)
from .memory_device import MemoryDevice
from .types import LinkNode, MetaDict, MetaList, StrNode


def show_path(parts) -> str:
    """把显示段序列拼成 `$根.段...` 形态（错误与地址回显用）。"""
    return "$" + ".".join(str(s) for s in parts)


def check_new_key(key) -> None:
    """新建地址的键语法（指令边界与 Memory.make 共用一套）。

    键是路径的一段，点号是段分隔符——带点的键建得出来但永远读不回来（模板：`$MEM.a.b`
    会被切成 a、b 两段）。空键同理，还会让 `$MEM.` 与读根混淆。
    """
    if not isinstance(key, str):
        raise IllegalOperationError(f"新地址的键必须是字符串，得到 {type(key).__name__}")
    if key == "":
        raise IllegalOperationError("新地址的键不能为空字符串")
    if "." in key:
        raise IllegalOperationError(f"新地址的键不能包含点号（点号是路径分隔符，该地址将无法寻址）：{key!r}")


def new_node(mem_type: str):
    """按 mem_type 造一个新节点（新节点不带 ctrl）。"""
    if mem_type == 'str':
        return StrNode("")
    if mem_type == 'dict':
        return MetaDict(data={})
    return MetaList(data=[])


def index_of(container, key):
    """列表容器把键转成整数下标；字典容器原样返回键。"""
    if isinstance(container, (list, MetaList)):
        try:
            return int(key)
        except (TypeError, ValueError):
            raise MemoryTypeError(f"列表索引必须是数字：{key}") from None
    return key


def index_key(node: Any, key: Any) -> Any:
    """列表节点按引用寻址时用整数下标取值。

    引用是一段字符串（`$MEM.a.0`），`_check_access` 已经确认这段是合法下标，
    但取值前必须把字符串转成整数——否则 `MetaList.__getitem__("0")` 会抛
    `TypeError: list indices must be integers`。设备列表自己会转，树里的列表靠这里。
    """
    return int(str(key)) if isinstance(node, (list, MetaList)) else key


def child(node: Any, key: Any, display: list, index: int):
    """(子节点, 是否存在)。node 不是容器时抛 MemoryTypeError（含请求路径，不含链接目标）。"""
    shown = show_path(display)
    if isinstance(node, (dict, MetaDict)):
        return node.get(key), (key in node)
    if isinstance(node, (list, MetaList)):
        idx = index_of(node, key)
        if idx < 0:
            raise MemoryIndexOutOfRangeError(
                f"负下标不支持：{shown}（-1 只表示追加位置）"
            )
        if idx >= len(node):
            raise MemoryIndexOutOfRangeError(
                f"索引越界：{idx}，列表长度：{len(node)}：{shown}"
            )
        return node[idx], True
    raise MemoryTypeError(
        f"无法写入：{shown} 是 {type(node).__name__} 类型，没有子节点 (.{key})"
    )


def assign(node: Any, key: Any, value: Any, display: list, index: int) -> None:
    """把 value 落到 node 的 key 上。已存在的容器节点按既有规则解释为改元数据。"""
    if isinstance(node, (dict, MetaDict)):
        existing = node.get(key)
        if isinstance(existing, (MetaList, MetaDict)):
            existing.set_metadata(value)
            return
        if isinstance(existing, str):
            # 写字符串叶子：只改内容，不动节点的 ctrl（写入不夺权）。
            keep = existing.get_ctrl() if isinstance(existing, StrNode) else None
            node[key] = StrNode(str(value), ctrl=keep)
            return
        node[key] = value
        return
    if isinstance(node, (list, MetaList)):
        idx = index_of(node, key)
        if idx < 0:
            raise IllegalOperationError("负下标不支持（-1 只表示追加位置）")
        if idx >= len(node):
            raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(node)}")
        existing = node[idx]
        if isinstance(existing, str):
            keep = existing.get_ctrl() if isinstance(existing, StrNode) else None
            node[idx] = StrNode(str(value), ctrl=keep)
            return
        node[idx] = value
        return
    raise MemoryTypeError(
        f"无法写入：{show_path(display)} 是 {type(node).__name__} 类型，不支持子路径"
    )


def check_access(current, key: str, path: list, shown: Optional[str] = None):
    """检查 key 是否可访问，不可则抛出对应的内存子类异常。

    `path` 是显示段序列（含根名）；`shown` 指定时直接用它做错误里的地址。
    """
    if isinstance(current, MemoryDevice):
        return
    partial = shown if shown is not None else show_path(path)
    if isinstance(current, str):
        raise MemoryTypeError(f"字符串没有子节点：{partial}")
    if isinstance(current, (dict, MetaDict)):
        if key == "-1":
            raise IllegalOperationError(f"字典里不能用 -1（-1 只表示列表追加位置）：{partial}")
        if key not in current:
            raise MemoryKeyNotFoundError(f"键 {key!r} 不存在：{partial}（可用 memory_make 创建）")
    elif isinstance(current, (list, MetaList)):
        try:
            idx = int(key)
        except ValueError:
            raise IllegalOperationError(f"列表索引必须是数字，got {key!r}：{partial}")
        if idx < 0:
            raise IllegalOperationError(f"负下标不支持：{partial}（-1 只表示追加位置）")
        if idx >= len(current):
            raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(current)}：{partial}")
    else:
        raise MemoryTypeError(f"无法访问子路径：父节点是 {type(current).__name__} 类型：{partial}")


def check_delete_target(ref: str, kind: str, display) -> None:
    """删除目标的静态拒绝：设备空间、根名本身、负下标。"""
    if kind == "device":
        raise IllegalOperationError(
            f"{ref} 落在设备上：不能在设备空间删除节点（设备内容由设备自己管理）"
        )
    if len(display) == 1:
        if display[0] == MEM_ROOT_NAME:
            raise IllegalOperationError("不能删除 $MEM 根节点")
        raise IllegalOperationError(f"不能删除根：{ref}（根是内核声明，不是内存节点）")
    # 请求段（根名之后）的负下标检查
    if any(seg == "-1" or (seg.startswith("-") and seg[1:].isdigit()) for seg in display[1:]):
        raise IllegalOperationError(f"负下标不支持：{ref}（-1 只表示追加位置）")


def check_make_args(mem_type: str, key: str) -> None:
    """建节点前先挡掉类型名写错与非法键。"""
    if mem_type not in ('str', 'dict', 'list'):
        raise IllegalOperationError(f"不支持的类型：{mem_type}，必须是 'str', 'dict', 'list' 之一")
    check_new_key(key)


def list_elem_type(node: list) -> str:
    """列表的元素类型（同类规则的依据）。混合列表报错：类型不唯一时不做猜测。"""
    kinds = set()
    for item in node:
        if isinstance(item, MetaDict) or isinstance(item, dict):
            kinds.add("dict")
        elif isinstance(item, MetaList) or isinstance(item, list):
            kinds.add("list")
        elif isinstance(item, LinkNode):
            kinds.add("link")
        else:
            kinds.add("str")
    if not kinds:
        return "str"  # 空列表默认 str（write 追加的默认类型）
    if len(kinds) > 1:
        raise VMMemoryError(
            f"列表元素类型不一致：{sorted(kinds)}，同类规则的依据不唯一（该列表来自旧数据，需重建）"
        )
    return kinds.pop()


def check_append_same_type(node: list, append_kind: str) -> None:
    """非空列表追加必须与已有元素同类型；空列表不检查（新列表可任意指定）。"""
    if not node:
        return
    elem = list_elem_type(node)
    if elem != append_kind:
        raise VMMemoryError(
            f"同类节点必须与已有元素同类型：列表元素是 {elem}，不能追加 {append_kind}"
        )


def link_target(node: LinkNode, address: str = "") -> tuple:
    """解析链接 target：返回 ("memory", segments) 或 ("device", 标识符)。"""
    target = node.target
    if not isinstance(target, str) or not target:
        raise VMMemoryError(f"链接 {address or '$MEM'} 的 target 必须是非空字符串")
    if target.startswith("$"):
        parts = target[1:].split(".")
        if not parts or parts[0] != "MEM":
            raise VMMemoryError(
                f"链接 {address or '$MEM'} 的 target 只支持 $MEM 地址或设备标识符：{target!r}"
            )
        return ("memory", parts[1:])
    return ("device", target)

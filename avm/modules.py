"""宿主模块目录、模块视图与导入裁决。

沙箱的导入能力分三层：

- **目录（闭集）**：宿主提供哪些模块、每个模块暴露哪些名字，全在这里定义。
  目录之外的模块名在任何执行点上都不存在；模块对象的内部属性（模块对象、文件、
  进程、时钟、熵源）不进入导出表。
- **视图（GuestModule）**：给 guest 的不是宿主真模块，而是只含导出表的只读视图；
  双下划线属性一律不可见，视图之间互不共享可变状态。
- **裁决（import_function）**：每个 PLM 实例一个 `__import__` 闭包，按「基线 ∪
  授予」白名单返回视图；未授予的名字抛 `SecurityError`。

模块目录设备 `$MEM.module_catalog` 与导入共用同一份目录数据（`BASE_MODULES` /
`MODULE_CATALOG`），供 AVM 内部查询「有什么模块可用、每个模块导出哪些名字」。

目录维护规则（新增模块时逐项审计）：

- 只列纯计算函数 / 类 / 不可变常量；不列模块对象、文件对象、进程/线程/网络/时钟/
  熵源相关的任何东西。
- 不列「按字符串取属性」的函数（`operator.attrgetter` / `operator.methodcaller`
  能绕过 AST 的双下划线属性规则），因此 `operator` 不进目录。
- 不列会改宿主全局状态的函数（`re.purge`、`gc.*`）。
- 不列返回宿主句柄或线程局部状态的入口（`decimal.getcontext` 之类）。
- 导出项若是真宿主对象，要接受「guest 可能改写其类属性」这一残余风险；不能有
  可被改写后影响宿主安全的属性。
"""

import hashlib
import importlib
import importlib.util
import json
import os
import types
from typing import Dict, Mapping, Optional

from .exceptions import VMMemoryError
from .memory_device import MemoryDevice
from .sandbox import SecurityError, safe_globals

# 兼容基线：历史镜像不写 import 而直接使用这些名字，运行时照旧注入（视图形态）。
# 这三个模块不属于 caps.imports 的授予面：所有 Python 对话都可用。
BASE_MODULES = ("math", "json", "re")

_RAW_GETATTR = types.ModuleType.__getattribute__


class GuestModule(types.ModuleType):
    """guest 可见的模块视图：只含导出表，双下划线属性不可见，只读。

    视图故意不暴露 `__name__` / `__dict__` / `__all__`：`from <视图> import x`
    的 `IMPORT_FROM` 实现会先取属性、取不到时再按 `__name__` 去宿主 `sys.modules`
    回退——屏蔽双下划线属性同时挡住了这条回退路径（未导出名只会得到 ImportError）。
    """

    def __init__(self, name: str, exports: Mapping[str, object]):
        types.ModuleType.__init__(self, name)
        _RAW_GETATTR(self, "__dict__").update(exports)

    def __getattribute__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return _RAW_GETATTR(self, name)
        except AttributeError:
            module_name = _RAW_GETATTR(self, "__name__")
            raise AttributeError(f"模块 {module_name!r} 未导出 {name!r}")

    def __setattr__(self, name, value):
        raise AttributeError("模块视图只读")

    def __repr__(self):
        return f"<module {_RAW_GETATTR(self, '__name__')!r} (avm view)>"


def _public(module, exclude=()) -> Dict[str, object]:
    """取模块的非下划线公开名（宿主侧构建导出表用）。"""
    out = {}
    for name in dir(module):
        if name.startswith("_") or name in exclude:
            continue
        out[name] = getattr(module, name)
    return out


def _pick(module, names) -> Dict[str, object]:
    """按名单取导出表；宿主 Python 版本缺少的名字跳过（目录自身不因此报错）。"""
    out = {}
    for name in names:
        if hasattr(module, name):
            out[name] = getattr(module, name)
    return out


def _math_exports():
    import math
    return _public(math)


def _json_exports():
    import json as _json
    return _pick(_json, ("loads", "dumps", "JSONDecodeError", "JSONDecoder", "JSONEncoder"))


def _re_exports():
    import re as _re
    return _pick(_re, (
        "search", "match", "fullmatch", "findall", "finditer", "compile",
        "sub", "subn", "split", "escape", "Pattern", "Match", "error", "RegexFlag",
        "A", "ASCII", "I", "IGNORECASE", "L", "LOCALE", "M", "MULTILINE",
        "S", "DOTALL", "U", "UNICODE", "X", "VERBOSE", "DEBUG", "NOFLAG",
    ))


def _itertools_exports():
    import itertools
    return _public(itertools)


def _functools_exports():
    import functools
    return _pick(functools, (
        "reduce", "partial", "partialmethod", "wraps", "update_wrapper",
        "lru_cache", "cache", "cached_property", "cmp_to_key", "total_ordering",
        "singledispatch", "singledispatchmethod",
    ))


def _collections_exports():
    import collections
    return _pick(collections, (
        "Counter", "OrderedDict", "defaultdict", "deque", "namedtuple",
        "ChainMap", "UserDict", "UserList", "UserString",
    ))


class ModuleSpec:
    """模块目录的一条：名字、类别（基线/可授予）、导出表、说明。

    `exports` 是构建时算好的静态表（名字 → 宿主对象），静态检查只用它的键；
    guest 拿到的是由它构建的 `GuestModule` 视图。
    """

    __slots__ = ("name", "kind", "exports", "note")

    def __init__(self, name, kind, builder, note):
        self.name = name
        self.kind = kind
        self.exports = builder()
        self.note = note


# 目录：名字 → ModuleSpec。基线模块 kind="base"；其余可被程序 caps.imports 授予。
_MODULE_SPECS = [
    ModuleSpec("math", "base", _math_exports, "数学函数与常量；无状态纯计算"),
    ModuleSpec("json", "base", _json_exports, "JSON 编解码；load/dump 需要文件对象，不导出"),
    ModuleSpec("re", "base", _re_exports, "正则；purge 会改宿主缓存，不导出"),
    ModuleSpec("itertools", "grantable", _itertools_exports, "迭代器工具"),
    ModuleSpec("functools", "grantable", _functools_exports, "高阶函数工具"),
    ModuleSpec("collections", "grantable", _collections_exports, "容器类型"),
]

MODULE_CATALOG: Dict[str, ModuleSpec] = {spec.name: spec for spec in _MODULE_SPECS}

# 运行期注册的模块描述（快照/journal 用）：name → {source_kind, source, exports, note, sha256}
REGISTERED_MODULE_DESCRIPTORS: Dict[str, dict] = {}

_GRANTABLE_CACHE = None
_EXPORTS_CACHE = None


def _invalidate_catalog_caches() -> None:
    global _GRANTABLE_CACHE, _EXPORTS_CACHE
    _GRANTABLE_CACHE = None
    _EXPORTS_CACHE = None


def grantable_module_names() -> frozenset:
    """当前可授予模块名（含运行期注册；实时读目录，注册后立即生效）。"""
    global _GRANTABLE_CACHE
    if _GRANTABLE_CACHE is None:
        _GRANTABLE_CACHE = frozenset(
            name for name, spec in MODULE_CATALOG.items() if spec.kind == "grantable"
        )
    return _GRANTABLE_CACHE


def catalog_exports() -> Dict[str, frozenset]:
    """静态检查用：模块名 → 可 from-import 的导出名集合（含基线与运行期注册）。"""
    global _EXPORTS_CACHE
    if _EXPORTS_CACHE is None:
        _EXPORTS_CACHE = {
            name: frozenset(spec.exports) for name, spec in MODULE_CATALOG.items()
        }
    return _EXPORTS_CACHE


def catalog_names() -> tuple:
    return tuple(sorted(MODULE_CATALOG))


def _check_export_names(name: str, export_names) -> None:
    if not isinstance(name, str) or not name:
        raise SecurityError("注册模块需要非空字符串名字")
    if "." in name or name.startswith("_"):
        raise SecurityError(f"模块名不能含点号或以 _ 开头：{name!r}")
    if name in BASE_MODULES:
        raise SecurityError(f"模块名 {name!r} 是基线模块，不可注册覆盖")
    if not export_names:
        raise SecurityError(f"模块 {name!r} 的导出名单不能为空")
    for item in export_names:
        if not isinstance(item, str) or not item:
            raise SecurityError(f"模块 {name!r} 的导出名必须是非空字符串：{item!r}")
        if item.startswith("_"):
            raise SecurityError(f"模块 {name!r} 的导出名不能以 _ 开头：{item!r}")


def register_module(name: str, exports: Optional[Mapping[str, object]] = None,
                    note: str = "") -> ModuleSpec:
    """把一个模块注册进宿主目录（运行期注册的唯一写入口的底层实现）。

    `exports` 是导出名 → 宿主对象；导出值不允许是模块对象（视图里再套真模块会
    绕开裁剪）。同描述重复注册由调用方（`register_module_from_source`）负责幂等；
    这里只做结构校验并写入目录。
    """
    if not isinstance(name, str) or not name:
        raise SecurityError("注册模块需要非空字符串名字")
    if name in BASE_MODULES:
        raise SecurityError(f"模块名 {name!r} 是基线模块，不可注册覆盖")
    if name in MODULE_CATALOG:
        raise SecurityError(f"模块 {name!r} 已在目录里，不能重复注册")
    if not isinstance(exports, Mapping):
        raise SecurityError(f"注册模块 {name!r} 需要导出映射（名字 → 宿主对象）")
    export_names = tuple(exports.keys())
    _check_export_names(name, export_names)
    for export_name, value in exports.items():
        if isinstance(value, types.ModuleType):
            raise SecurityError(
                f"模块 {name!r} 的导出 {export_name!r} 是模块对象：视图不能导出真模块"
            )
    frozen = dict(exports)
    spec = ModuleSpec(name, "grantable", lambda: dict(frozen), note or "运行期注册")
    MODULE_CATALOG[name] = spec
    _invalidate_catalog_caches()
    return spec


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module_source(source_kind: str, source: str):
    """按来源加载宿主模块（宿主代码，注册入口信任边界在上层）。"""
    if source_kind == "module":
        try:
            return importlib.import_module(source)
        except Exception as e:  # noqa: BLE001 - 宿主模块加载失败统一归类
            raise SecurityError(f"加载宿主模块 {source!r} 失败：{type(e).__name__}: {e}") from None
    if source_kind == "file":
        if not isinstance(source, str) or not source:
            raise SecurityError("file 来源需要非空路径")
        file_path = os.path.abspath(source)
        if not os.path.isfile(file_path):
            raise SecurityError(f"模块文件不存在：{file_path}")
        module_name = "_avm_host_module_" + hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:12]
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise SecurityError(f"无法加载模块文件：{file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except SecurityError:
            raise
        except Exception as e:  # noqa: BLE001 - 宿主模块加载失败统一归类
            raise SecurityError(f"加载模块文件 {file_path} 失败：{type(e).__name__}: {e}") from None
    raise SecurityError(f"未知模块来源：{source_kind!r}（file/module）")


def register_module_from_source(name: str, source_kind: str, source: str,
                                export_names, note: str = "") -> ModuleSpec:
    """从文件/已安装模块注册：按导出名单裁剪后进目录。

    同描述重复注册幂等（返回已有 spec，供 init 重跑、journal 重放使用）；
    同名不同描述报错（不做运行期替换）。
    """
    if not isinstance(export_names, (list, tuple)):
        raise SecurityError(f"模块 {name!r} 的导出名单必须是字符串列表")
    _check_export_names(name, export_names)
    existing = REGISTERED_MODULE_DESCRIPTORS.get(name)
    if existing is not None:
        if (existing.get("source_kind") == source_kind
                and existing.get("source") == source
                and existing.get("exports") == list(export_names)
                and existing.get("note", "") == (note or "")):
            return MODULE_CATALOG[name]
        raise SecurityError(f"模块 {name!r} 已用不同描述注册过，不能替换")
    if name in BASE_MODULES:
        raise SecurityError(f"模块名 {name!r} 是基线模块，不可注册覆盖")
    if name in MODULE_CATALOG:
        raise SecurityError(f"模块 {name!r} 已在目录里，不能重复注册")
    loaded = _load_module_source(source_kind, source)
    exports = {}
    missing = []
    for export_name in export_names:
        if hasattr(loaded, export_name):
            exports[export_name] = getattr(loaded, export_name)
        else:
            missing.append(export_name)
    if missing:
        raise SecurityError(f"模块 {name!r} 缺少导出名：{sorted(missing)}")
    spec = register_module(name, exports=exports, note=note)
    descriptor = {
        "name": name,
        "kind": "grantable",
        "source_kind": source_kind,
        "source": os.path.abspath(source) if source_kind == "file" else source,
        "exports": list(export_names),
        "note": note or "",
    }
    if source_kind == "file":
        descriptor["sha256"] = _sha256_file(descriptor["source"])
    REGISTERED_MODULE_DESCRIPTORS[name] = descriptor
    return spec


def registered_module_descriptors() -> Dict[str, dict]:
    """运行期注册的模块描述（快照/journal 用；JSON 安全副本）。"""
    return {
        name: {key: (list(value) if isinstance(value, list) else value)
               for key, value in descriptor.items()}
        for name, descriptor in REGISTERED_MODULE_DESCRIPTORS.items()
    }


def module_descriptor(name: str) -> dict:
    descriptor = REGISTERED_MODULE_DESCRIPTORS.get(name)
    if descriptor is None:
        raise SecurityError(f"模块 {name!r} 不是运行期注册的模块")
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in descriptor.items()}

# 事件处理器等没有实例缓存的位置使用的基线视图缓存（视图只读，可共享）。
_SHARED_VIEWS: Dict[str, GuestModule] = {}


def build_view(name: str):
    """按目录构建一个新的模块视图；目录外名字抛 SecurityError。"""
    spec = MODULE_CATALOG.get(name)
    if spec is None:
        raise SecurityError(
            f"模块 {name!r} 未登记在宿主模块目录（可用：{sorted(MODULE_CATALOG)}）"
        )
    return GuestModule(name, spec.exports)


def view(name: str, cache: Dict[str, GuestModule]) -> GuestModule:
    """按实例缓存取/建模块视图（每个 PLM 实例一份 cache）。"""
    existing = cache.get(name)
    if existing is not None:
        return existing
    built = build_view(name)
    cache[name] = built
    return built


def shared_view(name: str) -> GuestModule:
    """共享视图（事件处理器等无实例状态的位置用；视图本身只读）。"""
    existing = _SHARED_VIEWS.get(name)
    if existing is not None:
        return existing
    built = build_view(name)
    _SHARED_VIEWS[name] = built
    return built


def allowed_module_names(granted=None) -> frozenset:
    """一次执行可 import 的模块名：基线 ∪ 授予。"""
    return frozenset(BASE_MODULES) | frozenset(granted or ())


def import_function(allowed=None, cache: Optional[Dict[str, GuestModule]] = None):
    """构造一个实例的 `__import__` 闭包。

    语义与 CPython 的 `import` 对齐到本项目允许的子集：
    - 只支持顶层模块（`import a` / `from a import b` / `as` 别名）；
    - 相对导入、子模块导入、star import 在静态检查阶段已被拒绝，这里再兜一道；
    - 返回 GuestModule 视图，不返回宿主真模块。
    """
    allowed_set = frozenset(allowed if allowed is not None else BASE_MODULES)
    cache = {} if cache is None else cache

    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        if level:
            raise SecurityError("禁止相对导入")
        if not isinstance(name, str) or not name:
            raise SecurityError(f"非法的模块名：{name!r}")
        if "." in name:
            raise SecurityError(f"暂不支持子模块导入：{name!r}")
        if name not in allowed_set:
            raise SecurityError(
                f"模块 {name!r} 不在本对话的导入白名单里（允许：{sorted(allowed_set)}）"
            )
        return view(name, cache)

    return __import__


def sandbox_globals(extra=None, allowed=None, cache=None):
    """受限执行命名空间：安全 builtins + 基线视图注入 + 实例 `__import__`。

    `math` / `json` / `re` 三个基线视图照旧以原名注入（旧镜像不写 import 也能用），
    其余模块只能在 import 时按白名单出现。
    """
    namespace = dict(extra or {})
    if cache is None:
        cache = {}
    for name in BASE_MODULES:
        namespace[name] = view(name, cache)
    g = safe_globals(namespace)
    g["__builtins__"]["__import__"] = import_function(allowed, cache)
    return g


def module_catalog_summary() -> str:
    """目录设备的根摘要。"""
    base = "、".join(BASE_MODULES)
    grantable = [name for name in MODULE_CATALOG if name not in BASE_MODULES]
    listed = "、".join(grantable) if grantable else "（无）"
    return (
        f"模块目录（宿主）：基线不可裁剪 {base}；可授予 {len(grantable)} 个：{listed}。"
        f"读 $MEM.module_catalog.<模块名> 查导出名；本对话实际授予见 $MEM.caps.<cid>.imports"
    )


def module_entry(name: str) -> dict:
    """目录设备单模块条目（JSON 安全形状）。"""
    spec = MODULE_CATALOG.get(name)
    if spec is None:
        raise VMMemoryError(
            f"模块 {name!r} 不在宿主模块目录（可用：{sorted(MODULE_CATALOG)}）"
        )
    return {
        "name": spec.name,
        "kind": spec.kind,
        "note": spec.note,
        "exports": sorted(spec.exports),
    }


class ModuleCatalogDevice(MemoryDevice):
    """$MEM.module_catalog：宿主模块目录（只读）。

    AVM 内部据此回答「有什么模块可用、每个模块导出哪些名字」；真正的导入裁决
    仍由对话 caps 与模块视图执行。只读：不实现 `set_value`，写入由 Memory 拒绝。
    """

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return module_catalog_summary()

    def resolve_path(self, path):
        if not path:
            return self.to_llm_string()
        name = path[0]
        entry = module_entry(name)
        if len(path) == 1:
            return json.dumps(entry, ensure_ascii=False)
        if len(path) == 2:
            field = path[1]
            if field in entry:
                value = entry[field]
                return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            raise VMMemoryError(
                f"$MEM.module_catalog.<模块名> 支持 name/kind/note/exports，得到 {field!r}"
            )
        raise VMMemoryError("$MEM.module_catalog 支持 <模块名> 与 <模块名>.<字段>")

class ModuleLoaderDevice(MemoryDevice):
    """$MEM.module_loader：运行期模块注册入口（临时入口，审批流程后补）。

    写入一段 JSON 命令：

    ```json
    {"action": "register", "name": "mylib", "file": "lib/mylib.py",
     "exports": ["foo", "Bar"], "note": "可选说明"}
    {"action": "register", "name": "stats", "module": "statistics",
     "exports": ["mean"]}
    ```

    `file` 相对镜像目录解析；`module` 是宿主已安装模块名。注册 = 在宿主进程内
    import 该文件/模块，按 exports 名单裁剪后进目录（宿主代码，信任边界目前是
    「能写 $MEM 的内核态对话」；审批流程之后补，sha256 已记在描述里）。
    同描述重复注册幂等：镜像 init 在 reboot/resume 后重跑同一命令不会报错。
    """

    def __init__(self, core=None):
        self._core = core

    def attach_core(self, core) -> None:
        self._core = core

    def pretend_as_type(self) -> str:
        return "str"

    def to_llm_string(self) -> str:
        return (
            "module_loader：运行期模块注册入口。写 JSON 命令："
            "{\"action\": \"register\", \"name\": <名字>, "
            "\"file\": <相对镜像目录的 .py 路径> 或 \"module\": <已安装模块名>, "
            "\"exports\": [<导出名>...]}；注册后新创建的对话可按 caps.imports 使用"
        )

    def set_value(self, value) -> None:
        core = self._core
        if core is None:
            raise VMMemoryError("模块注册入口未接入 Core")
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise SecurityError("注册命令不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise SecurityError("注册命令必须是 JSON 对象")
        action = payload.get("action")
        if action != "register":
            raise SecurityError(f"注册命令 action 不支持：{action!r}")
        name = payload.get("name")
        exports = payload.get("exports")
        note = payload.get("note") or ""
        file_rel = payload.get("file")
        module_name = payload.get("module")
        if not isinstance(note, str):
            raise SecurityError("注册命令的 note 必须是字符串")
        if not isinstance(exports, list):
            raise SecurityError("注册命令的 exports 必须是字符串数组")
        has_file = isinstance(file_rel, str) and bool(file_rel)
        has_module = isinstance(module_name, str) and bool(module_name)
        if has_file == has_module:
            raise SecurityError("注册命令必须在 file 与 module 中二选一")
        if has_file:
            base_dir = getattr(core, "image_dir", None) or os.getcwd()
            source_kind, source = "file", (
                file_rel if os.path.isabs(file_rel) else os.path.normpath(os.path.join(base_dir, file_rel))
            )
        else:
            source_kind, source = "module", module_name
        try:
            core.register_module(
                name, source_kind=source_kind, source=source,
                export_names=tuple(exports), note=note,
            )
        except SecurityError as e:
            # 注册失败按可判定错误返回（不回显写入原文，错误文本来自注册表校验）
            raise VMMemoryError(f"模块注册失败：{e}") from None

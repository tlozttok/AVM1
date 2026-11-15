"""受限执行环境：防止 PLM 代码访问宿主系统（文件、网络、进程、时钟、熵源）。

最佳努力沙箱：安全 builtins 白名单 + AST 检查（双下划线属性、内省属性、危险调用名、
import 白名单）+ 运行时导入裁决。真正强隔离（子进程 + OS 级隔离）超出 VM 范围。

import 相关：`check_code(allow_import=True, imports=..., exports=...)` 允许 import
语法并做静态白名单检查；实际模块对象由调用方注入的 `__import__`（见
`avm/modules.py`）按实例白名单返回裁剪后的模块视图。
"""

import ast
import builtins
from typing import Mapping, Optional


class SecurityError(Exception):
    """PLM 代码触发了受限环境禁止的操作"""


# 允许的内置名字（不含 open/input/eval/exec/compile/__import__/getattr/type 等逃逸面）
SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hash", "hex", "int", "len",
    "list", "map", "max", "min", "oct", "ord", "pow", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "print",
    # 类型/迭代工具：处理器常用，且不构成逃逸面（不打开文件/网络、不做动态导入）
    "isinstance", "issubclass", "callable", "iter", "next",
    # 导入相关：白名单外模块与「未导出名字」都抛这两种异常，程序需要能捕获
    "ImportError", "ModuleNotFoundError",
    # 注意：这里不发 BaseException。Python 程序（PLM 代码、事件处理器）不该能触发
    # "紧急停止"——那会让一句 raise BaseException("x") 穿过内核所有错误处理。
    # 停机信号只来自宿主：人按 Ctrl+C、宿主主动退出。
    "ArithmeticError", "AssertionError", "AttributeError",
    "Exception", "KeyError", "IndexError", "NameError", "RuntimeError",
    "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
}


def make_safe_builtins() -> dict:
    b = {}
    for name in SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            b[name] = getattr(builtins, name)
    return b


_BLOCKED_CALL_NAMES = {
    "exec", "eval", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "hasattr", "vars", "dir",
    "globals", "locals", "type", "memoryview", "id", "breakpoint",
}

# 非双下划线的内省属性：它们能拿到帧、代码对象与 globals/builtins，
# 是「工具函数是宿主生成器」这条路径的出口（`g.gi_frame.f_globals`）。
# 属性名是静态数据；guest 没有动态取属性的入口（getattr 在禁用名单）。
_BLOCKED_ATTRIBUTE_NAMES = frozenset({
    # 生成器
    "gi_frame", "gi_code", "gi_yieldfrom", "gi_running",
    # 帧
    "f_back", "f_builtins", "f_code", "f_globals", "f_locals",
    "f_lasti", "f_lineno", "f_trace", "f_trace_lines", "f_trace_opcodes",
    # 代码对象
    "co_code", "co_consts", "co_names", "co_freevars", "co_cellvars",
    "co_varnames", "co_filename", "co_firstlineno", "co_stacksize", "co_flags",
    # traceback
    "tb_frame", "tb_next", "tb_lasti", "tb_lineno",
    # 旧式函数/方法属性（兼容 Python 2 术语，出现即拒绝）
    "func_globals", "func_code", "im_func", "im_self", "im_class",
})


class _RestrictedVisitor(ast.NodeVisitor):
    def __init__(self, allow_import: bool = False, imports=frozenset(),
                 exports: Optional[Mapping[str, frozenset]] = None):
        self.allow_import = allow_import
        self.imports = frozenset(imports)
        self.exports = exports or {}

    # ------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------

    def visit_Import(self, node):
        if not self.allow_import:
            raise SecurityError("禁止 import 语句")
        for alias in node.names:
            self._check_import_name(alias.name, alias.asname)

    def visit_ImportFrom(self, node):
        if not self.allow_import:
            raise SecurityError("禁止 from ... import 语句")
        if node.level:
            raise SecurityError("禁止相对导入")
        if node.module is None:
            raise SecurityError("from 导入缺少模块名")
        self._check_import_name(node.module, None)
        for alias in node.names:
            if alias.name == "*":
                raise SecurityError("禁止 from ... import *：显式导入需要的名字")
            if alias.name.startswith("__"):
                raise SecurityError(f"禁止导入双下划线名字: {alias.name}")
            if alias.asname and alias.asname.startswith("__"):
                raise SecurityError(f"禁止双下划线别名: {alias.asname}")
            exported = self.exports.get(node.module)
            if exported is not None and alias.name not in exported:
                raise SecurityError(
                    f"模块 {node.module!r} 未导出 {alias.name!r}"
                    f"（可用：{sorted(exported)}）"
                )

    def _check_import_name(self, name, asname):
        if not isinstance(name, str) or not name:
            raise SecurityError(f"非法的模块名：{name!r}")
        if "." in name:
            raise SecurityError(f"暂不支持子模块导入：{name!r}")
        if name not in self.imports:
            raise SecurityError(
                f"模块 {name!r} 不在本对话的导入白名单里（允许：{sorted(self.imports)}）"
            )
        if asname and asname.startswith("__"):
            raise SecurityError(f"禁止双下划线别名: {asname}")

    # ------------------------------------------------------------------
    # 属性 / 调用 / 名字
    # ------------------------------------------------------------------

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise SecurityError(f"禁止双下划线属性访问: {node.attr}")
        if node.attr in _BLOCKED_ATTRIBUTE_NAMES:
            raise SecurityError(f"禁止访问内省属性: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
            raise SecurityError(f"禁止调用 {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_CALL_NAMES:
            raise SecurityError(f"禁止调用 .{node.func.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        # guest 可以读 `__builtins__`（exec 机制需要它在 globals 里），但不能改写：
        # 换掉它等于换掉后续帧的 builtins。
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == "__builtins__":
            raise SecurityError("禁止改写 __builtins__")
        self.generic_visit(node)


def check_code(code: str, mode: str = "exec",
               allow_import: bool = False, imports=frozenset(),
               exports: Optional[Mapping[str, frozenset]] = None) -> None:
    """AST 检查：双下划线属性、内省属性、危险调用名；给 allow_import 时检查导入白名单。

    默认 allow_import=False：事件处理器、表达式求值等位置保持「禁 import」。
    """
    tree = ast.parse(code, mode=mode)
    _RestrictedVisitor(allow_import, imports, exports).visit(tree)


def safe_globals(extra: Optional[dict] = None, import_handler=None) -> dict:
    builtins_dict = make_safe_builtins()
    if import_handler is not None:
        builtins_dict["__import__"] = import_handler
    g = {"__builtins__": builtins_dict}
    if extra:
        g.update(extra)
    return g


def exec_safe(code: str, namespace: dict, filename: str = "<plm>",
              allow_import: bool = False, imports=frozenset(),
              exports: Optional[Mapping[str, frozenset]] = None) -> None:
    """受限 exec：AST 检查后执行（import 白名单参数透传给 check_code）。"""
    check_code(code, allow_import=allow_import, imports=imports, exports=exports)
    exec(compile(code, filename, "exec"), namespace)

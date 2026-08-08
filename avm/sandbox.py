"""受限执行环境：防止 PLM 代码访问宿主系统（文件、网络、进程、时钟、熵源）。

最佳努力沙箱：安全 builtins 白名单 + AST 检查（禁 import、禁双下划线属性、禁危险调用名）。
真正强隔离（子进程 + OS 级隔离）超出 VM 范围。
"""

import ast
import builtins


class SecurityError(Exception):
    """PLM 代码触发了受限环境禁止的操作"""


# 允许的内置名字（不含 open/input/eval/exec/compile/__import__/getattr/type 等逃逸面）
SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hash", "hex", "int", "len",
    "list", "map", "max", "min", "oct", "ord", "pow", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "print",
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
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


class _RestrictedVisitor(ast.NodeVisitor):
    def visit_Import(self, node):
        raise SecurityError("禁止 import 语句")

    def visit_ImportFrom(self, node):
        raise SecurityError("禁止 from ... import 语句")

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise SecurityError(f"禁止双下划线属性访问: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
            raise SecurityError(f"禁止调用 {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_CALL_NAMES:
            raise SecurityError(f"禁止调用 .{node.func.attr}")
        self.generic_visit(node)


def check_code(code: str, mode: str = "exec") -> None:
    """AST 检查：禁 import、禁双下划线属性、禁危险调用名。"""
    tree = ast.parse(code, mode=mode)
    _RestrictedVisitor().visit(tree)


def safe_globals(extra: dict = None) -> dict:
    g = {"__builtins__": make_safe_builtins()}
    if extra:
        g.update(extra)
    return g


def exec_safe(code: str, namespace: dict, filename: str = "<plm>") -> None:
    """受限 exec：AST 检查后执行。"""
    check_code(code)
    exec(compile(code, filename, "exec"), namespace)

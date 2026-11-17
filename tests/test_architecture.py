"""架构规则：服务层与 Core 的依赖方向（计划 plan-service-layer-2026-11-17）。

规则只检查 avm/ 内的 import 语句（含函数内 import）；外部依赖（openai 等）
单独检查。每条规则都在实施时做过"注入违规必须失败"的校准，结果见
report-service-layer-2026-11-17.md。
"""

import ast
from pathlib import Path

AVM = Path(__file__).resolve().parent.parent / "avm"


def _imports(path: Path):
    """返回 (全部相对导入的模块名, 顶层 import 的原始模块名)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative, top = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            relative.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                top.append(module)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            top.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top.append(alias.name.split(".")[0])
    return relative, top


def test_R1_service_modules_do_not_import_core():
    for name in ("llm", "llm_chat", "llm_responses", "llm_anthropic", "plm",
                 "executor", "services"):
        relative, _ = _imports(AVM / f"{name}.py")
        assert "core" not in relative, f"avm/{name}.py 不得 import core（含函数内）"


def test_R2_executor_does_not_import_llm_or_plm():
    relative, _ = _imports(AVM / "executor.py")
    assert "llm" not in relative and "plm" not in relative


def test_R2b_llm_and_plm_do_not_import_each_other():
    llm_rel, _ = _imports(AVM / "llm.py")
    plm_rel, _ = _imports(AVM / "plm.py")
    assert "plm" not in llm_rel, "avm/llm.py 不得 import plm"
    assert "llm" not in plm_rel, "avm/plm.py 不得 import llm"


def test_R2c_instructions_does_not_import_core():
    relative, _ = _imports(AVM / "instructions.py")
    assert "core" not in relative, "指令集（L3）不得反向 import Core"


def test_R3_core_top_level_has_no_protocol_or_service_impl():
    _, top = _imports(AVM / "core.py")
    forbidden = {"openai", "dotenv", "python_server", "llm", "plm"}
    assert not (forbidden & set(top)), f"core.py 顶层出现协议/服务实现 import：{sorted(forbidden & set(top))}"


def test_R4_l0_modules_do_not_import_openai():
    for name in ("types", "exceptions", "memory", "capabilities", "modules",
                 "sandbox", "memory_device"):
        relative, top = _imports(AVM / f"{name}.py")
        assert "openai" not in top and "openai" not in relative, f"L0 模块 {name}.py 依赖 openai"

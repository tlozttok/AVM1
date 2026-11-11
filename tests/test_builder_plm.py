"""base 镜像的构造器 PLM（`plm.builder`）测试。

覆盖：镜像装载（core.plms / 模型目录 / 预置 para 与程序节点）、端到端构造四类节点、
create 重复路径跳过、update 覆盖、非法规格不产生工具调用、calls 直通。
全程 Mock（构造器是确定性 Python 对话，不调 LLM）。
"""

import json
import os

import pytest

from avm.core import Core, _build_conversation
from avm.image import load_image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = os.path.join(REPO, "images", "base", "base.json")


class NoLMU:
    """构造器是 Python 对话；任何 LLM 调用都说明测试走错了路径。"""

    def exec(self, conv, para):
        raise AssertionError("builder 测试不应调用 LLM")


def load_core():
    core = load_image(Core(), BASE_IMAGE)
    core.lmu = NoLMU()
    # boot 在就绪队列里；这些用例只跑构造器对话，先清空，避免调度到 boot。
    core._ready_cids.clear()
    return core


def builder_conversation(core):
    init = core.get_conversation(0)
    assert init is not None, "base 镜像应创建 init 对话"
    return _build_conversation(
        core,
        "$MEM.sys.programs.builder",
        "$MEM.para.builder",
        parent=init,
        is_sub=False,
        space=init.caps.space(),
    )


def send(core, conv, spec):
    """投递一条规格，跑到构造器对话休眠；返回 (摘要, 回合数)。"""
    content = spec if isinstance(spec, str) else json.dumps(spec, ensure_ascii=False)
    conv.append_user_message(content)
    if conv.cid in core._dormant_cids:
        core._dormant_cids.remove(conv.cid)
    rounds = 0
    while conv.cid not in core._dormant_cids:
        if rounds >= 500:
            raise AssertionError("构造器对话没有在回合上限内休眠")
        core._active_cid = conv.cid
        core.advance_conversation()
        rounds += 1
    return json.loads(conv.messages[-1].content), rounds


CREATE_SPEC = {
    "mode": "create",
    "ops": [
        {
            "type": "settingup",
            "path": "$MEM.programs.hello",
            "name": "hello",
            "content": "print('hello')",
            "signature": "in -> out",
            "meta": "演示节点",
        },
        {
            "type": "python",
            "path": "$MEM.programs.py",
            "name": "py",
            "content": "def handle(cmd):\n    return cmd\n",
        },
        {
            "type": "handler",
            "path": "$MEM.handlers.h",
            "name": "h",
            "content": "def handle(event):\n    return {'action': 'allow'}\n",
            "condition": {"events": ["memory_write"], "phase": ["before"]},
        },
        {
            "type": "para",
            "path": "$MEM.para.builder_test",
            "fields": {"model": "plm.python", "temperature": 0.2},
        },
    ],
}


def test_image_declares_builder_plm_and_preset_nodes():
    core = load_core()
    cls = core.plms["plm.builder"]
    assert cls.__name__ == "NodeBuilderPLM"
    catalog = core.mem._data["model_catalog"]
    assert "plm.builder" in [entry["name"] for entry in catalog["plm"]]

    para = core.unwrap("$MEM.para.builder", for_llm=False)
    assert para.get_ctrl() == {"type": "para"}
    assert para.get("model") == "plm.builder"

    program = core.unwrap("$MEM.sys.programs.builder", for_llm=False)
    assert program.get_ctrl() == {"type": "python"}
    content = program.get("content")
    assert "构造器" in content and "JSON" in content
    assert program.get("para_ref") == "$MEM.para.builder"


def test_create_builds_all_node_kinds_end_to_end():
    core = load_core()
    conv = builder_conversation(core)
    summary, rounds = send(core, conv, CREATE_SPEC)

    assert rounds > 5, "构造器应该分多轮发工具调用"
    assert summary["ok"] is True
    assert summary["mode"] == "create"
    assert summary["errors"] == []
    assert summary["updated"] == []
    assert summary["skipped"] == []
    assert set(summary["built"]) == {
        "$MEM.programs.hello",
        "$MEM.programs.py",
        "$MEM.handlers.h",
        "$MEM.para.builder_test",
    }
    assert summary["verified"] == {
        "$MEM.programs.hello": {"expected": "settingup", "actual": "settingup", "ok": True},
        "$MEM.programs.py": {"expected": "python", "actual": "python", "ok": True},
        "$MEM.handlers.h": {"expected": "handler", "actual": "handler", "ok": True},
        "$MEM.para.builder_test": {"expected": "para", "actual": "para", "ok": True},
    }

    # 共享祖先 $MEM.programs 由第一个 op 建出，后续 op 不再重复 make。
    assert core.unwrap("$MEM.programs", for_llm=False) is not None

    hello = core.unwrap("$MEM.programs.hello", for_llm=False)
    assert hello.get_ctrl() == {"type": "settingup"}
    assert core.unwrap("$MEM.programs.hello.name", for_llm=False) == "hello"
    assert core.unwrap("$MEM.programs.hello.content", for_llm=False) == "print('hello')"
    assert core.unwrap("$MEM.programs.hello.signature", for_llm=False) == "in -> out"
    assert hello.get_metadata() == "演示节点"

    py = core.unwrap("$MEM.programs.py", for_llm=False)
    assert py.get_ctrl() == {"type": "python"}
    assert core.unwrap("$MEM.programs.py.name", for_llm=False) == "py"
    assert core.unwrap("$MEM.programs.py.content", for_llm=False).startswith("def handle")

    handler = core.unwrap("$MEM.handlers.h", for_llm=False)
    assert handler.get_ctrl() == {"type": "handler"}
    condition = json.loads(core.unwrap("$MEM.handlers.h.condition", for_llm=False))
    assert condition == {"events": ["memory_write"], "phase": ["before"]}
    assert core.unwrap("$MEM.handlers.h.name", for_llm=False) == "h"
    # condition 是合法 JSON，能被事件系统解析（节点在机制上可用）。
    parsed = core.events.parse_condition(0, "$MEM.handlers.h", kernel_mode=True)
    assert parsed["events"] == ["memory_write"]
    assert parsed["phases"] == ["before"]

    para = core.unwrap("$MEM.para.builder_test", for_llm=False)
    assert para.get_ctrl() == {"type": "para"}
    assert core.unwrap("$MEM.para.builder_test.model", for_llm=False) == "plm.python"
    assert core.unwrap("$MEM.para.builder_test.temperature", for_llm=False) == "0.2"

    # 构造出的 python 程序 + para 能真的建出对话：节点不只字段齐全，机制上也合法。
    init = core.get_conversation(0)
    child = _build_conversation(
        core, "$MEM.programs.py", "$MEM.para.builder_test",
        parent=init, is_sub=False, space=init.caps.space(),
    )
    assert child.is_python is True


def test_create_skips_existing_path_and_keeps_content():
    core = load_core()
    conv = builder_conversation(core)
    spec = {
        "ops": [
            {"type": "python", "path": "$MEM.workshop.py", "content": "v1"},
        ],
    }
    summary, _ = send(core, conv, spec)
    assert summary["ok"] is True
    assert summary["built"] == ["$MEM.workshop.py"]

    summary, _ = send(core, conv, spec)
    assert summary["ok"] is False
    assert summary["built"] == []
    assert summary["skipped"] == ["$MEM.workshop.py"]
    assert any("已存在" in error for error in summary["errors"])
    # create 不覆盖已有内容。
    assert core.unwrap("$MEM.workshop.py.content", for_llm=False) == "v1"


def test_update_rewrites_existing_content_and_fills_missing_field():
    core = load_core()
    conv = builder_conversation(core)
    create = {
        "ops": [
            {
                "type": "settingup",
                "path": "$MEM.workshop.hello",
                "name": "hello",
                "content": "v1",
            },
        ],
    }
    summary, _ = send(core, conv, create)
    assert summary["ok"] is True

    update = {
        "mode": "update",
        "ops": [
            {
                "type": "settingup",
                "path": "$MEM.workshop.hello",
                "content": "v2",
                "signature": "新增字段",
                "meta": "更新后的描述",
            },
        ],
    }
    summary, _ = send(core, conv, update)
    assert summary["ok"] is True
    assert summary["updated"] == ["$MEM.workshop.hello"]
    assert summary["errors"] == []
    assert core.unwrap("$MEM.workshop.hello.content", for_llm=False) == "v2"
    assert core.unwrap("$MEM.workshop.hello.signature", for_llm=False) == "新增字段"
    assert core.unwrap("$MEM.workshop.hello.name", for_llm=False) == "hello"
    assert core.unwrap("$MEM.workshop.hello", for_llm=False).get_metadata() == "更新后的描述"

    # update 缺失目标：记错误并跳过，不建节点。
    missing = {
        "mode": "update",
        "ops": [{"type": "para", "path": "$MEM.workshop.nope", "fields": {"model": "x"}}],
    }
    summary, _ = send(core, conv, missing)
    assert summary["ok"] is False
    assert summary["skipped"] == ["$MEM.workshop.nope"]
    assert any("不存在" in error for error in summary["errors"])


@pytest.mark.parametrize("bad", [
    "不是 JSON",
    {"mode": "remove", "ops": []},
    {"ops": [{"type": "nope", "path": "$MEM.x"}]},
    {"ops": [{"type": "python", "path": "$MEM.x"}]},
    {"ops": [{"type": "handler", "path": "$MEM.x", "content": "code"}]},
    {"calls": [{"tool": "not_an_instruction", "arguments": {}}]},
    {"calls": [{"tool": "memory_read", "arguments": "not-an-object"}]},
])
def test_invalid_spec_returns_error_without_tool_calls(bad):
    core = load_core()
    conv = builder_conversation(core)
    conv.append_user_message(bad if isinstance(bad, str) else json.dumps(bad, ensure_ascii=False))
    core._active_cid = conv.cid
    core.advance_conversation()

    # 一条用户消息 + 一条错误回执，没有 assistant 工具调用，也没有工具消息。
    assert [message.role for message in conv.messages] == ["system", "user", "assistant"]
    assert conv.messages[-1].content.startswith("错误: BuilderPLM:")
    assert conv.cid in core._dormant_cids
    # 规格非法时不执行任何指令：目标路径没有被建出来。
    assert "x" not in core.unwrap("$MEM.workshop", for_llm=False)


def test_calls_passthrough_and_summary_results():
    core = load_core()
    conv = builder_conversation(core)
    spec = {
        "calls": [
            {"tool": "memory_read", "arguments": {"ref": "$MEM.sys.power.state"}},
            {"tool": "memory_write",
             "arguments": {"ref": "$MEM.sys.outbox.-1", "content": "via-builder"}},
        ],
    }
    summary, _ = send(core, conv, spec)

    assert summary["ok"] is True
    assert [call["tool"] for call in summary["calls"]] == ["memory_read", "memory_write"]
    assert summary["calls"][0]["tool"] == "memory_read"
    assert summary["calls"][0]["arguments"] == {"ref": "$MEM.sys.power.state"}
    assert summary["calls"][0]["result"] == "on"
    assert summary["calls"][1]["tool"] == "memory_write"
    assert summary["calls"][1]["result"].startswith("Success")
    outbox = list(core.unwrap("$MEM.sys.outbox", for_llm=False))
    assert outbox[-1] == "via-builder"


def test_system_message_is_ignored():
    """程序节点的说明文字不是 JSON；system 消息必须忽略而不是报配置错误。"""
    core = load_core()
    plm = core.plms["plm.builder"]()
    history = [
        {"role": "system", "content": core.unwrap("$MEM.sys.programs.builder.content", for_llm=False)},
    ]
    assert plm.handle_messages(list(history), {}) == {"content": ""}

    spec = {"ops": [{"type": "para", "path": "$MEM.workshop.para", "fields": {"model": "x"}}]}
    history.append({"role": "user", "content": json.dumps(spec, ensure_ascii=False)})
    response = plm.handle_messages(list(history), {})
    assert response["tool_calls"][0]["function"]["name"] == "memory_read"

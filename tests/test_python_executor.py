"""Python 执行器接入 Core 的集成测试：创建计算器对话 → 发送表达式 → 返回结果"""

import pytest

from avm.core import Core
from avm.types import MetaDict


class MockLMU:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_index = 0

    def exec(self, conversation, para):
        if self.call_index >= len(self.responses):
            raise RuntimeError(f"MockLMU 第 {self.call_index} 次调用没有预设响应")
        resp = self.responses[self.call_index]
        self.call_index += 1
        return resp


def _batch(core, cid):
    conv = core._conv_by_cid[cid]
    return [(r.content, r.tool_call_id) for r in conv.user_batch.tool_responses]


def _setup():
    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "gpt-4o-mini"}, ctrl={"type": "para"})
    core.mem["calc_program"] = MetaDict(data={"name": "calc", "content": ""}, ctrl={"type": "python"})
    core.mem["calc_params"] = MetaDict(data={"model": "plm.simple"}, ctrl={"type": "para"})
    root = core.start("sys", "hello", "$MEM.model_params")
    root.name = "init"
    core._ready_cids.clear()
    core._active_cid = root.cid
    return core, root


def test_calc_driver_flow():
    core, root = _setup()
    core.lmu = MockLMU([
        (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.calc_program", "para_ref": "$MEM.calc_params"}}], None),
        (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "2 + 3"}}], None),
        ("done", [], None),
    ])

    # root: create_cmd → 计算器对话休眠（Python 执行器已接入），root 保持活跃
    core.advance_conversation()
    calc = core.get_conversation(1)
    assert calc.para_ref == "$MEM.calc_params"
    assert calc.messages[0].content == ""  # calc_program 节点的 content 作为系统提示词
    assert ("Success created: cid=1", "cc") in _batch(core, 0)
    assert core._active_cid == 0

    # root: send_instruction（简单投递）→ 计算器唤醒进就绪，root 收到确认、继续执行
    core.advance_conversation()
    assert core._active_cid == 0
    assert root.cid not in core._dormant_cids
    assert ("已投递到 calc", "si") in _batch(core, 0)
    assert calc.cid in core._ready_cids

    # root: 无工具收尾 → 休眠，计算器被调度
    core.advance_conversation()
    assert core._active_cid == 1
    assert root.cid in core._dormant_cids

    # 计算器（SimplePLM）把收到的内容直接当表达式求值：输出 5，无工具调用 → 交互结束休眠
    core.advance_conversation()
    assert 1 in core._executors  # PLM 执行器由 Core 按 cid 惰性创建并缓存
    assert core._active_cid is None  # 计算器输出结果后休眠，无返回、无唤醒
    assert 1 in core._dormant_cids

    # 没有 return_result：结果不自动回传给 root（root 的 batch 已随其收尾轮清空，无"5"投递）
    assert _batch(core, 0) == []


def test_llm_conversation_uses_core_lmu():
    """普通 LLM 对话仍走 core.lmu（executor 为 None）"""
    core, root = _setup()
    recorded = []

    class RecordingLMU(MockLMU):
        def exec(self, conversation, para):
            recorded.append((conversation.cid, para.get("model")))
            return "ok", [], conversation

    core.lmu = RecordingLMU()
    core.advance_conversation()
    assert recorded == [(0, "gpt-4o-mini")]  # root 用自己的 para_ref
    assert core._active_cid is None


def test_python_conversation_wrong_model_rejected_at_creation():
    """Python 程序的 model 不是 PLM 注册名：在创建时就报错，不静默回退到 lmu。

    2026-10-16 起这条走模型目录的命名空间校验（输入边界），不再等第一次推理。
    """
    core, root = _setup()
    core.mem["calc_params"] = MetaDict(data={"model": "not_a_plm"}, ctrl={"type": "para"})
    core.lmu = MockLMU([
        (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.calc_program", "para_ref": "$MEM.calc_params"}}], None),
    ])

    core.advance_conversation()  # root: create_cmd
    assert core.get_conversation(1) is None, "校验失败不得创建对话"
    (content, call_id), = _batch(core, root.cid)
    assert content.startswith("Error:") and "不是 PLM 注册名" in content
    assert call_id == "cc"


def test_python_conversation_model_changed_after_creation_records_frame():
    """漏网路径：创建后把 para 的 model 改成非法值，第一次推理仍会抛，但留监测帧。

    两个查找已移进 advance_conversation 的 try（问题 1 的位置修复）。
    """
    core, root = _setup()
    core.lmu = MockLMU([
        (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.calc_program", "para_ref": "$MEM.calc_params"}}], None),
    ])
    core.advance_conversation()  # root: create_cmd → 计算器就绪
    calc = core.get_conversation(1)
    assert calc.is_python is True
    # 创建时 para 是合法的 plm.simple；创建后把 model 改坏，执行器缓存还没建立
    core.mem.set("$MEM.calc_params.model", "not_a_plm")

    core.monitor.record_baseline(core)
    before = len(core.monitor.frames())
    core._active_cid = calc.cid
    with pytest.raises(ValueError, match="PLM_REGISTRY"):
        core.advance_conversation()
    frames = core.monitor.frames()
    assert len(frames) == before + 1, "错误应当留下监测帧"
    assert "PLM_REGISTRY" in frames[-1].lmu["error"]

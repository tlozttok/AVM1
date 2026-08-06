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
        (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "2 + 3", "wait": True}}], None),
        ("done", [], None),
    ])

    # root: create_cmd → 计算器对话休眠（Python 执行器已接入），root 保持活跃
    core.advance_conversation()
    calc = core.get_conversation(1)
    assert calc.para_ref == "$MEM.calc_params"
    assert calc.messages[0].content == ""  # calc_program 节点的 content 作为系统提示词
    assert ("Success created: cid=1", "cc") in _batch(core, 0)
    assert core._active_cid == 0

    # root: send_instruction(wait=true) → 计算器活跃，root 休眠
    core.advance_conversation()
    assert core._active_cid == 1
    assert root.cid in core._dormant_cids

    # 计算器（SimplePLM）求值 2+3=5 并 return_result → root 收到结果
    core.advance_conversation()
    assert 1 in core._executors  # PLM 执行器由 Core 按 cid 惰性创建并缓存
    assert ("5", "si") in _batch(core, 0)
    assert core._active_cid == 1  # 计算器还有确认回合

    # 计算器收尾（无工具）→ 休眠，root 被唤醒
    core.advance_conversation()
    assert core._active_cid == 0
    assert 1 in core._dormant_cids

    # root 收尾
    core.advance_conversation()
    assert core._active_cid is None


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


def test_python_conversation_wrong_model_raises():
    """Python 对话的 model 不在 PLM_REGISTRY 时报错，不静默回退到 lmu"""
    core, root = _setup()
    core.mem["calc_params"] = MetaDict(data={"model": "not_a_plm"}, ctrl={"type": "para"})
    core.lmu = MockLMU([
        (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.calc_program", "para_ref": "$MEM.calc_params"}}], None),
        (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "2 + 3", "wait": True}}], None),
    ])

    core.advance_conversation()  # root: create_cmd
    calc = core.get_conversation(1)
    assert calc.is_python is True
    core.advance_conversation()  # root: send_instruction → 计算器就绪
    with pytest.raises(ValueError, match="PLM_REGISTRY"):
        core.advance_conversation()  # 计算器首轮执行 → model 错误 → 报错

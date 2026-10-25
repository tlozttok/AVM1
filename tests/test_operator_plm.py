from avm.exceptions import MemoryKeyNotFoundError
"""操作员执行器测试

覆盖：动作往返（工具调用 → Core 执行 → 工具响应回到执行器）、结束动作让出执行权、
未注入通道时拒绝运行、形状不合法的动作报回控制台后继续等、未知工具名原样透传、
以及面板对话在调度里是一个普通参与者。
"""

import json
import queue

import pytest

from avm.core import Core
from avm.python_server import OperatorPLM, PLM_REGISTRY, create_plm


def _seed(mem, ref, value, owner_cid=None):
    """测试夹具：先 make 建链再写（memory_write 不再建节点）。"""
    try:
        return mem.set(ref, value, owner_cid=owner_cid)
    except MemoryKeyNotFoundError:
        parent, _, key = ref.rpartition(".")
        mem.make(parent, key, "str", owner_cid=owner_cid)
        return mem.set(ref, value, owner_cid=owner_cid)


class RecordingChannel:
    """最小的操作员通道：plm.operator 只需要 wait_action 与 publish。"""

    def __init__(self, actions=()):
        self._actions = queue.Queue()
        self.events = []
        for action in actions:
            self._actions.put(action)

    def wait_action(self):
        return self._actions.get()

    def submit(self, action):
        self._actions.put(action)

    def publish(self, kind, data):
        self.events.append((kind, data))

    def kinds(self):
        return [k for k, _ in self.events]

    def of(self, kind):
        return [d for k, d in self.events if k == kind]


SYSTEM_ONLY = [{"role": "system", "content": "操作员对话"}]


class TestRegistry:
    def test_registry_has_operator(self):
        assert PLM_REGISTRY["plm.operator"] is OperatorPLM

    def test_create_plm_forwards_kwargs(self):
        channel = RecordingChannel()
        plm = create_plm("plm.operator", channel=channel)
        assert isinstance(plm, OperatorPLM)

    def test_create_plm_still_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="未知 PLM 类型"):
            create_plm("plm.nope")

    def test_other_models_still_construct_without_kwargs(self):
        for name in ("plm.python", "plm.function", "plm.machine", "plm.command"):
            assert create_plm(name) is not None


class TestChannelRequired:
    def test_without_channel_returns_generic_error_instead_of_crashing(self, capsys):
        """通道缺失不该杀内核；对话只看到通用拒绝，宿主在 stderr 看到真实原因。"""
        plm = create_plm("plm.operator")
        resp = plm.handle_messages(SYSTEM_ONLY, {})
        assert resp == {"content": "错误: 操作被外部原因拒绝"}
        err = capsys.readouterr().err
        assert "操作员对话必须由控制台创建" in err
        assert "操作被外部原因拒绝" not in err


class TestResponseShape:
    def test_tool_action_becomes_a_tool_call(self):
        channel = RecordingChannel([{"tool": "memory_read", "arguments": {"ref": "$MEM.a"}}])
        plm = create_plm("plm.operator", channel=channel)
        resp = plm.handle_messages(SYSTEM_ONLY, {})
        assert resp["content"] == ""
        assert len(resp["tool_calls"]) == 1
        call = resp["tool_calls"][0]
        assert call["type"] == "function"
        assert call["function"]["name"] == "memory_read"
        assert json.loads(call["function"]["arguments"]) == {"ref": "$MEM.a"}
        assert channel.kinds() == ["tool_call"]

    def test_call_ids_are_unique(self):
        channel = RecordingChannel([
            {"tool": "memory_read", "arguments": {"ref": "$MEM.a"}},
            {"tool": "memory_read", "arguments": {"ref": "$MEM.b"}},
        ])
        plm = create_plm("plm.operator", channel=channel)
        first = plm.handle_messages(SYSTEM_ONLY, {})["tool_calls"][0]["id"]
        second = plm.handle_messages(SYSTEM_ONLY, {})["tool_calls"][0]["id"]
        assert first != second

    def test_end_action_returns_no_tool_calls(self):
        channel = RecordingChannel([{"end": True}])
        plm = create_plm("plm.operator", channel=channel)
        resp = plm.handle_messages(SYSTEM_ONLY, {})
        assert not resp.get("tool_calls")
        assert channel.kinds() == []

    def test_missing_arguments_default_to_empty_object(self):
        channel = RecordingChannel([{"tool": "memory_read"}])
        plm = create_plm("plm.operator", channel=channel)
        call = plm.handle_messages(SYSTEM_ONLY, {})["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == {}

    def test_unknown_tool_name_passes_through(self):
        """面板要能造出模型造得出的输入，包括未知工具名，以便观察 Core 的错误路径。"""
        channel = RecordingChannel([{"tool": "no_such_tool", "arguments": {}}])
        plm = create_plm("plm.operator", channel=channel)
        call = plm.handle_messages(SYSTEM_ONLY, {})["tool_calls"][0]
        assert call["function"]["name"] == "no_such_tool"


class TestShapeErrors:
    def _run(self, actions):
        channel = RecordingChannel(list(actions) + [{"end": True}])
        plm = create_plm("plm.operator", channel=channel)
        return plm.handle_messages(SYSTEM_ONLY, {}), channel

    def test_non_object_action_is_reported_and_skipped(self):
        resp, channel = self._run(["不是对象"])
        assert [d["reason"] for d in channel.of("action_error")] == [
            "动作必须是对象，得到 str"]
        assert not resp.get("tool_calls")

    def test_missing_tool_is_reported(self):
        resp, channel = self._run([{"arguments": {}}])
        assert channel.of("action_error")[0]["reason"] == "动作缺少字符串字段 tool"

    def test_empty_tool_name_is_reported(self):
        resp, channel = self._run([{"tool": ""}])
        assert channel.of("action_error")[0]["reason"] == "动作缺少字符串字段 tool"

    def test_non_object_arguments_is_reported(self):
        resp, channel = self._run([{"tool": "memory_read", "arguments": []}])
        assert channel.of("action_error")[0]["reason"] == "arguments 必须是对象"

    def test_bad_actions_do_not_end_the_conversation(self):
        """形状错误不能当成结束动作：那会静默地把执行权让回去。"""
        resp, channel = self._run([{}, {"tool": ""}, "x",
                                   {"tool": "memory_read", "arguments": {}}])
        assert len(channel.of("action_error")) == 3
        assert resp["tool_calls"][0]["function"]["name"] == "memory_read"


class TestThroughCore:
    def _panel(self, actions=()):
        """用真实路径激活面板：注册向量、挂起中断、派发。"""
        core = Core()
        channel = RecordingChannel(actions)
        panel = core.start_operator_conversation(channel)
        core.interrupts.register("operator", panel.cid)
        core.interrupts.raise_("operator")
        core.interrupts.dispatch()
        assert core._active_cid == panel.cid
        return core, panel, channel

    def test_panel_conversation_is_an_ordinary_participant(self):
        core, panel, channel = self._panel()
        assert core._conv_by_cid[panel.cid] is panel
        assert panel.is_python is True
        assert panel.is_sub is False
        assert panel.is_root is False
        assert panel.name is None, "面板对话不做特殊标记"
        assert core._executors[panel.cid].plm._channel is channel

    def test_panel_does_not_preempt_on_its_own(self):
        core = Core()
        channel = RecordingChannel()
        panel = core.start_operator_conversation(channel)
        assert panel.cid in core._dormant_cids
        assert panel.cid not in core._ready_cids

    def test_panel_cannot_be_checkpointed(self):
        core, panel, channel = self._panel()
        with pytest.raises(ValueError):
            core.save_conversation(panel.cid, "/tmp/avm-operator-checkpoint.json")

    def test_instruction_runs_and_result_returns_to_operator(self):
        core, panel, channel = self._panel([
            {"tool": "memory_write", "arguments": {"ref": "$MEM.x", "content": "hi"}},
            {"end": True},
        ])
        _seed(core.mem, "$MEM.x", "")
        core.advance_conversation()
        assert core.mem["x"] == "hi"
        assert core._active_cid == panel.cid, "有工具返回时对话保持活跃"
        assert core._activation_instructions == 1

        core.advance_conversation()
        assert core._active_cid is None, "结束动作让出执行权"
        assert panel.cid in core._dormant_cids

        assert channel.kinds() == ["tool_call", "tool_result"]
        assert "$MEM.x" in channel.of("tool_result")[0]["content"]

    def test_action_enters_the_event_stream_under_panel_identity(self):
        core, panel, channel = self._panel([
            {"tool": "memory_write", "arguments": {"ref": "$MEM.x", "content": "hi"}},
            {"end": True},
        ])
        _seed(core.mem, "$MEM.x", "")
        core.advance_conversation()
        writes = [e for e in core._turn_events if e["event_type"] == "memory_write"]
        assert [e["phase"] for e in writes] == ["before", "after"]
        assert writes[0]["source"] == {"kind": "core", "initiator_cid": panel.cid}
        assert writes[1]["source"] == {"kind": "memory"}
        assert writes[1]["target"] == "$MEM.x"

    def test_unknown_tool_name_is_reported_to_the_runner(self, capsys):
        """面板能造出模型造得出的输入，包括未知工具名。

        两条解析路径共用一个出口：未知名字在 `cmd_type not in registry` 分支报给
        运行者（stderr），对话侧只回一条中性占位——配对不破，错误细节不外传。
        """
        core, panel, channel = self._panel([
            {"tool": "no_such_tool", "arguments": {"a": 1}},
            {"end": True},
        ])
        core.advance_conversation()
        assert core._active_cid == panel.cid, "有工具调用，对话保持活跃"
        assert channel.kinds() == ["tool_call"], "本轮没有工具响应回到操作员手里"
        assert [(r.content, r.tool_call_id) for r in panel.user_batch.tool_responses] == [
            ("（该工具调用未执行）", "op-1")
        ]
        err = capsys.readouterr().err
        assert f"对话 {panel.cid}: 未知工具 no_such_tool" in err
        assert '{"a": 1}' in err

    def test_panel_history_records_the_operator_turns(self):
        core, panel, channel = self._panel([
            {"tool": "memory_read", "arguments": {"ref": "$MEM.x"}},
            {"end": True},
        ])
        core.advance_conversation()
        core.advance_conversation()
        roles = [m.role for m in panel.messages]
        assert roles == ["system", "assistant", "tool", "assistant"]

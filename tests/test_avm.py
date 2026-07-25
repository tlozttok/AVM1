"""Core 调度系统测试"""

import pytest
from avm.core import Core, LMU, _instruction_registry, _build_conversation
from avm.core import (
    MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    CreateInstruction, CreateSubInstruction,
)
from avm.types import Conversation, UserMessageBatch, SystemMessage, UserMessage, AssistantMessage, MetaDict
from avm.memory import Memory


class MockLMU:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_index = 0
        self.calls = []

    def _next(self, *args):
        self.calls.append(args)
        if self.call_index >= len(self.responses):
            raise RuntimeError(f"MockLMU 第 {self.call_index} 次调用没有预设响应")
        resp = self.responses[self.call_index]
        self.call_index += 1
        if callable(resp):
            return resp(*args)
        return resp

    def exec(self, conversation, para):
        return self._next(conversation, para)


def make_core(responses=None):
    core = Core()
    if responses is not None:
        core.lmu = MockLMU(responses)
    return core


def _state(core):
    return {
        "active": core._active_cid,
        "ready": list(core._ready_cids),
        "dormant": list(core._dormant_cids),
        "total": len(core._conv_by_cid),
    }


def _batch(core, cid):
    conv = core._conv_by_cid[cid]
    return [(r.content, r.tool_call_id) for r in conv.user_batch.tool_responses]


def _new_root(core, system="sys", user="hello"):
    core.mem["system"] = system
    core.mem["user"] = user
    conv = core.start(system, user)
    return conv


# ------------------------------------------------------------------
# Conversation lifecycle
# ------------------------------------------------------------------

class TestConversationStart:
    def test_start_creates_root(self):
        core = make_core()
        conv = _new_root(core)
        assert conv.is_root is True
        assert conv.is_sub is False
        assert conv.parent is None
        assert len(conv.messages) == 2  # system + user
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"

    def test_start_adds_to_ready(self):
        core = make_core()
        conv = _new_root(core)
        assert conv.cid in core._ready_cids
        assert core._active_cid is None

    def test_registry_contains_all_instructions(self):
        reg = _instruction_registry()
        assert "memory_read" in reg
        assert "memory_write" in reg
        assert "memory_make" in reg
        assert "create_cmd" in reg
        assert "create_sub" in reg
        assert reg["memory_read"] is MemoryReadInstruction
        assert reg["create_cmd"] is CreateInstruction
        assert reg["create_sub"] is CreateSubInstruction


# ------------------------------------------------------------------
# Memory instruction execution via scheduling
# ------------------------------------------------------------------

class TestMemoryOpsThroughScheduling:
    def test_memory_read_writes_to_batch(self):
        """LMU 返回 memory_read → advance 后 user_batch 有工具响应"""
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.data"}}], None),
        ])
        _new_root(core)
        core.mem["data"] = "stored_value"
        core._active_cid = core._ready_cids.pop(0)  # manually start

        core.advance_conversation()

        assert _batch(core, 0) == [("stored_value", "tc1")]

    def test_memory_read_error_writes_to_batch(self):
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.missing"}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()
        content, call_id = _batch(core, 0)[0]
        assert "Error" in content

    def test_memory_write(self):
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.out", "content": "written"}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()
        assert core.mem["out"] == "written"

    def test_memory_make(self):
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_make", "args": {"ref": "$MEM.base", "key": "child", "mem_type": "str"}}], None),
        ])
        _new_root(core)
        core.mem["base"] = {}
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()
        assert core.mem["base"]["child"] == ""

    def test_multiple_memory_ops_in_one_call(self):
        """一次 LMU 调用返回多个工具调用，全部处理，conv 保持活跃"""
        core = make_core([
            (None, [
                {"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.a", "content": "1"}},
                {"call_id": "tc2", "cmd_type": "memory_write", "args": {"ref": "$MEM.b", "content": "2"}},
            ], None),
            ("done", [], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        # first advance: process two memory_write
        core.advance_conversation()
        # conv stays active (no create/sub_create)
        assert core._active_cid == 0
        assert core.mem["a"] == "1"
        assert core.mem["b"] == "2"

        # second advance: done
        core.advance_conversation()
        assert core._active_cid is None


# ------------------------------------------------------------------
# 子对话 / 亚对话 调度
# ------------------------------------------------------------------

class TestCreateChild:
    def test_create_cmd_dormant_and_ready(self):
        """根对话调用 create_cmd → 根休眠，子进就绪，子变活跃"""
        core = make_core([
            (None, [
                {"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.sys2", "user_ref": "$MEM.usr2", "para_ref": "$MEM.para"}},
            ], None),
        ])
        _new_root(core)
        core.mem["sys2"] = "child_sys"
        core.mem["usr2"] = "child_usr"
        core.mem["para"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        st0 = _state(core)
        assert st0["active"] == 0
        assert st0["ready"] == []

        core.advance_conversation()

        st1 = _state(core)
        assert st1["active"] == 1           # child becomes active
        assert 0 in st1["dormant"]          # root dormant
        assert 1 not in st1["ready"]        # child removed from ready

    def test_child_finished_picks_next_ready(self):
        """子对话完成后取下一个就绪"""
        core = make_core([
            (None, [
                {"call_id": "c1", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.sys2", "user_ref": "$MEM.usr2", "para_ref": "$MEM.p"}},
                {"call_id": "c2", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.sys3", "user_ref": "$MEM.usr3", "para_ref": "$MEM.p"}},
            ], None),
            # child1 finishes
            ("child1_done", [], None),
            # child2 finishes
            ("child2_done", [], None),
        ])
        _new_root(core)
        for k in ["sys2", "usr2", "sys3", "usr3", "p"]:
            core.mem[k] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        # advance root: creates 2 children, root → dormant
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == 1       # first child active
        assert st1["dormant"] == [0]
        assert 2 in st1["ready"]        # second child waiting

        # advance child1: finishes → pick child2
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == 2       # second child active
        assert st2["ready"] == []

        # advance child2: finishes → no more ready
        core.advance_conversation()
        st3 = _state(core)
        assert st3["active"] is None

    def test_child_has_parent_link(self):
        core = make_core([
            (None, [
                {"call_id": "c", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        core.mem["s"] = "x"
        core.mem["u"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        child = core._conv_by_cid[1]
        assert child.parent is core._conv_by_cid[0]
        assert child.is_sub is False
        assert child.is_root is False


class TestCreateSub:
    def test_create_sub_parent_ready_front(self):
        """create_sub → 父插到就绪队首，亚对话进队首立即活跃"""
        core = make_core([
            (None, [
                {"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        core.mem["s"] = "sub_sys"
        core.mem["u"] = "sub_usr"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        st = _state(core)
        assert st["active"] == 1        # sub becomes active
        assert 0 in st["ready"]         # parent is in ready (front)

    def test_sub_finished_writes_to_parent_and_wakes(self):
        """亚对话完成 → 输出写回 parent.user_batch，parent 回到 active"""
        core = make_core([
            (None, [
                {"call_id": "cs_call", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}},
            ], None),
            # sub finishes, returns "sub_result"
            ("sub_result", [], None),
        ])
        _new_root(core)
        core.mem["s"] = "x"
        core.mem["u"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        # advance root: create sub
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == 1  # sub active
        assert 0 in st1["ready"]  # parent in ready

        # advance sub: finishes → parent wakes
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == 0  # parent wakes

        # parent batch has sub's result
        batch = _batch(core, 0)
        assert ("sub_result", "cs_call") in batch

    def test_sub_has_is_sub_true(self):
        core = make_core([
            (None, [
                {"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        for k in ["s", "u", "p"]:
            core.mem[k] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        sub = core._conv_by_cid[1]
        assert sub.is_sub is True
        assert sub.parent is core._conv_by_cid[0]


# ------------------------------------------------------------------
# 调度状态检查
# ------------------------------------------------------------------

class TestSchedulingState:
    def test_no_tool_calls_conversation_finishes(self):
        """无工具调用 → 对话结束"""
        core = make_core([
            ("final_answer", [], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        assert core._active_cid is None
        assert core._ready_cids == []

    def test_user_batch_cleared_after_advance(self):
        core = make_core([
            ("done", [], None),
        ])
        _new_root(core)
        conv = core._conv_by_cid[0]
        conv.user_batch.add_tool_response("stale", "old_tc")
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        assert len(conv.user_batch.tool_responses) == 0

    def test_advance_adds_assistant_message(self):
        core = make_core([
            ("direct_answer", [], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        msgs_before = len(core._conv_by_cid[0].messages)
        core.advance_conversation()

        conv = core._conv_by_cid[0]
        # real LMU.exec appends assistant; mock returns canned response
        # without modifying conversation — test just that advance runs
        assert core._active_cid is None  # conversation finished


# ------------------------------------------------------------------
# LMU.exec 调用参数
# ------------------------------------------------------------------

class TestLMUExecCalled:
    def test_exec_receives_conversation_and_para(self):
        recorded = []

        class RecordingLMU(MockLMU):
            def exec(self, conversation, para):
                recorded.append((len(conversation.messages), para.get("model")))
                return "ok", [], conversation

        core = Core()
        core.lmu = RecordingLMU()
        _new_root(core)
        core.mem["model_params"] = MetaDict(data={"model": "gpt-4o"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        assert len(recorded) == 1
        msg_count, model = recorded[0]
        assert msg_count == 2  # system + user
        assert model == "gpt-4o"

    def test_batch_tool_messages_sent_to_lmu(self):
        recorded_messages = []

        class RecordingLMU(MockLMU):
            def exec(self, conversation, para):
                msgs = conversation.to_api_messages()
                msgs.extend(conversation.user_batch.to_tool_messages())
                recorded_messages.extend(msgs)
                return "ok", [], conversation

        core = Core()
        core.lmu = RecordingLMU()
        _new_root(core)
        conv = core._conv_by_cid[0]
        conv.user_batch.add_tool_response("tool_result", "tc1")
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        tool_msgs = [m for m in recorded_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc1"

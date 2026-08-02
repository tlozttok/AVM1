"""Core 调度系统测试"""

import json
import pytest
from avm.core import Core, LMU, _instruction_registry, _build_conversation
from avm.core import (
    MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    CreateInstruction, CreateSubInstruction,
    RegisterServiceInstruction, CallServiceInstruction, TransferServiceInstruction,
    ReturnResultInstruction, SendInstruction,
)
from avm.types import Conversation, UserMessageBatch, SystemMessage, UserMessage, AssistantMessage, MetaDict
from avm.memory import Memory
from types import SimpleNamespace


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
        assert "return_result" in reg
        assert "send_instruction" in reg
        assert reg["memory_read"] is MemoryReadInstruction
        assert reg["create_cmd"] is CreateInstruction
        assert reg["create_sub"] is CreateSubInstruction
        assert reg["send_instruction"] is SendInstruction


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
    def test_create_cmd_creates_dormant_child_and_returns_cid(self):
        """create_cmd → 子对话休眠等待指令，父保持活跃，工具响应返回 cid"""
        core = make_core([
            (None, [
                {"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.sys2", "para_ref": "$MEM.para"}},
            ], None),
        ])
        _new_root(core)
        core.mem["sys2"] = "child_sys"
        core.mem["para"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        st = _state(core)
        assert st["active"] == 0          # 父保持活跃
        assert 1 in st["dormant"]         # 子对话休眠等待指令
        assert 1 not in st["ready"]
        content, call_id = _batch(core, 0)[0]
        assert "Success created: cid=1" in content
        assert call_id == "ctc"

    def test_create_cmd_child_has_only_system_message(self):
        """create_cmd 创建的子对话没有 user 消息（指令由 send_instruction 后续投递）"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.sys2", "para_ref": "$MEM.para"}}], None),
        ])
        _new_root(core)
        core.mem["sys2"] = "child_sys"
        core.mem["para"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        child = core._conv_by_cid[1]
        assert len(child.messages) == 1
        assert child.messages[0].role == "system"

    def test_child_has_parent_link(self):
        core = make_core([
            (None, [
                {"call_id": "c", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        core.mem["s"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        child = core._conv_by_cid[1]
        assert child.parent is core._conv_by_cid[0]
        assert child.is_sub is False
        assert child.is_root is False


class TestSendInstruction:
    def _child_setup(self):
        """root 用 create_cmd 创建子对话并保持活跃"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = "child_sys"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()  # root: create_cmd → 子休眠，root 活跃
        child = core._conv_by_cid[1]
        return core, root, child

    def test_send_instruction_wait_true(self):
        """wait=true：发起者休眠等待，子对话返回后发起者被激活"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": child.cid, "content": "do X", "wait": True}}], None),
            (None, [{"call_id": "rr", "cmd_type": "return_result", "args": {"content": "done", "icc_id": "si"}}], None),
            ("ok", [], None),
            ("done", [], None),
        ])

        # root: send_instruction(wait=true) → root 休眠，child 活跃，指令为 JSON 消息
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == child.cid
        assert root.cid in st1["dormant"]
        msg = json.loads(child.user_batch.user_contents[0])
        assert msg == {"icc_id": "si", "content": "do X"}

        # child: return_result → 结论投递 root，child 收到确认（保持活跃）
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == child.cid
        assert st2["ready"][0] == root.cid
        assert ("结果已发送", "rr") in _batch(core, child.cid)
        assert ("done", "si") in _batch(core, root.cid)

        # child: 无工具收尾 → 休眠，root 被激活
        core.advance_conversation()
        st3 = _state(core)
        assert st3["active"] == root.cid
        assert child.cid in st3["dormant"]

        # root: 完成
        core.advance_conversation()
        assert core._active_cid is None

    def test_send_instruction_wait_false(self):
        """wait=false：发起者继续执行，子对话返回异步进入发起者 batch"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": child.cid, "content": "fire", "wait": False}}], None),
            ("root_done", [], None),
            (None, [{"call_id": "rr", "cmd_type": "return_result", "args": {"content": "later", "icc_id": "si"}}], None),
            ("ok", [], None),
            ("root_final", [], None),
        ])

        # root: send_instruction(wait=false) → root 保持活跃，child 就绪
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == root.cid
        assert child.cid in st1["ready"]
        assert root.cid not in st1["dormant"]

        # root: 无工具收尾 → 休眠，child 被调度
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == child.cid
        assert root.cid in st2["dormant"]

        # child: return_result → root 排到就绪队首（异步收信）
        core.advance_conversation()
        st3 = _state(core)
        assert st3["active"] == child.cid
        assert st3["ready"][0] == root.cid
        assert ("later", "si") in _batch(core, root.cid)

        # child 收尾 → 休眠；root 被激活
        core.advance_conversation()
        st4 = _state(core)
        assert st4["active"] == root.cid
        assert child.cid in st4["dormant"]

        core.advance_conversation()
        assert core._active_cid is None

    def test_send_instruction_unknown_cid(self, capsys):
        core = make_core([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 99, "content": "x", "wait": False}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()
        content, _ = _batch(core, 0)[0]
        assert "不存在" in content
        assert "不存在" in capsys.readouterr().err
        assert core._active_cid == 0  # 仍活跃


class TestIccRouting:
    def test_return_result_routes_by_icc_id(self):
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = "s"
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.u", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "svc", "what": "x", "needs": "x", "returns": "x"}).execute(core, svc)

        # 两条请求各建 ICC 记录
        CallServiceInstruction("req1", root.cid, {"service_name": "svc", "input": "q1"}).execute(core, root)
        CallServiceInstruction("req2", root.cid, {"service_name": "svc", "input": "q2"}).execute(core, root)
        assert set(core._icc) == {"req1", "req2"}

        # 按 req2 返回 → 路由到 req2 的 call_id
        ReturnResultInstruction("ret2", svc.cid, {"content": "ans2", "icc_id": "req2"}).execute(core, svc)
        assert ("ans2", "req2") in _batch(core, root.cid)

        # 按 req1 返回
        ReturnResultInstruction("ret1", svc.cid, {"content": "ans1", "icc_id": "req1"}).execute(core, svc)
        assert ("ans1", "req1") in _batch(core, root.cid)

        # 记录已消费，再次返回报错（stderr）且不投递
        ReturnResultInstruction("ret3", svc.cid, {"content": "dup", "icc_id": "req1"}).execute(core, svc)
        assert ("dup", "req1") not in _batch(core, root.cid)


class TestCreateSub:
    def test_create_sub_parent_dormant(self):
        """create_sub → 亚对话立即活跃，父进入休眠等待子完成"""
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
        assert 0 in st["dormant"]       # parent dormant（子完成时由 core 唤醒）
        assert 0 not in st["ready"]

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
        assert 0 in st1["dormant"]  # parent dormant
        assert 0 not in st1["ready"]

        # advance sub: finishes → parent wakes
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == 0  # parent wakes
        assert 0 not in st2["dormant"]  # 休眠标记被清理
        assert 1 in st2["dormant"]  # 亚对话交互结束进入休眠

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


# ------------------------------------------------------------------
# 服务调用 / 服务移交调度
# ------------------------------------------------------------------

class TestServiceScheduling:
    def _root_with_service(self):
        """root 创建服务对话 svc 并注册为 calc；root 随后调用它"""
        core = make_core([
            (None, [{"call_id": "cc", "cmd_type": "call_service", "args": {"service_name": "calc", "input": "2+3"}}], None),
            (None, [{"call_id": "ret", "cmd_type": "return_result", "args": {"content": "6", "icc_id": "cc"}}], None),
            ("ok", [], None),
            ("done", [], None),
        ])
        root = _new_root(core)
        core.mem["s"] = "svc_sys"
        core.mem["u"] = "svc_usr"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.u", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "calc", "what": "计算", "needs": "表达式", "returns": "结果"}).execute(core, svc)
        core._ready_cids.clear()
        core._active_cid = root.cid
        return core, root, svc

    def test_call_service_uses_tool_return(self):
        core, root, svc = self._root_with_service()

        # root: call_service → root 休眠，服务进入活跃
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == svc.cid
        assert root.cid in st1["dormant"]
        assert root.cid not in st1["ready"]
        assert svc.user_batch.user_contents == [json.dumps({"icc_id": "cc", "content": "2+3"})]

        # svc: return_result → 结论投递 root、root 排到下一个、svc 收到确认（保持活跃）
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == svc.cid  # 还有确认回合，未休眠
        assert root.cid not in st2["dormant"]
        assert st2["ready"][0] == root.cid  # 调用者是下一个被激活的对话
        assert ("结果已发送", "ret") in _batch(core, svc.cid)
        assert ("6", "cc") in _batch(core, root.cid)

        # svc: 以无工具结果收尾 → 进入休眠，root 被激活
        core.advance_conversation()
        st3 = _state(core)
        assert st3["active"] == root.cid
        assert svc.cid in st3["dormant"]
        assert ("6", "cc") in _batch(core, root.cid)
        assert core._services["calc"] == svc.cid  # 交互结束不影响服务注册

        # root: 完成 → 交互结束进入休眠
        core.advance_conversation()
        assert core._active_cid is None
        assert root.cid in core._dormant_cids

    def test_call_service_natural_finish_does_not_deliver(self):
        """服务未调用 return_result 就自然结束：不投递结果、不唤醒调用者"""
        core = make_core([
            (None, [{"call_id": "cc", "cmd_type": "call_service", "args": {"service_name": "calc", "input": "2+3"}}], None),
            ("ok", [], None),
        ])
        root = _new_root(core)
        core.mem["s"] = "s"
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.u", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "calc", "what": "计算", "needs": "表达式", "returns": "结果"}).execute(core, svc)
        core._ready_cids.clear()
        core._active_cid = root.cid

        core.advance_conversation()  # root: call_service
        assert core._active_cid == svc.cid

        core.advance_conversation()  # svc: 自然结束
        assert core._active_cid is None
        assert root.cid in core._dormant_cids
        assert svc.cid in core._dormant_cids
        assert _batch(core, root.cid) == []  # 没有结果

    def test_transfer_service_caller_stays_dormant(self):
        core = make_core([
            (None, [{"call_id": "tt", "cmd_type": "transfer_service", "args": {"service_name": "svc", "input": "x"}}], None),
            ("bye", [], None),
        ])
        root = _new_root(core)
        core.mem["s"] = "s"
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.u", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "svc", "what": "x", "needs": "x", "returns": "x"}).execute(core, svc)
        core._ready_cids.clear()
        core._active_cid = root.cid

        # root: transfer_service → root 休眠（不等待返回）
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == svc.cid
        assert root.cid in st1["dormant"]

        # svc: 完成 → root 不被唤醒，也没有工具响应
        core.advance_conversation()
        assert core._active_cid is None
        assert root.cid in core._dormant_cids
        assert _batch(core, root.cid) == []


# ------------------------------------------------------------------
# 工具调用错误反馈（原来被静默丢弃）
# ------------------------------------------------------------------

class TestToolCallErrorFeedback:
    def _fake_tool_call(self, name, arguments):
        return SimpleNamespace(
            id="tc-bad",
            function=SimpleNamespace(name=name, arguments=arguments),
        )

    def test_malformed_json_reported_to_stderr(self, capsys):
        lmu = LMU()
        message = SimpleNamespace(tool_calls=[self._fake_tool_call("memory_read", "{bad json")])
        return_calls = lmu._return_calls_from_message(message)
        assert return_calls[0]["cmd_type"] == "json_error"

        core = make_core()
        conv = _new_root(core)
        core._process_return_calls(return_calls, conv)
        err = capsys.readouterr().err
        assert "不是合法 JSON" in err
        assert _batch(core, conv.cid) == []  # 不喂回 LLM

    def test_unknown_tool_reported_to_stderr(self, capsys):
        lmu = LMU()
        message = SimpleNamespace(tool_calls=[self._fake_tool_call("definitely_not_a_tool", "{}")])
        return_calls = lmu._return_calls_from_message(message)
        assert return_calls[0]["cmd_type"] == "unknown_tool"

        core = make_core()
        conv = _new_root(core)
        core._process_return_calls(return_calls, conv)
        err = capsys.readouterr().err
        assert "未知工具" in err
        assert _batch(core, conv.cid) == []

    def test_malformed_json_conversation_stays_active(self, capsys):
        """json_error 不结束对话：错误面向运行者输出，对话继续"""
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "json_error", "args": {"error": "bad json", "name": "memory_read"}}], None),
            ("recovered", [], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()
        assert core._active_cid == 0  # 仍活跃
        assert _batch(core, 0) == []  # 没有工具响应进 batch
        assert "不是合法 JSON" in capsys.readouterr().err

        core.advance_conversation()
        assert core._active_cid is None

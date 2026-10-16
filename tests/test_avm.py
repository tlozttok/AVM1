"""Core 调度系统测试"""

import json
import pytest
from avm.core import Core, LMU, _instruction_registry, _build_conversation
from avm.core import (
    MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    EditMetadataInstruction,
    CreateInstruction, CreateSubInstruction,
    SendInstruction, CloseInstruction,
)
from avm.types import Conversation, UserMessageBatch, SystemMessage, UserMessage, AssistantMessage, MetaDict, MetaList
from avm.memory import Memory
from avm.exceptions import VMMemoryError
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
    conv.name = "init"
    return conv


def _settingup(content: str) -> MetaDict:
    """构造 LLM 程序节点（create_cmd / create_sub / _build_conversation 的 system_ref 必须是它或 python 节点）"""
    return MetaDict(data={"content": content}, ctrl={"type": "settingup"})


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
        assert "send_instruction" in reg
        assert "close_conversation" in reg
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
        core.mem["sys2"] = _settingup("child_sys")
        core.mem["para"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
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
        core.mem["sys2"] = _settingup("child_sys")
        core.mem["para"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        child = core._conv_by_cid[1]
        assert len(child.messages) == 1
        assert child.messages[0].role == "system"

    def test_create_cmd_reads_name_from_settingup(self):
        """create_cmd 从 settingup 节点的 name 数据字段读取对话名字"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.prog", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["prog"] = MetaDict(data={"name": "Analyzer", "content": "prog"}, ctrl={"type": "settingup"})
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()
        child = core._conv_by_cid[1]
        assert child.name == "Analyzer"
        assert child.identity == "Analyzer"

    def test_child_has_parent_link(self):
        core = make_core([
            (None, [
                {"call_id": "c", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        child = core._conv_by_cid[1]
        assert child.parent is core._conv_by_cid[0]
        assert child.is_sub is False
        assert child.is_root is False

    def test_create_cmd_requires_program_node(self):
        """create_cmd 的 system_ref 必须是 settingup/python 程序节点；str 节点报错且不创建对话"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["s"] = "not_a_program"
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, call_id = _batch(core, 0)[0]
        assert "Error" in content and "settingup" in content
        assert call_id == "ctc"
        assert core._active_cid == 0          # 父保持活跃，可自纠
        assert len(core._conv_by_cid) == 1    # 未创建子对话

    def test_create_cmd_rejects_untyped_dict(self):
        """无 ctrl 类型的 dict 节点（如参考手册）不是程序节点，同样报错"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.manual", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["manual"] = MetaDict(data={"overview": "..."})
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "Error" in content and "settingup" in content
        assert len(core._conv_by_cid) == 1

    def test_create_cmd_requires_para_node(self):
        """create_cmd 的 para_ref 必须指向 ctrl.type='para' 的参数节点；指向程序节点报错且不创建"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = _settingup("not_para")  # 程序节点不是参数节点
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, call_id = _batch(core, 0)[0]
        assert "Error" in content and "para" in content
        assert call_id == "ctc"
        assert core._active_cid == 0
        assert len(core._conv_by_cid) == 1


class TestSendInstruction:
    def _child_setup(self):
        """root 用 create_cmd 创建子对话并保持活跃"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("child_sys")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()  # root: create_cmd → 子休眠，root 活跃
        child = core._conv_by_cid[1]
        return core, root, child

    def test_send_instruction_delivers_and_confirms(self):
        """简单投递：content 原样投递给目标并唤醒，发起者收到确认、不自动休眠"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": child.cid, "content": "do X"}}], None),
            ("root_done", [], None),
            ("child_done", [], None),
        ])

        # root: send_instruction → child 唤醒进就绪，root 收到确认、保持活跃
        core.advance_conversation()
        st1 = _state(core)
        assert st1["active"] == root.cid
        assert child.cid in st1["ready"]
        assert root.cid not in st1["dormant"]
        assert ("已投递到 init#1", "si") in _batch(core, root.cid)
        # content 原样投递，无 JSON 信封
        assert child.user_batch.user_contents == ["do X"]

        # root: 无工具收尾 → 休眠，child 被调度
        core.advance_conversation()
        st2 = _state(core)
        assert st2["active"] == child.cid
        assert root.cid in st2["dormant"]

        # child: 无工具收尾 → 休眠（不唤醒任何对话，root 不在就绪）
        core.advance_conversation()
        st3 = _state(core)
        assert st3["active"] is None
        assert child.cid in st3["dormant"]
        assert root.cid in st3["dormant"]
        assert root.cid not in st3["ready"]

    def test_send_instruction_caller_continues(self):
        """投递后发起者继续执行：不等待返回、不休眠"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": child.cid, "content": "fire"}}], None),
            ("root_next", [], None),
        ])

        core.advance_conversation()  # root: 投递
        assert core._active_cid == root.cid
        assert root.cid not in core._dormant_cids
        assert child.cid in core._ready_cids

        core.advance_conversation()  # root: 继续 → 休眠，child 被调度
        assert core._active_cid == child.cid
        assert root.cid in core._dormant_cids

    def test_send_instruction_unknown_cid(self, capsys):
        core = make_core([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 99, "content": "x"}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()
        content, _ = _batch(core, 0)[0]
        assert "不存在" in content
        assert "不存在" in capsys.readouterr().err
        assert core._active_cid == 0  # 仍活跃

    def test_send_instruction_multicast(self):
        """多播：每个目标各收一条原样消息，统一返回确认，发起者不等待"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "ctc2", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
            (None, [{"call_id": "ms", "cmd_type": "send_instruction", "args": {"cid": [1, 2], "content": "broadcast"}}], None),
        ])
        core.advance_conversation()  # root: 创建第二个子对话
        child2 = core._conv_by_cid[2]
        core.advance_conversation()  # root: 多播

        # 两个目标各收到原样消息；无 ICC 记录；发起者不休眠
        for t in (child, child2):
            assert t.user_batch.user_contents[-1] == "broadcast"
        assert ("已投递到 2", "ms") in _batch(core, root.cid)
        assert root.cid not in core._dormant_cids
        assert core._active_cid == root.cid
        assert child.cid in core._ready_cids
        assert child2.cid in core._ready_cids


class TestCreateSub:
    def test_create_sub_parent_dormant(self):
        """create_sub → 亚对话立即活跃，父进入休眠等待子完成"""
        core = make_core([
            (None, [
                {"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}},
            ], None),
        ])
        _new_root(core)
        core.mem["s"] = _settingup("sub_sys")
        core.mem["u"] = "sub_usr"
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        st = _state(core)
        assert st["active"] == 1        # sub becomes active
        assert 0 in st["dormant"]       # parent dormant（子完成时由 core 唤醒）
        assert 0 not in st["ready"]

    def test_create_sub_requires_program_node(self):
        """create_sub 的 system_ref 同样是程序节点；str 节点报错且不创建亚对话"""
        core = make_core([
            (None, [{"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["s"] = "not_a_program"
        core.mem["u"] = "sub_usr"
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, call_id = _batch(core, 0)[0]
        assert "Error" in content and "settingup" in content
        assert call_id == "cs"
        assert core._active_cid == 0          # 父保持活跃，可自纠
        assert len(core._conv_by_cid) == 1    # 未创建亚对话

    def test_create_sub_requires_para_node(self):
        """create_sub 的 para_ref 同样必须是参数节点；str 节点报错且不创建亚对话"""
        core = make_core([
            (None, [{"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["u"] = "sub_usr"
        core.mem["p"] = "not_a_para_node"
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "Error" in content and "para" in content
        assert len(core._conv_by_cid) == 1

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
        core.mem["s"] = _settingup("x")
        core.mem["u"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
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
        core.mem["s"] = _settingup("x")
        core.mem["u"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
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
        core.mem["model_params"] = MetaDict(data={"model": "gpt-4o"}, ctrl={"type": "para"})
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
# 工具调用错误反馈（错误面向运行者；协议层另回一条中性占位保住配对）
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
        # 错误细节只给运行者；对话侧只拿到中性占位，保住 tool_calls 与 tool 消息的配对
        assert _batch(core, conv.cid) == [("（该工具调用未执行）", "tc-bad")]

    def test_unknown_tool_reported_to_stderr(self, capsys):
        """解析器不查指令集：未知名字原样留在 cmd_type，由处理指令时判断。"""
        lmu = LMU()
        message = SimpleNamespace(tool_calls=[self._fake_tool_call("definitely_not_a_tool", '{"a": 1}')])
        return_calls = lmu._return_calls_from_message(message)
        assert return_calls[0]["cmd_type"] == "definitely_not_a_tool"
        assert return_calls[0]["args"] == {"a": 1}

        core = make_core()
        conv = _new_root(core)
        core._process_return_calls(return_calls, conv)
        err = capsys.readouterr().err
        assert "未知工具 definitely_not_a_tool" in err
        assert '{"a": 1}' in err
        assert _batch(core, conv.cid) == [("（该工具调用未执行）", "tc-bad")]

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
        assert _batch(core, 0) == [("（该工具调用未执行）", "tc1")]  # 只有中性占位
        assert "不是合法 JSON" in capsys.readouterr().err

        core.advance_conversation()
        assert core._active_cid is None


class TestQueueSchedulingAndInterrupt:
    def test_instruction_budget_yields_to_ready_queue(self):
        """核心中断：指令数超过预算后让出，进就绪队尾，轮到其他就绪对话"""
        core = make_core([
            (None, [
                {"call_id": "w1", "cmd_type": "memory_write", "args": {"ref": "$MEM.a", "content": "1"}},
                {"call_id": "w2", "cmd_type": "memory_write", "args": {"ref": "$MEM.b", "content": "2"}},
                {"call_id": "w3", "cmd_type": "memory_write", "args": {"ref": "$MEM.c", "content": "3"}},
            ], None),
        ])
        core._instruction_budget = 2
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        other = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._ready_cids.clear()
        core._ready_cids.append(other.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        assert core._active_cid == other.cid  # 中断后轮到 other
        assert root.cid in core._ready_cids   # root 进就绪队尾
        assert len(_batch(core, root.cid)) == 3  # 工具返回已加入 batch，状态完整

    def test_sub_inherits_parent_budget(self):
        """亚对话继承父的剩余预算：父+亚连续链总指令受限，超限后让位给无关对话"""
        core = make_core([
            (None, [
                {"call_id": "w1", "cmd_type": "memory_write", "args": {"ref": "$MEM.a", "content": "1"}},
                {"call_id": "w2", "cmd_type": "memory_write", "args": {"ref": "$MEM.b", "content": "2"}},
                {"call_id": "cs", "cmd_type": "create_sub", "args": {"system_ref": "$MEM.sub_sys", "user_ref": "task", "para_ref": "$MEM.sub_p"}},
            ], None),
            (None, [
                {"call_id": "s1", "cmd_type": "memory_write", "args": {"ref": "$MEM.c", "content": "3"}},
                {"call_id": "s2", "cmd_type": "memory_write", "args": {"ref": "$MEM.d", "content": "4"}},
                {"call_id": "s3", "cmd_type": "memory_write", "args": {"ref": "$MEM.e", "content": "5"}},
            ], None),
        ])
        core._instruction_budget = 5
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core.mem["sub_sys"] = _settingup("sub")
        core.mem["sub_p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        other = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._ready_cids.clear()
        core._ready_cids.append(other.cid)
        core._active_cid = root.cid

        # root: 2 条写入 + create_sub（亚对话前插）→ 计数 3，亚对话活跃（同链不重置）
        core.advance_conversation()
        assert core._active_cid == 2  # 亚对话（other 占 cid 1）
        assert core._activation_instructions == 3  # 继承了父的计数

        # 亚对话: 3 条写入 → 链总计数 6 ≥ 5 → 中断，亚对话进队尾，无关对话 other 被选中
        core.advance_conversation()
        assert core._active_cid == other.cid
        assert 2 in core._ready_cids
        assert len(_batch(core, 2)) == 3  # 工具返回已加入，状态完整


class TestInfoDevices:
    def test_info_devices_mounted_and_readable(self):
        core = make_core([("done", [], None)])
        _new_root(core)

        s = core.unwrap("$MEM.conversations")
        assert "cid=0" in s and "init" in s
        assert "state=" in core.unwrap("$MEM.conversations.0")

        s = core.unwrap("$MEM.scheduler")
        assert "active=" in s and "instruction_budget=" in s
        assert core.unwrap("$MEM.scheduler.budget") == "0/50"

        assert "顶层键" in core.unwrap("$MEM.memory")
        assert "监测" in core.unwrap("$MEM.monitor")

    def test_info_devices_readonly(self):
        core = make_core()
        _new_root(core)
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.scheduler", "x")


class TestReasoningContent:
    """思维链内容回传：历史保留、下一轮带回、last_call 记录"""

    def _lmu_with_fake_client(self, messages_log, reasoning="chain of thought", content="final"):
        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                messages_log.append(kwargs["messages"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=None,
                ))])

        class FakeChat:
            completions = FakeCompletions

        class FakeClient:
            chat = FakeChat

        lmu = LMU()
        lmu._client = FakeClient
        return lmu

    def test_exec_stores_and_returns_reasoning(self):
        messages_log = []
        lmu = self._lmu_with_fake_client(messages_log)
        core = make_core()
        core.lmu = lmu
        conv = _new_root(core)
        para = MetaDict(data={"model": "test"}, ctrl={"type": "para"})

        lmu.exec(conv, para)
        last = conv.messages[-1]
        assert isinstance(last, AssistantMessage)
        assert last.reasoning_content == "chain of thought"
        assert lmu.last_call["reasoning"] == "chain of thought"

        # 第二轮请求：assistant 消息带回 reasoning_content（有值带值）
        lmu.exec(conv, para)
        assistant_msgs = [m for m in messages_log[1] if m.get("role") == "assistant"]
        assert assistant_msgs
        assert assistant_msgs[0]["reasoning_content"] == "chain of thought"

    def test_exec_without_reasoning_returns_empty_string(self):
        messages_log = []
        lmu = self._lmu_with_fake_client(messages_log, reasoning=None)
        core = make_core()
        core.lmu = lmu
        conv = _new_root(core)
        para = MetaDict(data={"model": "test"}, ctrl={"type": "para"})

        lmu.exec(conv, para)
        assert conv.messages[-1].reasoning_content is None
        assert lmu.last_call["reasoning"] is None

        # 第二轮请求：无 reasoning 的 assistant 消息也带空字符串（回传所有）
        lmu.exec(conv, para)
        assistant_msgs = [m for m in messages_log[1] if m.get("role") == "assistant"]
        assert assistant_msgs
        assert assistant_msgs[0]["reasoning_content"] == ""


class TestCloseInstruction:
    """内核级关闭指令：个体结束、调度清理、执行器释放、服务注销、ICC 清理"""

    def _root_with_two(self, core):
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        c1 = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        c2 = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        return root, c1, c2

    def test_close_self_marks_finished_and_picks_next(self):
        """关闭自身：标记 finished、移出调度队列、监测器显示 finished、选择下一个活跃对话"""
        core = make_core([
            (None, [{"call_id": "cl", "cmd_type": "close_conversation", "args": {"cid": 0}}], None),
        ])
        root, other, _ = self._root_with_two(core)
        core._ready_cids.clear()
        core._ready_cids.append(other.cid)
        core._active_cid = root.cid
        core.monitor.record_baseline(core)

        core.advance_conversation()

        assert 0 in core._finished
        assert core._active_cid == other.cid
        assert 0 not in core._ready_cids and 0 not in core._dormant_cids
        assert core.monitor.frame(1).conversations[0]["state"] == "finished"
        content, call_id = _batch(core, 0)[0]
        assert "已关闭" in content and call_id == "cl"

    def test_close_child_cleans_executor_and_dormant(self):
        """关闭 Python 子对话：移出休眠、释放执行器（PLM 变量空间）"""
        core = make_core([
            (None, [{"call_id": "cl", "cmd_type": "close_conversation", "args": {"cid": 1}}], None),
        ])
        root = _new_root(core)
        core.mem["prog"] = MetaDict(data={"content": ""}, ctrl={"type": "python"})
        core.mem["p"] = MetaDict(data={"model": "plm.simple"}, ctrl={"type": "para"})
        child = _build_conversation(core, "$MEM.prog", "$MEM.p", parent=root, is_sub=False)
        core._get_executor(child, core._get_para(child))  # 预创建执行器并缓存
        assert child.cid in core._executors
        core._dormant_cids.append(child.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        assert child.cid in core._finished
        assert child.cid not in core._dormant_cids
        assert child.cid not in core._executors

    def test_close_nonexistent_errors(self):
        core = make_core([
            (None, [{"call_id": "cl", "cmd_type": "close_conversation", "args": {"cid": 99}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "不存在" in content
        assert core._active_cid == 0

    def test_close_finished_errors(self):
        core = make_core([
            (None, [{"call_id": "cl", "cmd_type": "close_conversation", "args": {"cid": 1}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._close_conversation(child.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "finished" in content

    def test_send_instruction_to_closed_errors(self, capsys):
        core = make_core([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "x"}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._close_conversation(child.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "已关闭" in content
        assert "已关闭" in capsys.readouterr().err
        assert core._active_cid == 0


class TestEditMetadata:
    """edit_metadata：编辑节点的 ctrl 元数据（set/get/del）"""

    def _root(self):
        core = make_core()
        root = _new_root(core)
        return core, root

    def _exec(self, core, root, args, call_id="em"):
        EditMetadataInstruction(call_id, root.cid, args).execute(core, root)
        return _batch(core, root.cid)[-1]

    def test_set_creates_ctrl(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={})
        content, call_id = self._exec(core, root, {"ref": "$MEM.a", "type": "set", "key": "type", "value": "settingup"})
        assert core.mem["a"].get_ctrl() == {"type": "settingup"}
        assert "已写入" in content and call_id == "em"

    def test_set_keeps_existing_keys(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"x": "1"})
        self._exec(core, root, {"ref": "$MEM.a", "type": "set", "key": "y", "value": "2"})
        assert core.mem["a"].get_ctrl() == {"x": "1", "y": "2"}

    def test_set_missing_value_errors(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "set", "key": "type"})
        assert "key 和 value 都必需" in content
        assert core.mem["a"].get_ctrl() is None

    def test_get_by_key(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"type": "para"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "get", "key": "type"})
        assert content == "para"

    def test_get_whole_ctrl(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"type": "para", "x": "1"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "get"})
        assert content == '{"type": "para", "x": "1"}'

    def test_get_empty_ctrl_returns_empty_object(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "get"})
        assert content == "{}"

    def test_get_missing_key_errors(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"type": "para"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "get", "key": "nope"})
        assert "ctrl 中没有键" in content

    def test_del(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"type": "para", "x": "1"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "del", "key": "type"})
        assert core.mem["a"].get_ctrl() == {"x": "1"}
        assert "已删除" in content

    def test_del_missing_key_errors(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={}, ctrl={"type": "para"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "del", "key": "nope"})
        assert "ctrl 中没有键" in content

    def test_invalid_type_errors(self):
        core, root = self._root()
        core.mem["a"] = MetaDict(data={})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "upsert"})
        assert "必须是 set/get/del" in content

    def test_str_node_errors(self):
        core, root = self._root()
        core.mem["a"] = "leaf"
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "set", "key": "type", "value": "x"})
        assert "不是 MetaDict/MetaList" in content

    def test_invalid_ref_errors(self):
        core, root = self._root()
        content, _ = self._exec(core, root, {"ref": "$MEM.nope", "type": "get"})
        assert "Error" in content

    def test_list_node_works(self):
        core, root = self._root()
        core.mem["a"] = MetaList(data=[], ctrl={"type": "para"})
        content, _ = self._exec(core, root, {"ref": "$MEM.a", "type": "get"})
        assert content == '{"type": "para"}'

    def test_edit_metadata_via_advance_keeps_conversation_active(self):
        """通过完整推进执行 edit_metadata：ctrl 生效，对话保持活跃可自纠"""
        core = make_core([
            (None, [{"call_id": "em", "cmd_type": "edit_metadata", "args": {"ref": "$MEM.a", "type": "set", "key": "type", "value": "settingup"}}], None),
        ])
        _new_root(core)
        core.mem["a"] = MetaDict(data={})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        assert core.mem["a"].get_ctrl() == {"type": "settingup"}
        assert core._active_cid == 0


class TestParaTypeCheck:
    def test_get_para_warns_and_falls_back_on_wrong_type(self, capsys):
        """para_ref 指向的节点不是 para 类型：stderr 警告并回退默认参数（不再静默）"""
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        child.para_ref = "$MEM.s"  # 把 para_ref 指向程序节点

        para = core._get_para(child)
        assert para.get("model") == "gpt-4"
        assert "不是 ctrl.type='para'" in capsys.readouterr().err

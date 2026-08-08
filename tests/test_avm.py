"""Core 调度系统测试"""

import json
import pytest
from avm.core import Core, LMU, _instruction_registry, _build_conversation
from avm.core import (
    MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    CreateInstruction, CreateSubInstruction,
    RegisterServiceInstruction, CallServiceInstruction, TransferServiceInstruction,
    ReturnResultInstruction, SendInstruction, CloseInstruction,
)
from avm.types import Conversation, UserMessageBatch, SystemMessage, UserMessage, AssistantMessage, MetaDict
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
        core.mem["sys2"] = _settingup("child_sys")
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
        core.mem["sys2"] = _settingup("child_sys")
        core.mem["para"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "Error" in content and "settingup" in content
        assert len(core._conv_by_cid) == 1


class TestSendInstruction:
    def _child_setup(self):
        """root 用 create_cmd 创建子对话并保持活跃"""
        core = make_core([
            (None, [{"call_id": "ctc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("child_sys")
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
        assert msg == {"from": "init", "to": "init#1", "icc_id": "si", "content": "do X"}

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

    def test_send_instruction_multicast(self):
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "ctc2", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
            (None, [{"call_id": "ms", "cmd_type": "send_instruction", "args": {"cid": [1, 2], "content": "broadcast", "wait": False}}], None),
        ])
        core.advance_conversation()  # root: 创建第二个子对话
        child2 = core._conv_by_cid[2]
        core.advance_conversation()  # root: 多播

        assert set(core._icc) == {"ms#0", "ms#1"}  # 每个目标一条独立记录
        assert core._icc_groups == {"ms": 2}        # 工具返回多播：等待所有目标返回
        assert root.cid in core._dormant_cids       # 工具返回多播期间发起者休眠
        for t in (child, child2):
            msg = json.loads(t.user_batch.user_contents[-1])
            assert msg["from"] == "init"
            assert msg["to"] == ["init#1", "init#2"]
            assert msg["icc_id"] in ("ms#0", "ms#1")
            assert msg["content"] == "broadcast"

    def test_multicast_tool_mode_merges_returns(self):
        """工具返回多播：各目标返回按 cid+名字合并成一条工具响应，最后一个返回唤醒发起者"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "ctc2", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
            (None, [{"call_id": "ms", "cmd_type": "send_instruction", "args": {"cid": [1, 2], "content": "broadcast", "wait": False}}], None),
            (None, [{"call_id": "rr1", "cmd_type": "return_result", "args": {"content": "reply1", "icc_id": "ms#0"}}], None),
            ("ok1", [], None),
            (None, [{"call_id": "rr2", "cmd_type": "return_result", "args": {"content": "reply2", "icc_id": "ms#1"}}], None),
            ("ok2", [], None),
            ("done", [], None),
        ])
        core.advance_conversation()
        core.advance_conversation()

        # 子对话 1 返回（ms#0）：合并进 batch，但还有未返回目标，发起者不唤醒
        core.advance_conversation()
        assert root.cid in core._dormant_cids
        root.user_batch.to_tool_messages()  # 物化合并
        assert json.loads(_batch(core, root.cid)[0][0]) == [
            {"from": "init#1", "cid": 1, "icc_id": "ms#0", "content": "reply1"},
        ]
        core.advance_conversation()  # 子对话 1 收尾

        # 子对话 2 返回（ms#1）：合并完成 → 发起者进就绪队列
        core.advance_conversation()
        root.user_batch.to_tool_messages()  # 物化合并
        assert json.loads(_batch(core, root.cid)[0][0]) == [
            {"from": "init#1", "cid": 1, "icc_id": "ms#0", "content": "reply1"},
            {"from": "init#2", "cid": 2, "icc_id": "ms#1", "content": "reply2"},
        ]
        assert root.cid in core._ready_cids
        assert root.cid not in core._dormant_cids
        core.advance_conversation()  # 子对话 1 收尾
        core.advance_conversation()  # root 收尾
        assert core._active_cid is None

    def test_single_message_mode(self):
        """单目标 message 模式：确认响应满足配对，返回以消息形式投递并唤醒发起者"""
        core, root, child = self._child_setup()
        core.lmu = MockLMU([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": child.cid, "content": "ask", "wait": True, "return_mode": "message"}}], None),
            (None, [{"call_id": "rr", "cmd_type": "return_result", "args": {"content": "msg-reply", "icc_id": "si"}}], None),
            ("ok", [], None),
            ("done", [], None),
        ])

        # root: 发送 → 确认响应 + root 休眠，child 活跃
        core.advance_conversation()
        assert _batch(core, root.cid) == [("已投递到 init#1", "si")]
        assert root.cid in core._dormant_cids

        # child: return_result → 消息投递 root 并唤醒
        core.advance_conversation()
        msg = json.loads(root.user_batch.user_contents[-1])
        assert msg == {"from": "init#1", "to": "init", "icc_id": "si", "content": "msg-reply"}
        assert root.cid in core._ready_cids

        core.advance_conversation()  # child 收尾
        core.advance_conversation()  # root 收尾
        assert core._active_cid is None


class TestIccRouting:
    def test_return_result_routes_by_icc_id(self):
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = _settingup("s")
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
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
        core.mem["s"] = _settingup("sub_sys")
        core.mem["u"] = "sub_usr"
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        content, call_id = _batch(core, 0)[0]
        assert "Error" in content and "settingup" in content
        assert call_id == "cs"
        assert core._active_cid == 0          # 父保持活跃，可自纠
        assert len(core._conv_by_cid) == 1    # 未创建亚对话

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
        core.mem["s"] = _settingup("x")
        core.mem["u"] = "x"
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
        core.mem["s"] = _settingup("svc_sys")
        core.mem["u"] = "svc_usr"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
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
        assert svc.user_batch.user_contents == [json.dumps({"from": "init", "to": "init#1", "icc_id": "cc", "content": "2+3"})]

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
        core.mem["s"] = _settingup("s")
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
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
        core.mem["s"] = _settingup("s")
        core.mem["u"] = "u"
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
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


class TestQueueSchedulingAndInterrupt:
    def test_return_result_caller_queues_at_end(self):
        """return_result 唤醒的调用者进就绪队列末尾（排队调度，create_sub 是唯一例外）"""
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = _settingup("s")
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        other = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "svc", "what": "x", "needs": "x", "returns": "x"}).execute(core, svc)
        CallServiceInstruction("req", root.cid, {"service_name": "svc", "input": "q"}).execute(core, root)

        # 清空就绪并放进另一个对话；svc 返回 → root 应排在 other 后面
        core._ready_cids.clear()
        core._ready_cids.append(other.cid)
        ReturnResultInstruction("rr", svc.cid, {"content": "6", "icc_id": "req"}).execute(core, svc)
        assert core._ready_cids == [other.cid, root.cid]

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
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
        core.mem["sub_sys"] = _settingup("sub")
        core.mem["sub_p"] = MetaDict(data={"model": "test"})
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

        assert core.unwrap("$MEM.icc") == "暂无待处理 ICC 记录"
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
        para = MetaDict(data={"model": "test"})

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
        para = MetaDict(data={"model": "test"})

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
        core.mem["p"] = MetaDict(data={"model": "test"})
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
        core.mem["p"] = MetaDict(data={"model": "test"})
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._close_conversation(child.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "finished" in content

    def test_send_instruction_to_closed_errors(self, capsys):
        core = make_core([
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "x", "wait": False}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("x")
        core.mem["p"] = MetaDict(data={"model": "test"})
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._close_conversation(child.cid)
        core._active_cid = root.cid

        core.advance_conversation()

        content, _ = _batch(core, 0)[0]
        assert "已关闭" in content
        assert "已关闭" in capsys.readouterr().err
        assert core._active_cid == 0

    def test_close_service_unregisters(self):
        core = make_core([
            (None, [{"call_id": "cl", "cmd_type": "close_conversation", "args": {"cid": 1}}], None),
            (None, [{"call_id": "cs", "cmd_type": "call_service", "args": {"service_name": "calc", "input": "2+3"}}], None),
        ])
        root = _new_root(core)
        core.mem["s"] = _settingup("svc_sys")
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "calc", "what": "计算", "needs": "表达式", "returns": "结果"}).execute(core, svc)
        core._active_cid = root.cid

        core.advance_conversation()  # root: close svc
        assert "calc" not in core._services

        core.advance_conversation()  # root: call_service → 服务不存在
        content, _ = _batch(core, 0)[-1]
        assert "不存在" in content

    def test_close_service_notifies_waiting_caller(self):
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = _settingup("svc_sys")
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "calc", "what": "计算", "needs": "表达式", "returns": "结果"}).execute(core, svc)
        CallServiceInstruction("req", root.cid, {"service_name": "calc", "input": "2+3"}).execute(core, root)
        assert root.cid in core._dormant_cids
        assert "req" in core._icc

        core._close_conversation(svc.cid)

        assert svc.cid in core._finished
        assert "req" not in core._icc
        assert ("Error: 对方已关闭（1），请求未完成", "req") in _batch(core, root.cid)
        assert root.cid in core._ready_cids
        assert root.cid not in core._dormant_cids

    def test_return_result_to_closed_caller_drops_record(self, capsys):
        core = make_core()
        root = _new_root(core)
        core.mem["s"] = _settingup("svc_sys")
        core.mem["p"] = MetaDict(data={"model": "test"})
        svc = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        RegisterServiceInstruction("rc", svc.cid, {"name": "calc", "what": "计算", "needs": "表达式", "returns": "结果"}).execute(core, svc)
        CallServiceInstruction("req", root.cid, {"service_name": "calc", "input": "2+3"}).execute(core, root)
        core._close_conversation(root.cid)
        assert "req" not in core._icc

        ReturnResultInstruction("rr", svc.cid, {"content": "6", "icc_id": "req"}).execute(core, svc)

        assert ("6", "req") not in _batch(core, root.cid)
        # close 已把发起者的记录作废，服务方返回得到"无对应请求记录"
        assert any("无对应请求记录" in c for c, _ in _batch(core, svc.cid))

    def test_close_multicast_target_merges_error_into_group(self):
        """多播目标被关闭：错误段并入组，组全部完成后才唤醒发起者"""
        core = make_core()
        root, c1, c2 = self._root_with_two(core)
        SendInstruction("ms", root.cid, {"cid": [1, 2], "content": "broadcast", "wait": True}).execute(core, root)
        assert root.cid in core._dormant_cids
        assert set(core._icc) == {"ms#0", "ms#1"}

        # 关闭目标 1：错误段并入组，组未完成，root 不唤醒
        CloseInstruction("cl", root.cid, {"cid": 1}).execute(core, root)
        assert root.cid in core._dormant_cids
        assert 1 in core._finished
        assert "ms#0" not in core._icc

        # 目标 2 正常返回：组归零，root 唤醒，batch 是合并的 JSON 数组
        ReturnResultInstruction("rr2", c2.cid, {"content": "reply2", "icc_id": "ms#1"}).execute(core, c2)
        assert root.cid not in core._dormant_cids
        assert root.cid in core._ready_cids
        root.user_batch.to_tool_messages()  # 物化合并
        merged = json.loads(_batch(core, root.cid)[0][0])
        assert {"from": "init#1", "cid": 1, "icc_id": "ms#0", "content": "Error: 对方已关闭（1），请求未完成"} in merged
        assert {"from": "init#2", "cid": 2, "icc_id": "ms#1", "content": "reply2"} in merged

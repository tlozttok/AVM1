"""亚对话严格链语义：父子之间不插其他对话、等待返回的对话不被唤醒、亚对话不能向主对话投递。

对应报告 report-sub-chain-strictness-2026-10-20.md 的验收。
"""

import json

from avm.core import Core
from avm.types import MetaDict


def _parent_with_sub():
    """父对话已声明 create_sub 并创建亚对话（父在等返回）。"""
    core = Core()
    parent = core.start("sys", "parent")
    other = core.start("sys", "other")
    core.mem["prog"] = MetaDict(data={"content": "x"}, ctrl={"type": "settingup"})
    core.mem["para"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
    core.mem["u"] = "任务"
    parent.append_assistant_message("", tool_calls=[{
        "id": "call_sub", "type": "function",
        "function": {"name": "create_sub",
                     "arguments": json.dumps({"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                              "para_ref": "$MEM.para"})},
    }])
    core._ready_cids.remove(parent.cid)  # 父对话已激活（start 只是先排进就绪）
    core._active_cid = parent.cid
    core._process_return_calls([{"cmd_type": "create_sub", "call_id": "call_sub",
                                 "args": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                          "para_ref": "$MEM.para"}}], parent)
    sub = [c for c in core._conv_by_cid.values() if c.is_sub][0]
    return core, parent, other, sub


class _Done:
    def __init__(self, result):
        self.result = result

    def exec(self, conversation, para):
        return self.result, [], conversation


def _finish(core, conv, result="结果"):
    """用一个最小执行器让对话完成一轮（无工具调用）。"""
    class _Done:
        def exec(self, conversation, para):
            return result, [], conversation

    core._executors[conv.cid] = _Done()
    core._active_cid = conv.cid
    core.advance_conversation()


class TestWaitingParentIsNotWoken:
    def test_notify_only_queues_while_waiting(self):
        core, parent, other, sub = _parent_with_sub()
        core.events.deliver_notifies({"notify": [{"cid": parent.cid, "content": "事件消息"}]})
        assert parent.user_batch.user_contents == ["事件消息"]
        assert parent.cid in core._dormant_cids, "等待返回的父对话不能被 notify 唤醒"
        assert parent.cid not in core._ready_cids

    def test_send_only_queues_while_waiting(self):
        core, parent, other, sub = _parent_with_sub()
        core._active_cid = other.cid
        core._process_return_calls(
            [{"cmd_type": "send_instruction", "call_id": "c2",
              "args": {"cid": parent.cid, "content": "投递消息"}}], other)
        assert parent.user_batch.user_contents == ["投递消息"]
        assert parent.cid in core._dormant_cids
        assert parent.cid not in core._ready_cids

    def test_scheduler_skips_waiting_conversation(self):
        core, parent, other, sub = _parent_with_sub()
        core._dormant_cids.remove(parent.cid)
        core._ready_cids.insert(0, parent.cid)  # 有人硬塞进就绪队列
        core._pick_next_active()
        assert core._active_cid != parent.cid
        assert parent.cid in core._dormant_cids, "被跳过时应退回休眠，而不是运行"


class TestStrictOrdering:
    def test_sub_runs_before_others_and_parent_resumes_immediately(self):
        core = Core()
        parent = core.start("sys", "parent")
        core.start("sys", "other")  # 另一个就绪对话
        core.mem["prog"] = MetaDict(data={"content": "x"}, ctrl={"type": "settingup"})
        core.mem["para"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        core.mem["u"] = "任务"
        parent.append_assistant_message("", tool_calls=[{
            "id": "call_sub", "type": "function",
            "function": {"name": "create_sub",
                         "arguments": json.dumps({"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                                  "para_ref": "$MEM.para"})},
        }])
        core._ready_cids.remove(parent.cid)
        core._active_cid = parent.cid
        core._process_return_calls([{"cmd_type": "create_sub", "call_id": "call_sub",
                                     "args": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                              "para_ref": "$MEM.para"}}], parent)
        sub = [c for c in core._conv_by_cid.values() if c.is_sub][0]
        assert core._ready_cids[0] == sub.cid, "亚对话应排在最前（父子之间没有其他对话）"
        _finish(core, sub)
        assert core._active_cid == parent.cid, "亚对话结束后父对话应立即运行"
        assert not core._is_waiting_for_return(parent), "返回已写回，父不再等待"


class TestSubCannotSendToParent:
    def test_sub_send_to_parent_is_illegal(self):
        core, parent, other, sub = _parent_with_sub()
        core._active_cid = sub.cid
        before = len(parent.user_batch.user_contents)
        core._process_return_calls(
            [{"cmd_type": "send_instruction", "call_id": "c3",
              "args": {"cid": parent.cid, "content": "偷偷给主对话发消息"}}], sub)
        resp = sub.user_batch.tool_responses[-1]
        assert resp.content.startswith("Illegal:")
        assert "只能用 create_sub 的返回通道" in resp.content
        assert len(parent.user_batch.user_contents) == before
        Core._check_pairing(sub)  # 响应成对


class TestSubLifecycle:
    def test_sub_deleted_after_completion(self):
        core, parent, other, sub = _parent_with_sub()
        sub_cid = sub.cid
        _finish(core, sub)
        assert sub_cid not in core._conv_by_cid, "亚对话完成后应当被删除"
        assert sub_cid not in core._dormant_cids
        assert parent.user_batch.tool_responses[0].content == "结果"

    def test_send_to_deleted_sub_reports_missing(self):
        core, parent, other, sub = _parent_with_sub()
        sub_cid = sub.cid
        _finish(core, sub)
        core._active_cid = other.cid
        core._process_return_calls(
            [{"cmd_type": "send_instruction", "call_id": "c9",
              "args": {"cid": sub_cid, "content": "hi"}}], other)
        resp = other.user_batch.tool_responses[-1]
        assert resp.content.startswith("Error:") and "不存在" in resp.content

    def test_sub_cannot_close_its_parent(self):
        core, parent, other, sub = _parent_with_sub()
        core._active_cid = sub.cid
        core._process_return_calls(
            [{"cmd_type": "close_conversation", "call_id": "c4", "args": {"cid": parent.cid}}], sub)
        resp = sub.user_batch.tool_responses[-1]
        assert resp.content.startswith("Illegal:")
        assert parent.cid in core._conv_by_cid, "父对话不能被亚对话关闭"

    def test_closing_sub_cascades_to_its_child_sub(self):
        core, parent, other, sub = _parent_with_sub()
        # sub 自己再创建一个亚对话（sub2）并等待返回
        core.mem["prog2"] = MetaDict(data={"content": "y"}, ctrl={"type": "settingup"})
        core.mem["para2"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        core._ready_cids.remove(sub.cid)
        core._active_cid = sub.cid
        core._process_return_calls(
            [{"cmd_type": "create_sub", "call_id": "call_sub2",
              "args": {"system_ref": "$MEM.prog2", "user_ref": "$MEM.u", "para_ref": "$MEM.para2"}}], sub)
        sub2 = [c for c in core._conv_by_cid.values() if c.is_sub and c.parent is sub][0]
        sub_cid, sub2_cid = sub.cid, sub2.cid

        core._active_cid = other.cid
        core._process_return_calls(
            [{"cmd_type": "close_conversation", "call_id": "c5", "args": {"cid": sub_cid}}], other)

        assert sub_cid not in core._conv_by_cid
        assert sub2_cid not in core._conv_by_cid, "子亚对话应随父亚对话级联删除"
        assert parent.user_batch.tool_responses[-1].tool_call_id == "call_sub"
        assert parent.user_batch.tool_responses[-1].content.startswith("Error:")


class TestMultipleSubsInOneTurn:
    """同一轮可以声明多个 create_sub：父要等全部返回才恢复，活跃指针不能留在已删个体上。"""

    def _parent_with_two_subs(self):
        core = Core()
        parent = core.start("sys", "parent")
        core.mem["prog"] = MetaDict(data={"content": "x"}, ctrl={"type": "settingup"})
        core.mem["para"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        core.mem["u"] = "任务"
        core._ready_cids.remove(parent.cid)
        parent.append_assistant_message("", tool_calls=[
            {"id": "call_a", "type": "function",
             "function": {"name": "create_sub", "arguments": json.dumps(
                 {"system_ref": "$MEM.prog", "user_ref": "$MEM.u", "para_ref": "$MEM.para"})}},
            {"id": "call_b", "type": "function",
             "function": {"name": "create_sub", "arguments": json.dumps(
                 {"system_ref": "$MEM.prog", "user_ref": "$MEM.u", "para_ref": "$MEM.para"})}},
        ])
        core._active_cid = parent.cid
        core._process_return_calls([
            {"cmd_type": "create_sub", "call_id": "call_a",
             "args": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u", "para_ref": "$MEM.para"}},
            {"cmd_type": "create_sub", "call_id": "call_b",
             "args": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u", "para_ref": "$MEM.para"}},
        ], parent)
        subs = [c for c in core._conv_by_cid.values() if c.is_sub]
        return core, parent, subs

    def test_parent_waits_for_all_subs(self):
        core, parent, subs = self._parent_with_two_subs()
        assert core._pending_call_ids(parent) == ["call_a", "call_b"]
        sub_a, sub_b = subs[0], subs[1]
        core._executors[sub_a.cid] = _Done("A")
        core._executors[sub_b.cid] = _Done("B")

        core._active_cid = sub_a.cid
        core.advance_conversation()
        # 第一个子完成：父还在等第二个，不能成为活跃对话；活跃指针不能留在已删个体上
        assert core._active_cid != sub_a.cid
        assert core._active_cid in (None, sub_b.cid)
        assert core._is_waiting_for_return(parent)
        assert sub_a.cid not in core._conv_by_cid

        core._active_cid = sub_b.cid
        core.advance_conversation()
        assert core._active_cid == parent.cid
        assert not core._is_waiting_for_return(parent)
        assert [r.tool_call_id for r in parent.user_batch.tool_responses] == ["call_a", "call_b"]

        for r in parent.user_batch.tool_responses:
            parent.append_tool_message(r.content, r.tool_call_id)
        parent.user_batch.clear()
        parent.append_assistant_message("done")
        Core._check_pairing(parent)  # 不抛

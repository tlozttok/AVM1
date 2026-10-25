"""指令边界与内核稳定性的回归用例。

对应 plan-kernel-stability-phase1-2026-10-16.md 的验收一节：把讨论案里的探针与
三件已定事项固化成用例。每条用例都回答同一个问题——这个输入会不会让进程退出。
"""

import json

import pytest

from avm.core import Core
from avm.exceptions import APIError, KernelPanic, RunnerStop, VMMemoryError
from avm.image import ImageError, load_image
from avm.memory_device import MemoryDevice, OutputsListDevice
from avm.python_server import create_plm
from avm.sandbox import exec_safe, safe_globals
from avm.types import AssistantMessage, LinkNode, MetaDict, MetaList, ToolMessage


def _core():
    core = Core()
    conv = core.start("sys", "user")
    core._active_cid = conv.cid
    return core, conv


def _call(core, conv, cmd, args, call_id="c1"):
    core._process_return_calls(
        [{"cmd_type": cmd, "call_id": call_id, "args": args}], conv
    )


def _responses(conv):
    return [(r.content, r.tool_call_id) for r in conv.user_batch.tool_responses]


class StubLMU:
    """最小执行器替身：前 fail_times 次抛 APIError，之后返回一个结果。"""

    def __init__(self, fail_times=1):
        self.n_calls = 0
        self.fail_times = fail_times
        self.resets = 0

    def exec(self, conversation, para):
        self.n_calls += 1
        if self.n_calls <= self.fail_times:
            raise APIError("boom")
        conversation.append_assistant_message("done")
        return "done", [], conversation

    def reset_client(self):
        self.resets += 1


# ----------------------------------------------------------------------
# 甲：指令参数仲裁
# ----------------------------------------------------------------------

BAD_CALLS = [
    ("memory_read", {}),                                            # 缺必填
    ("memory_read", []),                                            # 参数不是对象
    ("memory_write", {"ref": 42, "content": "x"}),                  # ref 类型不对
    ("memory_write", {"ref": "nope", "content": "x"}),              # ref 缺 $
    ("memory_write", {"ref": "$MEM.x"}),                            # 缺 content
    ("memory_make", {"ref": "$MEM", "key": "k", "mem_type": "wat"}),  # 枚举外
    ("send_instruction", {}),
    ("send_instruction", {"cid": {"a": 1}, "content": "x"}),
    ("close_conversation", {}),
    ("create_cmd", {}),
    ("create_sub", {"system_ref": "$MEM.s", "user_ref": "$MEM.u"}),
    ("edit_metadata", {"ref": "$MEM.x", "type": "set", "key": "k"}),   # set 缺 value
    ("edit_metadata", {"ref": "$MEM.x", "type": "wat"}),
    ("listen_event", {"action": "listen"}),                            # listen 缺三个参数
]


class TestArgumentArbitration:
    @pytest.mark.parametrize("cmd,args", BAD_CALLS)
    def test_bad_args_return_error_and_do_not_raise(self, cmd, args):
        core, conv = _core()
        _call(core, conv, cmd, args)
        assert len(_responses(conv)) == 1, "每个调用恰好回一条工具响应"
        content, call_id = _responses(conv)[0]
        assert content.startswith("Illegal:"), content
        assert call_id == "c1"

    def test_unknown_extra_keys_are_not_rejected(self):
        core, conv = _core()
        core.mem["x"] = "v"
        _call(core, conv, "memory_read", {"ref": "$MEM.x", "comment": "随便写的"})
        assert _responses(conv) == [("v", "c1")]

    def test_args_key_cannot_shadow_method_nor_spoof_call_id(self):
        """参数里的 execute 不能遮蔽方法，call_id 不能伪造配对 id。"""
        core, conv = _core()
        core.mem["x"] = "v"
        _call(core, conv, "memory_read", {"ref": "$MEM.x", "execute": 1, "call_id": "forged"})
        assert _responses(conv) == [("v", "c1")]

    def test_missing_call_id_skips_without_fabricating_one(self):
        core, conv = _core()
        core._process_return_calls(
            [{"cmd_type": "memory_read", "call_id": None, "args": {"ref": "$MEM.x"}}], conv
        )
        assert _responses(conv) == []


# ----------------------------------------------------------------------
# 甲：内存输入安全
# ----------------------------------------------------------------------

class TestMemoryInputSafety:
    def test_address_grammar_rejects_shadow_prefix(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEMM.x", "content": "v"})
        assert _responses(conv)[0][0].startswith("Illegal:")
        assert "MEMM" not in core.mem._data._data, "被拒绝的写入不得改状态"

    def test_address_grammar_rejects_empty_segment(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM.", "content": "v"})
        assert _responses(conv)[0][0].startswith("Illegal:")
        assert "" not in core.mem._data._data

    def test_string_parent_is_typed_error_not_typeerror(self):
        core, conv = _core()
        core.mem["s"] = "hello world"
        _call(core, conv, "memory_write", {"ref": "$MEM.s.ell.x", "content": "v"})
        assert "Error:" in _responses(conv)[0][0]

    def test_rejected_write_leaves_no_intermediate_nodes(self):
        core, conv = _core()
        core.mem["s"] = "hello world"
        before = set(core.mem._data._data.keys())
        _call(core, conv, "memory_write", {"ref": "$MEM.s.ell.x", "content": "v"})
        assert set(core.mem._data._data.keys()) == before

    def test_device_subpath_write_is_typed_error(self):
        core, conv = _core()
        core.mem.mount("outputs", OutputsListDevice())
        _call(core, conv, "memory_write", {"ref": "$MEM.outputs.0.1", "content": "x"})
        assert _responses(conv)[0][0].startswith("Error:")

    def test_nested_value_depth_is_capped(self):
        core, _conv = _core()
        value = "leaf"
        for _ in range(200):
            value = {"k": value}
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.deep", value)

    def test_address_segment_count_is_capped(self):
        core, _conv = _core()
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM." + ".".join(["a"] * 70), "v")

    def test_non_str_tool_content_is_rejected(self):
        """memory_read $conv.N 返回活对象：值域校验把它挡住，不让它进历史。"""
        core, conv = _core()
        _call(core, conv, "memory_read", {"ref": f"$conv.{conv.cid}"})
        content = _responses(conv)[0][0]
        assert content.startswith("Error:")  # 值域是世界状态问题，不是非法参数


# ----------------------------------------------------------------------
# 甲：协议配对
# ----------------------------------------------------------------------

class TestProtocolPairing:
    def test_json_error_gets_a_placeholder_response(self):
        core, conv = _core()
        core._process_return_calls(
            [{"call_id": "tc1", "cmd_type": "json_error", "args": {"error": "bad", "name": "memory_read"}}],
            conv,
        )
        assert _responses(conv) == [("（该工具调用未执行）", "tc1")]

    def test_unknown_tool_gets_a_placeholder_response(self):
        core, conv = _core()
        _call(core, conv, "definitely_not_a_tool", {"a": 1}, call_id="tc2")
        assert _responses(conv) == [("（该工具调用未执行）", "tc2")]

    def test_every_tool_call_gets_exactly_one_response(self):
        core, conv = _core()
        core.mem["x"] = "v"
        core._process_return_calls(
            [
                {"call_id": "a", "cmd_type": "json_error", "args": {"error": "bad"}},
                {"call_id": "b", "cmd_type": "no_such_tool", "args": {}},
                {"call_id": "c", "cmd_type": "memory_read", "args": {"ref": "$MEM.x"}},
                {"call_id": "d", "cmd_type": "memory_read", "args": {}},
            ],
            conv,
        )
        assert [cid for _c, cid in _responses(conv)] == ["a", "b", "c", "d"]


# ----------------------------------------------------------------------
# 乙（决定 1）：API 失败的运行者策略
# ----------------------------------------------------------------------

class TestApiErrorPolicy:
    def _ready(self, fail_times=1):
        core = Core()
        core.lmu = StubLMU(fail_times)
        conv = core.start("sys", "user")
        core._active_cid = core._ready_cids.pop(0)
        return core, conv

    def test_retry_reruns_the_same_turn(self):
        core, conv = self._ready(fail_times=1)
        seen = []
        core.api_error_policy = lambda error, attempt: (seen.append((str(error), attempt)) or "retry")
        core.advance_conversation()
        assert seen == [("boom", 1)]
        assert core.lmu.n_calls == 2
        assert [m.content for m in conv.messages] == ["sys", "user", "done"]

    def test_reload_resets_the_client(self):
        core, _conv = self._ready(fail_times=1)
        core.api_error_policy = lambda error, attempt: "reload"
        core.advance_conversation()
        assert core.lmu.resets == 1
        assert core.lmu.n_calls == 2

    def test_exit_raises_runner_stop(self):
        core, _conv = self._ready(fail_times=99)
        core.api_error_policy = lambda error, attempt: "exit"
        with pytest.raises(RunnerStop):
            core.advance_conversation()

    def test_no_policy_means_exit(self):
        core, _conv = self._ready(fail_times=99)
        with pytest.raises(RunnerStop):
            core.advance_conversation()

    def test_broken_policy_means_exit(self):
        core, _conv = self._ready(fail_times=99)

        def broken(error, attempt):
            raise RuntimeError("policy 自己坏了")

        core.api_error_policy = broken
        with pytest.raises(RunnerStop):
            core.advance_conversation()

    def test_unknown_action_means_exit(self):
        core, _conv = self._ready(fail_times=99)
        core.api_error_policy = lambda error, attempt: "再试试"
        with pytest.raises(RunnerStop):
            core.advance_conversation()

    def test_frame_records_the_api_error(self):
        core, _conv = self._ready(fail_times=99)
        core.monitor.record_baseline(core)
        core.api_error_policy = lambda error, attempt: "exit"
        with pytest.raises(RunnerStop):
            core.advance_conversation()
        assert core.monitor.frame(1).lmu["error"] == "boom"


# ----------------------------------------------------------------------
# 乙（决定 2）：沙箱里没有"紧急停止"
# ----------------------------------------------------------------------

class TestNoEmergencyStopInSandbox:
    def test_base_exception_not_in_palette(self):
        with pytest.raises(NameError):
            exec_safe("raise BaseException('x')", safe_globals())

    def test_plm_user_code_gets_an_error_text(self):
        plm = create_plm("plm.python")
        response = plm.handle_messages(
            [
                {"role": "system", "content": "x = 1"},
                {"role": "user", "content": "raise BaseException('x')"},
            ],
            {},
        )
        assert response["content"].startswith("错误:")

    def test_exception_class_still_available(self):
        exec_safe("try:\n    raise ValueError('v')\nexcept Exception:\n    ok = 1", safe_globals())


# ----------------------------------------------------------------------
# memory_delete（分批三，2026-10-18）：叶子/空容器/树链接可删
# ----------------------------------------------------------------------

class TestMemoryDelete:
    def test_delete_leaf_and_empty_containers(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "s", "mem_type": "str"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.s"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.d"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.L"})
        assert _responses(conv)[-3][0] == "Success deleted str at $MEM.s"
        assert _responses(conv)[-2][0] == "Success deleted dict at $MEM.d"
        assert _responses(conv)[-1][0] == "Success deleted list at $MEM.L"
        _call(core, conv, "memory_delete", {"ref": "$MEM.s"})
        assert "不存在" in _responses(conv)[-1][0]

    def test_non_empty_container_is_refused(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM.d", "key": "x", "mem_type": "str"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.d"})
        assert "非空容器不能删除" in _responses(conv)[-1][0] and "内核态对话" in _responses(conv)[-1][0]
        assert core.mem["d"]["x"] == ""

    def test_root_mount_and_device_space_are_refused(self):
        from avm.memory_device import StringDevice
        core, conv = _core()
        core.mem.mount("dev", StringDevice("v"))
        _call(core, conv, "memory_delete", {"ref": "$MEM"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.dev"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.dev.x"})
        assert "根节点" in _responses(conv)[-3][0]
        assert "挂载点" in _responses(conv)[-2][0]
        assert "设备" in _responses(conv)[-1][0]

    def test_link_node_is_deleted_but_target_survives(self):
        core, conv = _core()
        core.mem["real"] = "目标"
        core.mem._data._data["lk"] = LinkNode(target="$MEM.real")
        _call(core, conv, "memory_delete", {"ref": "$MEM.lk"})
        assert _responses(conv)[0][0] == "Success deleted link at $MEM.lk"
        assert "lk" not in core.mem._data._data and core.mem["real"] == "目标"

    def test_intermediate_link_is_followed(self):
        core, conv = _core()
        core.mem._data._data["box"] = MetaDict(data={"x": "v"})
        core.mem._data._data["alias"] = LinkNode(target="$MEM.box")
        _call(core, conv, "memory_delete", {"ref": "$MEM.alias.x"})
        assert _responses(conv)[0][0] == "Success deleted str at $MEM.alias.x"
        assert "x" not in core.mem["box"]  # 删的是目标里的节点

    def test_list_element_delete_shifts_indices(self):
        core, conv = _core()
        core.mem["L"] = MetaList(data=["a", "b", "c"])
        _call(core, conv, "memory_delete", {"ref": "$MEM.L.0"})
        assert _responses(conv)[0][0] == "Success deleted str at $MEM.L.0"
        assert core.mem["L"].to_list() == ["b", "c"]
        _call(core, conv, "memory_read", {"ref": "$MEM.L.0"})
        assert _responses(conv)[-1][0] == "b"

    def test_negative_index_refused(self):
        core, conv = _core()
        core.mem["L"] = MetaList(data=["a"])
        _call(core, conv, "memory_delete", {"ref": "$MEM.L.-1"})
        assert "负下标不支持" in _responses(conv)[0][0]

    def test_registry_includes_memory_delete(self):
        from avm.core import _instruction_registry, _build_tools
        assert "memory_delete" in _instruction_registry()
        assert "memory_delete" in [t["function"]["name"] for t in _build_tools()]


# ----------------------------------------------------------------------
# 亚对话在返回前被关闭：父对话那次 create_sub 必须补上响应（2026-10-20）
# ----------------------------------------------------------------------

class TestSubClosedBeforeReturn:
    def _parent_with_sub(self):
        import json as _json
        core = Core()
        parent = core.start("sys", "parent")
        other = core.start("sys", "other")
        core.mem["prog"] = MetaDict(data={"content": "x"}, ctrl={"type": "settingup"})
        core.mem["para"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        core.mem["u"] = "任务"
        parent.append_assistant_message("", tool_calls=[{
            "id": "call_sub", "type": "function",
            "function": {"name": "create_sub",
                         "arguments": _json.dumps({"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                                   "para_ref": "$MEM.para"})},
        }])
        core._active_cid = parent.cid
        core._process_return_calls([{"cmd_type": "create_sub", "call_id": "call_sub",
                                     "args": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u",
                                              "para_ref": "$MEM.para"}}], parent)
        sub = [c for c in core._conv_by_cid.values() if c.is_sub][0]
        return core, parent, other, sub

    def test_closing_waiting_sub_answers_parent(self):
        core, parent, other, sub = self._parent_with_sub()
        assert parent.user_batch.tool_responses == []  # 父在等亚对话返回

        core._active_cid = other.cid
        core._process_return_calls(
            [{"cmd_type": "close_conversation", "call_id": "c2", "args": {"cid": sub.cid}}], other
        )
        resp = parent.user_batch.tool_responses
        assert len(resp) == 1 and resp[0].tool_call_id == "call_sub"
        assert resp[0].content.startswith("Error:") and "被关闭" in resp[0].content
        assert parent.cid in core._ready_cids, "父应当被唤醒"
        assert sub.metadata.get("returned") is True

        # 模拟父下一轮推理：批次写进历史，再追加一条无工具调用的 assistant
        for r in parent.user_batch.tool_responses:
            parent.append_tool_message(r.content, r.tool_call_id)
        parent.user_batch.clear()
        parent.append_assistant_message("done")
        Core._check_pairing(parent)  # 不抛

    def test_close_after_normal_return_writes_nothing_extra(self):
        """亚对话已经正常返回过：关闭它不再补第二条响应。"""
        core, parent, other, sub = self._parent_with_sub()
        sub.metadata["returned"] = True  # 模拟已写回
        core._active_cid = other.cid
        core._process_return_calls(
            [{"cmd_type": "close_conversation", "call_id": "c2", "args": {"cid": sub.cid}}], other
        )
        assert parent.user_batch.tool_responses == []


# ----------------------------------------------------------------------
# 创建类指令的输入边界安全（2026-10-23 测试报告批一）
# ----------------------------------------------------------------------

def _seed_create_nodes(core, *, prog_type="settingup", prog_content="内容", model="m"):
    core.mem["prog"] = MetaDict(data={"content": prog_content}, ctrl={"type": prog_type})
    core.mem["para"] = MetaDict(data={"model": model}, ctrl={"type": "para"})


class TestCreateBoundarySafety:
    def test_missing_conversation_cid_is_error_not_crash(self):
        """$conv.<不存在的数字 cid> 曾抛 KeyError 逃逸并终止进程。"""
        core, conv = _core()
        _seed_create_nodes(core)
        _call(core, conv, "create_cmd", {"system_ref": "$conv.999", "para_ref": "$MEM.para"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Error:") and "不存在" in content

    def test_sub_user_ref_must_be_str_node(self):
        core, conv = _core()
        _seed_create_nodes(core)
        core.mem["d"] = MetaDict(data={})
        _call(core, conv, "create_sub",
              {"system_ref": "$MEM.prog", "user_ref": "$MEM.d", "para_ref": "$MEM.para"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Error:") and "不是字符串节点" in content
        assert core.get_conversation(1) is None

    def test_sub_user_ref_conversation_object_is_rejected(self):
        """$conv.<cid> 的值是 Conversation 对象，不能进 user 消息（否则下一安全点内核恐慌）。"""
        core, conv = _core()
        _seed_create_nodes(core)
        _call(core, conv, "create_sub",
              {"system_ref": "$MEM.prog", "user_ref": f"$conv.{conv.cid}", "para_ref": "$MEM.para"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Error:") and "不是字符串节点" in content
        assert core.get_conversation(1) is None
        Core._check_pairing(conv)  # 不抛

    def test_program_without_content_rejected(self):
        core, conv = _core()
        core.mem["prog"] = MetaDict(data={}, ctrl={"type": "settingup"})
        core.mem["para"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Error:") and "缺 content" in content
        assert core.get_conversation(1) is None

    def test_settingup_para_without_model_is_illegal(self):
        core, conv = _core()
        _seed_create_nodes(core)
        core.mem["para"] = MetaDict(data={}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Illegal:") and "缺少 model" in content
        assert core.get_conversation(1) is None


class TestConversationsDisplay:
    def test_list_groups_by_state_and_shows_finished_separately(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        child = int(_responses(conv)[-1][0].split("cid=")[1])
        out = core.unwrap("$MEM.conversations")
        assert "按状态分组" in out and "dormant:" in out
        assert "finished:" not in out
        _call(core, conv, "close_conversation", {"cid": child})
        out = core.unwrap("$MEM.conversations")
        assert "finished:" in out and f"cid={child}" in out


class TestSendAndCloseMessages:
    def _children(self, core, conv, count):
        sids = []
        for _i in range(count):
            _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
            sids.append(int(_responses(conv)[-1][0].split("cid=")[1]))
        return sids

    def test_single_ack_has_cid(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        (child,) = self._children(core, conv, 1)
        _call(core, conv, "send_instruction", {"cid": child, "content": "hi"})
        content = _responses(conv)[-1][0]
        assert content.startswith("已投递到 ") and f"（cid={child}）" in content

    def test_multicast_ack_lists_cids(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        a, b = self._children(core, conv, 2)
        _call(core, conv, "send_instruction", {"cid": [a, b], "content": "hi"})
        content = _responses(conv)[-1][0]
        assert content.startswith("已投递到 cid=[") and str(a) in content and str(b) in content

    def test_close_ready_reports_dropped_batch(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        (child,) = self._children(core, conv, 1)
        _call(core, conv, "send_instruction", {"cid": child, "content": "hi"})
        _call(core, conv, "close_conversation", {"cid": child})
        content = _responses(conv)[-1][0]
        assert "已关闭（finished）" in content and "待执行批次已丢弃" in content

    def test_close_idle_has_no_drop_hint(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        (child,) = self._children(core, conv, 1)
        _call(core, conv, "close_conversation", {"cid": child})
        content = _responses(conv)[-1][0]
        assert "已关闭（finished）" in content and "丢弃" not in content

    def test_closed_target_wording_consistent(self):
        core, conv = _core()
        _seed_create_nodes(core, prog_type="python", prog_content="pass", model="plm.simple")
        (child,) = self._children(core, conv, 1)
        _call(core, conv, "close_conversation", {"cid": child})
        _call(core, conv, "close_conversation", {"cid": child})
        assert _responses(conv)[-1][0] == f"Error: 对话 {child} 已关闭"
        _call(core, conv, "send_instruction", {"cid": child, "content": "hi"})
        content = _responses(conv)[-1][0]
        assert content.startswith("Error:") and "已关闭" in content


# ----------------------------------------------------------------------
# 宿主/面板对话：保留魔数 cid，对普通对话不可见、不可寻址（2026-10-23 批三）
# ----------------------------------------------------------------------

class _FakeChannel:
    def wait_action(self):
        return {"end": True}

    def publish(self, kind, data):
        pass


class TestPanelConversationHidden:
    def _core_with_panel(self):
        core, conv = _core()
        panel = core.start_operator_conversation(_FakeChannel())
        return core, conv, panel

    def test_panel_uses_reserved_cid(self):
        core, _conv, panel = self._core_with_panel()
        from avm.core import PANEL_CID
        assert panel.cid == PANEL_CID
        assert core.get_conversation(PANEL_CID) is panel, "宿主侧仍可直接拿到它"

    def test_hidden_from_conversations_and_scheduler(self):
        core, _conv, panel = self._core_with_panel()
        convs = core.unwrap("$MEM.conversations")
        sched = core.unwrap("$MEM.scheduler")
        assert f"cid={panel.cid} " not in convs
        assert "-1" not in sched, "调度队列里的面板 cid 也要过滤"

    def test_not_addressable_by_instructions(self):
        core, conv, panel = self._core_with_panel()
        _call(core, conv, "send_instruction", {"cid": panel.cid, "content": "hi"})
        assert "不存在" in _responses(conv)[-1][0]
        _call(core, conv, "close_conversation", {"cid": panel.cid})
        assert "不存在" in _responses(conv)[-1][0]
        with pytest.raises(VMMemoryError):
            core.unwrap(f"$conv.{panel.cid}")
        with pytest.raises(VMMemoryError):
            core.unwrap(f"$batch.{panel.cid}")

    def test_monitor_detail_filters_panel_and_does_not_advertise(self):
        core, _conv, panel = self._core_with_panel()
        core.monitor.record(core)
        seq = core.monitor.frames()[-1].seq
        summary = core.unwrap("$MEM.monitor")
        assert "frames." not in summary, "设备自述不宣传单帧读取路径"
        detail = core.unwrap(f"$MEM.monitor.frames.{seq}")
        assert f"[-1" not in detail and " -1," not in detail and "=-1" not in detail
        with pytest.raises(VMMemoryError):
            core.unwrap(f"$MEM.monitor.{seq}")


# ----------------------------------------------------------------------
# 错误分两档（2026-10-19 作者定）：Illegal = 改参数可重试；Error = 世界不配合
# ----------------------------------------------------------------------

class TestErrorTiers:
    def test_illegal_for_bad_input(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "nope", "content": "v"})
        assert _responses(conv)[-1][0].startswith("Illegal:")
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_read", {"ref": "$MEM.d.-1"})
        assert _responses(conv)[-1][0].startswith("Illegal:")  # 字典 -1 是参数误用
        _call(core, conv, "memory_delete", {"ref": "$MEM"})
        assert _responses(conv)[-1][0].startswith("Illegal:")  # 根不可删是策略

    def test_error_for_world_state(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM.missing", "content": "v"})
        assert _responses(conv)[-1][0].startswith("Error:")  # 键不存在：世界状态
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM.d", "key": "x", "mem_type": "str"})
        _call(core, conv, "memory_delete", {"ref": "$MEM.d"})
        assert _responses(conv)[-1][0].startswith("Error:")  # 非空容器：换办法

    def test_make_success_echoes_key(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "带 空格的键", "mem_type": "str"})
        assert _responses(conv)[-1][0] == "Success created str at $MEM.带 空格的键, key='带 空格的键'"

    def test_list_representation_has_element_type(self):
        from avm.types import MetaList
        assert "elem=str" in MetaList(data=[]).to_llm_string()
        assert "elem=dict" in MetaList(data=[{}]).to_llm_string()
        assert "elem=mixed" in MetaList(data=["a", {}]).to_llm_string()


# ----------------------------------------------------------------------
# memory_write 不建节点（分批二，2026-10-18）
# ----------------------------------------------------------------------

class TestWriteDoesNotCreate:
    def test_write_to_missing_path_errors_with_hint(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM.missing", "content": "v"})
        resp = _responses(conv)[0][0]
        assert resp.startswith("Error:") and "memory_make" in resp
        assert "missing" not in core.mem._data._data

    def test_write_to_missing_intermediate_leaves_no_partial(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "dict"})
        _call(core, conv, "memory_write", {"ref": "$MEM.a.b.c", "content": "v"})
        assert _responses(conv)[-1][0].startswith("Error:")
        assert core.mem["a"].to_dict() == {}

    def test_make_chain_then_write(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM.a.b", "key": "c", "mem_type": "str"})
        _call(core, conv, "memory_write", {"ref": "$MEM.a.b.c", "content": "v"})
        assert _responses(conv)[-1][0] == "Success set: $MEM.a.b.c"
        assert core.mem["a"]["b"]["c"] == "v"

    def test_write_existing_nodes_keeps_three_semantics(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "s", "mem_type": "str"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_write", {"ref": "$MEM.s", "content": "内容"})
        _call(core, conv, "memory_write", {"ref": "$MEM.d", "content": "描述"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-1", "content": "元素"})
        assert core.mem["s"] == "内容"                      # str 改内容
        assert core.unwrap("$MEM.d").startswith("dict[keys=[]")  # dict 改描述
        assert "描述" in core.unwrap("$MEM.d")
        assert core.mem["L"].to_list() == ["元素"]          # 列表追加（分批一）

    def test_write_root_changes_its_description(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM", "content": "根描述"})
        assert _responses(conv)[0][0] == "Success set: $MEM"
        assert "根描述" in core.unwrap("$MEM")

    def test_device_write_unaffected(self):
        from avm.memory_device import OutputsListDevice
        core, conv = _core()
        core.mem.mount("out", OutputsListDevice())
        _call(core, conv, "memory_write", {"ref": "$MEM.out.out", "content": "x"})
        assert _responses(conv)[0][0].startswith("Success")
        # 设备写只走命名接口：数字位置写入被拒
        _call(core, conv, "memory_write", {"ref": "$MEM.out.-1", "content": "y"})
        assert _responses(conv)[1][0].startswith("Error:")


# ----------------------------------------------------------------------
# 列表 -1 追加语义与 make 建链（2026-10-18 作者定）
# ----------------------------------------------------------------------

class TestListAppendSemantics:
    def test_write_appends_to_empty_list_with_real_address(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-1", "content": "a"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-1", "content": "b"})
        assert _responses(conv)[-2][0] == "Success appended str at $MEM.L.0"
        assert _responses(conv)[-1][0] == "Success appended str at $MEM.L.1"
        assert core.mem["L"].to_list() == ["a", "b"]

    def test_write_append_requires_same_type(self):
        """同类规则：非空列表元素是 dict 时不能追加 str，且不落笔。"""
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "-1", "mem_type": "dict"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-1", "content": "x"})
        assert "同类节点必须与已有元素同类型" in _responses(conv)[-1][0]
        assert len(core.mem["L"]) == 1

    def test_make_append_types_only_free_on_new_list(self):
        """仅新列表可随意指定类型；列表已有值后 make 与 write 一样受同类规则约束。"""
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "-1", "mem_type": "dict"})
        assert _responses(conv)[-1][0] == "Success created dict at $MEM.L.0"
        # 已有元素是 dict：make 也不能追加 str
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "-1", "mem_type": "str"})
        assert "同类节点必须与已有元素同类型" in _responses(conv)[-1][0]
        # 同类型可以
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "-1", "mem_type": "dict"})
        assert _responses(conv)[-1][0] == "Success created dict at $MEM.L.1"

    def test_make_creates_missing_chain(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM.a.b", "key": "c", "mem_type": "str"})
        assert _responses(conv)[-1][0] == "Success created str at $MEM.a.b.c, key='c'"
        assert type(core.mem["a"]).__name__ == "MetaDict"
        assert core.mem["a"]["b"]["c"] == ""

    def test_make_chain_appends_dict_into_dict_list(self):
        """作者例子：a 是字典列表时，ref=a.-1 追加一个 dict，再在里面建 b。"""
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "list"})
        _call(core, conv, "memory_make", {"ref": "$MEM.a", "key": "-1", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM.a.-1", "key": "b", "mem_type": "str"})
        assert _responses(conv)[-1][0] == "Success created str at $MEM.a.1.b, key='b'"
        assert core.mem["a"].to_list() == [{}, {"b": ""}]

    def test_make_chain_into_string_list_errors(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "list"})
        _call(core, conv, "memory_write", {"ref": "$MEM.a.-1", "content": "s"})
        _call(core, conv, "memory_make", {"ref": "$MEM.a.-1", "key": "b", "mem_type": "str"})
        assert "同类节点必须与已有元素同类型" in _responses(conv)[-1][0]
        assert core.mem["a"].to_list() == ["s"]

    def test_make_chain_empty_segment_after_dict_is_error(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM.a.-1", "key": "b", "mem_type": "str"})
        assert "字典里不能用 -1" in _responses(conv)[-1][0]
        assert core.mem["a"].to_dict() == {}

    def test_make_chain_leaves_no_half_state_on_error(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "str"})
        _call(core, conv, "memory_make", {"ref": "$MEM.a.b", "key": "c", "mem_type": "str"})
        assert _responses(conv)[-1][0].startswith("Error:")
        assert isinstance(core.mem["a"], str)

    def test_negative_indices_are_rejected(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_read", {"ref": "$MEM.L.-1"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-2", "content": "x"})
        assert "负下标不支持" in _responses(conv)[-2][0]
        assert "负下标不支持" in _responses(conv)[-1][0]

    def test_make_numeric_index_does_not_overwrite(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "-1", "mem_type": "str"})
        _call(core, conv, "memory_make", {"ref": "$MEM.L", "key": "0", "mem_type": "str"})
        assert "只能是 -1" in _responses(conv)[-1][0]

    def test_read_container_reports_length_for_latest_access(self):
        """读容器给 len，模型据此读 len-1 拿最新一条（不再有 -1 读）。"""
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "L", "mem_type": "list"})
        _call(core, conv, "memory_write", {"ref": "$MEM.L.-1", "content": "x"})
        _call(core, conv, "memory_read", {"ref": "$MEM.L"})
        assert "len=1" in _responses(conv)[-1][0]
        _call(core, conv, "memory_read", {"ref": "$MEM.L.0"})
        assert _responses(conv)[-1][0] == "x"


# ----------------------------------------------------------------------
# 一个工具调用只写一条响应（两条会让配对不变量停机）
# ----------------------------------------------------------------------

class TestSingleResponsePerCall:
    def test_publish_failure_does_not_write_a_second_response(self, capsys, monkeypatch):
        """memory_make：落笔成功 → 写成功响应 → 发布失败只记 stderr，不再写 Error。

        2026-10-16 之前这里会出两条同 call_id 的响应（Success + Error），
        下一次 API 请求因此被拒（配对不变量会先停机）。发布失败用注入制造：
        指令语法补完之后，正常输入已经走不到这条失败路径了。
        """
        def boom(*args, **kwargs):
            raise ValueError("发布坏了")

        monkeypatch.setattr("avm.core._publish_memory_after", boom)
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "b", "mem_type": "str"})
        assert [r.tool_call_id for r in conv.user_batch.tool_responses] == ["c1"]
        assert _responses(conv)[0][0].startswith("Success created")
        assert "after 事件发布失败" in capsys.readouterr().err
        Core._check_pairing(conv)  # 不抛

    def test_publish_failure_on_read_stays_one_response(self, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("发布坏了")

        monkeypatch.setattr("avm.core._publish_memory_after", boom)
        core, conv = _core()
        core.mem["x"] = "value"
        _call(core, conv, "memory_read", {"ref": "$MEM.x"})
        assert [r.tool_call_id for r in conv.user_batch.tool_responses] == ["c1"]
        assert _responses(conv)[0][0] == "value"

    def test_sub_is_deleted_after_completion(self):
        """亚对话用完即删：返回写回父对话后个体从对话表消失，父拿到一条响应。"""
        core = Core()
        parent = core.start("sys", "hello")
        core.mem["model_params"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
        core.mem["prog"] = MetaDict(data={"content": "x"}, ctrl={"type": "settingup"})
        core.mem["u"] = "任务"
        core._ready_cids.remove(parent.cid)
        core._active_cid = parent.cid
        _call(core, parent, "create_sub",
              {"system_ref": "$MEM.prog", "user_ref": "$MEM.u", "para_ref": "$MEM.model_params"})
        sub = [c for c in core._conv_by_cid.values() if c.is_sub][0]
        sub_cid = sub.cid

        class _Done:
            def exec(self, conversation, para):
                return "子结果", [], conversation

        core._executors[sub_cid] = _Done()
        core._active_cid = sub_cid
        core.advance_conversation()

        assert sub_cid not in core._conv_by_cid, "亚对话完成后应当被删除"
        assert sub_cid not in core._dormant_cids
        assert [r.tool_call_id for r in parent.user_batch.tool_responses] == ["c1"]
        assert parent.user_batch.tool_responses[0].content == "子结果"
        Core._check_pairing(parent)  # 不抛


# ----------------------------------------------------------------------
# 指令语法与创建语义：键语法、只创建不覆盖、读侧引用语法
# ----------------------------------------------------------------------

class TestMemoryGrammarAndCreateOnly:
    """2026-10-17 契约审查的一组输入边界修复（作者在 AVM 内用对话测试得到）。"""

    def test_read_ref_must_start_with_dollar(self):
        core, conv = _core()
        _call(core, conv, "memory_read", {"ref": "MEM"})
        assert _responses(conv)[0][0].startswith("Illegal: ref 必须是 $ 开头")

    def test_read_ref_rejects_empty_segment(self):
        core, conv = _core()
        _call(core, conv, "memory_read", {"ref": "$MEM..x"})
        assert "空段" in _responses(conv)[0][0]

    def test_make_key_must_be_non_empty(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "", "mem_type": "str"})
        assert "不能为空" in _responses(conv)[0][0]
        assert "" not in core.mem._data._data

    def test_make_key_must_not_contain_dot(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a.b", "mem_type": "str"})
        assert "点号" in _responses(conv)[0][0]
        assert "a.b" not in core.mem._data._data  # 不再造出不可寻址的字面键

    def test_make_does_not_overwrite_existing_dict(self):
        """静默覆盖会丢掉整棵子树；现在拒绝并保留原内容。"""
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "loss", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$MEM.loss", "key": "child", "mem_type": "str"})
        _call(core, conv, "memory_write", {"ref": "$MEM.loss.child", "content": "keep"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "loss", "mem_type": "dict"})
        assert "已存在" in _responses(conv)[-1][0]
        assert core.mem["loss"]["child"] == "keep"

    def test_make_does_not_overwrite_across_types(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "t", "mem_type": "str"})
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "t", "mem_type": "dict"})
        assert "已存在" in _responses(conv)[-1][0]
        assert isinstance(core.mem["t"], str)

    def test_make_cannot_clobber_a_device_mount_link(self):
        """键与设备挂载点名冲突：此前会覆写链接节点 → 设备表不一致 → 内核恐慌。"""
        class _Dev(MemoryDevice):
            def to_llm_string(self):
                return "dev"

        core, conv = _core()
        core.mem.mount("a", _Dev())
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "a", "mem_type": "str"})
        assert "已存在" in _responses(conv)[0][0]
        assert core.unwrap("$MEM.a") == "dev"
        core._check_invariants()  # 不抛：设备表一致

    def test_write_path_still_rejects_empty_segment(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM..x", "content": "v"})
        assert "空段" in _responses(conv)[0][0]

    def test_valid_make_still_works(self):
        core, conv = _core()
        _call(core, conv, "memory_make", {"ref": "$MEM", "key": "ok", "mem_type": "dict"})
        assert _responses(conv)[0][0].startswith("Success created dict at $MEM.ok")
        assert type(core.mem["ok"]).__name__ == "MetaDict"


# ----------------------------------------------------------------------
# 乙（决定 3）：内核缺陷不再被读路径吞掉
# ----------------------------------------------------------------------

class TestKernelDefectIsNotSwallowed:
    def test_read_path_defect_propagates(self):
        core, conv = _core()

        def boom(*args, **kwargs):
            raise RuntimeError("kernel defect")

        core.unwrap = boom
        with pytest.raises(RuntimeError):
            _call(core, conv, "memory_read", {"ref": "$MEM.x"})
        assert _responses(conv) == [], "内核缺陷不得伪装成工具响应"


# ----------------------------------------------------------------------
# 丁：启动期类型检查
# ----------------------------------------------------------------------

_BASE_IMAGE = {
    "meta": {"version": 1},
    "mem": {
        "model_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "gpt-4"}},
        },
        "seed": {"kind": "str", "value": "x"},
    },
}


def _write_image(tmp_path, doc, name="img.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


class TestImageStartupValidation:
    def test_meta_must_be_an_object(self, tmp_path):
        doc = dict(_BASE_IMAGE, meta=1)
        with pytest.raises(ImageError):
            load_image(Core(), _write_image(tmp_path, doc))

    def test_para_ref_must_be_a_non_empty_string(self, tmp_path):
        doc = dict(_BASE_IMAGE, para_ref=None)
        with pytest.raises(ImageError):
            load_image(Core(), _write_image(tmp_path, doc))

    def test_devices_entries_must_be_objects(self, tmp_path):
        doc = dict(_BASE_IMAGE, devices=["a"])
        with pytest.raises(ImageError):
            load_image(Core(), _write_image(tmp_path, doc))

    def test_duplicate_mount_path_is_rejected(self, tmp_path):
        """同一路径挂两台设备：拒绝在边界，不让设备表变得不一致。"""
        core = Core()
        dev_a = OutputsListDevice()
        dev_a.mount_path = "out"
        dev_a.device_id = "id-a"
        dev_b = OutputsListDevice()
        dev_b.mount_path = "out"
        dev_b.device_id = "id-b"
        core.mem.mount("out", dev_a)
        with pytest.raises(VMMemoryError):
            core.mem.mount("out", dev_b)


# ----------------------------------------------------------------------
# 丙：六条不变量
# ----------------------------------------------------------------------

class TestInvariants:
    def test_scheduling_mutex(self):
        core, conv = _core()
        core._ready_cids.extend([conv.cid, conv.cid])
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_budget_counter(self):
        core, _conv = _core()
        core._activation_instructions = -1
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_active_has_a_source(self):
        core, _conv = _core()
        core._active_cid = 999
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_history_pairing(self):
        core, conv = _core()
        conv.messages.append(
            AssistantMessage(
                content="",
                tool_calls=[{"id": "x1", "type": "function", "function": {"name": "memory_read", "arguments": "{}"}}],
            )
        )
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_pairing_counts_pending_batch(self):
        """工具响应先在 batch 里、下一轮才写进历史：待发内容也算配对。"""
        core, conv = _core()
        conv.messages.append(
            AssistantMessage(
                content="",
                tool_calls=[{"id": "x1", "type": "function", "function": {"name": "memory_read", "arguments": "{}"}}],
            )
        )
        conv.user_batch.add_tool_response("v", "x1")
        core._check_invariants()  # 不报

    def test_value_domain(self):
        core, conv = _core()
        # 绕过入队时的类型检查，直接放一个坏值：不变量要能自己认出来
        conv.user_batch.tool_responses.append(ToolMessage(content=123, tool_call_id="x1"))
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_tool_response_rejects_non_string(self):
        core, conv = _core()
        with pytest.raises(TypeError):
            conv.user_batch.add_tool_response(object(), "x1")

    def test_device_table(self):
        core, _conv = _core()
        core.mem._device_ids["ghost"] = object()
        with pytest.raises(KernelPanic):
            core._check_invariants()

    def test_normal_state_passes(self):
        core, _conv = _core()
        core._check_invariants()

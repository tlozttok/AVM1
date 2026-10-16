"""指令边界与内核稳定性的回归用例。

对应 plan-kernel-stability-phase1-2026-10-16.md 的验收一节：把讨论案里的探针与
三件已定事项固化成用例。每条用例都回答同一个问题——这个输入会不会让进程退出。
"""

import json

import pytest

from avm.core import Core
from avm.exceptions import APIError, KernelPanic, RunnerStop, VMMemoryError
from avm.image import ImageError, load_image
from avm.memory_device import OutputsListDevice
from avm.python_server import create_plm
from avm.sandbox import exec_safe, safe_globals
from avm.types import AssistantMessage, ToolMessage


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
        assert content.startswith("Error:"), content
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
        assert _responses(conv)[0][0].startswith("Error:")
        assert "MEMM" not in core.mem._data._data, "被拒绝的写入不得改状态"

    def test_address_grammar_rejects_empty_segment(self):
        core, conv = _core()
        _call(core, conv, "memory_write", {"ref": "$MEM.", "content": "v"})
        assert _responses(conv)[0][0].startswith("Error:")
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
        assert content.startswith("Error:")


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

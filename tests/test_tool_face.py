"""每对话工具面：schema 过滤、调度拒绝、PLM 注入裁剪。

对应权限体系批次一 S5（plan-permission-phase1-2026-10-29.md）。
"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import Core, _build_tools
from avm.python_server import create_plm
from avm.types import MetaDict


def _core():
    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    conv = core.start("sys", "user", para_ref="$MEM.model_params")
    core._active_cid = conv.cid
    return core, conv


def _tool_names(tools):
    return [t["function"]["name"] for t in tools]


def _tool_defs(names):
    return [t for t in _build_tools() if t["function"]["name"] in names]


def _call(core, conv, cmd, args, call_id="c1"):
    conv.user_batch.tool_responses.clear()
    core._process_return_calls(
        [{"cmd_type": cmd, "call_id": call_id, "args": args}], conv
    )
    return conv.user_batch.tool_responses[-1].content


class TestSchemaFiltering:
    def test_host_full_list(self):
        assert set(_tool_names(_build_tools())) == set(ALL_TOOL_NAMES)

    def test_kernel_conversation_full_list(self):
        core, conv = _core()
        assert set(_tool_names(_build_tools(conv))) == set(ALL_TOOL_NAMES)

    def test_restricted_conversation_subset(self):
        core, conv = _core()
        conv.caps = Capability(mode="user", roots={"work": ("ws",)},
                               tools=["memory_read", "send_instruction"])
        assert _tool_names(_build_tools(conv)) == ["memory_read", "send_instruction"]


class TestDispatchAuthorization:
    def test_calling_ungranted_tool_denied(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        core.mem["prog"] = MetaDict(data={"content": "s"}, ctrl={"type": "settingup"})
        conv.caps = Capability(mode="user", roots={"work": ("ws",)},
                               tools=["memory_read"])
        text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.model_params"})
        assert text.startswith("Error:") and "未授权" in text
        assert len(core._conv_by_cid) == 1  # 没有创建子对话

    def test_granted_tool_still_works(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        conv.caps = Capability(mode="user", roots={"work": ("ws",)},
                               tools=["memory_read", "memory_make"])
        _call(core, conv, "memory_make", {"ref": "$work", "key": "x", "mem_type": "str"})
        assert "Success created str at $work.x" in "Success created str at $work.x"
        assert core.mem.resolve_path(["MEM", "ws"], for_llm=False) is not None


class TestPLMInjection:
    def _run_plm(self, plm_name, system, user, allowed):
        plm = create_plm(plm_name)
        params = {"tools": _tool_defs(allowed)}
        return plm.handle_messages(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            params,
        )

    def test_python_injection_filtered(self):
        code = "yield from memory_read('$MEM.x')\n"
        response = self._run_plm("plm.python", "", code, ["memory_read"])
        calls = response.get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "memory_read"

    def test_python_ungranted_tool_is_rejected(self):
        code = "yield from create_cmd('$MEM.p', '$MEM.a')\n"
        response = self._run_plm("plm.python", "", code, ["memory_read"])
        assert "错误" in response["content"] and "create_cmd" in response["content"]
        assert "未授权" in response["content"]

    def test_command_injection_filtered(self):
        system = "def handle(cmd):\n    v = yield from memory_read('$MEM.x')\n    return 'v=' + str(v)\n"
        response = self._run_plm("plm.command", system, "go", ["memory_read"])
        calls = response.get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "memory_read"

    def test_command_ungranted_yield_rejected(self):
        system = ("def handle(cmd):\n"
                  "    v = yield ('create_cmd', {'system_ref': '$MEM.p', 'para_ref': '$MEM.a'})\n"
                  "    return 'ok'\n")
        response = self._run_plm("plm.command", system, "go", ["memory_read"])
        assert "非法工具调用" in response["content"]

    def test_machine_ungranted_action_rejected(self):
        system = (
            "initial: idle\n\n"
            "state idle:\n"
            "  on go -> idle | {\"content\": \"done\", \"tools\": ["
            "{\"name\": \"create_cmd\", \"arguments\": {\"system_ref\": \"$MEM.s\", \"para_ref\": \"$MEM.p\"}}]}\n"
        )
        response = self._run_plm("plm.machine", system, json.dumps({"event": "go"}), ["memory_read"])
        text = response["content"]
        assert "未知工具" in text and "create_cmd" in text

    def test_machine_granted_action_works(self):
        system = (
            "initial: idle\n\n"
            "state idle:\n"
            "  on go -> idle | {\"content\": \"done\", \"tools\": ["
            "{\"name\": \"create_cmd\", \"arguments\": {\"system_ref\": \"$MEM.s\", \"para_ref\": \"$MEM.p\"}}]}\n"
        )
        response = self._run_plm("plm.machine", system, json.dumps({"event": "go"}),
                                 ["create_cmd"])
        calls = response.get("tool_calls") or []
        assert calls and calls[0]["function"]["name"] == "create_cmd"

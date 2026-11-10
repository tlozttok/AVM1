"""每对话地址空间（根表）与显示纪律。

对应权限体系批次一 S2（plan-permission-phase1-2026-10-29.md）：
- Memory 的公开方法接受 AddressSpace；`space=None` 保持旧行为。
- 根名展开、真实目标与请求别名分离；错误/响应地址只写别名。
- 受限对话看不到 `$MEM`、设备简写与 `$conv`/`$batch`。
"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, AddressSpace, Capability
from avm.core import Core
from avm.exceptions import MemoryLinkBrokenError, VMMemoryError
from avm.memory import Memory
from avm.types import LinkNode, MetaDict, StrNode


def _core():
    core = Core()
    conv = core.start("sys", "user")
    core._active_cid = conv.cid
    return core, conv


def _call(core, conv, cmd, args, call_id="c1"):
    core._process_return_calls(
        [{"cmd_type": cmd, "call_id": call_id, "args": args}], conv
    )


def _last(conv):
    return conv.user_batch.tool_responses[-1].content


def _restrict(conv, roots, tools=None, groups=()):
    conv.caps = Capability(
        mode="user",
        groups=groups,
        roots=roots,
        tools=ALL_TOOL_NAMES if tools is None else tools,
    )


# ----------------------------------------------------------------------
# Memory 层：根展开与显示
# ----------------------------------------------------------------------

class TestMemoryAddressSpace:
    def _mem(self):
        mem = Memory()
        mem["a"] = MetaDict(data={"b": MetaDict(data={"data": "secret"})})
        return mem

    def test_resolve_root_alias(self):
        mem = self._mem()
        space = AddressSpace({"data": ("a", "b", "data")})
        assert mem.resolve_path(["data"], for_llm=False, space=space) == "secret"
        assert mem.unwrap(["$", "data"], space=space) == "secret"

    def test_unknown_root_is_uniform_error(self):
        mem = self._mem()
        space = AddressSpace({"data": ("a", "b", "data")})
        with pytest.raises(VMMemoryError) as ei:
            mem.resolve_path(["nope"], space=space)
        assert "引用根不可用" in str(ei.value)
        with pytest.raises(VMMemoryError) as ei2:
            mem.resolve_path(["other"], space=space)
        assert str(ei.value).split("：")[0] == str(ei2.value).split("：")[0]

    def test_broken_root_target_reports_root_invalid(self):
        mem = self._mem()
        space = AddressSpace({"data": ("missing", "node")})
        with pytest.raises(MemoryLinkBrokenError) as ei:
            mem.resolve_path(["data"], space=space)
        text = str(ei.value)
        assert "根无效" in text
        assert "missing" not in text

    def test_error_uses_alias_not_target(self):
        mem = self._mem()
        space = AddressSpace({"data": ("a", "b", "data")})
        with pytest.raises(VMMemoryError) as ei:
            mem.resolve_path(["data", "nope"], space=space)
        text = str(ei.value)
        assert "$data.nope" in text
        assert "a.b.data" not in text

    def test_write_read_append_make_delete_via_alias(self):
        mem = Memory()
        mem["ws"] = MetaDict(data={})
        space = AddressSpace({"work": ("ws",)})
        assert mem.make("$work", "n", "list", space=space) == "$work.n"
        assert mem.set("$work.n.-1", "v1", space=space) == "$work.n.0"
        assert mem.set("$work.n.-1", "v2", space=space) == "$work.n.1"
        assert mem.unwrap(["$", "work", "n", "1"], space=space) == "v2"
        assert mem.make("$work", "objs", "list", space=space) == "$work.objs"
        assert mem.make("$work.objs.-1", "inner", "dict", space=space) == "$work.objs.0.inner"
        assert mem.delete("$work.objs.0.inner", space=space) == ("dict", "$work.objs.0.inner")
        assert mem.delete("$work.objs.0", space=space) == ("dict", "$work.objs.0")
        assert mem.delete("$work.objs", space=space) == ("list", "$work.objs")
        assert mem.delete("$work.n.1", space=space) == ("str", "$work.n.1")
        assert mem.delete("$work.n.0", space=space) == ("str", "$work.n.0")
        assert mem.delete("$work.n", space=space) == ("list", "$work.n")

    def test_delete_bare_root_refused(self):
        mem = Memory()
        mem["ws"] = MetaDict(data={})
        space = AddressSpace({"work": ("ws",)})
        with pytest.raises(VMMemoryError) as ei:
            mem.delete("$work", space=space)
        assert "不能删除根" in str(ei.value)

    def test_split_and_follow_with_root(self):
        mem = Memory()
        mem["real"] = MetaDict(data={"x": "v"})
        mem["ws"] = MetaDict(data={})
        mem._data._data["ws"]._data["lk"] = LinkNode(target="$MEM.real")
        space = AddressSpace({"work": ("ws",)})
        assert mem.split_path("$work.lk", space=space) == ("memory", ["ws", "lk"])
        assert mem.follow_link_target("$work.lk.x", space=space) == ("memory", ["real", "x"])

    def test_device_shorthand_needs_mem(self):
        mem = Memory()
        mem["ws"] = MetaDict(data={})
        space = AddressSpace({"work": ("ws",)})
        with pytest.raises(VMMemoryError) as ei:
            mem.resolve_path(["inputs"], space=space)
        assert "引用根不可用" in str(ei.value)


# ----------------------------------------------------------------------
# 指令层：受限对话
# ----------------------------------------------------------------------

class TestRestrictedConversation:
    def test_user_conversation_uses_workspace_alias(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        _restrict(conv, {"work": ("ws",)})
        _call(core, conv, "memory_make", {"ref": "$work", "key": "x", "mem_type": "str"})
        assert _last(conv) == "Success created str at $work.x, key='x'"
        _call(core, conv, "memory_write", {"ref": "$work.x", "content": "hi"})
        assert _last(conv) == "Success set: $work.x"
        _call(core, conv, "memory_read", {"ref": "$work.x"})
        assert _last(conv) == "hi"
        # 工作空间外的东西不可见
        _call(core, conv, "memory_read", {"ref": "$MEM"})
        assert _last(conv).startswith("Error:") and "a.b.ws" not in _last(conv)
        _call(core, conv, "memory_read", {"ref": "$conversations"})
        assert _last(conv).startswith("Error:")

    def test_user_conversation_cannot_use_pseudo_roots(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        _restrict(conv, {"work": ("ws",)})
        _call(core, conv, "memory_read", {"ref": "$conv.0"})
        assert "引用根不可用" in _last(conv)

    def test_after_event_target_stays_alias(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        _restrict(conv, {"work": ("ws",)})
        core._turn_events.clear()
        _call(core, conv, "memory_make", {"ref": "$work", "key": "x", "mem_type": "str"})
        targets = [e.get("target") for e in core._turn_events
                   if e.get("event_type") == "memory_make" and e.get("phase") == "after"]
        assert targets == ["$work.x"]

    def test_listener_registration_resolves_alias(self):
        from avm.types import MetaDict as MD

        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        core.mem["handler"] = MD(
            data={"content": "def handle(event):\n    return {}\n",
                  "condition": StrNode(json.dumps({"preset": "self.lifecycle"}))},
            ctrl={"type": "handler"},
        )
        # handler_ref 经根表别名 $h 解析；监听条件用用户态 preset
        _restrict(conv, {"work": ("ws",), "h": ("handler",)})
        _call(core, conv, "memory_make", {"ref": "$work", "key": "d", "mem_type": "dict"})
        _call(core, conv, "memory_make", {"ref": "$work.d", "key": "x", "mem_type": "str"})
        _call(core, conv, "listen_event", {
            "action": "listen", "handler_ref": "$h",
        })
        assert "listener_id=" in _last(conv)

    def test_kernel_conversation_keeps_old_behavior(self):
        core, conv = _core()
        core.mem["k"] = MetaDict(data={"x": "v"})
        _call(core, conv, "memory_read", {"ref": "$MEM.k"})
        assert _last(conv).startswith("dict[keys=")
        _call(core, conv, "memory_read", {"ref": "$conversations"})
        assert _last(conv).startswith("对话列表")

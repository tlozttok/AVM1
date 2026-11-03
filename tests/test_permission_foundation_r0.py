"""R0 权限基础一致性修复测试（B2/B3/B7/B9/C1/C2）。

不调真实 API；全部用 Core 直调 + 指令执行。
"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import (
    CloseInstruction,
    Core,
    CreateInstruction,
    MemoryMakeInstruction,
    MemoryReadInstruction,
    MemoryWriteInstruction,
)
from avm.exceptions import VMMemoryError
from avm.types import LinkNode, MetaDict, MetaList, StrNode


def _caps_node(caps):
    data = {}
    for key in ("mode", "workspace"):
        if key in caps:
            data[key] = StrNode(str(caps[key]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            data[key] = MetaList([StrNode(str(x)) for x in caps[key]])
    return MetaDict(data=data)


def _core():
    core = Core()
    core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    core.mem["s"] = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    root = core.start("sys", "user", para_ref="$MEM.p")
    core._active_cid = root.cid
    return core, root


def _handler(code="def handle(event):\n    return {}\n"):
    return MetaDict(data={"content": code}, ctrl={"type": "handler"})


def _write(core, conv, ref, content, call_id="w1"):
    conv.user_batch.tool_responses.clear()
    MemoryWriteInstruction(call_id, conv.cid, {"ref": ref, "content": content}).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


def _make(core, conv, ref, key, mem_type="dict", call_id="m1"):
    conv.user_batch.tool_responses.clear()
    MemoryMakeInstruction(call_id, conv.cid, {
        "ref": ref, "key": key, "mem_type": mem_type,
    }).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


def _user_conv(core, root):
    """内核 root 创建一个用户态对话（私有工作空间），返回该对话。"""
    core.install_security_config({}, ("workspaces",))
    up = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    up["caps"] = _caps_node({"mode": "user", "workspace": "work",
                             "tools": list(ALL_TOOL_NAMES)})
    core.mem["up"] = up
    CreateInstruction("c1", root.cid, {
        "system_ref": "$MEM.up", "para_ref": "$MEM.p",
    }).execute(core, root)
    text = root.user_batch.tool_responses[-1].content
    assert text.startswith("Success created: cid="), text
    return core.get_conversation(int(text.split("=")[-1]))


def _observe(core):
    seen = []
    core.events.add_observer(seen.append)
    return seen


class TestB2TargetInfoContext:
    def test_make_before_has_parent_and_prefix(self):
        core, root = _core()
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        seen = _observe(core)
        core.mem["h"] = _handler()
        core.events.register(0, "memory_make", "$MEM.box.new", "$MEM.h", phase="before",
                             kernel_mode=True)
        _make(core, root, "$MEM.box", "new")
        before = [e for e in seen if e["event_type"] == "memory_make" and e["phase"] == "before"][-1]
        ti = before["target_info"]
        assert ti["exists"] is False
        assert ti["parent"]["type"] == "dict"
        assert ti["parent"]["alias"] == "$MEM.box"
        assert ti["longest_existing_prefix"]["alias"] == "$MEM.box"
        assert ti["longest_existing_prefix"]["depth"] == 1
        assert "$MEM.workspaces" not in json.dumps(ti)

    def test_append_before_has_list_parent(self):
        core, root = _core()
        core.mem.make("$MEM", "lst", "list", owner_cid=0)
        seen = _observe(core)
        core.mem["h"] = _handler()
        core.events.register(0, "memory_write", "$MEM.lst.-1", "$MEM.h", phase="before",
                             kernel_mode=True)
        _write(core, root, "$MEM.lst.-1", "v")
        before = [e for e in seen if e["event_type"] == "memory_write" and e["phase"] == "before"][-1]
        ti = before["target_info"]
        assert ti["parent"]["type"] == "list"
        assert ti["parent"]["alias"] == "$MEM.lst"
        assert ti["longest_existing_prefix"]["alias"] == "$MEM.lst"

    def test_delete_link_target_info_is_link_node(self):
        core, root = _core()
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.make("$MEM.box", "msg", "str", owner_cid=0)
        core.mem._data["alias"] = LinkNode(target="$MEM.box.msg")
        core.mem._data["alias"].set_ctrl({"owner": "0"})
        seen = _observe(core)
        core.mem["h"] = _handler()
        core.events.register(0, "memory_delete", "$MEM.alias", "$MEM.h", phase="before",
                             kernel_mode=True)
        from avm.core import MemoryDeleteInstruction
        MemoryDeleteInstruction("d1", 0, {"ref": "$MEM.alias"}).execute(core, root)
        before = [e for e in seen if e["event_type"] == "memory_delete" and e["phase"] == "before"][-1]
        ti = before["target_info"]
        assert ti["exists"] is True
        assert ti["type"] == "link"


class TestB3AncestorDisplay:
    def test_ancestors_have_no_real_path_and_depth_cap(self):
        core, root = _core()
        node = core.mem._data
        for i in range(40):
            child = MetaDict(data={})
            node[f"n{i}"] = child
            node = child
        core.mem._data["n0"].set_ctrl({"owner": "0"})
        target = "$MEM." + ".".join(f"n{i}" for i in range(40))
        seen = _observe(core)
        core.mem["h"] = _handler()
        core.events.register(0, "memory_read", target, "$MEM.h", phase="before",
                             kernel_mode=True)
        MemoryReadInstruction("r1", 0, {"ref": target}).execute(core, root)
        before = [e for e in seen if e["event_type"] == "memory_read" and e["phase"] == "before"][-1]
        ancestors = before["target_info"]["ancestors"]
        assert 0 < len(ancestors) <= 32
        assert all("ref" not in entry for entry in ancestors)
        assert ancestors[-1]["depth"] == 39
        assert "$MEM.n0" not in json.dumps(ancestors)


class TestB7NotifyRestriction:
    def _user_listener(self, core, root):
        user = _user_conv(core, root)
        # 用户在自己的工作空间建目标节点和 handler（owner 是用户自己）
        assert _make(core, user, "$work", "x", "str").startswith("Success")
        assert _make(core, user, "$work", "h", "dict").startswith("Success")
        handler = core.mem.resolve_path(["MEM"] + list(user.caps.roots["work"]) + ["h"],
                                        for_llm=False)
        handler["content"] = (
            "def handle(event):\n"
            "    return {'notify': [{'cid': 2, 'content': 'other'}]}\n"
        )
        handler.set_ctrl({"type": "handler"})
        lid = core.events.register(user.cid, "memory_write", "$work.x", "$work.h",
                                   phase="before", space=user.caps.space())
        return user, lid

    def test_user_listener_cannot_notify_kernel(self):
        core, root = _core()
        user, lid = self._user_listener(core, root)
        from avm.core import _build_conversation
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        target = "$MEM." + ".".join(user.caps.roots["work"] + ("x",))
        text = _write(core, root, target, "v")
        assert text.startswith("Error:") and "用户态监听器" in text
        assert core.events.listeners[-1]["listener_id"] == lid
        assert sink.user_batch.user_contents == []

    def test_user_listener_can_notify_self(self):
        core, root = _core()
        user, lid = self._user_listener(core, root)
        handler = core.mem.resolve_path(["MEM"] + list(user.caps.roots["work"]) + ["h"],
                                        for_llm=False)
        handler["content"] = (
            "def handle(event):\n"
            f"    return {{'notify': [{{'cid': {user.cid}, 'content': 'self-note'}}]}}\n"
        )
        core.events.cancel(lid, user.cid)
        core.events.register(user.cid, "memory_write", "$work.x", "$work.h",
                             phase="before", space=user.caps.space())
        target = "$MEM." + ".".join(user.caps.roots["work"] + ("x",))
        text = _write(core, root, target, "v")
        assert text.startswith("Success") or text.startswith("Success appended")
        assert "self-note" in user.user_batch.user_contents


class TestB9CloseRestriction:
    def test_user_cannot_close_kernel(self):
        core, root = _core()
        user = _user_conv(core, root)
        user.user_batch.tool_responses.clear()
        CloseInstruction("cl", user.cid, {"cid": 0}).execute(core, user)
        text = user.user_batch.tool_responses[-1].content
        assert text.startswith("Illegal:") and "内核态" in text
        assert 0 not in core._finished

    def test_user_can_close_user(self):
        core, root = _core()
        user = _user_conv(core, root)
        from avm.core import _build_conversation
        other = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(other.cid)
        other.caps = Capability(mode="user", roots={}, tools=list(ALL_TOOL_NAMES))
        user.user_batch.tool_responses.clear()
        CloseInstruction("cl", user.cid, {"cid": other.cid}).execute(core, user)
        assert other.cid in core._finished


class TestC1C2ReservedAreas:
    def test_image_registry_rejects_control_devices(self):
        core = Core()
        for target in (("conversations",), ("listeners",), ("roots",), ("caps",),
                       ("conversations", "0", "roots")):
            with pytest.raises(VMMemoryError):
                core.install_security_config({"admin": target}, ("workspaces",))

    def test_runtime_roots_rejects_control_devices(self):
        core, root = _core()
        user = _user_conv(core, root)
        for target in ("$MEM.conversations", "$MEM.listeners", "$MEM.roots",
                       "$MEM.caps", "$MEM.conversations.0.roots"):
            text = _write(core, root, f"$MEM.conversations.{user.cid}.roots",
                          json.dumps({"admin": target}))
            assert text.startswith("Illegal:"), (target, text)

    def test_observation_device_still_grantable(self):
        core, root = _core()
        user = _user_conv(core, root)
        text = _write(core, root, f"$MEM.conversations.{user.cid}.roots",
                      json.dumps({"sched": "$MEM.scheduler"}))
        assert text.startswith("Success"), text

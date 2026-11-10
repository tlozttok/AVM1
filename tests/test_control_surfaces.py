"""批次三：内核设备控制面（$MEM.conversations / $MEM.listeners 可写字段）。

不调真实 API：直接执行 memory_write 指令 + 检查 Core 运行时对象与事件载荷。
"""

import json

import pytest

from avm.core import Core, CreateInstruction, MemoryReadInstruction, MemoryWriteInstruction
from avm.exceptions import VMMemoryError
from avm.memory_device import StringDevice
from avm.types import MetaDict, MetaList, StrNode


def _caps_node(caps):
    data = {}
    if "mode" in caps:
        data["mode"] = StrNode(str(caps["mode"]))
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


def _read(core, conv, ref, call_id="r1"):
    conv.user_batch.tool_responses.clear()
    MemoryReadInstruction(call_id, conv.cid, {"ref": ref}).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


def _user_conv(core, root):
    """内核态 root 创建一个用户态对话，返回该对话。"""
    up = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    up["caps"] = _caps_node({"mode": "user"})
    core.mem["up"] = up
    conv = CreateInstruction("c1", root.cid, {
        "system_ref": "$MEM.up", "para_ref": "$MEM.p",
    })
    conv.execute(core, root)
    text = root.user_batch.tool_responses[-1].content
    assert text.startswith("Success created: cid="), text
    return core.get_conversation(int(text.split("=")[-1]))


class TestConversationDeviceWrites:
    def test_groups_write_and_read_back(self):
        core, root = _core()
        text = _write(core, root, "$MEM.conversations.0.groups", '["worker", "auditor"]')
        assert text == "Success set: $MEM.conversations.0.groups"
        assert root.caps.groups == ("worker", "auditor")
        assert json.loads(_read(core, root, "$MEM.conversations.0.groups")) == ["worker", "auditor"]
        # 组名去重
        _write(core, root, "$MEM.conversations.0.groups", '["worker", "worker"]')
        assert root.caps.groups == ("worker",)

    def test_new_groups_immediately_visible_in_event_payload(self):
        core, root = _core()
        core.mem["box"] = MetaDict(data={}, ctrl={"owner": "0"})
        core.mem["h"] = _handler(
            "def handle(event):\n"
            "    return {'notify': [{'cid': 1, 'content': "
            "json.dumps(event['source']['conversation']['groups'])}]}\n"
        )
        sink = core.start("s", "u", para_ref="$MEM.p")
        core._dormant_cids.append(sink.cid)
        core._ready_cids = [c for c in core._ready_cids if c != sink.cid]
        _write(core, root, "$MEM.conversations.0.groups", '["worker"]')
        core.events.register(0, "memory_read", "$MEM.box", "$MEM.h", phase="before")
        MemoryReadInstruction("r2", 0, {"ref": "$MEM.box"}).execute(core, root)
        assert json.loads(sink.user_batch.user_contents[-1]) == ["worker"]

    def test_before_listener_can_block_control_write(self):
        core, root = _core()
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': '禁止改组'}"
        )
        core.events.register(0, "memory_write", "$MEM.conversations.0.groups", "$MEM.h",
                             phase="before", tier="kernel", kernel_mode=True)
        text = _write(core, root, "$MEM.conversations.0.groups", '["auditor"]')
        assert text == "Interrupted: 禁止改组"
        assert root.caps.groups == ()

    def test_groups_validation(self):
        core, root = _core()
        assert _write(core, root, "$MEM.conversations.0.groups", "not-json").startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.groups", '{"a":1}').startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.groups", '[""]').startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.groups", "[1]").startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.mode", '"user"').startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.tools", "[]").startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.0.wat", '"x"').startswith("Illegal:")
        assert _write(core, root, "$MEM.conversations.99.groups", "[]").startswith("Error:")

    def test_meta_replace_and_single_key(self):
        core, root = _core()
        assert _write(core, root, "$MEM.conversations.0.meta", '{"role":"admin","level":"9"}') == \
            "Success set: $MEM.conversations.0.meta"
        assert root.attrs == {"role": "admin", "level": "9"}
        assert json.loads(_read(core, root, "$MEM.conversations.0.meta")) == {
            "role": "admin", "level": "9"}
        _write(core, root, "$MEM.conversations.0.meta.role", "auditor")
        assert root.attrs == {"role": "auditor", "level": "9"}
        assert _read(core, root, "$MEM.conversations.0.meta.role") == "auditor"
        assert _read(core, root, "$MEM.conversations.0.meta.missing").startswith("Error:")
        assert _write(core, root, "$MEM.conversations.0.meta", '["x"]').startswith("Illegal:")

    def test_kernel_roots_write_rejected(self):
        core, root = _core()
        text = _write(core, root, "$MEM.conversations.0.roots", '{"lib": "$MEM.lib"}')
        assert text.startswith("Illegal:") and "内核态" in text
        assert root.caps.roots == {"MEM": ()}

    def test_closed_conversation_returns_error(self):
        core, root = _core()
        from avm.core import _build_conversation
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)
        core._close_conversation(child.cid)
        assert _write(core, root, f"$MEM.conversations.{child.cid}.groups", "[]").startswith("Error:")
        assert "state=finished" in _read(core, root, f"$MEM.conversations.{child.cid}")


class TestRootsControl:
    def _user_with_root(self):
        core, root = _core()
        user = _user_conv(core, root)
        return core, root, user

    def test_roots_write_preserves_caps_root(self):
        core, root, user = self._user_with_root()
        caps_target = user.caps.roots["caps"]
        text = _write(core, root, "$MEM.conversations.1.roots", '{"lib": "$MEM.lib"}')
        assert text == "Success set: $MEM.conversations.1.roots", text
        assert user.caps.roots["lib"] == ("lib",)
        assert user.caps.roots["caps"] == caps_target
        data = json.loads(_read(core, root, "$MEM.conversations.1.roots"))
        assert data["caps"]["core"] is True
        assert data["lib"] == {"target": "$MEM.lib", "core": False}

    def test_granted_root_is_usable_after_write(self):
        core, root, user = self._user_with_root()
        core.mem["lib"] = MetaDict(data={"greeting": "hi"})
        _write(core, root, "$MEM.conversations.1.roots", '{"lib": "$MEM.lib"}')
        assert _read(core, user, "$lib.greeting") == "hi"

    def test_roots_validation(self):
        core, root, user = self._user_with_root()
        ref = "$MEM.conversations.1.roots"
        assert _write(core, root, ref, '{"MEM": "$MEM.x"}').startswith("Illegal:")
        assert _write(core, root, ref, '{"lib": "lib"}').startswith("Illegal:")
        assert _write(core, root, ref, '{"lib": "$MEM"}').startswith(("Illegal:", "Error:"))
        assert _write(core, root, ref, '{"caps": "$MEM.lib"}').startswith("Illegal:")  # core 根名
        assert _write(core, root, ref, '{"a": "$MEM.x", "b": "$MEM.x.y"}').startswith("Illegal:")
        assert _write(core, root, ref, '{"w": "$MEM.conversations"}').startswith("Illegal:")
        assert _write(core, root, ref, '{"w": "$MEM.conversations.0.roots"}').startswith("Illegal:")
        assert _write(core, root, ref, '{"w": "$MEM.roots"}').startswith("Illegal:")

    def test_roots_write_can_remove_root(self):
        core, root, user = self._user_with_root()
        core.mem["ws"] = MetaDict(data={"x": "v"})
        core.set_conversation_roots(user.cid, {"work": ["ws"]})
        assert _read(core, user, "$work.x") == "v"
        caps_target = user.caps.roots["caps"]
        text = _write(core, root, "$MEM.conversations.1.roots", "{}")
        assert text == "Success set: $MEM.conversations.1.roots", text
        assert "work" not in user.caps.roots
        assert user.caps.roots["caps"] == caps_target
        assert _read(core, user, "$work.x").startswith("Error:")

    def test_roots_cannot_cover_device_mount(self):
        core, root, user = self._user_with_root()
        core.mem.mount("box.dev", StringDevice("v"))
        text = _write(core, root, "$MEM.conversations.1.roots", '{"box": "$MEM.box"}')
        assert text.startswith("Illegal:") and "设备" in text
        # 显式授予设备本身允许
        text = _write(core, root, "$MEM.conversations.1.roots", '{"dev": "$MEM.box.dev"}')
        assert text.startswith("Success set"), text

    def test_roots_share_same_target_allowed_but_not_prefix(self):
        core, root, user = self._user_with_root()
        other = _user_conv(core, root)
        core.mem["lib"] = MetaDict(data={})
        _write(core, root, "$MEM.conversations.2.roots", '{"lib": "$MEM.lib"}')
        text = _write(core, root, "$MEM.conversations.1.roots", '{"lib": "$MEM.lib"}')
        assert text.startswith("Success set"), text
        # 与另一个对话的根目标前缀可比：拒绝
        core.mem.make("$MEM", "shared", "dict", owner_cid=0)
        core.set_conversation_roots(other.cid, {"lib": ["lib"], "work": ["shared", "sub"]})
        text = _write(core, root, "$MEM.conversations.1.roots", '{"w": "$MEM.shared.sub.x"}')
        assert text.startswith("Illegal:"), text


class TestListenerDeviceWrites:
    def _listener(self, core, root):
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}"
        )
        lid = core.events.register(0, "memory_read", "$MEM.box", "$MEM.h",
                                   phase="before", kernel_mode=True)
        return lid

    def test_priority_and_enabled_roundtrip(self):
        core, root = _core()
        lid = self._listener(core, root)
        assert _write(core, root, f"$MEM.listeners.{lid}.priority", "5") == \
            f"Success set: $MEM.listeners.{lid}.priority"
        assert core.events.listeners[-1]["priority"] == 5
        assert _read(core, root, f"$MEM.listeners.{lid}.priority") == "5"
        _write(core, root, f"$MEM.listeners.{lid}.enabled", "false")
        assert core.events.listeners[-1]["enabled"] is False
        assert _read(core, root, f"$MEM.listeners.{lid}.enabled") == "false"

    def test_disabled_listener_stops_firing(self):
        core, root = _core()
        core.mem["box"] = MetaDict(data={"x": "v"}, ctrl={"owner": "0"})
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}"
        )
        lid = core.events.register(0, "memory_read", "$MEM.box.x", "$MEM.h",
                                   phase="before", kernel_mode=True)
        _write(core, root, f"$MEM.listeners.{lid}.enabled", "false")
        assert _read(core, root, "$MEM.box.x") == "v"
        _write(core, root, f"$MEM.listeners.{lid}.enabled", "true")
        assert _read(core, root, "$MEM.box.x") == "Interrupted: blocked"

    def test_validation(self):
        core, root = _core()
        lid = self._listener(core, root)
        assert _write(core, root, f"$MEM.listeners.{lid}.priority", "-1").startswith("Illegal:")
        assert _write(core, root, f"$MEM.listeners.{lid}.priority", "99999").startswith("Illegal:")
        assert _write(core, root, f"$MEM.listeners.{lid}.priority", "abc").startswith("Illegal:")
        assert _write(core, root, f"$MEM.listeners.{lid}.enabled", "yes").startswith("Illegal:")
        assert _write(core, root, f"$MEM.listeners.{lid}.handler_ref", "$MEM.h").startswith("Illegal:")
        assert _write(core, root, "$MEM.listeners.99.priority", "1").startswith("Error:")

    def test_priority_change_takes_effect_in_next_dispatch(self):
        core, root = _core()
        core.mem["box"] = MetaDict(data={"x": "v"}, ctrl={"owner": "0"})
        core.mem["h2"] = _handler("def handle(event):\n    return {'action': 'allow'}")
        low = self._listener(core, root)  # replace "blocked"
        high = core.events.register(0, "memory_read", "$MEM.box", "$MEM.h2",
                                    phase="before", tier="kernel", kernel_mode=True)
        _write(core, root, f"$MEM.listeners.{low}.priority", "0")
        _write(core, root, f"$MEM.listeners.{high}.priority", "9")
        # 高优先级 allow 先短路 → 原 read 正常执行
        assert _read(core, root, "$MEM.box").startswith("dict[keys=")
        # 高优先级降回 0 后，先注册的 replace 先执行 → 被拦截
        _write(core, root, f"$MEM.listeners.{high}.priority", "0")
        assert _read(core, root, "$MEM.box") == "Interrupted: blocked"


class TestControlSurfaceEventsAndAuth:
    def test_after_event_payload_is_minimal(self):
        core, root = _core()
        seen = []
        core.events.add_observer(seen.append)
        _write(core, root, "$MEM.conversations.0.groups", '["worker"]')
        after = [e for e in seen if e["event_type"] == "memory_write" and e["phase"] == "after"]
        assert after
        payload = after[-1]["payload"]
        assert payload["cid"] == 0 and payload["field"] == "groups"
        assert payload["value"] == '["worker"]'
        assert "caps" not in payload and "conversation" not in payload
        assert after[-1]["source"]["kind"] == "device"

    def test_control_device_listener_requires_kernel(self):
        core, root = _core()
        core.mem["h"] = _handler()
        lid = core.events.register(0, "memory_write", "$MEM.conversations.0.groups",
                                   "$MEM.h", phase="before", kernel_mode=True)
        assert lid >= 1
        with pytest.raises(ValueError):
            core.events.register(1, "memory_write", "$MEM.conversations.0.groups",
                                 "$MEM.h", phase="before")

    def test_user_conversation_cannot_name_control_device(self):
        core, root = _core()
        user = _user_conv(core, root)
        text = _write(core, user, "$MEM.conversations.0.groups", '["worker"]')
        assert text.startswith("Error:") and "根" in text

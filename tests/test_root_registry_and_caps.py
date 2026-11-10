"""R2/R3：运行期根注册表 $MEM.roots 与只读能力自述 $MEM.caps。"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES
from avm.core import Core, CreateInstruction, MemoryReadInstruction, MemoryWriteInstruction
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


def _program(core, key, caps):
    node = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    node["caps"] = _caps_node(caps)
    core.mem[key] = node
    return f"$MEM.{key}"


def _write(core, conv, ref, content, call_id="w1"):
    conv.user_batch.tool_responses.clear()
    MemoryWriteInstruction(call_id, conv.cid, {"ref": ref, "content": content}).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


def _read(core, conv, ref, call_id="r1"):
    conv.user_batch.tool_responses.clear()
    MemoryReadInstruction(call_id, conv.cid, {"ref": ref}).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


def _user_conv(core, root, caps=None):
    _program(core, "up", caps or {"mode": "user", "tools": list(ALL_TOOL_NAMES)})
    root.user_batch.tool_responses.clear()
    CreateInstruction("c1", root.cid, {
        "system_ref": "$MEM.up", "para_ref": "$MEM.p",
    }).execute(core, root)
    text = root.user_batch.tool_responses[-1].content
    assert text.startswith("Success created: cid="), text
    return core.get_conversation(int(text.split("=")[-1]))


class TestRuntimeRootRegistry:
    def test_whole_replace_add_and_remove(self):
        core, root = _core()
        assert json.loads(_read(core, root, "$MEM.roots")) == {}
        assert _write(core, root, "$MEM.roots",
                      '{"lib": "$MEM.lib", "repo": "$MEM.repo"}') == \
            "Success set: $MEM.roots"
        assert core.root_registry == {"lib": ("lib",), "repo": ("repo",)}
        assert json.loads(_read(core, root, "$MEM.roots")) == {
            "lib": "$MEM.lib", "repo": "$MEM.repo"}
        assert _read(core, root, "$MEM.roots.lib") == "$MEM.lib"
        # 整个替换：缺省的 repo 被删除
        _write(core, root, "$MEM.roots", '{"lib": "$MEM.lib"}')
        assert core.root_registry == {"lib": ("lib",)}

    def test_per_name_write(self):
        core, root = _core()
        assert _write(core, root, "$MEM.roots.data", "$MEM.data") == \
            "Success set: $MEM.roots.data"
        assert core.root_registry == {"data": ("data",)}
        assert _read(core, root, "$MEM.roots.data") == "$MEM.data"

    def test_runtime_root_usable_by_program_caps(self):
        core, root = _core()
        core.mem["lib"] = MetaDict(data={"greeting": "hi"})
        _write(core, root, "$MEM.roots.lib", "$MEM.lib")
        user = _user_conv(core, root, {
            "mode": "user", "roots": ["lib"],
            "tools": list(ALL_TOOL_NAMES),
        })
        assert user.caps.roots["lib"] == ("lib",)
        assert _read(core, user, "$lib.greeting") == "hi"

    def test_validation(self):
        core, root = _core()
        _write(core, root, "$MEM.roots", '{"lib": "$MEM.lib"}')
        bad = [
            ("$MEM.roots", '{"wat": "lib"}'),                     # 目标不是 $MEM
            ("$MEM.roots", '{"MEM": "$MEM.x"}'),                  # 保留名
            ("$MEM.roots", '{"x": "$MEM"}'),                      # 根本身
            ("$MEM.roots", '{"x": "$MEM.conversations"}'),        # 控制面设备
            ("$MEM.roots", '{"lib": "$MEM.lib", "x": "$MEM.lib.sub"}'),  # 与已有根前缀冲突
            ("$MEM.roots.wat", "lib"),                            # 目标不是 $MEM
        ]
        for ref, content in bad:
            text = _write(core, root, ref, content)
            assert text.startswith(("Illegal:", "Error:")), (ref, content, text)
        assert core.root_registry == {"lib": ("lib",)}

    def test_validation_against_live_conversation_roots(self):
        core, root = _core()
        user = _user_conv(core, root)
        core.mem.make("$MEM", "ws", "dict")
        core.set_conversation_roots(user.cid, {"work": ["ws"]})
        text = _write(core, root, "$MEM.roots", '{"w": "$MEM.ws.sub"}')
        assert text.startswith("Illegal:"), text

    def test_user_conversation_cannot_name_roots_device(self):
        core, root = _core()
        user = _user_conv(core, root)
        text = _write(core, user, "$MEM.roots", '{"lib": "$MEM.lib"}')
        assert text.startswith("Error:") and "根" in text

    def test_before_listener_can_block_registry_write(self):
        core, root = _core()
        core.mem["h"] = MetaDict(
            data={"content": "def handle(event):\n    return {'action': 'replace', 'response': 'no'}\n"},
            ctrl={"type": "handler"},
        )
        core.events.register(0, "memory_write", "$MEM.roots", "$MEM.h",
                             phase="before", tier="kernel", kernel_mode=True)
        text = _write(core, root, "$MEM.roots", '{"lib": "$MEM.lib"}')
        assert text == "Interrupted: no"
        assert core.root_registry == {}

    def test_after_event_is_minimal(self):
        core, root = _core()
        seen = []
        core.events.add_observer(seen.append)
        core.mem["h"] = MetaDict(data={"content": "def handle(event):\n    return {}\n"},
                                 ctrl={"type": "handler"})
        core.events.register(0, "memory_write", "$MEM.roots", "$MEM.h",
                             phase="after", kernel_mode=True)
        _write(core, root, "$MEM.roots", '{"lib": "$MEM.lib"}')
        after = [e for e in seen if e["event_type"] == "memory_write" and e["phase"] == "after"]
        assert after
        assert "root_registry" not in after[-1]["payload"]


class TestCapsSelfDescription:
    def test_user_conversation_has_caps_root(self):
        core, root = _core()
        user = _user_conv(core, root)
        assert user.caps.roots["caps"] == ("caps", str(user.cid))

    def test_user_reads_own_caps(self):
        core, root = _core()
        user = _user_conv(core, root, {
            "mode": "user", "groups": ["worker"],
            "roots": [], "tools": ["memory_read"],
        })
        core.mem.make("$MEM", "ws", "dict")
        core.set_conversation_roots(user.cid, {"work": ["ws"]})
        data = json.loads(_read(core, user, "$caps"))
        assert data["cid"] == user.cid
        assert data["mode"] == "user"
        assert data["groups"] == ["worker"]
        assert data["tools"] == ["memory_read"]
        assert "workspace" not in data
        assert data["roots"]["work"]["core"] is False  # 动态授予根不是 core 根
        assert data["roots"]["caps"]["core"] is True
        assert _read(core, user, "$caps.mode") == "user"
        assert json.loads(_read(core, user, "$caps.tools")) == ["memory_read"]
        assert json.loads(_read(core, user, "$caps.roots"))["work"]["core"] is False

    def test_groups_change_visible_immediately(self):
        core, root = _core()
        user = _user_conv(core, root)
        _write(core, root, f"$MEM.conversations.{user.cid}.groups", '["auditor"]')
        data = json.loads(_read(core, user, "$caps"))
        assert data["groups"] == ["auditor"]

    def test_caps_device_is_read_only(self):
        core, root = _core()
        user = _user_conv(core, root)
        assert _write(core, user, "$caps", "{}").startswith("Error:")
        assert _write(core, user, "$caps.mode", "kernel").startswith("Error:")

    def test_kernel_can_read_any_caps(self):
        core, root = _core()
        user = _user_conv(core, root)
        data = json.loads(_read(core, root, f"$MEM.caps.{user.cid}"))
        assert data["cid"] == user.cid and data["mode"] == "user"

    def test_control_roots_cannot_overwrite_caps_root(self):
        core, root = _core()
        user = _user_conv(core, root)
        text = _write(core, root, f"$MEM.conversations.{user.cid}.roots",
                      '{"caps": "$MEM.lib"}')
        assert text.startswith("Illegal:"), text
        assert user.caps.roots["caps"][0] == "caps"

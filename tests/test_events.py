"""事件模型 v1 测试：owner、监听权限、waterfall、notify、after、错误、设备入口、信息设备。"""

import json
import pytest

from avm.core import (
    Core,
    MemoryReadInstruction,
    MemoryWriteInstruction,
    MemoryMakeInstruction,
    EditMetadataInstruction,
    CreateInstruction,
    CreateSubInstruction,
    SendInstruction,
    CloseInstruction,
    ListenEventInstruction,
    _build_tools,
    _instruction_registry,
    _build_conversation,
)
from avm.events import EventHandlerError, EVENT_TYPES
from avm.exceptions import VMMemoryError
from avm.memory import Memory
from avm.memory_device import MemoryDevice
from avm.types import MetaDict, MetaList


def _new_root(core):
    return core.start("sys", "hello")


def _python_node(code, ctrl=None):
    c = {"type": "python"}
    if ctrl:
        c.update(ctrl)
    return MetaDict(data={"content": code}, ctrl=c)


def _settingup(content="sys"):
    return MetaDict(data={"content": content}, ctrl={"type": "settingup"})


def _para():
    return MetaDict(data={"model": "test"}, ctrl={"type": "para"})


# ------------------------------------------------------------------
# 工具集
# ------------------------------------------------------------------

class TestToolSet:
    def test_nine_tools_and_listen_schema(self):
        tools = _build_tools()
        names = [t["function"]["name"] for t in tools]
        assert len(names) == 9
        assert names.count("listen_event") == 1
        assert "listen_event" in _instruction_registry()
        listen_def = next(t for t in tools if t["function"]["name"] == "listen_event")
        params = listen_def["function"]["parameters"]
        assert "action" in params["properties"]
        assert params["required"] == ["action"]

    def test_event_types_complete(self):
        names = [t["function"]["name"] for t in _build_tools()]
        assert sorted(names) == sorted(EVENT_TYPES)


# ------------------------------------------------------------------
# owner 记录
# ------------------------------------------------------------------

class TestOwner:
    def test_make_writes_owner_for_dict_and_list_but_not_str(self):
        mem = Memory()
        mem.make("$MEM", "d", "dict", owner_cid=7)
        mem.make("$MEM", "l", "list", owner_cid=7)
        mem.make("$MEM", "s", "str", owner_cid=7)
        assert mem["d"].get_ctrl()["owner"] == "7"
        assert mem["l"].get_ctrl()["owner"] == "7"
        assert mem["s"] == ""

    def test_set_implicit_intermediate_containers_get_owner(self):
        mem = Memory()
        mem.set("$MEM.a.b.c", "v", owner_cid=7)
        assert mem["a"].get_ctrl()["owner"] == "7"
        assert mem["a"]["b"].get_ctrl()["owner"] == "7"
        assert mem["a"]["b"]["c"] == "v"

    def test_set_does_not_change_existing_container_owner(self):
        mem = Memory()
        mem["a"] = MetaDict(data={}, ctrl={"owner": "5"})
        mem.set("$MEM.a.b", "v", owner_cid=7)
        assert mem["a"].get_ctrl()["owner"] == "5"

    def test_save_load_keeps_owner(self, tmp_path):
        mem = Memory()
        mem.make("$MEM", "d", "dict", owner_cid=3)
        mem.set("$MEM.d.x", "v", owner_cid=3)
        path = str(tmp_path / "mem.json")
        mem.save(path)
        loaded = Memory.load(path)
        assert loaded["d"].get_ctrl()["owner"] == "3"
        assert loaded["d"]["x"] == "v"


# ------------------------------------------------------------------
# 权限
# ------------------------------------------------------------------

class TestPermissions:
    def _setup(self):
        core = Core()
        root = _new_root(core)
        core.mem["handler"] = _python_node("def handle(event):\n    return {}")
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        return core, root

    def test_owner_can_register_non_owner_rejected(self):
        core, _ = self._setup()
        lid = core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.handler")
        assert lid == 1
        with pytest.raises(ValueError):
            core.events.register(1, "memory_read", "$MEM.box.msg", "$MEM.handler")

    def test_send_instruction_only_self(self):
        core, root = self._setup()
        core.events.register(0, "send_instruction", "cid:0", "$MEM.handler")
        with pytest.raises(ValueError):
            core.events.register(0, "send_instruction", "cid:1", "$MEM.handler")

    def test_modify_cancel_owner_only(self):
        core, _ = self._setup()
        lid = core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.handler")
        with pytest.raises(ValueError):
            core.events.modify(lid, 1, "$MEM.handler")
        with pytest.raises(ValueError):
            core.events.cancel(lid, 1)
        core.events.modify(lid, 0, "$MEM.handler")
        core.events.cancel(lid, 0)

    def test_unowned_path_rejected(self):
        core, _ = self._setup()
        core.mem["plain"] = MetaDict(data={"x": "v"})
        with pytest.raises(ValueError):
            core.events.register(0, "memory_read", "$MEM.plain.x", "$MEM.handler")

    def test_nonexistent_path_rejected_even_with_owned_ancestor(self):
        core, _ = self._setup()
        # $MEM.box 有主，但 $MEM.box.missing 不存在，不能注册 memory_read。
        with pytest.raises(ValueError):
            core.events.register(0, "memory_read", "$MEM.box.missing", "$MEM.handler")

    def test_memory_make_parent_must_exist(self):
        core, _ = self._setup()
        with pytest.raises(ValueError):
            core.events.register(0, "memory_make", "$MEM.box.missing.child", "$MEM.handler")

    def test_owner_check_does_not_touch_device_subpaths(self):
        class CountingDevice(MemoryDevice):
            def __init__(self):
                self.resolve_calls = 0

            def pretend_as_type(self):
                return "str"

            def to_llm_string(self):
                return "counting"

            def resolve_path(self, path):
                self.resolve_calls += 1
                return "side-effect"

        core = Core()
        _new_root(core)
        device = CountingDevice()
        core.mem.mount("inputs", device)
        core.mem["handler"] = _python_node("def handle(event):\n    return {}")

        with pytest.raises(ValueError, match="未授权"):
            core.events.register(0, "memory_read", "$MEM.inputs.-1", "$MEM.handler")
        with pytest.raises(ValueError, match="未授权"):
            core.events.register(0, "memory_make", "$MEM.inputs.job.child", "$MEM.handler")
        assert device.resolve_calls == 0


# ------------------------------------------------------------------
# before waterfall
# ------------------------------------------------------------------

class TestBeforeWaterfall:
    def _core_with_path(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "secret", owner_cid=0)
        return core, root

    def test_memory_read_replace_and_short_circuit(self):
        core, root = self._core_with_path()
        core.mem["h1"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': '请输入密码'}"
        )
        core.mem["h2"] = _python_node("def handle(event):\n    raise RuntimeError('should not run')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h1")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h2")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert [r.content for r in root.user_batch.tool_responses] == ["请输入密码"]

    def test_memory_write_payload_rewrite(self):
        core, root = self._core_with_path()
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'payload': {'content': 'changed'}}"
        )
        core.events.register(0, "memory_write", "$MEM.box.msg", "$MEM.h")

        MemoryWriteInstruction("tc", 0, {"ref": "$MEM.box.msg", "content": "new"}).execute(core, root)

        assert core.mem["box"]["msg"] == "changed"
        assert "Success set: $MEM.box.msg" in root.user_batch.tool_responses[-1].content

    def test_memory_write_replace_keeps_memory(self):
        core, root = self._core_with_path()
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no'}"
        )
        core.events.register(0, "memory_write", "$MEM.box.msg", "$MEM.h")

        MemoryWriteInstruction("tc", 0, {"ref": "$MEM.box.msg", "content": "new"}).execute(core, root)

        assert core.mem["box"]["msg"] == "secret"
        assert root.user_batch.tool_responses[-1].content == "no"

    def test_order_first_replace_wins(self):
        core, root = self._core_with_path()
        core.mem["h1"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'first'}"
        )
        core.mem["h2"] = _python_node("def handle(event):\n    raise RuntimeError('should not run')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h1")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h2")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == "first"

    def test_order_execute_then_replace(self):
        core, root = self._core_with_path()
        core.mem["h1"] = _python_node("def handle(event):\n    return {'action': 'execute'}")
        core.mem["h2"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'second'}"
        )
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h1")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h2")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == "second"


# ------------------------------------------------------------------
# notify
# ------------------------------------------------------------------

class TestNotify:
    def _setup_with_target(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(target.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        return core, root, target

    def test_notify_delivers_to_target_and_wakes(self):
        core, root, target = self._setup_with_target()
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'hello'}]}"
        )
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h", phase="before")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert target.user_batch.user_contents == ["hello"]
        assert target.cid in core._ready_cids
        assert target.cid not in core._dormant_cids
        assert all(e["event_type"] != "send_instruction" for e in core._turn_events)


# ------------------------------------------------------------------
# send_instruction 拦截
# ------------------------------------------------------------------

class TestSendInstructionIntercept:
    def _children(self, core, root, count):
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        children = []
        for _ in range(count):
            child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
            core._dormant_cids.append(child.cid)
            children.append(child)
        return children

    def test_all_targets_replaced(self):
        core = Core()
        root = _new_root(core)
        children = self._children(core, root, 2)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}"
        )
        for child in children:
            core.events.register(child.cid, "send_instruction", f"cid:{child.cid}", "$MEM.h")

        SendInstruction("tc", 0, {"cid": [1, 2], "content": "hi"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == "blocked"
        assert all(c.user_batch.user_contents == [] for c in children)
        assert all(c.cid in core._dormant_cids for c in children)

    def test_partial_replaced(self):
        core = Core()
        root = _new_root(core)
        children = self._children(core, root, 2)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}"
        )
        core.events.register(children[0].cid, "send_instruction", f"cid:{children[0].cid}", "$MEM.h")

        SendInstruction("tc", 0, {"cid": [1, 2], "content": "hi"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == "已投递到 1，1 个被拦截，0 个处理失败"
        assert children[0].user_batch.user_contents == []
        assert children[1].user_batch.user_contents == ["hi"]
        assert children[0].cid in core._dormant_cids
        assert children[1].cid in core._ready_cids


# ------------------------------------------------------------------
# modify / cancel / close 清理
# ------------------------------------------------------------------

class TestLifecycle:
    def test_modify_changes_handler(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h1"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'one'}")
        core.mem["h2"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'two'}")
        lid = core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h1")
        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "one"

        core.events.modify(lid, 0, "$MEM.h2")
        root.user_batch.clear()
        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "two"

        core.events.cancel(lid, 0)
        root.user_batch.clear()
        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "v"

    def test_close_cancels_owner_listeners(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=1)
        core.mem.set("$MEM.box.msg", "v", owner_cid=1)
        core.mem["h"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}")
        core.events.register(child.cid, "memory_read", "$MEM.box.msg", "$MEM.h")
        assert len(core.events.listeners) == 1

        core._close_conversation(child.cid)

        assert core.events.listeners == []
        assert "listener_id=1" not in core.mem.unwrap(["$", "MEM", "listeners"], for_llm=True)
        # 行为：根对话现在可以正常读取（监听已撤销）
        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "v"


# ------------------------------------------------------------------
# after 事件
# ------------------------------------------------------------------

class TestAfterEvents:
    def _core(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        return core, root

    def test_create_cmd_after_notifies(self):
        core, root = self._core()
        core._active_cid = root.cid
        core._ready_cids.remove(root.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'created'}]}"
        )
        core.events.register(0, "create_cmd", "*", "$MEM.h")

        CreateInstruction("tc", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)

        child = core._conv_by_cid[1]
        assert child.user_batch.user_contents == ["created"]
        assert child.cid in core._ready_cids
        # 源对话保持 active，不因 notify 进入 ready。
        assert core._active_cid == 0
        assert 0 not in core._ready_cids
        assert root.user_batch.tool_responses[-1].content.startswith("Success created")

    def test_close_after_notifies(self):
        core, root = self._core()
        close_target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(close_target.cid)
        notify_target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(notify_target.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 2, 'content': 'closed'}]}"
        )
        core.events.register(0, "close_conversation", "*", "$MEM.h")

        CloseInstruction("tc", 0, {"cid": close_target.cid}).execute(core, root)

        assert notify_target.user_batch.user_contents == ["closed"]
        assert notify_target.cid in core._ready_cids
        assert close_target.cid in core._finished
        assert root.user_batch.tool_responses[-1].content == f"对话 {close_target.cid} 已关闭（finished）"

    def test_listen_event_after_notifies(self):
        core, root = self._core()
        notify_target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(notify_target.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'listener changed'}]}"
        )
        core.events.register(0, "listen_event", "*", "$MEM.h")
        core.mem["h2"] = _python_node("def handle(event):\n    return {}")

        ListenEventInstruction("tc", 0, {
            "action": "listen",
            "event_type": "send_instruction",
            "target": "cid:0",
            "handler_ref": "$MEM.h2",
        }).execute(core, root)

        # listen_event 的 after 事件通配监听会收到通知（非自我投递）
        assert notify_target.user_batch.user_contents == ["listener changed"]
        assert notify_target.cid in core._ready_cids
        assert "已注册" in root.user_batch.tool_responses[-1].content


# ------------------------------------------------------------------
# 处理器错误
# ------------------------------------------------------------------

class TestHandlerErrors:
    def test_before_fail_closed(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        content = root.user_batch.tool_responses[-1].content
        assert "Error: 事件监听器 1 处理失败" in content
        assert "RuntimeError: boom" in content
        assert "boom" in capsys.readouterr().err

    def test_before_non_dict_fail_closed(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    return 'not-a-dict'")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert "Error: 事件监听器 1 处理失败" in root.user_batch.tool_responses[-1].content

    def test_after_fail_open(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        core.mem["h"] = _python_node("def handle(event):\n    raise RuntimeError('after boom')")
        core.events.register(0, "create_cmd", "*", "$MEM.h")

        CreateInstruction("tc", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content.startswith("Success created")
        assert "after boom" in capsys.readouterr().err


# ------------------------------------------------------------------
# owner 键保护
# ------------------------------------------------------------------

class TestOwnerKeyProtection:
    def test_set_del_owner_rejected_get_allowed(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem["box"].set_ctrl({"owner": "0", "type": "para"})

        EditMetadataInstruction("tc1", 0, {"ref": "$MEM.box", "type": "set", "key": "owner", "value": "9"}).execute(core, root)
        assert "不允许通过 edit_metadata 修改" in root.user_batch.tool_responses[-1].content

        EditMetadataInstruction("tc2", 0, {"ref": "$MEM.box", "type": "del", "key": "owner"}).execute(core, root)
        assert "不允许通过 edit_metadata 修改" in root.user_batch.tool_responses[-1].content

        root.user_batch.clear()
        EditMetadataInstruction("tc3", 0, {"ref": "$MEM.box", "type": "get", "key": "owner"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "0"


# ------------------------------------------------------------------
# 设备入口
# ------------------------------------------------------------------

class _DummyDevice(MemoryDevice):
    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return "dummy"

    def authorize_listener(self, event_type, remainder, owner_cid):
        return True

    def after_event(self, event_type, remainder, result):
        return None


class TestEmitDeviceEvent:
    def test_source_and_target_validation_and_replace(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.mount("box.dev", _DummyDevice())
        core.mem["h"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}")
        core.events.register(0, "memory_write", "$MEM.box.dev", "$MEM.h")

        event = core.emit_device_event("box.dev", "$MEM.box.dev", {"ref": "$MEM.box.dev", "content": "x"})

        assert event["source"] == {"kind": "device", "path": "$MEM.box.dev"}
        assert event["action"] == "replace"
        with pytest.raises(Exception):
            core.emit_device_event("box.dev", "$MEM.other", {"content": "x"})

    def test_device_event_target_prefix_boundary(self):
        core = Core()
        _new_root(core)
        core.mem.mount("box", _DummyDevice())

        with pytest.raises(VMMemoryError):
            core.emit_device_event("box", "$MEM.box2", {"ref": "$MEM.box2", "content": "x"})

        assert core.emit_device_event("box", "$MEM.box", {"ref": "$MEM.box", "content": "x"})["action"] == "execute"
        assert core.emit_device_event("box", "$MEM.box.sub", {"ref": "$MEM.box.sub", "content": "x"})["action"] == "execute"


# ------------------------------------------------------------------
# 信息设备与 monitor
# ------------------------------------------------------------------

class TestInfoAndMonitor:
    def test_listeners_device(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    return {}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")

        text = core.mem.unwrap(["$", "MEM", "listeners"], for_llm=True)
        assert "listener_id=1" in text
        detail = core.mem.unwrap(["$", "MEM", "listeners", "1"], for_llm=True)
        assert "event_type=memory_read" in detail
        assert "handler_ref=$MEM.h" in detail

    def test_monitor_frame_has_events_summary(self):
        class MockLMU:
            def __init__(self):
                self.last_call = None

            def exec(self, conversation, para):
                self.last_call = {
                    "result": "ok",
                    "tool_calls": [{"call_id": "tc", "cmd_type": "memory_read", "args": {"ref": "$MEM.box.msg"}}],
                }
                return "ok", [{"call_id": "tc", "cmd_type": "memory_read", "args": {"ref": "$MEM.box.msg"}}], conversation

        core = Core()
        core.lmu = MockLMU()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        core.advance_conversation()

        frame = core.monitor.frame(1)
        assert frame.lmu["events"]
        ev = frame.lmu["events"][0]
        assert ev["event_type"] == "memory_read"
        assert ev["phase"] == "before"
        assert ev["target"] == "$MEM.box.msg"
        assert ev["action"] == "execute"


# ------------------------------------------------------------------
# P1 收尾：自我投递、handler_error 事件结果、多播部分投递
# ------------------------------------------------------------------

class TestSelfDelivery:
    def test_send_instruction_to_self_rejected(self, capsys):
        core = Core()
        root = _new_root(core)
        SendInstruction("tc", 0, {"cid": 0, "content": "hi"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == \
            "Error: send_instruction 不允许自我投递（cid=0 是当前对话）"
        assert root.user_batch.user_contents == []
        assert "不允许自我投递" in capsys.readouterr().err

    def test_send_instruction_self_in_multicast_rejected(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)

        SendInstruction("tc", 0, {"cid": [0, 1], "content": "hi"}).execute(core, root)

        assert "不允许自我投递" in root.user_batch.tool_responses[-1].content
        assert child.user_batch.user_contents == []
        assert child.cid in core._dormant_cids


class TestNotifySelf:
    def test_before_notify_self_is_handler_error(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 0, 'content': 'self'}]}"
        )
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        content = root.user_batch.tool_responses[-1].content
        assert "Error: 事件监听器 1 处理失败" in content
        assert "notify 不允许自我投递" in content
        assert root.user_batch.user_contents == []
        assert any(
            e["event_type"] == "memory_read" and e["action"] == "handler_error"
            for e in core._turn_events
        )

    def test_after_notify_self_dropped_other_delivered(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 0, 'content': 'self'}, {'cid': 1, 'content': 'other'}]}"
        )
        core.events.register(0, "create_cmd", "*", "$MEM.h")

        CreateInstruction("tc", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)

        child = core._conv_by_cid[1]
        assert root.user_batch.user_contents == []
        assert child.user_batch.user_contents == ["other"]
        summary = core._turn_events[-1]
        assert summary["action"] == "execute"
        assert any("自我投递" in h["error"] for h in summary.get("handler_errors", []))


class TestBeforeHandlerErrorEventResult:
    def test_first_notify_kept_and_second_error_stops(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(target.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h1"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'first'}]}"
        )
        core.mem["h2"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h1")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h2")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert target.user_batch.user_contents == ["first"]
        assert target.cid in core._ready_cids
        content = root.user_batch.tool_responses[-1].content
        assert "Error: 事件监听器 2 处理失败" in content
        assert "RuntimeError: boom" in content
        assert core._turn_events[-1]["action"] == "handler_error"
        assert core._turn_events[-1].get("error") is not None

    def test_stderr_reported_once(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        err = capsys.readouterr().err
        assert err.count("事件监听器 1 处理失败") == 1
        assert "boom" in err

    def test_monitor_frame_records_handler_error(self):
        class MockLMU:
            def __init__(self):
                self.last_call = None

            def exec(self, conversation, para):
                self.last_call = {
                    "result": "ok",
                    "tool_calls": [{"call_id": "tc", "cmd_type": "memory_read", "args": {"ref": "$MEM.box.msg"}}],
                }
                return "ok", [{"call_id": "tc", "cmd_type": "memory_read", "args": {"ref": "$MEM.box.msg"}}], conversation

        core = Core()
        core.lmu = MockLMU()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h")
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        core.advance_conversation()

        ev = core.monitor.frame(1).lmu["events"][0]
        assert ev["action"] == "handler_error"
        assert "boom" in ev["error"]


class TestMulticastPartial:
    def test_execute_replace_handler_error(self, capsys):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        children = []
        for _ in range(3):
            child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
            core._dormant_cids.append(child.cid)
            children.append(child)
        core.mem["block"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'blocked'}"
        )
        core.mem["boom"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(children[0].cid, "send_instruction", f"cid:{children[0].cid}", "$MEM.block")
        core.events.register(children[1].cid, "send_instruction", f"cid:{children[1].cid}", "$MEM.boom")

        SendInstruction("tc", 0, {"cid": [1, 2, 3], "content": "hi"}).execute(core, root)

        assert root.user_batch.tool_responses[-1].content == "已投递到 1，1 个被拦截，1 个处理失败"
        assert children[0].user_batch.user_contents == []
        assert children[1].user_batch.user_contents == []
        assert children[2].user_batch.user_contents == ["hi"]
        assert children[0].cid in core._dormant_cids
        assert children[1].cid in core._dormant_cids
        assert children[2].cid in core._ready_cids
        assert "boom" in capsys.readouterr().err


class TestMissingCoverage:
    def test_create_sub_after_notifies(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        core.mem["u"] = "x"
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'sub created'}]}"
        )
        core.events.register(0, "create_sub", "*", "$MEM.h")

        CreateSubInstruction("tc", 0, {
            "system_ref": "$MEM.s",
            "user_ref": "$MEM.u",
            "para_ref": "$MEM.p",
        }).execute(core, root)

        # CreateSubInstruction 没有工具响应；事件 notify 投递给新亚对话
        sub = core._conv_by_cid[1]
        assert sub.user_batch.user_contents == ["sub created"]
        assert sub.cid in core._ready_cids

    def test_memory_make_before_replace_and_execute(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no make'}"
        )
        core.events.register(0, "memory_make", "$MEM.box.new", "$MEM.h")

        MemoryMakeInstruction("tc", 0, {"ref": "$MEM.box", "key": "new", "mem_type": "dict"}).execute(core, root)
        assert "new" not in core.mem["box"]
        assert root.user_batch.tool_responses[-1].content == "no make"

        # 清除 replace 监听，验证 execute 正常创建
        core.events.cancel(1, 0)
        root.user_batch.clear()
        MemoryMakeInstruction("tc2", 0, {"ref": "$MEM.box", "key": "new", "mem_type": "dict"}).execute(core, root)
        assert "new" in core.mem["box"]
        assert core.mem["box"]["new"].get_ctrl()["owner"] == "0"

    def test_edit_metadata_before_replace_and_value_rewrite(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no edit'}"
        )
        core.events.register(0, "edit_metadata", "$MEM.box", "$MEM.h")

        EditMetadataInstruction("tc", 0, {"ref": "$MEM.box", "type": "set", "key": "type", "value": "settingup"}).execute(core, root)
        assert (core.mem["box"].get_ctrl() or {}).get("type") is None
        assert root.user_batch.tool_responses[-1].content == "no edit"

        # 改为 value 改写
        core.events.cancel(1, 0)
        core.mem["h2"] = _python_node(
            "def handle(event):\n    return {'payload': {'value': 'python'}}"
        )
        core.events.register(0, "edit_metadata", "$MEM.box", "$MEM.h2")
        root.user_batch.clear()
        EditMetadataInstruction("tc2", 0, {"ref": "$MEM.box", "type": "set", "key": "type", "value": "settingup"}).execute(core, root)
        assert core.mem["box"].get_ctrl()["type"] == "python"

    def test_send_instruction_before_rewrites_content(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'payload': {'content': 'rewritten'}}"
        )
        core.events.register(child.cid, "send_instruction", f"cid:{child.cid}", "$MEM.h")

        SendInstruction("tc", 0, {"cid": child.cid, "content": "hi"}).execute(core, root)

        assert child.user_batch.user_contents == ["rewritten"]
        assert child.cid in core._ready_cids
        assert "已投递到" in root.user_batch.tool_responses[-1].content


# ------------------------------------------------------------------
# 发布者分工：split_path / after 事实事件 / phase / 设备授权
# ------------------------------------------------------------------

class TestSplitPath:
    def test_split_path_boundaries(self):
        from avm.memory import Memory

        mem = Memory()
        mem.mount("box", _DummyDevice())
        mem.mount("box2", _DummyDevice())

        assert mem.split_path("$MEM.box") == ("device", "box", [])
        assert mem.split_path("$MEM.box2") == ("device", "box2", [])
        assert mem.split_path("$MEM.box.x") == ("device", "box", ["x"])
        assert mem.split_path("$MEM.a.b") == ("memory", ["a", "b"])

    def test_mount_under_existing_mount_rejected(self):
        """链接节点里放不下第二个挂载点：box 已挂载时不能再挂 box.dev。"""
        from avm.memory import Memory

        mem = Memory()
        mem.mount("box", _DummyDevice())
        with pytest.raises(VMMemoryError):
            mem.mount("box.dev", _DummyDevice())
        assert "box.dev" not in mem._devices


class TestMemoryAfterEvents:
    def _core_with_sink(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        # 捕获 after payload：各事件把 payload JSON 通知给 sink
        core.mem["h_read"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.mem["h_write"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.mem["h_make"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.mem["h_edit"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h_read", phase="after")
        core.events.register(0, "memory_write", "$MEM.box.msg", "$MEM.h_write", phase="after")
        core.events.register(0, "memory_make", "$MEM.box.child", "$MEM.h_make", phase="after")
        core.events.register(0, "edit_metadata", "$MEM.box", "$MEM.h_edit", phase="after")
        return core, root, sink

    def test_read_write_make_edit_after_events(self):
        core, root, sink = self._core_with_sink()

        MemoryReadInstruction("tc1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        MemoryWriteInstruction("tc2", 0, {"ref": "$MEM.box.msg", "content": "new"}).execute(core, root)
        MemoryMakeInstruction("tc3", 0, {"ref": "$MEM.box", "key": "child", "mem_type": "dict"}).execute(core, root)
        EditMetadataInstruction("tc4", 0, {"ref": "$MEM.box", "type": "set", "key": "type", "value": "para"}).execute(core, root)

        after = [e for e in core._turn_events if e["phase"] == "after"]
        types = [e["event_type"] for e in after]
        assert types == ["memory_read", "memory_write", "memory_make", "edit_metadata"]
        assert all(e["source"] == {"kind": "memory"} for e in after)

        payloads = [json.loads(c) for c in sink.user_batch.user_contents]
        assert len(payloads) == 4
        assert all(p.get("actor_cid") == 0 for p in payloads)
        assert payloads[0]["result"] == "v"
        assert payloads[1]["content"] == "new"
        assert payloads[2]["path"] == "$MEM.box.child"
        assert payloads[3]["type"] == "set"
        assert payloads[3]["value"] == "para"

    def test_replace_and_handler_error_no_after(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'no'}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h", phase="before")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert not [e for e in core._turn_events if e["phase"] == "after"]

        core.mem["h2"] = _python_node("def handle(event):\n    raise RuntimeError('boom')")
        core.events.register(0, "memory_write", "$MEM.box.msg", "$MEM.h2", phase="before")
        root.user_batch.clear()
        MemoryWriteInstruction("tc2", 0, {"ref": "$MEM.box.msg", "content": "x"}).execute(core, root)

        assert not [e for e in core._turn_events if e["event_type"] == "memory_write" and e["phase"] == "after"]


class TestDeviceAfterEvents:
    def test_device_after_event_wrapped_by_core(self):
        class AfterDevice(_DummyDevice):
            def after_event(self, event_type, remainder, result):
                return {"result": "device-result", "evil": "x"}

        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.mount("box.dev", AfterDevice())
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.events.register(0, "memory_read", "$MEM.box.dev", "$MEM.h", phase="after")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.dev"}).execute(core, root)

        after = [e for e in core._turn_events if e["phase"] == "after" and e["event_type"] == "memory_read"]
        assert len(after) == 1
        ev = after[0]
        assert ev["source"] == {"kind": "device", "path": "$MEM.box.dev"}
        payload = json.loads(sink.user_batch.user_contents[0])
        assert payload["result"] == "device-result"
        assert payload["ref"] == "$MEM.box.dev"
        assert payload["actor_cid"] == 0

    def test_device_after_event_none_no_after(self):
        class NoAfterDevice(_DummyDevice):
            def after_event(self, event_type, remainder, result):
                return None

        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.mount("box.dev", NoAfterDevice())

        MemoryWriteInstruction("tc", 0, {"ref": "$MEM.box.dev", "content": "x"}).execute(core, root)

        assert not [e for e in core._turn_events if e["phase"] == "after" and e["event_type"] == "memory_write"]


class TestDeviceAuthorization:
    def test_authorize_listener_called_and_enforced_without_resolve(self):
        class AuthDevice(MemoryDevice):
            def __init__(self):
                self.calls = []
                self.resolve_calls = 0

            def pretend_as_type(self):
                return "str"

            def to_llm_string(self):
                return "auth"

            def authorize_listener(self, event_type, remainder, owner_cid):
                self.calls.append((event_type, remainder, owner_cid))
                return False

            def resolve_path(self, path):
                self.resolve_calls += 1
                return "side-effect"

        core = Core()
        _new_root(core)
        dev = AuthDevice()
        core.mem.mount("box", dev)
        core.mem["h"] = _python_node("def handle(event):\n    return {}")

        with pytest.raises(ValueError, match="未授权"):
            core.events.register(0, "memory_read", "$MEM.box.data", "$MEM.h", phase="after")
        assert dev.calls == [("memory_read", ["data"], 0)]
        assert dev.resolve_calls == 0

        dev.authorize_listener = lambda event_type, remainder, owner_cid: True
        lid = core.events.register(0, "memory_read", "$MEM.box.data", "$MEM.h", phase="after")
        assert lid == 1


class TestPhaseMatching:
    def test_before_after_all_listeners(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        targets = []
        for _ in range(3):
            t = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
            core._dormant_cids.append(t.cid)
            targets.append(t)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h_before"] = _python_node("def handle(event):\n    return {'notify': [{'cid': 1, 'content': 'b'}]}")
        core.mem["h_after"] = _python_node("def handle(event):\n    return {'notify': [{'cid': 2, 'content': 'a'}]}")
        core.mem["h_all"] = _python_node("def handle(event):\n    return {'notify': [{'cid': 3, 'content': event['phase']}]}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h_before", phase="before")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h_after", phase="after")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h_all", phase="all")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert targets[0].user_batch.user_contents == ["b"]
        assert targets[1].user_batch.user_contents == ["a"]
        assert targets[2].user_batch.user_contents == ["before", "after"]

    def test_after_only_type_rejects_before(self):
        core = Core()
        _new_root(core)
        core.mem["h"] = _python_node("def handle(event):\n    return {}")
        with pytest.raises(ValueError, match="只有 after"):
            core.events.register(0, "create_cmd", "*", "$MEM.h", phase="before")


class TestLifecycleAfterSourceCore:
    def test_create_cmd_after_source_core(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.events.register(0, "create_cmd", "*", "$MEM.h", phase="after")

        CreateInstruction("tc", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)

        after = [e for e in core._turn_events if e["event_type"] == "create_cmd" and e["phase"] == "after"]
        assert len(after) == 1
        assert after[0]["source"] == {"kind": "core"}
        assert json.loads(sink.user_batch.user_contents[0]) == {"cid": 2, "actor_cid": 0}

    def test_send_instruction_after_only_for_delivered(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 2, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.events.register(child.cid, "send_instruction", f"cid:{child.cid}", "$MEM.h", phase="after")

        SendInstruction("tc", 0, {"cid": child.cid, "content": "hi"}).execute(core, root)

        after = [e for e in core._turn_events if e["event_type"] == "send_instruction" and e["phase"] == "after"]
        assert len(after) == 1
        assert after[0]["source"] == {"kind": "core"}
        assert after[0]["target"] == "cid:1"
        assert json.loads(sink.user_batch.user_contents[0]) == {"cid": 1, "content": "hi", "actor_cid": 0}

    def test_send_instruction_blocked_no_after(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        child = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(child.cid)
        core.mem["h"] = _python_node("def handle(event):\n    return {'action': 'replace', 'response': 'no'}")
        core.events.register(child.cid, "send_instruction", f"cid:{child.cid}", "$MEM.h", phase="before")

        SendInstruction("tc", 0, {"cid": child.cid, "content": "hi"}).execute(core, root)

        assert not [e for e in core._turn_events if e["event_type"] == "send_instruction" and e["phase"] == "after"]


class TestNewSourceSelfNotify:
    def test_before_source_uses_initiator_cid(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        core.mem["h"] = _python_node("def handle(event):\n    return {'notify': [{'cid': 0, 'content': 'self'}]}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h", phase="before")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.box.msg"}).execute(core, root)

        assert "Error: 事件监听器 1 处理失败" in root.user_batch.tool_responses[-1].content

    def test_after_source_uses_actor_cid(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = _para()
        core.mem["s"] = _settingup()
        target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(target.cid)
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 0, 'content': 'self'}, {'cid': 1, 'content': 'other'}]}"
        )
        core.events.register(0, "create_cmd", "*", "$MEM.h", phase="after")

        CreateInstruction("tc", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)

        # create_cmd 的 after 事件中 actor_cid=0；self notify 被丢弃，other 投递给新建子对话
        child = core._conv_by_cid[1]
        assert root.user_batch.user_contents == []
        assert child.user_batch.user_contents == ["other"]

"""批次二事件层权限机制测试：payload 分层、处理器分级、priority、allow、
before 覆盖全部 10 条工具、canonical 匹配、监听注册权限、$MEM.listeners 展示。

不调真实 API：全部用指令直接执行 + 处理器 notify 观察。
"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import (
    CloseInstruction,
    Core,
    CreateInstruction,
    CreateSubInstruction,
    ListenEventInstruction,
    MemoryReadInstruction,
    MemoryWriteInstruction,
)
from avm.events import KERNEL_TIER, NORMAL_TIER
from avm.types import MetaDict, MetaList, StrNode


def _core():
    core = Core()
    core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    core.mem["s"] = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    root = core.start("sys", "user", para_ref="$MEM.p")
    core._active_cid = root.cid
    return core, root


def _caps_node(caps):
    data = {}
    if "mode" in caps:
        data["mode"] = StrNode(str(caps["mode"]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            data[key] = MetaList([StrNode(str(x)) for x in caps[key]])
    return MetaDict(data=data)


def _handler(code):
    return MetaDict(data={"content": code}, ctrl={"type": "handler"})


def _set_condition(core, ref, condition):
    node = core.unwrap(ref, for_llm=False)
    node["condition"] = StrNode(json.dumps(condition))


def _sink(core, root):
    """建一个只收 notify 的空对话，返回 cid。"""
    from avm.core import _build_conversation
    sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
    core._dormant_cids.append(sink.cid)
    core._ready_cids = [c for c in core._ready_cids if c != sink.cid]
    return sink


def _notify_json_handler(expr: str) -> str:
    """处理器把 expr 的结果 JSON 序列化后 notify 给 cid=1。"""
    return (
        "def handle(event):\n"
        "    def val():\n"
        f"        return {expr}\n"
        "    return {'notify': [{'cid': 1, 'content': json.dumps(val(), ensure_ascii=False, default=str)}]}\n"
    )


def _seed_box(core):
    core.mem.make("$MEM", "box", "dict", owner_cid=0)
    core.mem.make("$MEM.box", "msg", "str", owner_cid=0)
    core.mem.set("$MEM.box.msg", "v")


class TestPayloadTiers:
    def test_normal_listener_gets_metadata_but_not_kernel_blocks(self):
        core, root = _core()
        _seed_box(core)
        sink = _sink(core, root)
        core.mem["h"] = _handler(_notify_json_handler(
            "{'has_full': 'target_full' in event, 'has_ctx': 'source_context' in event, "
            "'ti': event.get('target_info'), 'src': event.get('source')}"
        ))
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h", phase="before")
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        data = json.loads(sink.user_batch.user_contents[-1])
        assert data["has_full"] is False
        assert data["has_ctx"] is False
        assert data["ti"]["kind"] == "memory"
        assert data["ti"]["type"] == "str"
        assert "canonical" not in data["ti"], "normal 载荷不暴露真实规范化地址"
        assert isinstance(data["ti"]["ancestors"], list) and data["ti"]["ancestors"]
        assert data["src"]["kind"] == "core"
        conv = data["src"]["conversation"]
        assert conv["cid"] == 0 and conv["mode"] == "kernel" and "groups" in conv

    def test_kernel_listener_gets_full_target_and_context(self):
        core, root = _core()
        _seed_box(core)
        sink = _sink(core, root)
        core.mem["h"] = _handler(_notify_json_handler(
            "{'has_full': 'target_full' in event, 'has_ctx': 'source_context' in event, "
            "'full': event.get('target_full'), 'ctx': event.get('source_context')}"
        ))
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h",
                             phase="before", tier=KERNEL_TIER, kernel_mode=True)
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        data = json.loads(sink.user_batch.user_contents[-1])
        assert data["has_full"] is True and data["has_ctx"] is True
        assert data["full"]["kind"] == "str" and data["full"]["value"] == "v"
        assert data["ctx"]["total"] >= 1 and data["ctx"]["truncated"] is False

    def test_normal_view_hidden_when_kernel_listener_also_matched(self):
        core, root = _core()
        _seed_box(core)
        sink = _sink(core, root)
        core.mem["hn"] = _handler(_notify_json_handler("'target_full' in event"))
        core.mem["hk"] = _handler("def handle(event):\n    return {}\n")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.hn", phase="before")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.hk",
                             phase="before", tier=KERNEL_TIER, kernel_mode=True)
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert json.loads(sink.user_batch.user_contents[-1]) is False


class TestPriorityAndDecisions:
    def _intercept_core(self, handler_code, priority=0, tier=NORMAL_TIER, phase="before"):
        core, root = _core()
        _seed_box(core)
        core.mem["h"] = _handler(handler_code)
        lid = core.events.register(
            0, "memory_read", "$MEM.box.msg", "$MEM.h", phase=phase,
            tier=tier, priority=priority, kernel_mode=(tier == KERNEL_TIER),
        )
        return core, root, lid

    def test_high_priority_allow_short_circuits_lower_replace(self):
        core, root, _ = self._intercept_core(
            "def handle(event):\n    return {'action': 'allow'}", priority=10, tier=KERNEL_TIER)
        core.mem["h_low"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'low'}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h_low", phase="before")
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "v"

    def test_same_priority_registration_order_decides(self):
        core, root, _ = self._intercept_core(
            "def handle(event):\n    return {'action': 'replace', 'response': 'first'}")
        core.mem["h2"] = _handler("def handle(event):\n    return {'action': 'allow'}")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h2",
                             phase="before", tier=KERNEL_TIER, kernel_mode=True)
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "Interrupted: first"

    def test_normal_listener_allow_is_handler_error(self):
        core, root, _ = self._intercept_core(
            "def handle(event):\n    return {'action': 'allow'}", tier=NORMAL_TIER)
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        text = root.user_batch.tool_responses[-1].content
        assert text.startswith("Error:") and "只有内核级监听器" in text

    def test_after_decision_actions_ignored(self):
        core, root, _ = self._intercept_core(
            "def handle(event):\n    return {'action': 'replace', 'response': 'nope'}", phase="after")
        MemoryReadInstruction("c1", 0, {"ref": "$MEM.box.msg"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "v"
        after = [e for e in core._turn_events if e["phase"] == "after" and e["event_type"] == "memory_read"]
        assert after and after[0]["action"] == "execute"


class TestBeforeCoverageAllTools:
    def test_create_cmd_before_can_block(self):
        core, root = _core()
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no create'}")
        core.events.register(0, "create_cmd", "*", "$MEM.h", phase="before", kernel_mode=True)
        CreateInstruction("c1", 0, {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "Interrupted: no create"
        assert len(core._conv_by_cid) == 1

    def test_create_sub_before_can_block(self):
        core, root = _core()
        core.mem["u"] = "task"
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no sub'}")
        core.events.register(0, "create_sub", "*", "$MEM.h", phase="before", kernel_mode=True)
        CreateSubInstruction("c1", 0, {
            "system_ref": "$MEM.s", "user_ref": "$MEM.u", "para_ref": "$MEM.p",
        }).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "Interrupted: no sub"
        assert not any(c.is_sub for c in core._conv_by_cid.values())

    def test_close_before_can_block(self):
        from avm.core import _build_conversation

        core, root = _core()
        target = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(target.cid)
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no close'}")
        core.events.register(0, "close_conversation", f"cid:{target.cid}", "$MEM.h",
                             phase="before", kernel_mode=True)
        CloseInstruction("c1", 0, {"cid": target.cid}).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "Interrupted: no close"
        assert target.cid in core._conv_by_cid

    def test_listen_event_before_can_block(self):
        core, root = _core()
        _seed_box(core)
        core.mem["h"] = _handler(
            "def handle(event):\n    return {'action': 'replace', 'response': 'no listen'}")
        _set_condition(core, "$MEM.h", {"events": ["listen_event"], "phase": ["before"]})
        core.events.register(0, "listen_event", "*", "$MEM.h", phase="before", kernel_mode=True)
        ListenEventInstruction("c1", 0, {
            "action": "listen", "event_type": "memory_read",
            "target": "$MEM.box.msg", "handler_ref": "$MEM.h",
        }).execute(core, root)
        assert root.user_batch.tool_responses[-1].content == "Interrupted: no listen"
        assert len(core.events.listeners) == 1  # 只有拦截用的那条


class TestListenerRegistrationPermissions:
    def _user_conv(self, core):
        core.mem["ws"] = MetaDict(data={"x": "", "h": MetaDict(
            data={"content": "def handle(event):\n    return {}\n",
                  "condition": StrNode(json.dumps({"preset": "own_nodes.memory"}))},
            ctrl={"type": "handler"})},
            ctrl={"owner": "1"})
        conv = core._conv_by_cid[1] if 1 in core._conv_by_cid else core.start("u", "u", para_ref="$MEM.p")
        conv.caps = Capability(mode="user", roots={"work": ("ws",)}, tools=ALL_TOOL_NAMES)
        return conv

    def test_user_conversation_cannot_register_kernel_tier(self):
        core, root = _core()
        conv = self._user_conv(core)
        ListenEventInstruction("c1", conv.cid, {
            "action": "listen", "handler_ref": "$work.h", "tier": "kernel",
        }).execute(core, conv)
        assert "内核级监听器" in conv.user_batch.tool_responses[-1].content

    def test_user_conversation_cannot_set_priority(self):
        core, root = _core()
        conv = self._user_conv(core)
        ListenEventInstruction("c2", conv.cid, {
            "action": "listen", "handler_ref": "$work.h", "priority": 5,
        }).execute(core, conv)
        assert "priority>0" in conv.user_batch.tool_responses[-1].content

    def test_user_conversation_cannot_use_custom_condition(self):
        core, root = _core()
        conv = self._user_conv(core)
        handler = core.mem.resolve_path(["MEM", "ws", "h"], for_llm=False)
        handler["condition"] = StrNode(json.dumps({"events": ["memory_read"]}))
        ListenEventInstruction("c3", conv.cid, {
            "action": "listen", "handler_ref": "$work.h",
        }).execute(core, conv)
        assert "命名条件子集" in conv.user_batch.tool_responses[-1].content

    def test_normal_conversation_can_listen_own_node(self):
        core, root = _core()
        conv = self._user_conv(core)
        ListenEventInstruction("c4", conv.cid, {
            "action": "listen", "handler_ref": "$work.h",
        }).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content.startswith("listener_id=")
        record = core.events.listeners[-1]
        assert record["tier"] == NORMAL_TIER and record["priority"] == 0

    def test_kernel_conversation_can_register_kernel_tier_any_target(self):
        core, root = _core()
        _seed_box(core)
        core.mem["h"] = _handler("def handle(event):\n    return {}\n")
        _set_condition(core, "$MEM.h",
                       {"events": ["memory_read"],
                        "target": {"kind": "exact", "ref": "$MEM.box.msg"}})
        ListenEventInstruction("c5", 0, {
            "action": "listen", "handler_ref": "$MEM.h", "tier": "kernel", "priority": 7,
        }).execute(core, root)
        assert root.user_batch.tool_responses[-1].content.startswith("listener_id=")
        record = core.events.listeners[-1]
        assert record["tier"] == KERNEL_TIER and record["priority"] == 7

    def test_modify_priority_only_for_kernel_listener(self):
        core, root = _core()
        conv = self._user_conv(core)
        ListenEventInstruction("c6", conv.cid, {
            "action": "listen", "handler_ref": "$work.h",
        }).execute(core, conv)
        lid = core.events.listeners[-1]["listener_id"]
        ListenEventInstruction("c7", conv.cid, {
            "action": "modify", "listener_id": lid, "handler_ref": "$work.h", "priority": 3,
        }).execute(core, conv)
        assert "priority>0" in conv.user_batch.tool_responses[-1].content


    def test_kernel_listener_can_modify_priority_only(self):
        core, root = _core()
        _seed_box(core)
        core.mem["h"] = _handler("def handle(event):\n    return {}\n")
        _set_condition(core, "$MEM.h",
                       {"events": ["memory_read"],
                        "target": {"kind": "exact", "ref": "$MEM.box.msg"}})
        ListenEventInstruction("c1", 0, {
            "action": "listen", "handler_ref": "$MEM.h", "tier": "kernel",
        }).execute(core, root)
        lid = core.events.listeners[-1]["listener_id"]
        ListenEventInstruction("c2", 0, {
            "action": "modify", "listener_id": lid, "priority": 4,
        }).execute(core, root)
        assert core.events.listeners[-1]["priority"] == 4


class TestKernelSupervisionAcrossSpaces:
    def test_kernel_listener_on_real_path_hits_user_work_alias(self):
        """内核态安全对话注册在真实路径上的监听器，命中用户态对话的 $work 别名事件。"""
        from avm.core import MemoryMakeInstruction, _canonical_ref

        core, root = _core()
        prog = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
        prog["caps"] = _caps_node({"mode": "user"})
        core.mem["prog"] = prog
        CreateInstruction("c1", 0, {
            "system_ref": "$MEM.prog", "para_ref": "$MEM.p",
        }).execute(core, root)
        child = core.get_conversation(int(root.user_batch.tool_responses[-1].content.split("=")[-1]))
        assert child.caps.mode == "user"
        # 工作目录由软件建、根由控制面动态授予（内核不再分配工作空间）
        core.mem.make("$MEM", "ws", "dict", owner_cid=0)
        core.set_conversation_roots(child.cid, {"work": ["ws"]})
        real = ("MEM",) + tuple(child.caps.roots["work"]) + ("x",)
        real_ref = "$" + ".".join(real)

        MemoryMakeInstruction("c2", child.cid, {
            "ref": "$work", "key": "x", "mem_type": "str",
        }).execute(core, child)
        assert child.user_batch.tool_responses[-1].content.startswith("Success created str")

        core.mem["h"] = _handler(
            "def handle(event):\n"
            "    conv = (event.get('source') or {}).get('conversation') or {}\n"
            "    if conv.get('mode') == 'user':\n"
            "        return {'action': 'replace', 'response': '超级用户拒绝'}\n"
            "    return {}\n"
        )
        core.events.register(0, "memory_read", real_ref, "$MEM.h", phase="before",
                             tier=KERNEL_TIER, priority=10, kernel_mode=True)

        MemoryReadInstruction("c3", child.cid, {"ref": "$work.x"}).execute(core, child)
        assert child.user_batch.tool_responses[-1].content == "Interrupted: 超级用户拒绝"
        lst = core.events.listeners[-1]
        assert lst["target_ref"] == ("memory", tuple(child.caps.roots["work"]) + ("x",))
        # 用户态对话自己的监听器不会命中同一次事件：它的 root 表里没有真实路径
        assert _canonical_ref(core, "$work.x", child.caps.space()) == lst["target_ref"]


class TestListenerDeviceDisplay:
    def test_priority_and_tier_are_visible(self):
        core, root = _core()
        _seed_box(core)
        core.mem["h"] = _handler("def handle(event):\n    return {}\n")
        core.events.register(0, "memory_read", "$MEM.box.msg", "$MEM.h",
                             tier=KERNEL_TIER, priority=9, kernel_mode=True)
        text = core.unwrap("$MEM.listeners")
        assert "priority=9" in text and "tier=kernel" in text

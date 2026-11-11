"""Guard 监听器：监听注册授权从 Core 的 owner 检查移到软件层。

覆盖：
- listen_event before 事件先解析 handler 的 condition，载荷带解析后的条件；
- 内核态 Guard 监听 listen_event before，可 replace / allow 普通注册；
- 无 Guard 时注册不受限（默认行为）；
- handler 节点使用已删除条件形态（owned_by_owner、source.cid、preset）时注册报明确错误；
- 宿主入口 EventSystem.register 不做 memory owner 授权（受信任入口）。
"""

import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import Core, ListenEventInstruction
from avm.journal import Journal, JournalError
from avm.exceptions import IllegalOperationError
from avm.types import MetaDict, StrNode


def _handler(condition, code="def handle(event):\n    return {}\n"):
    return MetaDict(
        data={"content": code, "condition": StrNode(json.dumps(condition))},
        ctrl={"type": "handler"},
    )


def _core():
    core = Core()
    core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    core.mem["s"] = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    root = core.start("sys", "user", para_ref="$MEM.p")
    core._active_cid = root.cid
    core.mem.make("$MEM", "box", "dict")
    core.mem.make("$MEM.box", "x", "str")
    return core, root


def _memory_handler(core, ref="$MEM.h", event="memory_read"):
    core.mem["h"] = _handler({
        "events": [event],
        "phase": ["before"],
        "target": {"kind": "exact", "ref": "$MEM.box.x"},
    })


def _listen_memory(core, conv, call_id="l1"):
    conv.user_batch.tool_responses.clear()
    ListenEventInstruction(call_id, conv.cid, {
        "action": "listen", "handler_ref": "$MEM.h",
    }).execute(core, conv)
    return conv.user_batch.tool_responses[-1].content


GUARD_REPLACE_MEMORY = (
    "def handle(event):\n"
    "    condition = (event.get('payload') or {}).get('condition') or {}\n"
    "    target = condition.get('target') or {}\n"
    "    if target.get('kind') in ('exact', 'subtree'):\n"
    "        return {'action': 'replace', 'response': 'Guard: 禁止监听内存路径'}\n"
    "    return {}\n"
)

GUARD_ALLOW = "def handle(event):\n    return {'action': 'allow'}\n"


def _add_guard(core, code, priority=10):
    core.mem["guard"] = _handler({"events": ["listen_event"], "phase": ["before"]}, code)
    return core.events.register_condition(
        0, "$MEM.guard", tier="kernel", priority=priority, kernel_mode=True,
    )


class TestListenBeforePayload:
    def test_before_payload_contains_resolved_condition(self):
        core, root = _core()
        _memory_handler(core)
        seen = []
        core.events.add_observer(seen.append)

        text = _listen_memory(core, root)
        assert text.startswith("listener_id="), text

        before = [e for e in seen if e["event_type"] == "listen_event" and e["phase"] == "before"][-1]
        condition = before["payload"]["condition"]
        assert condition["events"] == ["memory_read"]
        assert condition["phases"] == ["before"]
        assert condition["target"]["kind"] == "exact"
        # 条件里的引用已解析成 canonical 目标
        assert condition["target"]["refs"]["normal"] == ("memory", ("box", "x"))
        assert condition["source"] is None
        assert before["payload"]["handler_ref"] == "$MEM.h"
        assert before["payload"]["action"] == "listen"
        # before 事件的目标是第一个 condition target 的 canonical ref
        assert before["target"] == "$MEM.box.x"
        assert before["target_ref"] == ("memory", ("box", "x"))

    def test_before_target_is_wildcard_without_condition_target(self):
        core, root = _core()
        core.mem["h"] = _handler({"events": ["memory_read"], "phase": ["before"]})
        seen = []
        core.events.add_observer(seen.append)

        assert _listen_memory(core, root).startswith("listener_id=")
        before = [e for e in seen if e["event_type"] == "listen_event" and e["phase"] == "before"][-1]
        assert before["target"] == "*"
        assert before["target_ref"] == ("*",)

    def test_before_target_is_cid_for_user_preset(self):
        core, root = _core()
        user = core.start("sys", "u2")
        user.caps = Capability(mode="user", roots={"uh": ("uh",)},
                               tools=list(ALL_TOOL_NAMES))
        core._active_cid = user.cid
        core.mem["uh"] = _handler({"preset": "self.lifecycle"})
        seen = []
        core.events.add_observer(seen.append)

        user.user_batch.tool_responses.clear()
        ListenEventInstruction("l1", user.cid, {
            "action": "listen", "handler_ref": "$uh",
        }).execute(core, user)
        assert user.user_batch.tool_responses[-1].content.startswith("listener_id=")

        before = [e for e in seen if e["event_type"] == "listen_event" and e["phase"] == "before"][-1]
        assert before["target"] == f"cid:{user.cid}"
        assert before["target_ref"] == ("cid", user.cid)
        assert before["payload"]["condition"]["target"] == {
            "kind": "self_lifecycle", "cid": user.cid,
        }


class TestGuardListener:
    def test_guard_replaces_protected_listen(self):
        core, root = _core()
        _memory_handler(core)
        guard_id = _add_guard(core, GUARD_REPLACE_MEMORY)

        text = _listen_memory(core, root)
        assert text == "Interrupted: Guard: 禁止监听内存路径"
        assert [l["listener_id"] for l in core.events.listeners] == [guard_id]

    def test_guard_allow_then_registration_succeeds(self):
        core, root = _core()
        _memory_handler(core)
        _add_guard(core, GUARD_ALLOW)

        text = _listen_memory(core, root)
        assert text.startswith("listener_id="), text
        assert len(core.events.listeners) == 2

    def test_without_guard_registration_succeeds(self):
        """默认行为：镜像没装 Guard 时 Core 不拦截 memory 监听注册。"""
        core, root = _core()
        _memory_handler(core)
        text = _listen_memory(core, root)
        assert text.startswith("listener_id="), text
        assert len(core.events.listeners) == 1

    def test_guard_sees_normal_conversation_listen(self):
        """普通对话（用户态）的 listen_event 请求也受 Guard 裁决。"""
        core, root = _core()
        user = core.start("sys", "u2")
        user.caps = Capability(mode="user", roots={"uh": ("uh",)},
                               tools=list(ALL_TOOL_NAMES))
        core._active_cid = user.cid
        core.mem["uh"] = _handler({"preset": "self.lifecycle"})
        guard_id = _add_guard(core, (
            "def handle(event):\n"
            "    source = (event.get('source') or {}).get('conversation') or {}\n"
            "    if source.get('mode') == 'user':\n"
            "        return {'action': 'replace', 'response': 'Guard: 用户态禁止注册'}\n"
            "    return {}\n"
        ))

        user.user_batch.tool_responses.clear()
        ListenEventInstruction("l1", user.cid, {
            "action": "listen", "handler_ref": "$uh",
        }).execute(core, user)
        assert user.user_batch.tool_responses[-1].content == "Interrupted: Guard: 用户态禁止注册"
        assert len(core.events.listeners) == 1  # 只有 Guard 自己

        # 换一条放行策略的 Guard 后同一注册成功（处理器代码是注册时的副本）
        core.events.cancel(guard_id, 0)
        _add_guard(core, GUARD_ALLOW)
        user.user_batch.tool_responses.clear()
        ListenEventInstruction("l2", user.cid, {
            "action": "listen", "handler_ref": "$uh",
        }).execute(core, user)
        assert user.user_batch.tool_responses[-1].content.startswith("listener_id=")


class TestRemovedConditionForms:
    def _register(self, core, condition):
        core.mem["bad"] = _handler(condition)
        with pytest.raises(IllegalOperationError) as ei:
            core.events.register_condition(0, "$MEM.bad", kernel_mode=True)
        return str(ei.value)

    def test_owned_by_owner_rejected(self):
        core, _ = _core()
        text = self._register(core, {
            "events": ["memory_read"],
            "target": {"kind": "owned_by_owner"},
        })
        assert "已移除" in text and "owned_by_owner" in text

    def test_source_cid_rejected(self):
        core, _ = _core()
        text = self._register(core, {
            "events": ["memory_read"],
            "source": {"cid": [0]},
        })
        assert "source.cid 已移除" in text

    def test_own_nodes_preset_rejected(self):
        core, _ = _core()
        text = self._register(core, {"preset": "own_nodes.memory"})
        assert "已移除" in text and "own_nodes.memory" in text


# ----------------------------------------------------------------------
# 格式拒绝：旧版本输入不能回流
# ----------------------------------------------------------------------

class TestOldFormatRejected:
    def test_old_journal_record_version_is_rejected(self, tmp_path):
        core, root = _core()
        path = str(tmp_path / "run.journal")
        core.attach_journal(path)
        Journal(path).append({"v": 2, "generation": 1, "cid": root.cid, "changes": []})
        with pytest.raises(JournalError, match="不支持的记录版本"):
            core.replay_journal()


class TestHostRegisterIsTrusted:
    def test_register_does_not_check_memory_owner(self):
        """EventSystem.register 是宿主/内部受信任入口，不再做 memory owner 授权。"""
        core, _ = _core()
        core.mem["h"] = _handler({"events": ["memory_read"]})
        assert core.events.register(1, "memory_read", "$MEM.box.x", "$MEM.h") == 1

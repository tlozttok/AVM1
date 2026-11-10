"""R4：声明式条件监听（handler 节点 condition 数据、用户态 named presets）。"""

import json

from avm.capabilities import ALL_TOOL_NAMES
from avm.core import (
    Core,
    CreateInstruction,
    ListenEventInstruction,
    MemoryReadInstruction,
    MemoryWriteInstruction,
)
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


def _handler(condition, code="def handle(event):\n    return {}\n"):
    return MetaDict(data={"content": code,
                          "condition": StrNode(json.dumps(condition))},
                    ctrl={"type": "handler"})


def _call(core, conv, cmd, args, call_id="c1"):
    conv.user_batch.tool_responses.clear()
    core._process_return_calls([{"cmd_type": cmd, "call_id": call_id, "args": args}], conv)
    return conv.user_batch.tool_responses[-1].content


def _user_conv(core, root, caps):
    prog = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    prog["caps"] = _caps_node(caps)
    core.mem["up"] = prog
    text = _call(core, root, "create_cmd",
                 {"system_ref": "$MEM.up", "para_ref": "$MEM.p"})
    return core.get_conversation(int(text.split("=")[-1]))


class TestKernelCondition:
    def test_exact_and_phase(self):
        core, root = _core()
        core.mem.make("$MEM", "box", "dict")
        core.mem.make("$MEM.box", "x", "str")
        seen = []
        core.events.add_observer(seen.append)
        h = _handler({"events": ["memory_read"], "phase": ["after"],
                      "target": {"kind": "exact", "ref": "$MEM.box.x"}})
        core.mem["h"] = h
        core.events.register_condition(0, "$MEM.h", tier="kernel", priority=1, kernel_mode=True)
        MemoryReadInstruction("r1", 0, {"ref": "$MEM.box.x"}).execute(core, root)
        after = [e for e in seen if e["event_type"] == "memory_read" and e["phase"] == "after"]
        before = [e for e in seen if e["event_type"] == "memory_read" and e["phase"] == "before"]
        assert len(after) == 1
        assert before and "target_info" not in before[0]

    def test_subtree_and_source_filter(self):
        core, root = _core()
        core.mem.make("$MEM", "box", "dict")
        core.mem.make("$MEM.box", "x", "str")
        seen = []
        core.events.add_observer(seen.append)
        h = _handler({"events": ["memory_write"], "phase": ["before"],
                      "target": {"kind": "subtree", "ref": "$MEM.box"},
                      "source": {"mode": ["kernel"]}})
        core.mem["h"] = h
        core.events.register_condition(0, "$MEM.h", tier="kernel", kernel_mode=True)
        MemoryWriteInstruction("w1", 0, {"ref": "$MEM.box.x", "content": "v"}).execute(core, root)
        writes = [e for e in seen if e["event_type"] == "memory_write" and e["phase"] == "before"]
        assert len(writes) == 1


class TestUserPresets:
    def _user(self, core, root, condition):
        user = _user_conv(core, root, {
            "mode": "user", "groups": ["worker"],
            "tools": list(ALL_TOOL_NAMES),
        })
        # 工作目录由软件建、根由控制面动态授予：handler 放在用户可见的目录里
        core.mem.make("$MEM", "workdirs", "dict")
        core.mem.make("$MEM.workdirs", f"w{user.cid}", "dict")
        core.set_conversation_roots(user.cid, {"work": ["workdirs", f"w{user.cid}"]})
        work = core.mem.resolve_path(["MEM", "workdirs", f"w{user.cid}"], for_llm=False)
        work["h"] = _handler(condition)
        work["h"].set_ctrl({"type": "handler", "owner": str(user.cid)})
        return user

    def _listen(self, core, user):
        return _call(core, user, "listen_event", {
            "action": "listen", "handler_ref": "$work.h",
        })

    def test_own_work_preset_removed(self):
        core, root = _core()
        user = self._user(core, root, {"preset": "own_work.memory"})
        text = self._listen(core, user)
        assert text.startswith(("Illegal:", "Error:")), text
        assert "已移除" in text
        assert core.events.listeners == []

    def test_own_nodes_preset_removed(self):
        core, root = _core()
        user = self._user(core, root, {"preset": "own_nodes.memory"})
        text = self._listen(core, user)
        assert text.startswith(("Illegal:", "Error:")), text
        assert "已移除" in text and "owned" not in text
        assert core.events.listeners == []

    def test_user_preset_self_lifecycle_registers(self):
        core, root = _core()
        user = self._user(core, root, {"preset": "self.lifecycle"})
        assert self._listen(core, user).startswith("listener_id=")
        record = core.events.listeners[-1]
        assert record["condition"]["target"] == {"kind": "self_lifecycle", "cid": user.cid}

    def test_user_custom_condition_rejected(self):
        core, root = _core()
        user = self._user(core, root, {"events": ["memory_read"]})
        text = self._listen(core, user)
        assert "命名条件子集" in text

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


def _handler(condition, code="def handle(event):\n    return {}\n"):
    return MetaDict(data={"content": code,
                          "condition": StrNode(json.dumps(condition))},
                    ctrl={"type": "handler"})


def _call(core, conv, cmd, args, call_id="c1"):
    conv.user_batch.tool_responses.clear()
    core._process_return_calls([{"cmd_type": cmd, "call_id": call_id, "args": args}], conv)
    return conv.user_batch.tool_responses[-1].content


def _user_conv(core, root, caps):
    core.install_security_config({}, ("workspaces",))
    prog = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
    prog["caps"] = _caps_node(caps)
    core.mem["up"] = prog
    text = _call(core, root, "create_cmd",
                 {"system_ref": "$MEM.up", "para_ref": "$MEM.p"})
    return core.get_conversation(int(text.split("=")[-1]))


class TestKernelCondition:
    def test_exact_and_phase(self):
        core, root = _core()
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.make("$MEM.box", "x", "str", owner_cid=0)
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
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.make("$MEM.box", "x", "str", owner_cid=0)
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
    def _user(self, core, root):
        user = _user_conv(core, root, {
            "mode": "user", "workspace": "work", "groups": ["worker"],
            "tools": list(ALL_TOOL_NAMES),
        })
        core.mem["h"] = _handler({"preset": "own_work.memory"})
        # handler 必须对用户可见：放到其工作空间
        user.user_batch.tool_responses.clear()
        root_path = "$MEM." + ".".join(user.caps.roots["work"]) + ".h"
        node = core.mem.resolve_path(root_path.split(".")[0:0] or ["MEM"] + list(user.caps.roots["work"]), for_llm=False)
        node["h"] = _handler({"preset": "own_work.memory"})
        node["h"].set_ctrl({"type": "handler", "owner": str(user.cid)})
        _call(core, user, "listen_event", {"action": "listen", "handler_ref": "$work.h"})
        assert user.user_batch.tool_responses[-1].content.startswith("listener_id=")
        return user

    def test_user_preset_registers_and_matches_own_work(self):
        core, root = _core()
        user = self._user(core, root)
        seen = []
        core.events.add_observer(seen.append)
        _call(core, user, "memory_make", {"ref": "$work", "key": "x", "mem_type": "str"})
        _call(core, user, "memory_read", {"ref": "$work.x"})
        reads = [e for e in seen if e["event_type"] == "memory_read"]
        assert {e["phase"] for e in reads} == {"before", "after"}

    def test_user_custom_condition_rejected(self):
        core, root = _core()
        user = _user_conv(core, root, {
            "mode": "user", "workspace": "work", "tools": list(ALL_TOOL_NAMES),
        })
        ws = core.mem.resolve_path(["MEM"] + list(user.caps.roots["work"]), for_llm=False)
        ws["h"] = _handler({"events": ["memory_read"]})
        _call(core, user, "listen_event", {"action": "listen", "handler_ref": "$work.h"})
        assert "命名条件子集" in user.user_batch.tool_responses[-1].content

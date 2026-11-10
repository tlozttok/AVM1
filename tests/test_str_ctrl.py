"""字符串叶子的控制数据（StrNode）：只有 ctrl，没有描述数据。

批次二前置：str 需要携带控制数据（owner、后续的 ACL 等），但不带描述数据。
"""

import copy
import json

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import Core
from avm.image import ImageError, load_image
from avm.memory import Memory
from avm.types import MetaDict, MetaList, StrNode


def _caps_node(caps):
    data = {}
    if "mode" in caps:
        data["mode"] = StrNode(str(caps["mode"]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            data[key] = MetaList([StrNode(str(x)) for x in caps[key]])
    return MetaDict(data=data)


class TestStrNode:
    def test_str_semantics_preserved(self):
        node = StrNode("hello", ctrl={"owner": "3"})
        assert isinstance(node, str)
        assert node == "hello"
        assert json.dumps(node) == '"hello"'
        assert node.upper() == "HELLO"
        assert node.get_ctrl() == {"owner": "3"}

    def test_copy_and_deepcopy_preserve_ctrl(self):
        node = StrNode("v", ctrl={"owner": "7"})
        assert copy.copy(node).get_ctrl() == {"owner": "7"}
        assert copy.deepcopy(node).get_ctrl() == {"owner": "7"}
        wrapped = {"k": node}
        assert copy.deepcopy(wrapped)["k"].get_ctrl() == {"owner": "7"}

    def test_plain_str_is_accepted_where_metadata_is_read(self):
        mem = Memory()
        mem["plain"] = "x"
        assert isinstance(mem["plain"], StrNode)
        assert mem["plain"].get_ctrl() is None


class TestMemoryStrCtrl:
    def test_make_str_records_owner(self):
        mem = Memory()
        assert mem.make("$MEM", "s", "str", owner_cid=5) == "$MEM.s"
        node = mem.unwrap(["$", "MEM", "s"], for_llm=False)
        assert isinstance(node, StrNode)
        assert node.get_ctrl()["owner"] == "5"

    def test_write_preserves_ctrl(self):
        mem = Memory()
        mem.make("$MEM", "s", "str", owner_cid=5)
        mem.set("$MEM.s", "new")
        node = mem.unwrap(["$", "MEM", "s"], for_llm=False)
        assert node == "new"
        assert node.get_ctrl()["owner"] == "5"

    def test_append_records_owner(self):
        mem = Memory()
        mem.make("$MEM", "L", "list", owner_cid=5)
        assert mem.set("$MEM.L.-1", "leaf", owner_cid=9) == "$MEM.L.0"
        node = mem.resolve_path(["MEM", "L", "0"], for_llm=False)
        assert node.get_ctrl()["owner"] == "9"

    def test_edit_metadata_on_str(self):
        core = Core()
        conv = core.start("sys", "user")
        core._active_cid = conv.cid
        core.mem.make("$MEM", "s", "str", owner_cid=conv.cid)
        conv.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "edit_metadata", "call_id": "c1",
            "args": {"ref": "$MEM.s", "type": "set", "key": "flag", "value": "1"},
        }], conv)
        assert "已写入" in conv.user_batch.tool_responses[-1].content
        node = core.mem.resolve_path(["MEM", "s"], for_llm=False)
        assert node.get_ctrl()["flag"] == "1"
        conv.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "edit_metadata", "call_id": "c2",
            "args": {"ref": "$MEM.s", "type": "get", "key": "flag"},
        }], conv)
        assert conv.user_batch.tool_responses[-1].content == "1"

    def test_save_load_roundtrip_keeps_str_ctrl(self, tmp_path):
        mem = Memory()
        mem.make("$MEM", "s", "str", owner_cid=5)
        mem.make("$MEM", "d", "dict", owner_cid=6)
        path = tmp_path / "mem.json"
        mem.save(str(path))
        loaded = Memory.load(str(path))
        node = loaded.resolve_path(["MEM", "s"], for_llm=False)
        assert isinstance(node, StrNode)
        assert node.get_ctrl()["owner"] == "5"


class TestStrInImage:
    def _image(self, str_node):
        return {
            "meta": {"version": 1},
            "mem": {
                "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                                 "value": {"model": {"kind": "str", "value": "test"}}},
                "s": str_node,
                "u": {"kind": "str", "value": "go"},
                "programs": {"kind": "dict", "value": {
                    "init": {"kind": "dict", "ctrl": {"type": "settingup"},
                             "value": {"content": {"kind": "str", "value": "sys"}}},
                }},
            },
            "para_ref": "$MEM.model_params",
            "init": {"system_ref": "$MEM.programs.init", "user_ref": "$MEM.u"},
        }

    def test_str_node_accepts_ctrl(self, tmp_path):
        path = tmp_path / "i.json"
        path.write_text(json.dumps(self._image({
            "kind": "str", "value": "v", "ctrl": {"owner": "0"},
        })), encoding="utf-8")
        core = load_image(Core(), str(path))
        node = core.mem.resolve_path(["MEM", "s"], for_llm=False)
        assert isinstance(node, StrNode)
        assert node.get_ctrl() == {"owner": "0"}

    def test_str_node_rejects_meta(self, tmp_path):
        path = tmp_path / "i.json"
        path.write_text(json.dumps(self._image({
            "kind": "str", "value": "v", "meta": "描述",
        })), encoding="utf-8")
        with pytest.raises(ImageError) as ei:
            load_image(Core(), str(path))
        assert "描述" in str(ei.value) and "meta" in str(ei.value)


class TestStrOwnerFixesWorkspaceListener:
    def test_user_owned_str_can_be_listened(self):
        """动态授予的工作目录由软件建成；用户对话自己建的 str 带自己的 owner。"""
        core = Core()
        core.mem["model_params"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        parent = core.start("sys", "parent", para_ref="$MEM.model_params")
        core._active_cid = parent.cid
        prog = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
        prog["caps"] = _caps_node({"mode": "user"})
        core.mem["prog"] = prog
        parent.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "create_cmd", "call_id": "c1",
            "args": {"system_ref": "$MEM.prog", "para_ref": "$MEM.model_params"},
        }], parent)
        child = core.get_conversation(int(parent.user_batch.tool_responses[-1].content.split("=")[-1]))
        core._active_cid = child.cid
        # 工作目录由软件建、根动态授予（内核不再分配工作空间）
        core.mem["ws"] = MetaDict(data={})
        core.set_conversation_roots(child.cid, {"work": ["ws"]})
        assert child.caps.tools == ALL_TOOL_NAMES or child.caps.mode == "user"
        # 用户对话在自己的工作目录建 str，owner 应是它自己
        child.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "memory_make", "call_id": "c2",
            "args": {"ref": "$work", "key": "s", "mem_type": "str"},
        }], child)
        node = core.mem.resolve_path(["MEM"] + list(child.caps.roots["work"]) + ["s"], for_llm=False)
        assert node.get_ctrl()["owner"] == str(child.cid)
        # 监听自己的 str 不再被 owner 检查挡住（handler 也在工作空间里）
        child.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "memory_make", "call_id": "c3",
            "args": {"ref": "$work", "key": "h", "mem_type": "dict"},
        }], child)
        handler = core.mem.resolve_path(["MEM"] + list(child.caps.roots["work"]) + ["h"], for_llm=False)
        handler["content"] = "def handle(event):\n    return {}\n"
        handler["condition"] = StrNode(json.dumps({"preset": "own_nodes.memory"}))
        handler.set_ctrl({"type": "handler"})
        child.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "listen_event", "call_id": "c4",
            "args": {"action": "listen", "handler_ref": "$work.h"},
        }], child)
        text = child.user_batch.tool_responses[-1].content
        assert "listener_id=" in text

"""内存树链接节点（LinkNode）与"挂载即物化链接"的测试。

对应计划 plan-link-node-2026-10-12 的验收：挂载物化、根读可枚举、读写与
元数据编辑进入目标、标识符/IPC 解析、成环检测、序列化往返、卸载清理、挂载规则。
"""

import json

import pytest

from avm.core import Core
from avm.exceptions import (
    MemoryCircularReferenceError,
    MemoryKeyNotFoundError,
    MemoryTypeError,
    VMMemoryError,
)
from avm.memory import Memory
from avm.memory_device import LinkDevice, MemoryDevice
from avm.types import MetaDict, MetaList, LinkNode


class BoxDevice(MemoryDevice):
    """最小可读写设备：内部是一个 dict，支持下钻与整值写入。"""

    def __init__(self, data=None):
        self._data = dict(data or {})

    def to_llm_string(self):
        return f"box[keys={sorted(self._data)}]"

    def resolve_path(self, path):
        current = self
        for key in path:
            current = current[key]
        return current

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def set_value(self, value, owner_cid=None):
        self._data = dict(value)


def _mount_link_device(core, name, target):
    """挂一个设备层别名（LinkDevice），并把它接到 Core 上。"""
    device = LinkDevice(target)
    core.mem.mount(name, device)
    device.mount_path = name
    device.attach_core(core)
    return device


class TestMountMaterializesLink:
    def test_mount_plants_link_node(self):
        core = Core()
        device = BoxDevice({"a": "1"})
        core.mem.mount("box", device)

        link = core.mem._data["box"]
        assert isinstance(link, LinkNode)
        assert link.target == "box"
        assert link.get_metadata() == "BoxDevice"
        # 设备表保留地址，另有标识符索引
        assert core.mem._devices["box"] is device
        assert core.mem._device_ids["box"] is device

    def test_root_read_lists_mount_points(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))

        text = core.mem.resolve_path(["MEM"], for_llm=True)
        assert "box" in text
        for name in ("conversations", "scheduler", "memory", "monitor", "listeners"):
            assert name in text
            assert isinstance(core.mem._data[name], LinkNode)

    def test_memory_summary_counts_links(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))
        summary = core.mem.resolve_path(["MEM", "memory"], for_llm=True)
        # 五个内建设备 + box
        assert "'link': 6" in summary

    def test_parent_segments_materialized(self):
        core = Core()
        core.mem.mount("cfg.dev", BoxDevice({"a": "1"}))

        assert isinstance(core.mem._data["cfg"], MetaDict)
        assert isinstance(core.mem._data["cfg"]["dev"], LinkNode)
        assert core.unwrap("$MEM.cfg.dev.a") == "1"

    def test_parent_link_rejects_second_mount(self):
        core = Core()
        core.mem.mount("box", BoxDevice({}))
        with pytest.raises(VMMemoryError):
            core.mem.mount("box.dev", BoxDevice({}))
        assert "box.dev" not in core.mem._devices

    def test_same_named_data_node_rejected(self):
        core = Core()
        core.mem["box"] = {"a": "1"}
        with pytest.raises(VMMemoryError):
            core.mem.mount("box", BoxDevice({}))
        assert "box" not in core.mem._devices

    def test_duplicate_or_dollar_identifier_rejected(self):
        core = Core()
        first, second, third = BoxDevice({}), BoxDevice({}), BoxDevice({})
        second.device_id = "box"
        third.device_id = "$MEM.x"

        core.mem.mount("box", first)
        with pytest.raises(VMMemoryError):
            core.mem.mount("other", second)
        with pytest.raises(VMMemoryError):
            core.mem.mount("third", third)


class TestTransparentAccess:
    def test_read_enters_target(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))

        # 无剩余段：进入设备本身（设备自述），不返回链接描述
        assert core.unwrap("$MEM.box") == "box[keys=['a']]"
        assert core.unwrap("$MEM.box.a") == "1"

    def test_write_enters_target(self):
        core = Core()
        device = BoxDevice({"a": "1"})
        core.mem.mount("box", device)

        core.mem.set("$MEM.box.a", "2")
        assert device["a"] == "2"
        core.mem.set("$MEM.box", {"b": "3"})
        assert device._data == {"b": "3"}

    def test_link_to_memory_address(self):
        core = Core()
        core.mem["real"] = {"x": "1"}
        core.mem._data["rooms"] = LinkNode(target="$MEM.real")

        assert core.unwrap("$MEM.rooms", for_llm=False) is core.unwrap("$MEM.real", for_llm=False)
        assert core.unwrap("$MEM.rooms.x") == "1"

    def test_memory_make_and_metadata_edit_enter_target(self):
        core = Core()
        core.mem["real"] = MetaDict(data={"x": "1"}, metadata="旧")
        core.mem._data["rooms"] = LinkNode(target="$MEM.real")

        # memory_make 落在链接下 → 进入目标创建
        core.mem.make("$MEM.rooms", "child", "str")
        assert core.mem["real"]["child"] == ""
        # 元数据编辑解析到目标节点本身（指令走 core.unwrap）
        node = core.unwrap("$MEM.rooms", for_llm=False)
        assert node is core.mem["real"]
        node.set_metadata("新")
        assert core.mem["real"].get_metadata() == "新"

    def test_unknown_device_identifier(self):
        core = Core()
        core.mem._data["ghost"] = LinkNode(target="nobody")
        with pytest.raises(VMMemoryError) as ei:
            core.unwrap("$MEM.ghost")
        assert "设备标识符未知" in str(ei.value)

    def test_error_messages_do_not_leak_target_path(self):
        """链接是 capability 边界：报错写请求路径，不写真实目标路径。"""
        core = Core()
        core.mem["game"] = MetaDict(data={"map": MetaDict(data={"rooms": MetaDict(data={})})})
        core.mem._data["rooms"] = LinkNode(target="$MEM.game.map.rooms")

        with pytest.raises(MemoryKeyNotFoundError) as ei:
            core.unwrap("$MEM.rooms.ghost")
        assert "rooms.ghost" in str(ei.value)
        assert "game.map" not in str(ei.value)

        core.mem["leaf"] = "s"
        core.mem._data["alias"] = LinkNode(target="$MEM.leaf")
        with pytest.raises(MemoryTypeError) as ei:
            core.mem.set("$MEM.alias.deep", "v")
        assert "alias" in str(ei.value)
        assert "leaf" not in str(ei.value)

    def test_follow_link_target_reports_real_kind(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))
        core.mem["real"] = {"x": "1"}
        core.mem._data["rooms"] = LinkNode(target="$MEM.real")

        assert core.mem.follow_link_target("$MEM.box.a") == ("device", "box", ["a"])
        assert core.mem.follow_link_target("$MEM.rooms.x") == ("memory", ["real", "x"])


class TestCircularLinks:
    def test_two_links_cycle(self):
        core = Core()
        core.mem._data["l1"] = LinkNode(target="$MEM.l2")
        core.mem._data["l2"] = LinkNode(target="$MEM.l1")

        with pytest.raises(MemoryCircularReferenceError) as ei:
            core.unwrap("$MEM.l1")
        message = str(ei.value)
        assert "链接成环" in message
        assert "l1" in message and "l2" in message

    def test_self_cycle(self):
        core = Core()
        core.mem._data["l1"] = LinkNode(target="$MEM.l1")
        with pytest.raises(MemoryCircularReferenceError):
            core.unwrap("$MEM.l1")

    def test_cycle_through_subpath(self):
        core = Core()
        core.mem["a"] = MetaDict(data={})
        core.mem._data["a"]["l1"] = LinkNode(target="$MEM.a.l1.x")

        with pytest.raises(MemoryCircularReferenceError):
            core.unwrap("$MEM.a.l1.y")

    def test_cycle_through_device_alias(self):
        core = Core()
        _mount_link_device(core, "alias", "$MEM.l1")
        core.mem._data["l1"] = LinkNode(target="alias")

        with pytest.raises(MemoryCircularReferenceError):
            core.unwrap("$MEM.l1")

    def test_write_cycle_raises(self):
        core = Core()
        core.mem._data["l1"] = LinkNode(target="$MEM.l2")
        core.mem._data["l2"] = LinkNode(target="$MEM.l1")

        with pytest.raises(MemoryCircularReferenceError):
            core.mem.set("$MEM.l1.x", "v")

    def test_link_device_write_cycle_raises(self):
        core = Core()
        core.mem["real"] = MetaDict(data={})
        _mount_link_device(core, "a", "$MEM.b")
        _mount_link_device(core, "b", "$MEM.a")

        with pytest.raises(MemoryCircularReferenceError):
            core.mem.set("$MEM.a.sub", "x")


class TestUnmountAndMask:
    def test_unmount_clears_address_identifier_and_link(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))
        core.mem.unmount("box")

        assert "box" not in core.mem._data
        assert "box" not in core.mem._devices
        assert "box" not in core.mem._device_ids
        with pytest.raises(VMMemoryError):
            core.mem.unmount("box")

    def test_remount_restores_link(self):
        core = Core()
        device = BoxDevice({"a": "1"})
        core.mem.mount("box", device)
        core.mem.unmount("box")
        core.mem.mount("box", device)

        assert isinstance(core.mem._data["box"], LinkNode)
        assert core.unwrap("$MEM.box.a") == "1"
        assert core.mem.resolve_path(["MEM"], for_llm=True).count("box") >= 1

    def test_materialize_mount_links_restores_after_tree_replace(self):
        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))
        # 模拟镜像装载：整树替换后再按设备表重建挂载点
        core.mem._data = MetaDict(data={"system": "sys"})
        core.mem.materialize_mount_links()

        assert isinstance(core.mem._data["box"], LinkNode)
        assert core.unwrap("$MEM.box.a") == "1"


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        core = Core()
        core.mem["real"] = {"x": "1"}
        core.mem._data["rooms"] = LinkNode(target="$MEM.real", metadata="别名", ctrl={"owner": "0"})

        out = str(tmp_path / "mem.json")
        core.mem.save(out)
        raw = json.load(open(out, encoding="utf-8"))
        assert raw["value"]["rooms"]["kind"] == "link"
        assert raw["value"]["rooms"]["target"] == "$MEM.real"
        assert raw["value"]["rooms"]["meta"] == "别名"

        loaded = Memory.load(out)
        link = loaded._data["rooms"]
        assert isinstance(link, LinkNode)
        assert link.target == "$MEM.real"
        assert link.get_ctrl() == {"owner": "0"}

    def test_deserialize_rejects_empty_target(self):
        with pytest.raises(VMMemoryError):
            Memory._deserialize_node({"kind": "link", "target": ""})

    def test_to_dict_does_not_follow(self):
        core = Core()
        core.mem["real"] = {"x": "1"}
        core.mem._data["rooms"] = LinkNode(target="$MEM.real")

        assert core.mem._data.to_dict()["rooms"] == {"$link": "$MEM.real"}
        assert MetaList(data=[LinkNode(target="$MEM.real")]).to_list() == [{"$link": "$MEM.real"}]


class TestConsoleRendering:
    def test_jsonable_and_walk_show_link_kind(self):
        from avm.console import _jsonable, _walk

        core = Core()
        core.mem.mount("box", BoxDevice({"a": "1"}))

        rendered = _jsonable(core.mem._data)["children"]["box"]
        assert rendered == {"kind": "link", "target": "box", "metadata": "BoxDevice", "ctrl": None}

        flat = _walk(core.mem._data)
        assert flat["box"] == {"kind": "link", "target": "box"}
        # 链接不跟随：面板看到的是树本身，不是设备内部
        assert "box.a" not in flat

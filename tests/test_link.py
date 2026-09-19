"""LinkDevice 链接设备测试：透明读写、链式、成环、错误隐藏、镜像、事件。"""

import json

import pytest

from avm.core import Core, MemoryReadInstruction, EditMetadataInstruction, _build_conversation
from avm.events import EVENT_TYPES
from avm.exceptions import MemoryCircularReferenceError, VMMemoryError
from avm.image import ImageError, load_image
from avm.memory import Memory
from avm.memory_device import LinkDevice, BUILTIN_DEVICE_CLASSES
from avm.types import MetaDict, MetaList


def _python_node(code):
    return MetaDict(data={"content": code}, ctrl={"type": "python"})


def _mount(core, path, target):
    link = LinkDevice(target)
    link.mount_path = path
    core.mem.mount(path, link)
    link.attach_core(core)
    return link


def _new_root(core):
    return core.start("sys", "hello")


class TestBasicTransparency:
    def test_read_str_dict_list(self):
        core = Core()
        core.mem["deep"] = {"str": "hello", "d": {"k": "v"}, "l": [1, 2]}
        _mount(core, "alias", "$MEM.deep")

        assert core.mem.resolve_path(["MEM", "alias"], for_llm=False) == core.mem["deep"]
        assert core.mem.resolve_path(["MEM", "alias", "str"], for_llm=True) == "hello"
        node = core.mem.resolve_path(["MEM", "alias", "d"], for_llm=False)
        assert isinstance(node, MetaDict) and node["k"] == "v"
        lst = core.mem.resolve_path(["MEM", "alias", "l"], for_llm=False)
        assert isinstance(lst, MetaList) and lst[0] == 1

    def test_subpath_read_write(self):
        core = Core()
        core.mem["deep"] = {"a": {"b": "orig"}}
        _mount(core, "alias", "$MEM.deep")

        # 读子路径
        assert core.unwrap("$MEM.alias.a.b", for_llm=False) == "orig"
        # 写子路径
        core.mem.set("$MEM.alias.a.c", "new", owner_cid=0)
        assert core.mem["deep"]["a"]["c"] == "new"
        # 精确别名写（目标为 dict 时是写 meta，沿用既有语义）
        core.mem.set("$MEM.alias", "whole", owner_cid=0)
        assert core.mem["deep"].get_metadata() == "whole"

    def test_non_mem_alias(self):
        core = Core()
        core.mem["secret"] = "data"
        _mount(core, "rooms", "$MEM.secret")
        assert core.mem.unwrap(["$", "rooms"], for_llm=True) == "data"


class TestMakeAndEdit:
    def test_memory_make_via_alias_writes_owner(self):
        core = Core()
        root = _new_root(core)
        core.mem["deep"] = {}
        _mount(core, "alias", "$MEM.deep")

        MemoryReadInstruction  # noqa: F401
        core.mem.make("$MEM.alias", "child", "dict", owner_cid=0)
        assert isinstance(core.mem["deep"]["child"], MetaDict)
        assert core.mem["deep"]["child"].get_ctrl()["owner"] == "0"
        core.mem.make("$MEM.alias", "s", "str", owner_cid=0)
        assert core.mem["deep"]["s"] == ""

    def test_edit_metadata_via_alias(self):
        core = Core()
        root = _new_root(core)
        core.mem.make("$MEM", "deep", "dict", owner_cid=0)
        _mount(core, "alias", "$MEM.deep")

        EditMetadataInstruction("em", root.cid, {"ref": "$MEM.alias", "type": "set", "key": "type", "value": "para"}).execute(core, root)
        assert core.mem["deep"].get_ctrl()["type"] == "para"

        root.user_batch.clear()
        EditMetadataInstruction("em2", root.cid, {"ref": "$MEM.alias", "type": "set", "key": "owner", "value": "9"}).execute(core, root)
        assert "不允许通过 edit_metadata 修改" in root.user_batch.tool_responses[-1].content
        assert core.mem["deep"].get_ctrl()["owner"] == "0"


class TestChainAndCycle:
    def test_chain_a_to_b_to_c(self):
        core = Core()
        core.mem["real"] = {"value": "x"}
        _mount(core, "c", "$MEM.real.value")
        _mount(core, "b", "$MEM.c")
        _mount(core, "a", "$MEM.b")

        assert core.unwrap("$MEM.a", for_llm=True) == "x"
        assert core.unwrap("$MEM.a" if False else "$MEM.b", for_llm=True) == "x"

    def test_cycle_raises_with_aliases_only(self):
        core = Core()
        core.mem["real"] = {"x": "1"}
        _mount(core, "a", "$MEM.b")
        _mount(core, "b", "$MEM.a")

        with pytest.raises(MemoryCircularReferenceError) as ei:
            core.unwrap("$MEM.a", for_llm=True)
        assert "链接成环" in str(ei.value)
        assert "$MEM.real" not in str(ei.value)
        assert "a" in str(ei.value) and "b" in str(ei.value)

    def test_write_cycle_raises(self):
        core = Core()
        core.mem["real"] = {}
        _mount(core, "a", "$MEM.b")
        _mount(core, "b", "$MEM.a")

        with pytest.raises(MemoryCircularReferenceError):
            core.mem.set("$MEM.a.sub", "x")


class TestErrorHiding:
    def test_missing_target_error_uses_alias(self):
        core = Core()
        core.mem["real"] = {"a": "1"}
        _mount(core, "alias", "$MEM.real.missing")

        with pytest.raises(VMMemoryError) as ei:
            core.unwrap("$MEM.alias.sub", for_llm=True)
        # 错误信息应只含别名，不泄露真实路径
        assert "$MEM.alias" in str(ei.value)
        assert "$MEM.real" not in str(ei.value)


class TestImageBuiltin:
    def _image(self, rooms_target="$MEM.game.map.rooms", extra=None):
        image = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {
                    "kind": "dict",
                    "ctrl": {"type": "para"},
                    "value": {"model": {"kind": "str", "value": "test"}},
                },
                "game": {
                    "kind": "dict",
                    "value": {
                        "map": {
                            "kind": "dict",
                            "value": {
                                "rooms": {"kind": "dict", "meta": "rooms", "value": {}}
                            },
                        }
                    },
                },
                "rooms": {"kind": "device"},
            },
            "para_ref": "$MEM.model_params",
            "devices": [
                {
                    "path": "rooms",
                    "class": "LinkDevice",
                    "builtin": True,
                    "args": {"target": rooms_target},
                }
            ],
        }
        if extra:
            image["devices"][0].update(extra)
        return image

    def test_builtin_load_success(self, tmp_path):
        path = tmp_path / "link.json"
        path.write_text(json.dumps(self._image()), encoding="utf-8")
        core = load_image(str(path))
        assert isinstance(core.mem._devices["rooms"], LinkDevice)
        assert core.unwrap("$MEM.rooms", for_llm=True) == "dict[keys=[],metadata='rooms']"

    def test_builtin_with_file_rejected(self, tmp_path):
        path = tmp_path / "link.json"
        path.write_text(json.dumps(self._image(extra={"file": "foo.py"})), encoding="utf-8")
        with pytest.raises(ImageError, match="builtin=true 不允许提供 file"):
            load_image(str(path))

    def test_unreachable_target_fails_load(self, tmp_path):
        path = tmp_path / "link.json"
        path.write_text(json.dumps(self._image(rooms_target="$MEM.nope")), encoding="utf-8")
        with pytest.raises(ImageError, match="目标不可达"):
            load_image(str(path))

    def test_builtin_class_registry(self):
        assert "LinkDevice" in BUILTIN_DEVICE_CLASSES


class TestPersistence:
    def test_save_does_not_include_link(self, tmp_path):
        core = Core()
        core.mem["deep"] = {"x": "1"}
        _mount(core, "alias", "$MEM.deep")
        out = str(tmp_path / "mem.json")
        core.mem.save(out)

        loaded = Memory.load(out)
        # 设备不持久化；alias 不应出现在数据中
        assert "alias" not in loaded._data
        assert loaded._devices == {}


class TestEventTransparency:
    def test_alias_register_by_real_owner_and_after_memory(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core.mem["s"] = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        _mount(core, "alias", "$MEM.box.msg")

        core.mem["h"] = _python_node(
            "def handle(event):\n    return {}"
        )
        # 真实 owner(0) 可通过别名注册
        lid = core.events.register(0, "memory_read", "$MEM.alias", "$MEM.h", phase="before")
        assert lid == 1
        # 非 owner 通过别名注册被拒
        with pytest.raises(ValueError):
            core.events.register(1, "memory_read", "$MEM.alias", "$MEM.h", phase="before")

        # before 事件 target 是别名
        MemoryReadInstruction("tc", 0, {"ref": "$MEM.alias"}).execute(core, root)
        before = [e for e in core._turn_events if e["phase"] == "before"]
        assert before[-1]["target"] == "$MEM.alias"
        assert before[-1]["source"] == {"kind": "core", "initiator_cid": 0}
        # after 事件 source 是 memory（真实目标为 memory），target 仍是别名
        after = [e for e in core._turn_events if e["phase"] == "after"]
        assert after[-1]["source"] == {"kind": "memory"}
        assert after[-1]["target"] == "$MEM.alias"

    def test_alias_after_payload_ref_is_alias(self):
        core = Core()
        root = _new_root(core)
        core.mem["p"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
        core.mem["s"] = MetaDict(data={"content": "sys"}, ctrl={"type": "settingup"})
        sink = _build_conversation(core, "$MEM.s", "$MEM.p", parent=root, is_sub=False)
        core._dormant_cids.append(sink.cid)
        core.mem.make("$MEM", "box", "dict", owner_cid=0)
        core.mem.set("$MEM.box.msg", "v", owner_cid=0)
        _mount(core, "alias", "$MEM.box.msg")
        core.mem["h"] = _python_node(
            "def handle(event):\n    return {'notify': [{'cid': 1, 'content': json.dumps(event['payload'], ensure_ascii=False)}]}"
        )
        core.events.register(0, "memory_read", "$MEM.alias", "$MEM.h", phase="after")

        MemoryReadInstruction("tc", 0, {"ref": "$MEM.alias"}).execute(core, root)

        payload = json.loads(sink.user_batch.user_contents[0])
        assert payload["ref"] == "$MEM.alias"
        assert "$MEM.box" not in payload["ref"]

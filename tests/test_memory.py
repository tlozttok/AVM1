"""内存系统测试"""

import pytest
from avm.memory import Memory, _wrap_value
from avm.memory_device import StringDevice, MetaListDevice, MetaDictDevice, InputsListDevice, OutputsListDevice, MemoryDevice
from avm.types import MetaDict, MetaList
from avm.exceptions import VMMemoryError, MemoryKeyNotFoundError, MemoryIndexOutOfRangeError, MemoryTypeError


class TestMemoryBasic:
    def test_set_and_get(self):
        mem = Memory()
        mem["k"] = "v"
        assert mem["k"] == "v"

    def test_in_operator(self):
        mem = Memory()
        mem["a"] = 1
        assert "a" in mem
        assert "b" not in mem

    def test_del_item(self):
        mem = Memory()
        mem["a"] = 1
        del mem["a"]
        assert "a" not in mem

    def test_get_default(self):
        mem = Memory()
        assert mem.get("x", "default") == "default"

    def test_setdefault_creates(self):
        mem = Memory()
        mem.setdefault("x", "v")
        assert mem["x"] == "v"

    def test_setdefault_no_overwrite(self):
        mem = Memory()
        mem["x"] = "original"
        mem.setdefault("x", "new")
        assert mem["x"] == "original"

    def test_cannot_delete_device(self):
        mem = Memory()
        mem.mount("dev", StringDevice("val"))
        with pytest.raises(VMMemoryError):
            del mem["dev"]


class TestWrapValue:
    def test_dict_wrapped(self):
        result = _wrap_value({"a": {"b": "c"}})
        assert isinstance(result, MetaDict)
        assert isinstance(result["a"], MetaDict)

    def test_list_wrapped(self):
        result = _wrap_value([1, 2])
        assert isinstance(result, MetaList)

    def test_nested_structures(self):
        result = _wrap_value({"lst": [{"k": "v"}]})
        assert isinstance(result, MetaDict)
        assert isinstance(result["lst"], MetaList)
        assert isinstance(result["lst"][0], MetaDict)

    def test_string_unchanged(self):
        assert _wrap_value("hello") == "hello"


class TestUnwrap:
    def test_mem_root(self):
        mem = Memory()
        mem["key"] = "val"
        result = mem.unwrap(["$", "MEM", "key"])
        assert result == "val"

    def test_mem_nested(self):
        mem = Memory()
        mem["a"] = {"b": "nested"}
        result = mem.unwrap(["$", "MEM", "a", "b"])
        assert result == "nested"

    def test_mem_key_not_found(self):
        mem = Memory()
        with pytest.raises(MemoryKeyNotFoundError):
            mem.unwrap(["$", "MEM", "nonexistent"])

    def test_mem_string_no_subkey(self):
        mem = Memory()
        mem["s"] = "hello"
        with pytest.raises(MemoryTypeError):
            mem.unwrap(["$", "MEM", "s", "child"])

    def test_non_mem_device(self):
        mem = Memory()
        dev = StringDevice("hello")
        mem.mount("io", dev)
        result = mem.unwrap(["$", "io"])
        assert result == "hello"

    def test_non_mem_device_subpath(self):
        mem = Memory()
        dev = InputsListDevice(data=["a", "b", "c"])
        mem.mount("inputs", dev)
        result = mem.unwrap(["$", "inputs", "0"])
        assert result == "a"

    def test_non_mem_unknown_raises(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.unwrap(["$", "unknown_device"])

    def test_mem_with_device_on_path(self):
        mem = Memory()
        mem["cfg"] = MetaDict(data={"key": "val"})
        dev = StringDevice("device_val")
        mem.mount("cfg.dev", dev)
        # 走到 cfg.dev 命中设备，返回设备值
        result = mem.unwrap(["$", "MEM", "cfg", "dev"])
        assert result == "device_val"

    def test_for_llm_false_returns_raw(self):
        mem = Memory()
        mem["d"] = MetaDict(data={"k": "v"})
        result = mem.unwrap(["$", "MEM", "d"], for_llm=False)
        assert isinstance(result, MetaDict)
        assert result["k"] == "v"

    def test_no_prefix_passthrough(self):
        mem = Memory()
        mem["key"] = "val"
        # 无 $ 前缀直接走 resolve_path，非 MEM 路径需要设备
        with pytest.raises(VMMemoryError):
            mem.unwrap(["key"])


class TestDeviceResolvePath:
    def test_inputs_device_resolve_path(self):
        dev = InputsListDevice(data=["x", "y"])
        assert dev.resolve_path(["0"]) == "x"
        assert dev.resolve_path(["1"]) == "y"

    def test_inputs_device_multilevel_raises(self):
        dev = InputsListDevice(data=["x"])
        with pytest.raises(VMMemoryError):
            dev.resolve_path(["0", "child"])

    def test_outputs_device_resolve_path(self):
        dev = OutputsListDevice(data=["a", "b"])
        assert dev.resolve_path(["1"]) == "b"

    def test_outputs_device_append_prints(self, capsys):
        dev = OutputsListDevice()
        dev.append("hello")
        assert dev.resolve_path(["0"]) == "hello"
        assert capsys.readouterr().out == "hello\n"

    def test_outputs_device_write_minus_one_prints(self, capsys):
        dev = OutputsListDevice()
        dev[-1] = "printed"
        assert dev.resolve_path(["0"]) == "printed"
        assert capsys.readouterr().out == "printed\n"

    def test_string_device_subpath_raises(self):
        dev = StringDevice("val")
        with pytest.raises(VMMemoryError):
            dev.resolve_path(["child"])


class TestMemoryDeviceMount:
    def test_mount_and_read(self):
        mem = Memory()
        dev = StringDevice("hello")
        mem.mount("path", dev)
        assert mem.is_device_path(["path"])

    def test_mount_invalid_type_raises(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.mount("path", "not_a_device")

    def test_unmount(self):
        mem = Memory()
        dev = StringDevice("v")
        mem.mount("path", dev)
        mem.unmount("path")
        assert not mem.is_device_path(["path"])

    def test_unmount_nonexistent_raises(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.unmount("nonexistent")

    def test_get_device(self):
        mem = Memory()
        dev = StringDevice("x")
        mem.mount("a", dev)
        assert mem.get_device(["a"]) is dev
        assert mem.get_device(["nonexistent"]) is None


class TestMemorySet:
    def test_set_top_level(self):
        mem = Memory()
        mem.set("$MEM.key", "val")
        assert mem["key"] == "val"

    def test_set_nested(self):
        mem = Memory()
        mem["parent"] = {}
        mem.set("$MEM.parent.child", "deep")
        assert mem["parent"]["child"] == "deep"

    def test_set_creates_intermediate(self):
        mem = Memory()
        mem.set("$MEM.a.b.c", "val")
        assert mem["a"]["b"]["c"] == "val"

    def test_set_without_dollar_raises(self):
        mem = Memory()
        with pytest.raises(ValueError):
            mem.set("no_dollar", "val")

    def test_set_by_path_device_root(self):
        mem = Memory()
        dev = StringDevice("old")
        mem.mount("dev", dev)
        mem.set_by_path(["dev"], "new")
        assert dev.get_value() == "new"

    def test_set_device_nested_through_data(self):
        mem = Memory()
        mem["cfg"] = MetaDict(data={"inner": "old"})
        dev = StringDevice("device_val")
        mem.mount("cfg.inner", dev)
        mem.set_by_path(["cfg", "inner"], "overwritten")
        assert dev.get_value() == "overwritten"


class TestMemoryMake:
    def test_make_string_on_dict(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "str")
        assert mem["base"]["child"] == ""

    def test_make_dict_on_dict(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "dict")
        assert mem["base"]["child"] == {}

    def test_make_list_on_dict(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "list")
        assert mem["base"]["child"] == []

    def test_make_on_list(self):
        mem = Memory()
        mem["base"] = [None]
        mem.make("$MEM.base", "0", "str")
        assert mem["base"][0] == ""

    def test_make_on_string_raises(self):
        mem = Memory()
        mem["base"] = "hello"
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "child", "dict")

    def test_make_on_nonexistent_raises(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.nonex", "child", "dict")

    def test_make_list_bad_index_raises(self):
        mem = Memory()
        mem["base"] = [None]
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "abc", "str")

    def test_make_list_out_of_range_raises(self):
        mem = Memory()
        mem["base"] = [None]
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "5", "str")


class TestMemorySaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = Memory()
        mem["a"] = "hello"
        mem["nested"] = MetaDict(data={"k": MetaList(data=["x", "y", "z"])})
        mem.save(str(path))

        loaded = Memory.load(str(path))
        assert loaded["a"] == "hello"
        assert loaded["nested"]["k"] == ["x", "y", "z"]

    def test_load_nonexistent_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        mem = Memory.load(str(path))
        # empty memory is fine
        assert list(mem._data.keys()) == []

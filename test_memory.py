"""memory 模块单元测试"""

import pytest
from memory import Memory
from memory_device import StringDevice, MetaListDevice, MetaDictDevice
from exceptions import VMMemoryError
from avm_types import MetaDict, MetaList


class TestMemoryBasic:
    def test_set_get(self):
        mem = Memory()
        mem["key"] = "value"
        assert mem["key"] == "value"

    def test_contains(self):
        mem = Memory()
        mem["a"] = 1
        assert "a" in mem
        assert "b" not in mem

    def test_delitem(self):
        mem = Memory()
        mem["a"] = 1
        del mem["a"]
        assert "a" not in mem

    def test_get_default(self):
        mem = Memory()
        assert mem.get("x", "default") == "default"

    def test_setdefault(self):
        mem = Memory()
        mem.setdefault("x", "v")
        assert mem["x"] == "v"


class TestMemoryNested:
    def test_nested_set_get(self):
        mem = Memory()
        mem["a"] = {}
        mem.set("$MEM.a.b", "nested")
        assert mem["a"]["b"] == "nested"

    def test_set_creates_intermediate_dicts(self):
        mem = Memory()
        mem.set("$MEM.a.b.c", "deep")
        assert mem["a"]["b"]["c"] == "deep"


class TestMemoryUnwrap:
    def test_dollar_unwrap(self):
        mem = Memory()
        mem["a"] = "hello"
        assert mem.unwrap(["$", "MEM", "a"], for_llm=True) == "hello"

    def test_dollar_unwrap_without_mem_prefix(self):
        mem = Memory()
        mem["a"] = "hello"
        assert mem.unwrap(["$", "a"], for_llm=True) == "hello"

    def test_recursive_dereference(self):
        mem = Memory()
        mem["a"] = "$MEM.b"
        mem["b"] = "final"
        assert mem.unwrap(["$", "MEM", "a"], for_llm=True) == "final"

    def test_ampersand_one_level(self):
        mem = Memory()
        mem["a"] = "$MEM.b"
        mem["b"] = "final"
        assert mem.unwrap(["&", "MEM", "a"], for_llm=True) == "$MEM.b"

    def test_unwrap_no_prefix(self):
        mem = Memory()
        mem["a"] = "val"
        assert mem.unwrap(["a"], for_llm=True) == "val"

    def test_unwrap_metadict(self):
        mem = Memory()
        mem["d"] = MetaDict(data={"k": "v"})
        result = mem.unwrap(["$", "d"], for_llm=True)
        assert result == "dict[keys=['k']]"

    def test_unwrap_metalist(self):
        mem = Memory()
        mem["l"] = MetaList(data=[1, 2, 3])
        result = mem.unwrap(["$", "l"], for_llm=True)
        assert result == "list[len=3]"


class TestMemoryDevice:
    def test_mount_read(self):
        mem = Memory()
        dev = StringDevice("val")
        mem.mount("io.test", dev)
        assert mem.unwrap(["$", "MEM", "io", "test"], for_llm=True) == "val"

    def test_mount_write_string_device(self):
        mem = Memory()
        dev = StringDevice("old")
        mem.mount("io.test", dev)
        mem.set("$MEM.io.test", "new")
        assert dev.get_value() == "new"

    def test_mount_invalid_device(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.mount("path", "not_a_device")

    def test_unmount(self):
        mem = Memory()
        dev = StringDevice("v")
        mem.mount("path", dev)
        mem.unmount("path")
        assert not mem.is_device_path(["path"])

    def test_unmount_nonexistent(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.unmount("nonexistent")

    def test_is_device_path(self):
        mem = Memory()
        mem.mount("a.b", StringDevice("x"))
        assert mem.is_device_path(["a", "b"])
        assert not mem.is_device_path(["a"])

    def test_get_device(self):
        mem = Memory()
        dev = StringDevice("x")
        mem.mount("a", dev)
        assert mem.get_device(["a"]) is dev


class TestMemoryMake:
    def test_make_dict(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "dict")
        assert mem["base"]["child"] == {}

    def test_make_list(self):
        mem = Memory()
        mem["base"] = [None]
        mem.make("$MEM.base", "0", "str")
        assert mem["base"][0] == ""

    def test_make_str(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "str")
        assert mem["base"]["child"] == ""

    def test_make_on_string_raises(self):
        mem = Memory()
        mem["base"] = "hello"
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "child", "dict")

    def test_make_on_nonexistent_raises(self):
        mem = Memory()
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.nonexistent", "child", "dict")

    def test_make_list_bad_index(self):
        mem = Memory()
        mem["base"] = [None]
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "abc", "str")

    def test_make_list_out_of_range(self):
        mem = Memory()
        mem["base"] = [None]
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "5", "str")

    def test_make_invalid_type(self):
        mem = Memory()
        mem["base"] = {}
        with pytest.raises(VMMemoryError):
            mem.make("$MEM.base", "child", "invalid_type")


class TestMemorySet:
    def test_set_invalid_ref(self):
        mem = Memory()
        with pytest.raises(ValueError):
            mem.set("no_dollar", "val")

    def test_set_by_path_device(self):
        mem = Memory()
        dev = StringDevice("old")
        mem.mount("io", dev)
        mem.set_by_path(["io"], "new")
        assert dev.get_value() == "new"

    def test_set_by_path_device_not_string(self):
        mem = Memory()
        dev = MetaListDevice()
        mem.mount("io", dev)
        with pytest.raises(VMMemoryError):
            mem.set_by_path(["io"], "val")

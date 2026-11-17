"""memory_nodes：节点规则可以直接对着裸 dict/list 调，不用建一棵树。

这批函数从 Memory 搬出来才这么可测——在类里时它们是 @staticmethod，但名字藏在
内存系统后面，测一条"越界报什么"得先造出整棵内存树和设备表。现在参数就是那个
节点本身。
"""

import pytest

from avm.exceptions import (
    IllegalOperationError,
    MemoryIndexOutOfRangeError,
    MemoryKeyNotFoundError,
    MemoryTypeError,
)
from avm.memory_nodes import (
    assign,
    check_access,
    check_append_same_type,
    check_new_key,
    child,
    index_key,
    index_of,
    list_elem_type,
    new_node,
    show_path,
)
from avm.types import LinkNode, MetaDict, MetaList, StrNode

SHOWN = ["MEM", "box"]


def test_show_path_joins_segments():
    assert show_path(SHOWN) == "$MEM.box"
    assert show_path([]) == "$"


def test_child_reads_dict_and_reports_presence():
    node = {"a": 1}
    assert child(node, "a", SHOWN + ["a"], 1) == (1, True)
    # 键不存在不算错：存在性由第二项告诉调用方（写入路径要靠它决定建不建）
    assert child(node, "b", SHOWN + ["b"], 1) == (None, False)


def test_child_reads_list_with_string_index():
    """引用里的下标永远是一段字符串，取值前必须转成整数。"""
    assert child([10, 20], "1", SHOWN + ["1"], 1) == (20, True)


def test_child_rejects_out_of_range_and_negative():
    with pytest.raises(MemoryIndexOutOfRangeError):
        child([10], "5", SHOWN + ["5"], 1)
    with pytest.raises(MemoryIndexOutOfRangeError):
        child([10], "-1", SHOWN + ["-1"], 1)


def test_child_rejects_non_container():
    with pytest.raises(MemoryTypeError):
        child("叶子", "a", SHOWN + ["a"], 1)


def test_index_of_converts_only_for_lists():
    assert index_of([1, 2], "1") == 1
    assert index_of({"k": 1}, "k") == "k"     # 字典原样返回键
    with pytest.raises(MemoryTypeError):
        index_of([1, 2], "x")


def test_index_key_converts_only_for_list_nodes():
    assert index_key([1, 2], "0") == 0
    assert index_key({"0": 1}, "0") == "0"


def test_assign_replaces_string_leaf_but_keeps_ctrl():
    """写字符串叶子只改内容，不动 ctrl（写入不夺权）。"""
    node = {"a": StrNode("old", ctrl={"type": "para"})}
    assign(node, "a", "new", SHOWN + ["a"], 1)
    assert node["a"] == "new"
    assert node["a"].get_ctrl() == {"type": "para"}


def test_assign_on_container_writes_metadata_not_data():
    """对已存在的容器赋值 = 改它的描述数据，子树本身不动。"""
    node = {"a": MetaDict(data={"inner": "v"})}
    assign(node, "a", {"desc": "说明"}, SHOWN + ["a"], 1)
    assert node["a"]["inner"] == "v"
    assert node["a"].get_metadata() == {"desc": "说明"}


def test_assign_writes_list_slot_and_rejects_out_of_range():
    node = [StrNode("x")]
    assign(node, "0", "y", SHOWN + ["0"], 1)
    assert node[0] == "y"
    with pytest.raises(MemoryIndexOutOfRangeError):
        assign(node, "3", "z", SHOWN + ["3"], 1)


def test_assign_rejects_non_container():
    with pytest.raises(MemoryTypeError):
        assign("叶子", "a", 1, SHOWN, 1)


def test_check_access_reports_missing_key_with_address():
    with pytest.raises(MemoryKeyNotFoundError) as e:
        check_access({"a": 1}, "b", SHOWN + ["b"])
    assert "$MEM.box.b" in str(e.value)


def test_check_access_blocks_minus_one_on_dict():
    with pytest.raises(IllegalOperationError):
        check_access({"a": 1}, "-1", SHOWN)


def test_check_access_on_list_index_bounds():
    check_access([1], "0", SHOWN)          # 合法下标不抛
    with pytest.raises(MemoryIndexOutOfRangeError):
        check_access([1], "2", SHOWN)
    with pytest.raises(IllegalOperationError):
        check_access([1], "x", SHOWN)


def test_check_access_on_string_leaf():
    with pytest.raises(MemoryTypeError):
        check_access("叶子", "a", SHOWN)


def test_new_node_makes_the_three_types():
    assert isinstance(new_node("dict"), MetaDict)
    assert isinstance(new_node("list"), MetaList)
    assert isinstance(new_node("str"), StrNode)
    # 新节点不带 ctrl：寄存器类节点由内核自己设，建节点这一步不猜
    assert new_node("str").get_ctrl() is None


def test_list_elem_type_defaults_to_str_and_rejects_mixed():
    assert list_elem_type([]) == "str"
    assert list_elem_type([MetaDict(), MetaDict()]) == "dict"
    assert list_elem_type([LinkNode("dev")]) == "link"
    with pytest.raises(Exception):
        list_elem_type([MetaDict(), MetaList()])


def test_check_append_same_type_only_checks_non_empty():
    check_append_same_type([], "dict")                    # 空列表随便追加
    check_append_same_type([MetaDict()], "dict")          # 同类型放行
    with pytest.raises(Exception):
        check_append_same_type([MetaDict()], "list")


def test_check_new_key_rejects_dot_and_empty():
    check_new_key("ok")
    with pytest.raises(IllegalOperationError):
        check_new_key("a.b")     # 点号是段分隔符，建得出来但永远读不回来
    with pytest.raises(IllegalOperationError):
        check_new_key("")

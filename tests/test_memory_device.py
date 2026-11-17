"""MetaListDevice 的预览串：形状说明，不是内容。

这个方法是给面板/信息设备看的"这个列表长什么样"——长度、元素类型、元数据。
它曾经对任何非空列表都抛 NameError（`MetaDict` / `MetaList` 在本模块里没导入，
而这两者的导入不能放模块顶部：types -> capabilities -> modules -> memory_device
会绕回来），被 console 的 try 吞成预览里的一行错误。这几条守着它别再退回去。
"""

from avm.memory_device import MetaListDevice
from avm.types import MetaDict, MetaList


def test_empty_list_previews_as_str_element():
    """空列表默认 str——与 write 追加的默认类型一致。"""
    assert MetaListDevice().to_llm_string() == "list[len=0,elem=str,metadata=None]"


def test_homogeneous_str_list():
    assert MetaListDevice(data=["a", "b"]).to_llm_string() == "list[len=2,elem=str,metadata=None]"


def test_dict_elements():
    """这一条就是原来抛 NameError 的那条路径。"""
    dev = MetaListDevice(data=[MetaDict(data={"a": 1})])
    assert "elem=dict" in dev.to_llm_string()


def test_list_elements():
    dev = MetaListDevice(data=[MetaList(data=[])])
    assert "elem=list" in dev.to_llm_string()


def test_mixed_list_says_mixed_instead_of_guessing():
    dev = MetaListDevice(data=[MetaDict(data={}), MetaList(data=[])])
    assert "elem=mixed(dict,list)" in dev.to_llm_string()


def test_metadata_is_shown():
    dev = MetaListDevice(data=[], metadata={"type": "inputs"})
    assert "metadata={'type': 'inputs'}" in dev.to_llm_string()

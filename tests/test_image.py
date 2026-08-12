"""系统镜像加载器测试：显式节点格式、设备插件、para、init、持久化"""

import json

import pytest

from avm.image import load_image, ImageError
from avm.memory import Memory
from avm.types import MetaDict, MetaList

DEVICE_CODE = '''
from avm.memory_device import MemoryDevice

class DummyDevice(MemoryDevice):
    def __init__(self, label=""):
        self.label = label

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return f"dummy:{self.label}"
'''


class MockLMU:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_index = 0
        self.calls = []

    def exec(self, conversation, para):
        self.calls.append(para)
        if self.call_index >= len(self.responses):
            raise RuntimeError(f"MockLMU 第 {self.call_index} 次调用没有预设响应")
        resp = self.responses[self.call_index]
        self.call_index += 1
        return resp


def _write(tmp_path, image, devices_code=None):
    if devices_code is not None:
        d = tmp_path / "devices"
        d.mkdir(exist_ok=True)
        (d / "dummy.py").write_text(devices_code, encoding="utf-8")
    path = tmp_path / "image.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _sample_image():
    return {
        "meta": {"name": "demo", "version": 1},
        "mem": {
            "system": {
                "kind": "dict",
                "meta": "init 的 LLM 提示词程序",
                "ctrl": {"type": "settingup"},
                "value": {
                    "name": {"kind": "str", "value": "init"},
                    "content": {"kind": "str", "value": "sys"},
                },
            },
            "user": {"kind": "str", "value": "hello"},
            "model_params": {
                "kind": "dict",
                "meta": "模型参数",
                "ctrl": {"type": "para"},
                "value": {
                    "model": {"kind": "str", "value": "test-model"},
                    "temperature": {"kind": "str", "value": "0.5"},
                },
            },
            "inputs": {"kind": "device"},
            "nested": {
                "kind": "dict",
                "value": {"k": {"kind": "list", "value": [{"kind": "str", "value": "a"}]}},
            },
        },
        "para_ref": "$MEM.model_params",
        "devices": [
            {"path": "inputs", "file": "devices/dummy.py", "class": "DummyDevice", "args": {"label": "io"}},
        ],
        "init": {"name": "init", "system_ref": "$MEM.system", "user_ref": "$MEM.user"},
    }


class TestLoadImage:
    def test_load_mem_para_init(self, tmp_path):
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        core = load_image(path)

        system = core.mem["system"]
        assert isinstance(system, MetaDict)
        assert system.get_ctrl() == {"type": "settingup"}
        assert system["content"] == "sys"
        assert core.mem["nested"]["k"] == ["a"]
        para = core.mem["model_params"]
        assert isinstance(para, MetaDict)
        assert para.get_ctrl() == {"type": "para"}
        assert para.get_metadata() == "模型参数"
        assert core.para_ref == "$MEM.model_params"

        conv = core.get_conversation(0)
        assert conv.name == "init"
        assert conv.is_root
        assert conv.cid == 0
        assert conv.messages[0].content == "sys"
        assert conv.messages[1].content == "hello"
        assert 0 in core._ready_cids

        assert core.unwrap("$MEM.inputs") == "dummy:io"

    def test_init_literal_system_user(self, tmp_path):
        image = _sample_image()
        image["init"] = {"name": "boot", "system": "S", "user": "U"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(path)
        conv = core.get_conversation(0)
        assert conv.name == "boot"
        assert conv.messages[0].content == "S"
        assert conv.messages[1].content == "U"

    def test_run_uses_para_ref_and_mock_lmu(self, tmp_path):
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        core = load_image(path)
        lmu = MockLMU([("done", [], None)])
        core.lmu = lmu
        core.run()
        assert core._active_cid is None
        assert lmu.calls[0].get("model") == "test-model"

    def test_device_class_must_inherit_memorydevice(self, tmp_path):
        image = _sample_image()
        image["devices"][0]["class"] = "NotADevice"
        path = _write(tmp_path, image, devices_code=DEVICE_CODE + "\n\nclass NotADevice:\n    pass\n")
        with pytest.raises(ImageError, match="继承 MemoryDevice"):
            load_image(path)

    def test_para_must_be_typed(self, tmp_path):
        image = _sample_image()
        del image["mem"]["model_params"]["ctrl"]
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="ctrl.type='para'"):
            load_image(path)

    def test_unknown_kind_rejected(self, tmp_path):
        image = _sample_image()
        image["mem"]["bad"] = {"kind": "file", "value": "x"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="未知 kind"):
            load_image(path)

    def test_device_marker_requires_declaration(self, tmp_path):
        image = _sample_image()
        image["devices"] = []
        path = _write(tmp_path, image)
        with pytest.raises(ImageError, match="未在 devices 段声明"):
            load_image(path)

    def test_device_marker_nested_allowed(self, tmp_path):
        image = _sample_image()
        image["mem"]["game"] = {
            "kind": "dict",
            "value": {
                "map": {
                    "kind": "dict",
                    "value": {"rooms": {"kind": "device"}},
                }
            },
        }
        image["devices"].append({
            "path": "game.map.rooms", "file": "devices/dummy.py", "class": "DummyDevice", "args": {"label": "rooms"},
        })
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(path)
        assert core.unwrap("$MEM.game.map.rooms") == "dummy:rooms"
        # 设备不写入数据树
        assert "rooms" not in core.mem["game"]["map"]

    def test_str_node_must_be_string(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "str", "value": 123}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="value 必须是字符串"):
            load_image(path)

    def test_init_ref_must_be_ref(self, tmp_path):
        image = _sample_image()
        image["init"] = {"system_ref": "not_a_ref", "user_ref": "$MEM.user"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="开头的内存引用"):
            load_image(path)

    def test_init_system_ref_requires_settingup(self, tmp_path):
        """init 的 system_ref 指向 str 节点：与 create_cmd 一致，要求 settingup 程序节点"""
        image = _sample_image()
        image["mem"]["system"] = {"kind": "str", "value": "sys"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="settingup"):
            load_image(path)

    def test_init_system_ref_rejects_untyped_dict(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "dict", "value": {"content": {"kind": "str", "value": "sys"}}}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="settingup"):
            load_image(path)

    def test_init_settingup_requires_content(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "dict", "ctrl": {"type": "settingup"}, "value": {}}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="content"):
            load_image(path)

    def test_init_user_ref_must_be_str_node(self, tmp_path):
        image = _sample_image()
        image["mem"]["user"] = {
            "kind": "dict",
            "ctrl": {"type": "settingup"},
            "value": {"content": {"kind": "str", "value": "hello"}},
        }
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="str 节点"):
            load_image(path)

    def test_device_file_missing(self, tmp_path):
        image = _sample_image()
        image["devices"] = [{"path": "inputs", "file": "devices/nope.py", "class": "X", "args": {}}]
        path = _write(tmp_path, image)
        with pytest.raises(ImageError, match="设备文件不存在"):
            load_image(path)


class TestParaCoercion:
    def test_filter_api_params_coerces_numeric_strings(self):
        from avm.core import LMU

        lmu = LMU()
        out = lmu._filter_api_params({
            "model": "gpt-4o-mini",
            "temperature": "0.5",
            "max_tokens": "128",
            "n": "2",
            "stop": "END",
        })
        assert out["temperature"] == 0.5
        assert out["max_tokens"] == 128
        assert out["n"] == 2
        assert out["stop"] == "END"
        assert "model" not in out  # model 由 para.get("model") 单独读取


class TestPersistRoundtrip:
    def test_save_load_preserves_meta_ctrl(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = Memory()
        mem["a"] = "hello"
        mem["cfg"] = MetaDict(data={"k": "v"}, metadata="描述", ctrl={"type": "para"})
        mem["tags"] = MetaList(data=["x", "y"], metadata="标签")
        mem.save(str(path))

        loaded = Memory.load(str(path))
        assert loaded["a"] == "hello"
        assert loaded["cfg"]["k"] == "v"
        assert loaded["cfg"].get_metadata() == "描述"
        assert loaded["cfg"].get_ctrl() == {"type": "para"}
        assert loaded["tags"].get_metadata() == "标签"

    def test_load_legacy_format(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps({
            "__type": "MetaDict",
            "meta": None,
            "data": {"a": {"__type": "MetaList", "meta": "m", "data": ["x", "y"]}},
        }), encoding="utf-8")
        loaded = Memory.load(str(path))
        assert loaded["a"] == ["x", "y"]
        assert loaded["a"].get_metadata() == "m"

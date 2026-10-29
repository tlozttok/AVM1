"""系统镜像加载器测试：显式节点格式、设备插件、para、init、持久化"""

import json

import pytest

from avm.core import Core
from avm.exceptions import VMMemoryError
from avm.image import load_image, ImageError
from avm.memory import Memory
from avm.types import MetaDict, MetaList, LinkNode

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


FLAG_DEVICE_CODE = '''
import json, os
from avm.memory_device import MemoryDevice

class FlagDevice(MemoryDevice):
    """把 attach_core 那一刻读到的启动标志写进 devices/seen.json。"""

    def attach_core(self, core):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seen.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(core.host_flags), f)

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return "flag-device"
'''


def _flag_image():
    image = _sample_image()
    image["devices"] = [
        {"path": "inputs", "file": "devices/dummy.py", "class": "FlagDevice"},
    ]
    return image


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


COMMAND_PROGRAM = '''
def handle(cmd):
    return 'echo: ' + cmd
'''


def _plm_init_image():
    """init 指向 python 程序节点：0 号是 plm.command 对话，para 写在程序节点的 para_ref 里。"""
    image = _sample_image()
    image["mem"]["speaker"] = {
        "kind": "dict",
        "meta": "PLM 入口程序",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": COMMAND_PROGRAM},
            "para_ref": {"kind": "str", "value": "$MEM.plm_params"},
        },
    }
    image["mem"]["plm_params"] = {
        "kind": "dict",
        "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "plm.command"}},
    }
    image["init"] = {"name": "speaker", "system_ref": "$MEM.speaker", "user_ref": "$MEM.user"}
    return image


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


class TestHostFlags:
    """启动标志：宿主事实，在挂载设备之前写入 Core，核心启动后不再变化。"""

    def _seen(self, tmp_path):
        return json.loads((tmp_path / "devices" / "seen.json").read_text(encoding="utf-8"))

    def test_flags_are_visible_to_devices_at_attach(self, tmp_path):
        path = _write(tmp_path, _flag_image(), devices_code=FLAG_DEVICE_CODE)
        core = load_image(Core(), path, host_flags={"console": True})
        assert core.host_flags == {"console": True}
        assert self._seen(tmp_path) == {"console": True}, "attach_core 时标志已经就位"

    def test_default_is_empty(self, tmp_path):
        path = _write(tmp_path, _flag_image(), devices_code=FLAG_DEVICE_CODE)
        core = load_image(Core(), path)
        assert core.host_flags == {}
        assert self._seen(tmp_path) == {}, "不传标志时设备看到的与改动前一致"

    def test_flags_are_copied(self, tmp_path):
        path = _write(tmp_path, _flag_image(), devices_code=FLAG_DEVICE_CODE)
        given = {"console": True}
        core = load_image(Core(), path, host_flags=given)
        given["console"] = False
        assert core.host_flags == {"console": True}, "调用方的字典进不来"

    def test_flags_must_be_an_object(self, tmp_path):
        path = _write(tmp_path, _flag_image(), devices_code=FLAG_DEVICE_CODE)
        with pytest.raises(ImageError, match="host_flags 必须是对象"):
            load_image(Core(), path, host_flags=["console"])

    def test_flag_keys_must_be_non_empty_strings(self, tmp_path):
        path = _write(tmp_path, _flag_image(), devices_code=FLAG_DEVICE_CODE)
        with pytest.raises(ImageError, match="host_flags 的键"):
            load_image(Core(), path, host_flags={"": True})


class TestLoadImage:
    def test_load_mem_para_init(self, tmp_path):
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        core = load_image(Core(), path)

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
        core = load_image(Core(), path)
        conv = core.get_conversation(0)
        assert conv.name == "boot"
        assert conv.messages[0].content == "S"
        assert conv.messages[1].content == "U"

    # ---- init 指向 python 程序节点：0 号对话是 PLM（Python 对话） ----

    def test_init_python_program_runs_as_plm(self, tmp_path):
        """init 可以是 PLM：0 号建 Python 对话，程序是 system、user_ref 是首条用户消息。"""
        image = _plm_init_image()
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(Core(), path)

        conv = core.get_conversation(0)
        assert conv.is_python is True
        assert conv.name == "speaker"
        assert conv.para_ref == "$MEM.plm_params"
        assert conv.is_root is True
        assert conv.messages[0].content == COMMAND_PROGRAM
        assert conv.messages[1].content == "hello"
        assert core._ready_cids == [0]

        # 跑到底：镜像里没有 LLM 对话，这一跑不碰 API
        core.run()
        assert conv.messages[-1].content == "echo: hello"
        assert 0 in core._dormant_cids

    def test_init_python_requires_para_ref(self, tmp_path):
        """python 入口没有 para 引用：早报错，不拿顶层 LLM para 去撞命名空间校验。"""
        image = _plm_init_image()
        image["mem"]["speaker"]["value"].pop("para_ref")
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="para 引用"):
            load_image(Core(), path)

    def test_init_para_ref_overrides_program_node(self, tmp_path):
        image = _plm_init_image()
        image["mem"]["other_params"] = {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "plm.command"}},
        }
        image["init"]["para_ref"] = "$MEM.other_params"
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(Core(), path)
        assert core.get_conversation(0).para_ref == "$MEM.other_params"

    def test_init_python_model_must_be_plm(self, tmp_path):
        """python 程序配 LLM 模型名：模型命名空间交叉校验拦住。"""
        image = _plm_init_image()
        image["mem"]["plm_params"]["value"]["model"] = {"kind": "str", "value": "test-model"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(VMMemoryError, match="PLM"):
            load_image(Core(), path)

    def test_init_settingup_para_ref_field(self, tmp_path):
        """LLM 入口也能用 init.para_ref 指定自己的 para。"""
        image = _sample_image()
        image["mem"]["system"]["value"]["para_ref"] = {"kind": "str", "value": "$MEM.model_params"}
        image["mem"]["other_params"] = {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        }
        image["init"] = {
            "system_ref": "$MEM.system",
            "user_ref": "$MEM.user",
            "para_ref": "$MEM.other_params",
        }
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(Core(), path)
        assert core.get_conversation(0).para_ref == "$MEM.other_params"
        assert core.get_conversation(0).is_python is False

    def test_run_uses_para_ref_and_mock_lmu(self, tmp_path):
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        core = load_image(Core(), path)
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
            load_image(Core(), path)

    def test_para_must_be_typed(self, tmp_path):
        image = _sample_image()
        del image["mem"]["model_params"]["ctrl"]
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="ctrl.type='para'"):
            load_image(Core(), path)

    def test_unknown_kind_rejected(self, tmp_path):
        image = _sample_image()
        image["mem"]["bad"] = {"kind": "file", "value": "x"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="未知 kind"):
            load_image(Core(), path)

    def test_device_marker_requires_declaration(self, tmp_path):
        image = _sample_image()
        image["devices"] = []
        path = _write(tmp_path, image)
        with pytest.raises(ImageError, match="未在 devices 段声明"):
            load_image(Core(), path)

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
        core = load_image(Core(), path)
        assert core.unwrap("$MEM.game.map.rooms") == "dummy:rooms"
        # 挂载点物化为链接节点：数据树里看得见挂载点，读它是进设备空间
        link = core.mem["game"]["map"]["rooms"]
        assert isinstance(link, LinkNode)
        assert link.target == "game.map.rooms"

    def test_str_node_must_be_string(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "str", "value": 123}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="value 必须是字符串"):
            load_image(Core(), path)

    def test_link_node_declared_in_mem(self, tmp_path):
        """mem 段可以声明链接节点：target 为 $MEM 地址时按内存树地址解析。"""
        image = _sample_image()
        image["mem"]["alias"] = {"kind": "link", "target": "$MEM.system", "meta": "别名"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(Core(), path)

        link = core.mem._data["alias"]
        assert isinstance(link, LinkNode)
        assert link.get_metadata() == "别名"
        assert core.unwrap("$MEM.alias") == core.unwrap("$MEM.system")

    def test_link_node_requires_target(self, tmp_path):
        image = _sample_image()
        image["mem"]["alias"] = {"kind": "link", "value": None}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="target"):
            load_image(Core(), path)

    def test_device_id_override(self, tmp_path):
        """devices 条目的 id 覆盖缺省标识符（缺省取挂载地址）。"""
        image = _sample_image()
        image["devices"].append({
            "path": "box", "file": "devices/dummy.py", "class": "DummyDevice",
            "args": {"label": "box"}, "id": "box-1",
        })
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        core = load_image(Core(), path)

        assert core.mem._data["box"].target == "box-1"
        assert core.mem._device_ids["box-1"] is core.mem._devices["box"]

    def test_init_ref_must_be_ref(self, tmp_path):
        image = _sample_image()
        image["init"] = {"system_ref": "not_a_ref", "user_ref": "$MEM.user"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="开头的内存引用"):
            load_image(Core(), path)

    def test_init_system_ref_requires_settingup(self, tmp_path):
        """init 的 system_ref 指向 str 节点：与 create_cmd 一致，要求 settingup 程序节点"""
        image = _sample_image()
        image["mem"]["system"] = {"kind": "str", "value": "sys"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="settingup"):
            load_image(Core(), path)

    def test_init_system_ref_rejects_untyped_dict(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "dict", "value": {"content": {"kind": "str", "value": "sys"}}}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="settingup"):
            load_image(Core(), path)

    def test_init_settingup_requires_content(self, tmp_path):
        image = _sample_image()
        image["mem"]["system"] = {"kind": "dict", "ctrl": {"type": "settingup"}, "value": {}}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="content"):
            load_image(Core(), path)

    def test_init_user_ref_must_be_str_node(self, tmp_path):
        image = _sample_image()
        image["mem"]["user"] = {
            "kind": "dict",
            "ctrl": {"type": "settingup"},
            "value": {"content": {"kind": "str", "value": "hello"}},
        }
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="str 节点"):
            load_image(Core(), path)

    def test_checkpoint_in_skips_init_and_restores_ready(self, tmp_path):
        """启动时加载检查点：跳过 init，恢复的对话为 cid 0 且就绪（直接续跑）"""
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        core0 = load_image(Core(), path)
        conv0 = core0.get_conversation(0)
        conv0.append_assistant_message("done")
        cp = tmp_path / "cp.json"
        core0.save_conversation(conv0.cid, str(cp))

        core = load_image(Core(), path, checkpoint_in=str(cp))

        assert len(core._conv_by_cid) == 1  # init 被跳过，只有恢复的对话
        restored = core.get_conversation(0)
        assert restored.name == "init"
        assert restored.cid in core._ready_cids
        assert restored.to_api_messages()[-1]["content"] == "done"

    def test_checkpoint_in_missing_file(self, tmp_path):
        path = _write(tmp_path, _sample_image(), devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="检查点"):
            load_image(Core(), path, checkpoint_in=str(tmp_path / "nope.json"))

    def test_device_file_missing(self, tmp_path):
        image = _sample_image()
        image["devices"] = [{"path": "inputs", "file": "devices/nope.py", "class": "X", "args": {}}]
        path = _write(tmp_path, image)
        with pytest.raises(ImageError, match="设备文件不存在"):
            load_image(Core(), path)


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


PLM_CODE = '''
from avm.python_server import PLMServer


class EchoPLM(PLMServer):
    """测试用 PLM：不做推理，只回一句固定内容。"""

    def handle_messages(self, raw_messages, params):
        self._prev_input = raw_messages
        return {"content": "echo from plm.demo"}

    def _execute_system_message(self, content):
        return content

    def _execute_user_message(self, content):
        return content

    def _execute_tool_message(self, content, call_id):
        return content
'''

NOT_A_PLM_CODE = '''
class NotAPlm:
    """不是 PLMServer 的子类。"""
'''


def _plm_declaration_image(plms, model="plm.demo"):
    """在 _plm_init_image 基础上加 plms 段，并把 0 号对话的 model 指过去。"""
    image = _plm_init_image()
    image["mem"]["plm_params"]["value"]["model"] = {"kind": "str", "value": model}
    image["plms"] = plms
    return image


def _write_plm(tmp_path, image, code=PLM_CODE):
    (tmp_path / "plm_impl.py").write_text(code, encoding="utf-8")
    return _write(tmp_path, image, devices_code=DEVICE_CODE)


def _catalog_plm_names(core):
    return [entry["name"] for entry in core.mem._data["model_catalog"]["plm"]]


class TestImagePlms:
    """镜像的 plms 段：自定义 PLM 由镜像声明，装载进本核的 Core.plms。"""

    def test_declaration_lands_in_core_and_catalog(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.demo", "file": "plm_impl.py", "class": "EchoPLM"},
        ])
        core = load_image(Core(), _write_plm(tmp_path, image))
        assert core.plms["plm.demo"].__name__ == "EchoPLM"
        assert core.plms["plm.demo"].__module__.startswith("_avm_image_plm_")
        # 模型目录是 PLM 注册表的可见面：声明的 PLM 必须出现在这里，
        # 否则 create_sub 的命名空间校验会拒绝 para 里的 model。
        assert "plm.demo" in _catalog_plm_names(core)
        # 内建表不被镜像改写
        from avm.python_server import PLM_REGISTRY
        assert "plm.demo" not in PLM_REGISTRY

    def test_declared_plm_runs_as_conversation(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.demo", "file": "plm_impl.py", "class": "EchoPLM"},
        ])
        core = load_image(Core(), _write_plm(tmp_path, image))
        conv = core.get_conversation(0)
        core.run()
        assert conv.messages[-1].content == "echo from plm.demo"

    def test_absent_section_changes_nothing(self, tmp_path):
        image = _plm_init_image()
        core = load_image(Core(), _write(tmp_path, image, devices_code=DEVICE_CODE))
        assert core.plms == {}
        assert "plm.demo" not in _catalog_plm_names(core)

    def test_file_must_exist(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.demo", "file": "nope.py", "class": "EchoPLM"},
        ])
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="文件不存在"):
            load_image(Core(), path)

    def test_class_must_be_a_plm_server(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.demo", "file": "plm_impl.py", "class": "NotAPlm"},
        ])
        core = load_image  # 只是为了避免未使用告警
        with pytest.raises(ImageError, match="继承 PLMServer"):
            load_image(Core(), _write_plm(tmp_path, image, code=NOT_A_PLM_CODE))

    def test_model_must_not_shadow_a_builtin(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.python", "file": "plm_impl.py", "class": "EchoPLM"},
        ], model="plm.python")
        with pytest.raises(ImageError, match="与内建 PLM 重名"):
            load_image(Core(), _write_plm(tmp_path, image))

    def test_duplicate_model_is_rejected(self, tmp_path):
        image = _plm_declaration_image([
            {"model": "plm.demo", "file": "plm_impl.py", "class": "EchoPLM"},
            {"model": "plm.demo", "file": "plm_impl.py", "class": "EchoPLM"},
        ])
        with pytest.raises(ImageError, match="重复声明"):
            load_image(Core(), _write_plm(tmp_path, image))

    def test_missing_field_is_rejected(self, tmp_path):
        image = _plm_declaration_image([{"model": "plm.demo"}])
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="非空 file"):
            load_image(Core(), path)

    def test_section_must_be_an_array(self, tmp_path):
        image = _plm_init_image()
        image["plms"] = {"model": "plm.demo"}
        path = _write(tmp_path, image, devices_code=DEVICE_CODE)
        with pytest.raises(ImageError, match="plms 段必须是数组"):
            load_image(Core(), path)

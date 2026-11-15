"""端到端：同一 plm.python、不同 caps.imports 的两个对话；镜像声明面与恢复路径。

对应计划 `docs/private/Agents/proposed/plan-sandbox-imports-2026-11-15.md` 的阶段 C。
"""

import json

import pytest

from avm.core import Core, _build_conversation
from avm.image import ImageError, load_image
from avm.types import MetaDict, MetaList, StrNode


def _core():
    core = Core()
    core.mem["plm_params"] = MetaDict(
        data={"model": "plm.python"}, ctrl={"type": "para"})
    return core


def _python_conv(core, imports, content="x = 1"):
    parent = core.start("sys", "user", para_ref="$MEM.plm_params")
    node = MetaDict(data={"content": content}, ctrl={"type": "python"})
    node["caps"] = MetaDict(
        data={"imports": MetaList([StrNode(name) for name in imports])})
    core.mem["prog"] = node
    return _build_conversation(core, "$MEM.prog", "$MEM.plm_params", parent, is_sub=False)


def _exec(core, conv, code):
    # 真实调度里用户消息在 user_batch，进历史发生在 exec 之后；预热恢复要发生在
    # 用户消息进入历史之前，否则新的用户消息会被当成"已处理"的历史。
    para = core.unwrap("$MEM.plm_params", for_llm=False)
    executor = core._get_executor(conv, para)
    conv.append_user_message(code)
    result, _calls, _conv = executor.exec(conv, para)
    return result


class TestTwoConversationsSameModel:
    def test_same_model_different_import_whitelists(self):
        core = _core()
        conv_a = _python_conv(core, ["itertools"])
        conv_b = _python_conv(core, ["collections"])
        code = "import itertools\nprint(list(itertools.islice(itertools.count(), 2)))"

        assert _exec(core, conv_a, code) == "[0, 1]\n"
        denied = _exec(core, conv_b, code)
        assert "错误" in denied and "SecurityError" in denied and "itertools" in denied

        assert _exec(core, conv_b, "from collections import Counter\nprint(Counter('aab')['a'])") == "2\n"

    def test_system_message_import_sees_grant(self):
        core = _core()
        conv = _python_conv(core, ["itertools"], content="import itertools\nx = 1")
        assert _exec(core, conv, "print(next(itertools.count()))") == "0\n"

    def test_warm_restore_replays_import_whitelist(self):
        core = _core()
        # 跨轮状态必须落在 $MEM（命名空间不序列化）：把 import 放进 setup，
        # 恢复时 setup 重放会重新按白名单导入。
        conv = _python_conv(core, ["itertools"], content="import itertools")
        assert _exec(core, conv, "x = 1") == ""
        # 模拟重启：执行器被回收，Python 对话恢复时重放 setup
        core._executors.pop(conv.cid)
        core._warmup_needed.add(conv.cid)
        assert _exec(core, conv, "print(next(itertools.count()))") == "0\n"


def _image(program_content, user_code, caps_imports=None):
    program = {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": program_content},
            "para_ref": {"kind": "str", "value": "$MEM.plm_params"},
        },
    }
    if caps_imports is not None:
        program["value"]["caps"] = {
            "kind": "dict",
            "value": {"imports": {"kind": "list", "value": [
                {"kind": "str", "value": name} for name in caps_imports
            ]}},
        }
    mem = {
        "prog": program,
        "plm_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "plm.python"}},
        },
        "user": {"kind": "str", "value": user_code},
        "model_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
    }
    return {
        "meta": {"version": 1},
        "mem": mem,
        "para_ref": "$MEM.model_params",
        "init": {"name": "runner", "system_ref": "$MEM.prog", "user_ref": "$MEM.user"},
    }


def _write(tmp_path, image):
    path = tmp_path / "image.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


class TestImageDeclaredImports:
    def test_init_program_imports_with_declared_grant(self, tmp_path):
        image = _image("x = 1", "import itertools\nprint(next(itertools.islice(itertools.count(), 2)))",
                       caps_imports=["itertools"])
        core = load_image(Core(), _write(tmp_path, image))
        core.run()
        conv = core.get_conversation(0)
        assert conv.messages[-1].content == "0\n"
        assert conv.caps.imports == {"itertools"}

    def test_unknown_module_rejected_at_image_load(self, tmp_path):
        image = _image("x = 1", "x = 1", caps_imports=["numpy"])
        with pytest.raises(ImageError, match="能力声明非法"):
            load_image(Core(), _write(tmp_path, image))

    def test_baseline_module_declaration_rejected_at_image_load(self, tmp_path):
        image = _image("x = 1", "x = 1", caps_imports=["math"])
        with pytest.raises(ImageError, match="能力声明非法"):
            load_image(Core(), _write(tmp_path, image))

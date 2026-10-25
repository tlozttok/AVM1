"""模型目录（$MEM.model_catalog）与命名空间校验的回归用例。

对应 plan-model-catalog-2026-10-16.md 的验收一节。目录是启动自带的普通内存节点
（不是设备），两侧分别是可用的 LLM 模型与本地 PLM 注册表。
"""

import json

import pytest

from avm.core import MODEL_CATALOG_KEY, Core, LMU
from avm.image import ImageError, load_image
from avm.types import MetaDict, MetaList


def _core():
    core = Core()
    conv = core.start("sys", "user")
    core._active_cid = conv.cid
    return core, conv


def _catalog(core):
    return core.mem._data[MODEL_CATALOG_KEY]


class FakeRaw:
    """假的原始响应：payload 是对象就返回它，是异常就抛（模拟非 JSON）。"""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeHttpResponse:
    """httpx 风格响应：只有 .json()。"""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class NewSdkRaw:
    """openai 2.x 的 LegacyAPIResponse：没有 .json()，只有 .http_response 与 .text。"""

    def __init__(self, payload):
        self.http_response = _FakeHttpResponse(payload)
        self.text = json.dumps(payload, ensure_ascii=False)


class TextOnlyRaw:
    """更保守的形状：没有 .json()/.http_response，只有 .text。"""

    def __init__(self, payload):
        self.text = json.dumps(payload, ensure_ascii=False)


class UnparseableRaw:
    """什么取 JSON 的入口都没有。"""


class _RawModels:
    def __init__(self, raw):
        self._raw = raw

    @property
    def with_raw_response(self):
        return self

    def list(self, timeout=None):
        return self._raw


class _RawClient:
    def __init__(self, raw):
        self.models = _RawModels(raw)


def _install_raw_client(core, raw):
    core.lmu._client = _RawClient(raw)


class FakeModels:
    def __init__(self, payload):
        self._payload = payload

    @property
    def with_raw_response(self):
        return self

    def list(self, timeout=None):
        return FakeRaw(self._payload)


class FakeClient:
    def __init__(self, payload):
        self.models = FakeModels(payload)


def _install_fake_client(core, payload):
    core.lmu._client = FakeClient(payload)


def _call(core, conv, cmd, args, call_id="c1"):
    core._process_return_calls([{"cmd_type": cmd, "call_id": call_id, "args": args}], conv)


def _responses(conv):
    return [r.content for r in conv.user_batch.tool_responses]


# ----------------------------------------------------------------------
# 节点本身
# ----------------------------------------------------------------------

class TestCatalogNode:
    def test_installed_at_startup(self):
        core, _conv = _core()
        catalog = _catalog(core)
        assert catalog.get_metadata(), "节点要有 meta 说明"
        plm_names = [e["name"] for e in catalog["plm"]]
        # 对话可见的 PLM 列表：宿主独占的 plm.operator 不列（模型目录是对话可见面）
        assert set(plm_names) == {
            "plm.simple", "plm.python", "plm.function", "plm.machine", "plm.command"
        }
        assert "plm.operator" not in plm_names
        assert catalog["llm"]["source"] == "unavailable"
        assert catalog["llm"]["required"] == "model"
        assert set(catalog["llm"]["params"].keys()) == set(LMU._API_PARAM_KEYS)
        assert len(catalog["llm"]["models"]) == 0

    def test_dot_names_live_in_fields_not_keys(self):
        """plm.simple 这类名字带点，不能当 MetaDict 的键（$MEM 寻址按点切段）。"""
        core, _conv = _core()
        catalog = _catalog(core)
        assert isinstance(catalog["plm"], MetaList)
        assert all(isinstance(e, MetaDict) for e in catalog["plm"])
        assert all("." not in k for k in catalog.keys())

    def test_install_is_idempotent(self):
        core, _conv = _core()
        before = _catalog(core)["plm"]
        core.install_model_catalog()
        core.install_model_catalog()
        assert len(_catalog(core)["plm"]) == len(before)
        assert [e["name"] for e in _catalog(core)["plm"]] == [e["name"] for e in before]

    def test_image_declared_fields_win(self):
        """镜像或检查点里已有的字段不覆盖，只补缺失。"""
        core, _conv = _core()
        _catalog(core)["llm"]["params"]["temperature"] = "自定义"
        core.install_model_catalog()
        assert _catalog(core)["llm"]["params"]["temperature"] == "自定义"

    def test_survives_image_load(self):
        """load_image 会整树替换，目录必须重装。"""
        import tempfile, os

        doc = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                                 "value": {"model": {"kind": "str", "value": "gpt-4"}}},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(doc, f)
            path = f.name
        try:
            core = load_image(Core(), path)
        finally:
            os.unlink(path)
        assert MODEL_CATALOG_KEY in core.mem._data
        assert core.mem._data[MODEL_CATALOG_KEY]["llm"]["source"] == "unavailable"

    def test_load_image_requires_an_empty_core(self):
        core, _conv = _core()
        with pytest.raises(ImageError, match="空核心"):
            load_image(core, "whatever.json")


# ----------------------------------------------------------------------
# 刷新（拉一次上游 /models）
# ----------------------------------------------------------------------

class TestRefresh:
    def test_openai_envelope_populates_models(self):
        core, _conv = _core()
        _install_fake_client(core, {
            "object": "list",
            "data": [
                {"id": "deepseek-flash", "object": "model", "owned_by": "deepseek",
                 "context_window": 128000, "max_output_tokens": 8192,
                 "input_modalities": ["text", "image"]},
                {"id": "m2"},
            ],
        })
        result = core.refresh_model_catalog()
        assert result == {"source": "api", "count": 2, "skipped": 0}
        llm = _catalog(core)["llm"]
        assert llm["source"] == "api"
        assert "reason" not in llm
        first = llm["models"][0]
        assert first["id"] == "deepseek-flash"
        assert first["context_window"] == "128000"
        assert first["input_modalities"] == "text, image"

    def test_deepseek_item_without_created_or_owned_by_is_accepted(self):
        """DeepSeek 的条目没有 created；OpenRouter 的没有 owned_by——都不能因此拒绝。"""
        core, _conv = _core()
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        assert core.refresh_model_catalog()["source"] == "api"

    def test_new_sdk_raw_response_without_json_method(self):
        """openai 2.x 的原始响应没有 .json()（现场：AttributeError: 'LegacyAPIResponse' ...)，
        要能从 .http_response 取回 JSON。"""
        core, _conv = _core()
        _install_raw_client(core, NewSdkRaw({"object": "list", "data": [{"id": "m1"}]}))
        assert core.refresh_model_catalog()["source"] == "api"
        assert _catalog(core)["llm"]["models"][0]["id"] == "m1"

    def test_text_only_raw_response(self):
        core, _conv = _core()
        _install_raw_client(core, TextOnlyRaw({"object": "list", "data": [{"id": "m2"}]}))
        assert core.refresh_model_catalog()["source"] == "api"

    def test_unparseable_raw_response_is_unavailable(self):
        core, _conv = _core()
        _install_raw_client(core, UnparseableRaw())
        result = core.refresh_model_catalog()
        assert result["source"] == "unavailable"
        assert "无法从" in result["reason"]

    def test_deepseek_documented_metadata_fields(self):
        """按 DeepSeek /models 文档：没有 created，带 name/context_window/effort 等可选字段。"""
        core, _conv = _core()
        _install_raw_client(core, NewSdkRaw({
            "object": "list",
            "data": [{
                "id": "deepseek-chat", "object": "model", "owned_by": "deepseek",
                "name": "DeepSeek Chat", "context_window": 131072, "max_output_tokens": 8192,
                "input_modalities": ["text"], "output_modalities": ["text"],
                "effort": {"supported_levels": ["low", "medium", "high"], "default_level": "medium"},
            }],
        }))
        assert core.refresh_model_catalog()["source"] == "api"
        entry = _catalog(core)["llm"]["models"][0]
        assert entry["name"] == "DeepSeek Chat"
        assert entry["context_window"] == "131072"
        assert entry["reasoning_effort_levels"] == "low, medium, high"
        assert entry["default_effort_level"] == "medium"

    def test_bare_array_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, [{"id": "m"}])
        result = core.refresh_model_catalog()
        assert result["source"] == "unavailable"
        assert "信封" in result["reason"]

    def test_non_openai_envelope_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, {"models": [{"name": "m"}]})
        assert core.refresh_model_catalog()["source"] == "unavailable"

    def test_non_json_body_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, ValueError("Expecting value: line 1 column 1"))
        result = core.refresh_model_catalog()
        assert result["source"] == "unavailable"
        assert "Expecting value" in result["reason"]

    def test_missing_client_is_unavailable_not_fatal(self, monkeypatch):
        """没有 key / 网络不通：目录退回 unavailable，不抛。"""
        core, _conv = _core()

        class Boom:
            @property
            def models(self):
                raise RuntimeError("no api key")

        core.lmu._client = Boom()
        assert core.refresh_model_catalog()["source"] == "unavailable"

    def test_dotted_model_id_is_findable(self):
        core, _conv = _core()
        _install_fake_client(core, {"object": "list", "data": [{"id": "meta-llama/Llama-3.1-8B"}]})
        core.refresh_model_catalog()
        assert _catalog(core)["llm"]["models"][0]["id"] == "meta-llama/Llama-3.1-8B"

    def test_entries_are_addressable_by_index(self):
        """目录的用处是让对话读得到：列表项要能按引用下标寻址。"""
        core, _conv = _core()
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        core.refresh_model_catalog()
        assert core.unwrap("$MEM.model_catalog.llm.source") == "api"
        assert core.unwrap("$MEM.model_catalog.llm.models.0.id") == "deepseek-flash"
        assert core.unwrap("$MEM.model_catalog.plm.0.name") == "plm.simple"
        assert core.unwrap("$MEM.model_catalog.plm.4.name") == "plm.command"


# ----------------------------------------------------------------------
# 命名空间校验（输入边界）
# ----------------------------------------------------------------------

def _with_nodes(core):
    core.mem["para_llm"] = MetaDict(data={"model": "deepseek-flash"}, ctrl={"type": "para"})
    core.mem["para_plm"] = MetaDict(data={"model": "plm.simple"}, ctrl={"type": "para"})
    core.mem["prog_py"] = MetaDict(data={"content": "print(1)"}, ctrl={"type": "python"})
    core.mem["prog_setup"] = MetaDict(data={"content": "提示词"}, ctrl={"type": "settingup"})


class TestNamespaceCheck:
    def test_python_program_accepts_plm_name(self):
        core, conv = _core()
        _with_nodes(core)
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_py", "para_ref": "$MEM.para_plm"})
        assert _responses(conv) == ["Success created: cid=1"]

    def test_host_only_plm_rejected_generically(self):
        """plm.operator 是宿主独占：对话拿不到详细理由，也建不出对话。"""
        core, conv = _core()
        _with_nodes(core)
        core.mem["para_host"] = MetaDict(data={"model": "plm.operator"}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_py", "para_ref": "$MEM.para_host"})
        assert _responses(conv) == ["Illegal: 操作被外部原因拒绝"]
        assert core.get_conversation(1) is None
        assert "plm.operator" not in [e["name"] for e in _catalog(core)["plm"]]

    def test_python_program_rejects_llm_model(self):
        core, conv = _core()
        _with_nodes(core)
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_py", "para_ref": "$MEM.para_llm"})
        assert "不是 PLM 注册名" in _responses(conv)[0]
        assert core.get_conversation(1) is None

    def test_settingup_program_rejects_unlisted_model_when_list_known(self):
        core, conv = _core()
        _with_nodes(core)
        _install_fake_client(core, {"object": "list", "data": [{"id": "other-model"}]})
        core.refresh_model_catalog()
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_llm"})
        assert "不在上游模型清单里" in _responses(conv)[0]
        assert core.get_conversation(1) is None

    def test_settingup_program_passes_when_list_unavailable(self, capsys):
        core, conv = _core()
        _with_nodes(core)
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_llm"})
        assert _responses(conv) == ["Success created: cid=1"]
        assert "跳过 LLM 模型名校验" in capsys.readouterr().err

    def test_python_program_without_model_is_rejected(self):
        core, conv = _core()
        _with_nodes(core)
        core.mem["para_bare"] = MetaDict(data={}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_py", "para_ref": "$MEM.para_bare"})
        assert "缺少 model" in _responses(conv)[0]

    def test_refresh_warns_when_default_para_is_unlisted(self, capsys):
        core, _conv = _core()
        core.mem["model_params"] = MetaDict(data={"model": "deepseek-v4-flash"}, ctrl={"type": "para"})
        core.para_ref = "$MEM.model_params"
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        core.refresh_model_catalog()
        err = capsys.readouterr().err
        assert "不在上游清单里" in err and "deepseek-v4-flash" in err

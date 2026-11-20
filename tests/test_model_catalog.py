"""模型目录（$MEM.model_catalog）与命名空间校验的回归用例。

I6 起目录按 LLM 接口分栏（llm.interfaces.<api>）；每个接口的参数与模型清单
由该接口的适配器声明。目录仍是启动自带的普通内存节点，不是设备。
"""

import json

import pytest

from avm.core import MODEL_CATALOG_KEY, Core, LMU
from avm.image import ImageError, load_image
from avm.llm import LLMService
from avm.services import Services
from avm.types import MetaDict, MetaList


def _core():
    core = Core()
    conv = core.start("sys", "user")
    core._active_cid = conv.cid
    return core, conv


def _catalog(core):
    return core.mem._data[MODEL_CATALOG_KEY]


def _llm(core):
    return _catalog(core)["llm"]


def _iface(core, api="chat"):
    return _llm(core)["interfaces"][api]


class FakeRaw:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class NewSdkRaw:
    def __init__(self, payload):
        self.http_response = _FakeHttpResponse(payload)
        self.text = json.dumps(payload, ensure_ascii=False)


class TextOnlyRaw:
    def __init__(self, payload):
        self.text = json.dumps(payload, ensure_ascii=False)


class UnparseableRaw:
    pass


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


class FakeAdapter:
    """一个最小的第二接口适配器（只实现目录与参数声明的部分）。"""

    def __init__(self, payload, *, fail=False):
        self.payload = payload
        self.fail = fail
        self.last_dropped = []

    def api_param_names(self):
        return ("max_output_tokens", "reasoning_effort")

    def numeric_param_names(self):
        return ("max_output_tokens",)

    def fetch_models_raw(self, timeout=10.0):
        if self.fail:
            raise RuntimeError("no key")
        return self.payload

    def parse_models(self, payload):
        return [{"id": item["id"]} for item in payload.get("data", [])]


def _core_with_interfaces(**adapters):
    all_adapters = {"chat": LMU()}
    all_adapters.update(adapters)
    return Core(services=Services(llm=LLMService(adapters=all_adapters)))


# ----------------------------------------------------------------------
# 节点本身
# ----------------------------------------------------------------------

class TestCatalogNode:
    def test_installed_at_startup(self):
        core, _conv = _core()
        llm = _llm(core)
        assert llm["default"] == "chat"
        chat = _iface(core)
        assert chat["source"] == "unavailable"
        assert chat["required"] == "model"
        assert isinstance(chat["params"], MetaDict) and "temperature" in chat["params"]
        assert chat["params"]["temperature"] == "数值型"
        assert isinstance(chat["models"], MetaList) and len(chat["models"]) == 0
        assert isinstance(_catalog(core)["plm"], MetaList)

    def test_only_configured_interfaces_are_listed(self):
        core, _conv = _core()
        assert sorted(_llm(core)["interfaces"].keys()) == ["chat"]

    def test_second_interface_is_listed_with_its_params(self):
        core = _core_with_interfaces(responses=FakeAdapter({"object": "list", "data": []}))
        interfaces = _llm(core)["interfaces"]
        assert sorted(interfaces.keys()) == ["chat", "responses"]
        assert "max_output_tokens" in interfaces["responses"]["params"]
        assert interfaces["responses"]["params"]["max_output_tokens"] == "数值型"

    def test_install_is_idempotent(self):
        core, _conv = _core()
        before = [_e["name"] for _e in _catalog(core)["plm"]]
        core.install_model_catalog()
        core.install_model_catalog()
        assert [_e["name"] for _e in _catalog(core)["plm"]] == before
        assert sorted(_llm(core)["interfaces"].keys()) == ["chat"]

    def test_image_declared_fields_win(self):
        core, _conv = _core()
        _iface(core)["params"]["temperature"] = "自定义"
        core.install_model_catalog()
        assert _iface(core)["params"]["temperature"] == "自定义"

    def test_survives_image_load(self):
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
        assert _iface(core)["source"] == "unavailable"

    def test_load_image_requires_an_empty_core(self):
        core, _conv = _core()
        with pytest.raises(ImageError, match="空核心"):
            load_image(core, "whatever.json")


# ----------------------------------------------------------------------
# 刷新（各接口拉一次上游 /models）
# ----------------------------------------------------------------------

class TestRefresh:
    def test_openai_envelope_populates_chat_interface(self):
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
        assert result["interfaces"]["chat"] == {"source": "api", "count": 2}
        chat = _iface(core)
        assert chat["source"] == "api" and "reason" not in chat
        assert chat["models"][0]["id"] == "deepseek-flash"
        assert chat["models"][0]["context_window"] == "128000"
        assert chat["models"][0]["input_modalities"] == "text, image"

    def test_deepseek_item_without_created_or_owned_by_is_accepted(self):
        core, _conv = _core()
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        assert core.refresh_model_catalog()["interfaces"]["chat"]["source"] == "api"

    def test_new_sdk_raw_response_without_json_method(self):
        core, _conv = _core()
        _install_raw_client(core, NewSdkRaw({"object": "list", "data": [{"id": "m1"}]}))
        assert core.refresh_model_catalog()["interfaces"]["chat"]["source"] == "api"
        assert _iface(core)["models"][0]["id"] == "m1"

    def test_text_only_raw_response(self):
        core, _conv = _core()
        _install_raw_client(core, TextOnlyRaw({"object": "list", "data": [{"id": "m2"}]}))
        assert core.refresh_model_catalog()["interfaces"]["chat"]["source"] == "api"

    def test_unparseable_raw_response_is_unavailable(self):
        core, _conv = _core()
        _install_raw_client(core, UnparseableRaw())
        result = core.refresh_model_catalog()
        assert result["interfaces"]["chat"]["source"] == "unavailable"
        assert "无法从" in result["interfaces"]["chat"]["reason"]

    def test_deepseek_documented_metadata_fields(self):
        core, _conv = _core()
        _install_raw_client(core, NewSdkRaw({
            "object": "list",
            "data": [{
                "id": "deepseek-chat", "object": "model", "owned_by": "deepseek",
                "name": "DeepSeek Chat", "context_window": 131072, "max_output_tokens": 8192,
                "input_modalities": ["text"], "output_modalities": ["text"],
                "effort": {"supported_levels": ["low", "medium", "high"],
                           "default_level": "medium"},
            }],
        }))
        core.refresh_model_catalog()
        entry = _iface(core)["models"][0]
        assert entry["name"] == "DeepSeek Chat"
        assert entry["context_window"] == "131072"
        assert entry["reasoning_effort_levels"] == "low, medium, high"
        assert entry["default_effort_level"] == "medium"

    def test_bare_array_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, [{"id": "m"}])
        result = core.refresh_model_catalog()
        assert result["interfaces"]["chat"]["source"] == "unavailable"
        assert "信封" in result["interfaces"]["chat"]["reason"]

    def test_non_openai_envelope_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, {"models": [{"name": "m"}]})
        assert core.refresh_model_catalog()["interfaces"]["chat"]["source"] == "unavailable"

    def test_non_json_body_is_unavailable(self):
        core, _conv = _core()
        _install_fake_client(core, ValueError("Expecting value: line 1 column 1"))
        result = core.refresh_model_catalog()
        assert "Expecting value" in result["interfaces"]["chat"]["reason"]

    def test_missing_client_is_unavailable_not_fatal(self):
        core, _conv = _core()

        class Boom:
            @property
            def models(self):
                raise RuntimeError("no api key")

        core.lmu._client = Boom()
        assert core.refresh_model_catalog()["interfaces"]["chat"]["source"] == "unavailable"

    def test_interface_failure_does_not_block_other_interfaces(self):
        core = _core_with_interfaces(
            responses=FakeAdapter({"object": "list", "data": [{"id": "resp-model"}]}))
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        result = core.refresh_model_catalog()
        assert result["interfaces"]["chat"]["source"] == "api"
        assert result["interfaces"]["responses"]["source"] == "api"
        assert _iface(core, "responses")["models"][0]["id"] == "resp-model"

    def test_dotted_model_id_is_findable(self):
        core, _conv = _core()
        _install_fake_client(core, {"object": "list",
                                    "data": [{"id": "meta-llama/Llama-3.1-8B"}]})
        core.refresh_model_catalog()
        assert _iface(core)["models"][0]["id"] == "meta-llama/Llama-3.1-8B"

    def test_entries_are_addressable_by_index(self):
        core, _conv = _core()
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        core.refresh_model_catalog()
        assert core.unwrap("$MEM.model_catalog.llm.default") == "chat"
        assert core.unwrap("$MEM.model_catalog.llm.interfaces.chat.source") == "api"
        assert core.unwrap("$MEM.model_catalog.llm.interfaces.chat.models.0.id") == "deepseek-flash"
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
        core, conv = _core()
        _with_nodes(core)
        core.mem["para_host"] = MetaDict(data={"model": "plm.operator"},
                                         ctrl={"type": "para"})
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
        assert "不在 chat 接口的模型清单里" in _responses(conv)[0]
        assert core.get_conversation(1) is None

    def test_settingup_program_passes_when_list_unavailable(self, capsys):
        core, conv = _core()
        _with_nodes(core)
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_llm"})
        assert _responses(conv) == ["Success created: cid=1"]
        assert "跳过模型名校验" in capsys.readouterr().err

    def test_unknown_api_is_rejected(self):
        core, conv = _core()
        _with_nodes(core)
        core.mem["para_api"] = MetaDict(data={"model": "some-model", "api": "anthropic"},
                                        ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_api"})
        assert "api 'anthropic' 未配置" in _responses(conv)[0]
        assert core.get_conversation(1) is None

    def test_second_interface_model_is_validated_against_its_own_list(self):
        core = _core_with_interfaces(
            responses=FakeAdapter({"object": "list", "data": [{"id": "resp-model"}]}))
        conv = core.start("sys", "user")
        core._active_cid = conv.cid
        _with_nodes(core)
        core.refresh_model_catalog()
        core.mem["para_resp"] = MetaDict(data={"model": "resp-model", "api": "responses"},
                                         ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_resp"},
              call_id="c1")
        assert _responses(conv) == ["Success created: cid=1"]

        core.mem["para_resp_bad"] = MetaDict(
            data={"model": "other-model", "api": "responses"}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_setup", "para_ref": "$MEM.para_resp_bad"},
              call_id="c2")
        assert "不在 responses 接口的模型清单里" in _responses(conv)[1]

    def test_python_program_without_model_is_rejected(self):
        core, conv = _core()
        _with_nodes(core)
        core.mem["para_bare"] = MetaDict(data={}, ctrl={"type": "para"})
        _call(core, conv, "create_cmd",
              {"system_ref": "$MEM.prog_py", "para_ref": "$MEM.para_bare"})
        assert "缺少 model" in _responses(conv)[0]

    def test_refresh_warns_when_default_para_is_unlisted(self, capsys):
        core, _conv = _core()
        core.mem["model_params"] = MetaDict(data={"model": "deepseek-v4-flash"},
                                            ctrl={"type": "para"})
        core.para_ref = "$MEM.model_params"
        _install_fake_client(core, {"object": "list", "data": [{"id": "deepseek-flash"}]})
        core.refresh_model_catalog()
        err = capsys.readouterr().err
        assert "不在 chat 接口的清单里" in err and "deepseek-v4-flash" in err


"""服务层组合根与 Core 接线（计划 plan-service-layer-2026-11-17 批次 S5）。"""

from avm.core import Core
from avm.types import Conversation, MetaDict, SystemMessage


class FakeLLMExecutor:
    def __init__(self):
        self.last_call = None
        self.calls = []
        self.api_param_names = lambda: ("model", "temperature")
        self.numeric_param_names = lambda: ("temperature",)

    def exec(self, conversation, para):
        self.calls.append((conversation.cid, para.get("model")))
        self.last_call = {
            "model": para.get("model"), "messages": [], "result": "ok",
            "tool_calls": [], "error": None,
        }
        return "ok", [], conversation

    def reset_client(self):
        self.reset = True


class FakePLMExecutor:
    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.last_call = None
        self.warm_calls = []

    def warm_restore(self, messages, params=None):
        self.warm_calls.append((messages, params))

    def exec(self, conversation, para):
        return "", [], conversation


class FakePLMService:
    def __init__(self):
        self.created = []

    def available(self, extra=None):
        return ["plm.fake"]

    def is_host_only(self, model):
        return model == "plm.operator"

    def create_executor(self, model, extra=None, **kwargs):
        executor = FakePLMExecutor(model, kwargs)
        self.created.append(executor)
        return executor

    def catalog_entries(self, extra=None):
        return [MetaDict(data={"name": "plm.fake", "required": "model"})]


class FakeServices:
    def __init__(self):
        self.llm = FakeLLMExecutor()
        self.plm = FakePLMService()


def _core():
    return Core(services=FakeServices())


def test_core_advances_with_injected_llm_service():
    core = _core()
    conv = core.start("sys", "hi")
    core._active_cid = conv.cid
    core.mem["para"] = MetaDict(data={"model": "fake-model"}, ctrl={"type": "para"})
    conv.para_ref = "$MEM.para"

    core.advance_conversation()

    assert core.services.llm.calls == [(conv.cid, "fake-model")]
    assert core.monitor.frames()[-1].lmu["result"] == "ok"


def test_core_lmu_property_delegates_to_services():
    core = _core()
    fake = core.lmu
    assert fake is core.services.llm
    replacement = FakeLLMExecutor()
    core.lmu = replacement
    assert core.services.llm is replacement


def test_model_catalog_uses_injected_service_param_names():
    core = _core()
    params = core.mem._data["model_catalog"]["llm"]["params"]
    assert "temperature" in params and params["temperature"] == "数值型"
    assert "top_p" not in params  # 假服务没有声明


def test_get_executor_for_python_uses_plm_service_and_warm_restore():
    core = _core()
    conv = Conversation(messages=[SystemMessage("setup")], cid=7)
    conv.is_python = True
    core._conv_by_cid[7] = conv
    para = MetaDict(data={"model": "plm.fake"}, ctrl={"type": "para"})
    core._warmup_needed.add(7)

    executor = core._get_executor(conv, para)

    assert executor is core.services.plm.created[-1]
    assert executor.warm_calls and set(executor.warm_calls[0][1]) == {"tools", "imports"}


def test_get_executor_unknown_plm_keeps_error_text():
    core = _core()
    conv = Conversation(messages=[SystemMessage("setup")], cid=8)
    conv.is_python = True
    core._conv_by_cid[8] = conv
    para = MetaDict(data={"model": "plm.missing"}, ctrl={"type": "para"})
    try:
        core._get_executor(conv, para)
    except ValueError as e:
        assert "不在 PLM_REGISTRY 中" in str(e)
        assert "plm.fake" in str(e)
    else:  # pragma: no cover
        raise AssertionError("未知 PLM 名应报错")


def test_bind_console_conversation_uses_plm_service():
    core = _core()
    conv = core.bind_console_conversation(channel=object())
    assert conv.cid == -1
    assert core.services.plm.created[-1].model == "plm.operator"

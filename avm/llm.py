"""LLM 服务入口（L4）：适配器选择、流式事件汇合与 last_call 维护。

计划 plan-ir-multiprotocol-2026-11-17 批次 I2：本模块不再实现具体协议。
- 非流式路径（observer 未装）：直接调用适配器的 `exec`，行为与拆分前一致；
- 流式路径（observer 已装）：迭代适配器 `stream()` 的统一事件，交给
  `executor.assistant_from_events` 组装 assistant IR 消息并提交。

Core 通过 `services.llm` 使用本服务（`core.lmu` 是它的兼容属性）。
"""

import time
from typing import Optional

from .api_errors import classify_api_error
from .exceptions import APIError
from .executor import (
    assistant_from_events, return_calls_from_assistant,
)
from .llm_chat import LMU
from .types import Conversation, MetaDict


class LLMService:
    """LLM 服务：把「一次推理」分派给某个协议适配器。"""

    def __init__(self, adapters: Optional[dict] = None, observer=None):
        if adapters is None:
            adapters = {"chat": LMU()}
        self.adapters = dict(adapters)
        self.observer = observer
        self.last_call: Optional[dict] = None

    # ------------------------------------------------------------------
    # 兼容面（chat 适配器 = 拆分前的 LMU）
    # ------------------------------------------------------------------

    @property
    def chat(self):
        return self.adapters.get("chat")

    @property
    def _client(self):
        """兼容：测试与旧代码通过 core.lmu._client 注入假客户端。"""
        return self.chat._client

    @_client.setter
    def _client(self, value):
        self.chat._client = value

    @property
    def client(self):
        return self.chat.client

    def reset_client(self) -> None:
        if self.chat is not None:
            self.chat.reset_client()

    def api_param_names(self) -> tuple:
        return self.chat.api_param_names()

    def numeric_param_names(self) -> tuple:
        return self.chat.numeric_param_names()

    def fetch_models_raw(self, timeout: float = 10.0) -> dict:
        return self.chat.fetch_models_raw(timeout)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def adapter_for(self, para: MetaDict):
        api = (para.get("api") or "chat") if hasattr(para, "get") else "chat"
        adapter = self.adapters.get(api)
        if adapter is None:
            raise APIError(
                f"LLM 接口 {api!r} 未配置（可用: {sorted(self.adapters)}）",
                category="other",
                detail={"category": "other", "message": f"接口 {api!r} 未配置",
                        "status_code": None, "context_limit": None},
            )
        return adapter

    def exec(self, conversation: Conversation, para: MetaDict):
        adapter = self.adapter_for(para)
        if self.observer is None:
            result = adapter.exec(conversation, para)
            self.last_call = adapter.last_call
            return result
        return self._exec_stream(conversation, para, adapter)

    def _exec_stream(self, conversation: Conversation, para: MetaDict, adapter):
        events = []
        start = time.time()
        try:
            for event in adapter.stream(conversation, para):
                events.append(event)
                self.observer(event)
            assistant, usage, stop_reason = assistant_from_events(events)
            return_calls = return_calls_from_assistant(assistant)
            from .executor import commit_round
            commit_round(conversation, assistant=assistant)
        except Exception as e:
            self._record_failure(adapter, para, e, start)
            if isinstance(e, APIError):
                raise
            category, detail = classify_api_error(e)
            raise APIError(
                f"{type(e).__name__}: {e}",
                category=category,
                status_code=detail["status_code"],
                detail=detail,
            ) from e

        request = getattr(adapter, "last_request", None) or {}
        self.last_call = {
            "model": request.get("model", para.get("model")),
            "messages": request.get("messages", []),
            "result": assistant.content,
            "reasoning": assistant.reasoning_content,
            "tool_calls": return_calls,
            "elapsed_ms": (time.time() - start) * 1000,
            "error": None,
            "usage": usage,
            "stop_reason": stop_reason,
        }
        return assistant.content, return_calls, conversation

    def _record_failure(self, adapter, para, error, start: float) -> None:
        request = getattr(adapter, "last_request", None) or {}
        self.last_call = {
            "model": request.get("model", para.get("model")),
            "messages": request.get("messages", []),
            "result": None,
            "tool_calls": [],
            "elapsed_ms": (time.time() - start) * 1000,
            "error": f"{type(error).__name__}: {error}",
        }


def _raw_response_json(raw):  # pragma: no cover - 兼容旧导入路径
    from .llm_chat import _raw_response_json as impl
    return impl(raw)


__all__ = ["LLMService", "LMU"]

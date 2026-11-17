"""OpenAI Responses API 协议适配器（L4）。

计划 plan-ir-multiprotocol-2026-11-17 批次 I3。无状态用法（`store=False`）；
把 IR 渲染成 Responses 的 input items，把语义流式事件转成统一事件
（见 executor.assistant_from_events）。注意：
- reasoning item 的 id/encrypted_content 原样放在 IR reasoning part 的
  `id`（额外字段）与 `opaque` 上，请求时原样回传；
- 来自其它接口的 reasoning/provider 块不渲染，记入 `last_dropped` 供声明；
- 未识别的 para 参数暂不报错（I6 统一按接口白名单校验）。
"""

import copy
import json
import time
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from .api_errors import classify_api_error
from .exceptions import APIError
from .instructions import build_tools
from .types import Conversation, MetaDict

_API_PARAM_KEYS = {
    "max_output_tokens", "temperature", "top_p", "reasoning_effort",
    "extra_body", "timeout",
}


def _num(value):
    if not isinstance(value, str):
        return value
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def _usage_dict(usage) -> Optional[dict]:
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _api_error(message: str, detail: Optional[dict] = None) -> APIError:
    return APIError(
        message, category="other",
        status_code=(detail or {}).get("status_code"),
        detail={"category": "other", "message": message,
                "status_code": (detail or {}).get("status_code"),
                "context_limit": None},
    )


class ResponsesAdapter:
    """Responses API 适配器：只有流式路径（`LLMService` 负责 drain 与提交）。"""

    def __init__(self):
        self._client = None
        self.last_call: Optional[dict] = None
        self.last_request: Optional[dict] = None
        self.last_dropped: list = []

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    def api_param_names(self) -> tuple:
        return tuple(sorted(_API_PARAM_KEYS))

    def fetch_models_raw(self, timeout: float = 10.0) -> dict:
        """拉一次上游 `/models`（I6 按接口装配目录时使用）。"""
        from .llm_chat import _raw_response_json
        raw = self.client.models.with_raw_response.list(timeout=timeout)
        return _raw_response_json(raw)

    # ------------------------------------------------------------------
    # 请求渲染
    # ------------------------------------------------------------------

    def _build_request(self, conversation: Conversation, para: MetaDict) -> dict:
        systems = []
        items = []
        dropped = []
        for message in conversation.messages:
            if message.role == "system":
                systems.append(message.content)
                continue
            items.extend(self._items_for(message, dropped))
        for response in conversation.user_batch.tool_responses:
            items.append({"type": "function_call_output",
                          "call_id": response.tool_call_id,
                          "output": response.content})
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            items.append({"role": "user", "content": user_content})
        if not items and not systems:
            raise _api_error("Responses 适配器：对话没有可发送的输入")

        request = {
            "model": para.get("model", "gpt-4"),
            "input": items,
            "tools": self._tools(conversation),
            "store": False,          # 无状态：AVM 自己持有完整历史
            "stream": True,
        }
        if systems:
            request["instructions"] = "\n\n".join(systems)
        use_tool = para.get("use_tool")
        if use_tool:
            request["tool_choice"] = use_tool
        values = para.to_dict()
        if values.get("max_output_tokens") is not None:
            request["max_output_tokens"] = _num(values["max_output_tokens"])
        if values.get("temperature") is not None:
            request["temperature"] = _num(values["temperature"])
        if values.get("top_p") is not None:
            request["top_p"] = _num(values["top_p"])
        if values.get("reasoning_effort") is not None:
            request["reasoning"] = {"effort": str(values["reasoning_effort"])}
        if isinstance(values.get("extra_body"), dict):
            request.update(copy.deepcopy(values["extra_body"]))
        if values.get("timeout") is not None:
            request["timeout"] = _num(values["timeout"])
        self.last_dropped = dropped
        return request

    def _items_for(self, message, dropped: list) -> list:
        if message.role == "user":
            return [{"role": "user", "content": message.content}]
        if message.role == "tool":
            return [{"type": "function_call_output", "call_id": message.tool_call_id,
                     "output": message.content}]
        if message.role != "assistant":
            return []
        items = []
        for part in message.parts:
            kind = part.get("type")
            if kind == "text":
                if part.get("text"):
                    items.append({"role": "assistant",
                                  "content": [{"type": "output_text", "text": part["text"]}]})
            elif kind == "reasoning":
                if part.get("interface") != "responses":
                    dropped.append({"type": "reasoning", "interface": part.get("interface"),
                                    "reason": "cross-interface"})
                    continue
                item = {"type": "reasoning"}
                if part.get("id"):
                    item["id"] = part["id"]
                if part.get("opaque"):
                    item["encrypted_content"] = part["opaque"]
                if part.get("text"):
                    item["summary"] = [{"type": "summary_text", "text": part["text"]}]
                items.append(item)
            elif kind == "tool_call":
                item = {"type": "function_call",
                        "call_id": part.get("call_id"),
                        "name": part.get("name"),
                        "arguments": part.get("args", "")}
                if part.get("id"):
                    item["id"] = part["id"]
                items.append(item)
            elif kind == "provider" and part.get("interface") == "responses":
                data = part.get("data") or {}
                if isinstance(data, dict):
                    items.append(copy.deepcopy(data))
            else:
                dropped.append({"type": kind, "interface": part.get("interface"),
                                "reason": "unsupported-part"})
        return items

    @staticmethod
    def _tools(conversation: Conversation) -> list:
        tools = []
        for tool in build_tools(conversation):
            fn = tool.get("function") or {}
            item = {"type": "function", "name": fn.get("name"),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    "strict": False}
            if fn.get("description"):
                item["description"] = fn["description"]
            tools.append(item)
        return tools

    # ------------------------------------------------------------------
    # 流式路径
    # ------------------------------------------------------------------

    def stream(self, conversation: Conversation, para: MetaDict):
        request = self._build_request(conversation, para)
        self.last_request = {"model": request["model"], "messages": request["input"]}
        start = time.time()
        try:
            events = self.client.responses.create(**request)
        except Exception as e:
            self._failed(e, request, start)
            raise
        yield {"type": "start"}
        index = 0
        text_index = None
        reasoning_index = None
        tool_indexes = {}   # item_id → part index
        finished = False
        try:
            for event in events:
                etype = getattr(event, "type", None)
                if etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", None) == "function_call":
                        part_index = index
                        index += 1
                        tool_indexes[getattr(item, "id", None)] = part_index
                        yield {"type": "tool_call_start", "index": part_index,
                               "call_id": getattr(item, "call_id", None) or "",
                               "name": getattr(item, "name", None) or ""}
                elif etype == "response.output_text.delta":
                    if text_index is None:
                        text_index = index
                        index += 1
                        yield {"type": "part_start", "index": text_index, "part": "text"}
                    yield {"type": "text_delta", "index": text_index,
                           "text": getattr(event, "delta", None) or ""}
                elif etype in ("response.reasoning_summary_text.delta",
                               "response.reasoning_text.delta"):
                    if reasoning_index is None:
                        reasoning_index = index
                        index += 1
                        yield {"type": "part_start", "index": reasoning_index,
                               "part": "reasoning", "interface": "responses"}
                    yield {"type": "reasoning_delta", "index": reasoning_index,
                           "text": getattr(event, "delta", None) or ""}
                elif etype == "response.function_call_arguments.delta":
                    part_index = tool_indexes.get(getattr(event, "item_id", None))
                    if part_index is not None:
                        yield {"type": "tool_call_args_delta", "index": part_index,
                               "args": getattr(event, "delta", None) or ""}
                elif etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", None) if item is not None else None
                    if item_type == "function_call":
                        part_index = tool_indexes.get(getattr(item, "id", None))
                        if part_index is not None:
                            extra = {"id": item.id} if getattr(item, "id", None) else None
                            yield {"type": "part_end", "index": part_index,
                                   "extra": extra}
                    elif item_type == "reasoning" and reasoning_index is not None:
                        extra = {"id": item.id} if getattr(item, "id", None) else None
                        yield {"type": "part_end", "index": reasoning_index,
                               "opaque": getattr(item, "encrypted_content", None),
                               "extra": extra}
                        reasoning_index = None
                    elif item_type == "message" and text_index is not None:
                        yield {"type": "part_end", "index": text_index}
                        text_index = None
                elif etype == "response.completed":
                    response = getattr(event, "response", None)
                    if text_index is not None:
                        yield {"type": "part_end", "index": text_index}
                        text_index = None
                    if reasoning_index is not None:
                        yield {"type": "part_end", "index": reasoning_index}
                        reasoning_index = None
                    yield {"type": "finish", "stop_reason": "stop",
                           "usage": _usage_dict(getattr(response, "usage", None)) or {}}
                    finished = True
                elif etype == "response.incomplete":
                    response = getattr(event, "response", None)
                    details = getattr(response, "incomplete_details", None)
                    reason = getattr(details, "reason", None) if details else None
                    stop_reason = "length" if reason == "max_output_tokens" else "content_filter"
                    if text_index is not None:
                        yield {"type": "part_end", "index": text_index}
                        text_index = None
                    if reasoning_index is not None:
                        yield {"type": "part_end", "index": reasoning_index}
                        reasoning_index = None
                    yield {"type": "finish", "stop_reason": stop_reason,
                           "usage": _usage_dict(getattr(response, "usage", None)) or {}}
                    finished = True
                elif etype in ("response.failed", "error"):
                    error = getattr(event, "error", None)
                    if error is None:
                        response = getattr(event, "response", None)
                        error = getattr(response, "error", None)
                    message = getattr(error, "message", None) if error is not None else None
                    if message is None:
                        message = getattr(event, "message", None) or getattr(event, "code", None)
                    raise _api_error(f"Responses 调用失败：{message or etype}")
            if not finished:
                raise _api_error("Responses 流在终止事件之前结束")
        except Exception as e:
            self._failed(e, request, start)
            raise

    def _failed(self, error, request: dict, start: float) -> None:
        self.last_call = {
            "model": request.get("model"),
            "messages": request.get("input", []),
            "result": None,
            "tool_calls": [],
            "elapsed_ms": (time.time() - start) * 1000,
            "error": f"{type(error).__name__}: {error}",
            "dropped_parts": list(self.last_dropped),
        }
        if isinstance(error, APIError):
            return
        category, detail = classify_api_error(error)
        raise APIError(
            f"{type(error).__name__}: {error}",
            category=category,
            status_code=detail["status_code"],
            detail=detail,
        ) from error


__all__ = ["ResponsesAdapter"]

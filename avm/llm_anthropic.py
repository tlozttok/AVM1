"""Anthropic Messages 协议适配器（L4）。

计划 plan-ir-multiprotocol-2026-11-17 批次 I4。用 httpx 直接发 HTTP；
SSE 事件转成统一事件（见 executor.assistant_from_events）。要点：
- system 消息渲染为顶层 system；中段 system 作为消息内的 system 角色（2026-10-21 起支持）；
- assistant 的 thinking 块（interface="anthropic"）带 signature 原样回传；
- tool_result 折叠进下一轮 user 消息，且在文本之前；
- 参数白名单按 Anthropic 2026-10-21：temperature/top_p/top_k 已移除，不发送；
- 同一接口无状态请求，AVM 自持完整历史。
"""

import json
import os
import time
from typing import Optional

import httpx

from .api_errors import classify_api_error
from .exceptions import APIError
from .instructions import build_tools
from .types import Conversation, MetaDict

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_VERSION = "2026-10-21"
_DEFAULT_MAX_TOKENS = 4096
_API_PARAM_KEYS = {"max_tokens", "extra_body", "timeout"}


def _num(value):
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        return value


class _HttpFailure(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _json_args(raw: str) -> dict:
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class AnthropicAdapter:
    """Anthropic Messages API 适配器（只有流式路径）。"""

    def __init__(self, client: Optional[httpx.Client] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 version: Optional[str] = None):
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        self._version = version
        self.last_call: Optional[dict] = None
        self.last_request: Optional[dict] = None
        self.last_dropped: list = []

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise APIError(
                    "ANTHROPIC_API_KEY 未配置", category="other",
                    detail={"category": "other", "message": "ANTHROPIC_API_KEY 未配置",
                            "status_code": None, "context_limit": None},
                )
            base = (self._base_url or os.environ.get("ANTHROPIC_BASE_URL")
                    or _DEFAULT_BASE_URL).rstrip("/")
            version = self._version or os.environ.get("ANTHROPIC_VERSION") or _DEFAULT_VERSION
            self._client = httpx.Client(
                base_url=base,
                headers={"x-api-key": key, "anthropic-version": version,
                         "content-type": "application/json"},
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
        return self._client

    def api_param_names(self) -> tuple:
        return tuple(sorted(_API_PARAM_KEYS))

    def fetch_models_raw(self, timeout: float = 10.0) -> dict:
        response = self.client.get("/v1/models", timeout=timeout)
        if response.status_code >= 400:
            raise self._http_failure(response)
        return response.json()

    # ------------------------------------------------------------------
    # 请求渲染
    # ------------------------------------------------------------------

    def _build_request(self, conversation: Conversation, para: MetaDict) -> dict:
        dropped = []
        systems = []
        messages = []
        pending_user = None
        names_by_call = {}
        for message in conversation.messages:
            if message.role == "assistant":
                for part in message.parts:
                    if part.get("type") == "tool_call":
                        names_by_call[part.get("call_id")] = part.get("name")
        for message in conversation.messages:
            if message.role == "system":
                if pending_user is not None or messages:
                    messages.append({"role": "system",
                                     "content": [{"type": "text", "text": message.content}]})
                else:
                    systems.append(message.content)
                continue
            if message.role == "user":
                if pending_user is None:
                    pending_user = []
                if message.content:
                    pending_user.append({"type": "text", "text": message.content})
                continue
            if message.role == "tool":
                if pending_user is None:
                    pending_user = []
                pending_user.append({"type": "tool_result",
                                     "tool_use_id": message.tool_call_id,
                                     "content": message.content})
                continue
            if message.role != "assistant":
                continue
            if pending_user is not None:
                messages.append({"role": "user", "content": pending_user})
                pending_user = None
            blocks = self._assistant_blocks(message, dropped)
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        for response in conversation.user_batch.tool_responses:
            if pending_user is None:
                pending_user = []
            pending_user.append({"type": "tool_result",
                                 "tool_use_id": response.tool_call_id,
                                 "content": response.content})
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            if pending_user is None:
                pending_user = []
            pending_user.append({"type": "text", "text": user_content})
        if pending_user is not None:
            messages.append({"role": "user", "content": pending_user})

        values = para.to_dict()
        request = {
            "model": para.get("model", "claude-sonnet-5"),
            "max_tokens": _num(values.get("max_tokens")) or _DEFAULT_MAX_TOKENS,
            "messages": messages,
            "tools": self._tools(conversation),
            "stream": True,
        }
        if systems:
            request["system"] = "\n\n".join(systems)
        use_tool = para.get("use_tool")
        if use_tool == "none":
            request["tool_choice"] = {"type": "none"}
        elif use_tool == "required":
            request["tool_choice"] = {"type": "any"}
        elif use_tool == "auto":
            request["tool_choice"] = {"type": "auto"}
        if isinstance(values.get("extra_body"), dict):
            request.update(json.loads(json.dumps(values["extra_body"])))
        if values.get("timeout") is not None:
            request["timeout"] = _num(values["timeout"])
        self.last_dropped = dropped
        return request

    def _assistant_blocks(self, message, dropped: list) -> list:
        reasoning, others = [], []
        for part in message.parts:
            kind = part.get("type")
            if kind == "reasoning":
                if part.get("interface") != "anthropic":
                    dropped.append({"type": "reasoning",
                                    "interface": part.get("interface"),
                                    "reason": "cross-interface"})
                    continue
                if not part.get("opaque"):
                    dropped.append({"type": "reasoning", "interface": "anthropic",
                                    "reason": "missing-signature"})
                    continue
                reasoning.append({"type": "thinking",
                                  "thinking": part.get("text") or "",
                                  "signature": part["opaque"]})
            elif kind == "text":
                if part.get("text"):
                    others.append({"type": "text", "text": part["text"]})
            elif kind == "tool_call":
                raw = part.get("args", "")
                args = _json_args(raw)
                if not args and raw.strip() not in ("", "{}"):
                    dropped.append({"type": "tool_call", "call_id": part.get("call_id"),
                                    "reason": "malformed-args"})
                others.append({"type": "tool_use", "id": part.get("call_id"),
                               "name": part.get("name"), "input": args})
            elif kind == "provider" and part.get("interface") == "anthropic":
                data = part.get("data")
                if isinstance(data, dict):
                    others.append(json.loads(json.dumps(data)))
            else:
                dropped.append({"type": kind, "interface": part.get("interface"),
                                "reason": "unsupported-part"})
        return reasoning + others

    @staticmethod
    def _tools(conversation: Conversation) -> list:
        tools = []
        for tool in build_tools(conversation):
            fn = tool.get("function") or {}
            item = {"name": fn.get("name"),
                    "input_schema": fn.get("parameters") or {"type": "object",
                                                             "properties": {}}}
            if fn.get("description"):
                item["description"] = fn["description"]
            tools.append(item)
        return tools

    # ------------------------------------------------------------------
    # 流式路径
    # ------------------------------------------------------------------

    def stream(self, conversation: Conversation, para: MetaDict):
        request = self._build_request(conversation, para)
        self.last_request = {"model": request["model"], "messages": request["messages"]}
        start = time.time()
        try:
            with self.client.stream("POST", "/v1/messages", json=request) as response:
                if response.status_code >= 400:
                    response.read()
                    raise self._http_failure(response)
                yield from self._map_stream(response, start)
        except Exception as e:
            self._failed(e, request, start)
            raise

    def _map_stream(self, response, start: float):
        yield {"type": "start"}
        index = 0
        block_parts = {}      # content_block index → part index
        block_kinds = {}      # content_block index → text|thinking|tool_use
        signatures = {}       # content_block index → signature
        usage = {}
        stop_reason = None
        finished = False
        for event_name, data in _iter_sse(response.iter_lines()):
            etype = data.get("type") or event_name
            if etype == "message_start":
                msg_usage = (data.get("message") or {}).get("usage") or {}
                if msg_usage.get("input_tokens") is not None:
                    usage["input_tokens"] = msg_usage["input_tokens"]
            elif etype == "content_block_start":
                block = data.get("content_block") or {}
                block_type = block.get("type")
                part_index = index
                index += 1
                block_parts[data.get("index")] = part_index
                block_kinds[data.get("index")] = block_type
                if block_type == "text":
                    yield {"type": "part_start", "index": part_index, "part": "text"}
                elif block_type == "thinking":
                    yield {"type": "part_start", "index": part_index,
                           "part": "reasoning", "interface": "anthropic"}
                elif block_type == "tool_use":
                    yield {"type": "tool_call_start", "index": part_index,
                           "call_id": block.get("id") or "",
                           "name": block.get("name") or ""}
            elif etype == "content_block_delta":
                block_index = data.get("index")
                part_index = block_parts.get(block_index)
                delta = data.get("delta") or {}
                delta_type = delta.get("type")
                if part_index is None:
                    continue
                if delta_type == "text_delta":
                    yield {"type": "text_delta", "index": part_index,
                           "text": delta.get("text") or ""}
                elif delta_type == "thinking_delta":
                    yield {"type": "reasoning_delta", "index": part_index,
                           "text": delta.get("thinking") or ""}
                elif delta_type == "signature_delta":
                    signatures[block_index] = (signatures.get(block_index) or "") + (
                        delta.get("signature") or "")
                elif delta_type == "input_json_delta":
                    yield {"type": "tool_call_args_delta", "index": part_index,
                           "args": delta.get("partial_json") or ""}
            elif etype == "content_block_stop":
                block_index = data.get("index")
                part_index = block_parts.get(block_index)
                if part_index is None:
                    continue
                event = {"type": "part_end", "index": part_index}
                if block_kinds.get(block_index) == "thinking" and signatures.get(block_index):
                    event["opaque"] = signatures[block_index]
                yield event
            elif etype == "message_delta":
                delta = data.get("delta") or {}
                if delta.get("stop_reason"):
                    stop_reason = delta["stop_reason"]
                delta_usage = data.get("usage") or {}
                if delta_usage.get("output_tokens") is not None:
                    usage["output_tokens"] = delta_usage["output_tokens"]
            elif etype == "message_stop":
                if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
                    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
                yield {"type": "finish", "stop_reason": _stop_reason(stop_reason),
                       "usage": usage}
                finished = True
            elif etype == "error":
                error = data.get("error") or {}
                raise APIError(
                    f"Anthropic 流式错误：{error.get('message') or error}",
                    category="other",
                    detail={"category": "other",
                            "message": f"Anthropic 流式错误：{error.get('message') or error}",
                            "status_code": None, "context_limit": None},
                )
        if not finished:
            raise APIError("Anthropic 流在 message_stop 之前结束", category="other",
                           detail={"category": "other",
                                   "message": "Anthropic 流在 message_stop 之前结束",
                                   "status_code": None, "context_limit": None})

    def _http_failure(self, response: httpx.Response) -> APIError:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        error = (body or {}).get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            error_type = error.get("type") or "error"
            message = error.get("message") or error_type
        else:
            error_type, message = "http_error", f"HTTP {status}"
        failure = _HttpFailure(status, f"Anthropic {error_type}: {message}")
        category, detail = classify_api_error(failure)
        return APIError(str(failure), category=category,
                        status_code=status, detail=detail)

    def _failed(self, error, request: dict, start: float) -> None:
        self.last_call = {
            "model": request.get("model"),
            "messages": request.get("messages", []),
            "result": None,
            "tool_calls": [],
            "elapsed_ms": (time.time() - start) * 1000,
            "error": f"{type(error).__name__}: {error}",
            "dropped_parts": list(self.last_dropped),
        }


def _iter_sse(lines):
    event_name = None
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.rstrip("\r")
        if not line:
            event_name = None
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            yield event_name, data


def _stop_reason(reason) -> str:
    if reason == "max_tokens" or reason == "model_context_window_exceeded":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "refusal":
        return "content_filter"
    return "stop"


__all__ = ["AnthropicAdapter"]

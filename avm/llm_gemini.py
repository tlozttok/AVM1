"""Google Gemini 原生 generateContent 协议适配器（L4）。

计划 plan-ir-multiprotocol-2026-11-17 批次 I5。用 httpx 直接发 HTTP；
`streamGenerateContent?alt=sse` 的 chunk 转成统一事件。要点：
- generateContent 是无状态原生格式（其被标 legacy，但仍是原生格式）；
- contents 只有 user/model 两种 role；工具结果折叠进 user 的 functionResponse；
- 思考 part 用 `thought=True` + `thoughtSignature`（放 IR reasoning 的 opaque）；
- functionCall.args 是对象（由 IR 的原始 args 字符串解析而来）；
- functionResponse.response 必须是对象（字符串结果包成 {"result": ...}）；
- API key 只出现在请求 query，不进 last_call / 错误文本。
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

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_DEFAULT_VERSION = "v1beta"
_API_PARAM_KEYS = {"temperature", "top_p", "max_output_tokens", "extra_body", "timeout"}
_SAFETY_FINISH = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
    "IMAGE_OTHER", "CONTENT_FILTER",
}


class _HttpFailure(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _num(value):
    if not isinstance(value, str):
        return value
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def _gemini_schema(schema):
    """JSON Schema → Gemini Schema（类型名大写；未声明形状不做假设）。"""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key in ("properties", "patternProperties") and isinstance(value, dict):
            out[key] = {name: _gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _gemini_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            out[key] = [_gemini_schema(sub) for sub in value]
        else:
            out[key] = value
    return out


def _args_json(args) -> str:
    return json.dumps(args or {}, ensure_ascii=False, sort_keys=True)


class GeminiAdapter:
    """Gemini generateContent 适配器（只有流式路径）。"""

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
            base = (self._base_url or os.environ.get("GEMINI_BASE_URL")
                    or _DEFAULT_BASE_URL).rstrip("/")
            self._client = httpx.Client(base_url=base,
                                        timeout=httpx.Timeout(600.0, connect=10.0))
        return self._client

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise APIError(
                "GEMINI_API_KEY 未配置", category="other",
                detail={"category": "other", "message": "GEMINI_API_KEY 未配置",
                        "status_code": None, "context_limit": None},
            )
        return key

    def api_param_names(self) -> tuple:
        return tuple(sorted(_API_PARAM_KEYS))

    def numeric_param_names(self) -> tuple:
        return ("max_output_tokens", "temperature", "top_p")

    def fetch_models_raw(self, timeout: float = 10.0) -> dict:
        version = self._version or os.environ.get("GEMINI_API_VERSION") or _DEFAULT_VERSION
        response = self.client.get(f"/{version}/models",
                                   params={"key": self.api_key}, timeout=timeout)
        if response.status_code >= 400:
            raise self._http_failure(response)
        return response.json()

    # ------------------------------------------------------------------
    # 请求渲染
    # ------------------------------------------------------------------

    def _build_request(self, conversation: Conversation, para: MetaDict) -> dict:
        dropped = []
        names_by_call = {}
        for message in conversation.messages:
            if message.role == "assistant":
                for part in message.parts:
                    if part.get("type") == "tool_call":
                        names_by_call[part.get("call_id")] = part.get("name")
        systems = []
        contents = []

        def append_content(role: str, parts: list):
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for message in conversation.messages:
            if message.role == "system":
                if message.content:
                    systems.append(message.content)
                continue
            if message.role == "user":
                append_content("user", [{"text": message.content}])
                continue
            if message.role == "tool":
                name = names_by_call.get(message.tool_call_id)
                if not name:
                    dropped.append({"type": "tool_result",
                                    "call_id": message.tool_call_id,
                                    "reason": "unknown-tool-name"})
                    name = message.tool_call_id
                append_content("user", [{"functionResponse": {
                    "name": name,
                    "response": {"result": message.content},
                }}])
                continue
            if message.role != "assistant":
                continue
            append_content("model", self._assistant_parts(message, dropped))
        for response in conversation.user_batch.tool_responses:
            name = names_by_call.get(response.tool_call_id) or response.tool_call_id
            append_content("user", [{"functionResponse": {
                "name": name,
                "response": {"result": response.content},
            }}])
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            append_content("user", [{"text": user_content}])

        values = para.to_dict()
        request = {"contents": contents}
        if systems:
            request["systemInstruction"] = {"parts": [{"text": "\n\n".join(systems)}]}
        declarations = []
        for tool in build_tools(conversation):
            fn = tool.get("function") or {}
            declaration = {"name": fn.get("name"),
                           "parameters": _gemini_schema(fn.get("parameters") or
                                                        {"type": "object", "properties": {}})}
            if fn.get("description"):
                declaration["description"] = fn["description"]
            declarations.append(declaration)
        if declarations:
            request["tools"] = [{"functionDeclarations": declarations}]
        use_tool = para.get("use_tool")
        mode = {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(use_tool)
        if mode:
            request["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        generation = {}
        if values.get("temperature") is not None:
            generation["temperature"] = _num(values["temperature"])
        if values.get("top_p") is not None:
            generation["topP"] = _num(values["top_p"])
        if values.get("max_output_tokens") is not None:
            generation["maxOutputTokens"] = _num(values["max_output_tokens"])
        if generation:
            request["generationConfig"] = generation
        if isinstance(values.get("extra_body"), dict):
            request.update(json.loads(json.dumps(values["extra_body"])))
        self.last_dropped = dropped
        return request

    def _assistant_parts(self, message, dropped: list) -> list:
        reasoning, others = [], []
        for part in message.parts:
            kind = part.get("type")
            if kind == "text":
                if part.get("text"):
                    others.append({"text": part["text"]})
            elif kind == "reasoning":
                if part.get("interface") != "gemini":
                    dropped.append({"type": "reasoning",
                                    "interface": part.get("interface"),
                                    "reason": "cross-interface"})
                    continue
                item = {"text": part.get("text") or "", "thought": True}
                if part.get("opaque"):
                    item["thoughtSignature"] = part["opaque"]
                reasoning.append(item)
            elif kind == "tool_call":
                raw = part.get("args", "")
                try:
                    args = json.loads(raw) if raw else {}
                except ValueError:
                    args = {}
                    dropped.append({"type": "tool_call",
                                    "call_id": part.get("call_id"),
                                    "reason": "malformed-args"})
                item = {"functionCall": {"name": part.get("name"),
                                         "args": args if isinstance(args, dict) else {}}}
                if part.get("call_id"):
                    item["functionCall"]["id"] = part["call_id"]
                others.append(item)
            elif kind == "provider" and part.get("interface") == "gemini":
                data = part.get("data")
                if isinstance(data, dict):
                    others.append(json.loads(json.dumps(data)))
            else:
                dropped.append({"type": kind, "interface": part.get("interface"),
                                "reason": "unsupported-part"})
        return reasoning + others

    # ------------------------------------------------------------------
    # 流式路径
    # ------------------------------------------------------------------

    def stream(self, conversation: Conversation, para: MetaDict):
        request = self._build_request(conversation, para)
        model = para.get("model", "gemini-4")
        version = self._version or os.environ.get("GEMINI_API_VERSION") or _DEFAULT_VERSION
        url = f"/{version}/models/{model}:streamGenerateContent"
        self.last_request = {"model": model,
                             "messages": request.get("contents", [])}
        start = time.time()
        try:
            with self.client.stream("POST", url, json=request,
                                    params={"key": self.api_key, "alt": "sse"}) as response:
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
        text_index = None
        reasoning_index = None
        signature = None
        tool_count = 0
        stop_reason = None
        usage = {}
        for _event_name, data in _iter_sse(response.iter_lines()):
            if isinstance(data, dict) and data.get("error"):
                error = data["error"]
                raise APIError(
                    f"Gemini 流式错误：{error.get('message') or error}",
                    category="other",
                    detail={"category": "other",
                            "message": f"Gemini 流式错误：{error.get('message') or error}",
                            "status_code": None, "context_limit": None},
                )
            candidates = (data or {}).get("candidates") or []
            meta = (data or {}).get("usageMetadata") or {}
            if meta:
                usage = {
                    "input_tokens": meta.get("promptTokenCount"),
                    "output_tokens": meta.get("candidatesTokenCount"),
                    "total_tokens": meta.get("totalTokenCount"),
                    "reasoning_tokens": meta.get("thoughtsTokenCount"),
                }
            prompt_feedback = (data or {}).get("promptFeedback") or {}
            if not candidates:
                if prompt_feedback.get("blockReason"):
                    raise APIError(
                        f"Gemini 拒绝：{prompt_feedback['blockReason']}", category="other",
                        detail={"category": "other",
                                "message": f"Gemini 拒绝：{prompt_feedback['blockReason']}",
                                "status_code": None, "context_limit": None},
                    )
                continue
            candidate = candidates[0]
            finish = candidate.get("finishReason")
            if finish:
                stop_reason = finish
            parts = ((candidate.get("content") or {}).get("parts")) or []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                thought_signature = part.get("thoughtSignature")
                if thought_signature:
                    signature = thought_signature
                if part.get("functionCall"):
                    call = part["functionCall"] or {}
                    part_index = index
                    index += 1
                    tool_count += 1
                    call_id = call.get("id") or f"gemini-call-{tool_count}"
                    yield {"type": "tool_call_start", "index": part_index,
                           "call_id": call_id, "name": call.get("name") or ""}
                    yield {"type": "tool_call_args_delta", "index": part_index,
                           "args": _args_json(call.get("args"))}
                    extra = {"id": call["id"]} if call.get("id") else None
                    yield {"type": "part_end", "index": part_index, "extra": extra}
                    continue
                text = part.get("text")
                if part.get("thought") is True:
                    if reasoning_index is None:
                        reasoning_index = index
                        index += 1
                        yield {"type": "part_start", "index": reasoning_index,
                               "part": "reasoning", "interface": "gemini"}
                    if text:
                        yield {"type": "reasoning_delta", "index": reasoning_index,
                               "text": text}
                elif text is not None:
                    if reasoning_index is not None:
                        yield {"type": "part_end", "index": reasoning_index,
                               "opaque": signature}
                        reasoning_index = None
                        signature = None
                    if text_index is None:
                        text_index = index
                        index += 1
                        yield {"type": "part_start", "index": text_index, "part": "text"}
                    yield {"type": "text_delta", "index": text_index, "text": text}
        if reasoning_index is not None:
            yield {"type": "part_end", "index": reasoning_index, "opaque": signature}
        if text_index is not None:
            yield {"type": "part_end", "index": text_index}
        yield {"type": "finish", "stop_reason": _stop_reason(stop_reason),
               "usage": usage}

    def _http_failure(self, response: httpx.Response) -> APIError:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        error = (body or {}).get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            message = error.get("message") or error.get("status") or "http_error"
            status_text = error.get("status") or ""
        else:
            message, status_text = f"HTTP {status}", ""
        failure = _HttpFailure(status, f"Gemini {status_text}: {message}")
        category, detail = classify_api_error(failure)
        return APIError(str(failure), category=category,
                        status_code=status, detail=detail)

    def _failed(self, error, request: dict, start: float) -> None:
        self.last_call = {
            "model": self.last_request.get("model") if self.last_request else None,
            "messages": (self.last_request or {}).get("messages", []),
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
                yield event_name, json.loads(payload)
            except ValueError:
                continue


def _stop_reason(reason) -> str:
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _SAFETY_FINISH:
        return "content_filter"
    return "stop"


__all__ = ["GeminiAdapter"]

"""Gemini generateContent 适配器：请求渲染、SSE chunk 映射与错误（计划 I5）。

用 httpx.MockTransport，不调真实 API。
"""

import json

import httpx
import pytest

from avm.exceptions import APIError
from avm.llm import LLMService
from avm.llm_gemini import GeminiAdapter
from avm.types import (
    AssistantMessage, Conversation, MetaDict, SystemMessage, ToolMessage,
    UserMessage, reasoning_part, text_part, tool_call_part,
)


def _sse(chunks):
    return "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                   for chunk in chunks).encode("utf-8")


def _client(chunks, captured, status=200, body=None):
    def handler(request):
        captured.append({"url": str(request.url),
                         "body": json.loads(request.content)})
        if status >= 400:
            return httpx.Response(status, json=body or {})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_sse(chunks))
    return httpx.Client(transport=httpx.MockTransport(handler),
                        base_url="https://gemini.test")


def _para(**extra):
    data = {"model": "gemini-test", "api": "gemini"}
    data.update(extra)
    return MetaDict(data=data, ctrl={"type": "para"})


def _history():
    return Conversation(messages=[
        SystemMessage("sys-g"),
        UserMessage("q"),
        AssistantMessage(parts=[
            reasoning_part("chat", text="foreign"),
            reasoning_part("gemini", text="plan", opaque="sig-1"),
            text_part("earlier"),
            tool_call_part("g_1", "memory_read", '{"ref": "$MEM"}'),
        ]),
        ToolMessage(content="tool-result", tool_call_id="g_1"),
    ])


STREAM_CHUNKS = [
    {"candidates": [{"content": {"parts": [
        {"text": "plan", "thought": True, "thoughtSignature": "sig-2"}]}}]},
    {"candidates": [{"content": {"parts": [{"text": "do"}]}}]},
    {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "memory_write",
                          "args": {"ref": "$MEM.out"}, "id": "g_2"}}]}}]},
    {"candidates": [{"finishReason": "STOP"}],
     "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3,
                       "totalTokenCount": 8, "thoughtsTokenCount": 2}},
]


class TestGeminiRequest:
    def test_request_rendering(self):
        captured = []
        adapter = GeminiAdapter(client=_client([], captured), api_key="k")
        request = adapter._build_request(_history(), _para(
            temperature="0.3", max_output_tokens="256", use_tool="required"))
        assert request["systemInstruction"] == {"parts": [{"text": "sys-g"}]}
        contents = request["contents"]
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "q"}]
        model_parts = contents[1]["parts"]
        assert model_parts[0] == {"text": "plan", "thought": True,
                                  "thoughtSignature": "sig-1"}
        assert {"text": "earlier"} in model_parts
        call = [part for part in model_parts if part.get("functionCall")]
        assert call == [{"functionCall": {"name": "memory_read",
                                          "args": {"ref": "$MEM"},
                                          "id": "g_1"}}]
        assert contents[2] == {"role": "user", "parts": [
            {"functionResponse": {"name": "memory_read",
                                  "response": {"result": "tool-result"}}}]}
        declaration = request["tools"][0]["functionDeclarations"][0]
        assert declaration["name"]
        assert declaration["parameters"]["type"] == "OBJECT"
        assert request["generationConfig"] == {"temperature": 0.3,
                                               "maxOutputTokens": 256}
        assert request["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
        extras = [item for item in adapter.last_dropped
                  if item.get("interface") == "chat"]
        assert extras and extras[0]["reason"] == "cross-interface"
        # API key 只在 query，不在请求体（工具 schema 里出现 key 字段名不算）
        assert "key" not in request


class TestGeminiStream:
    def test_stream_maps_chunks_and_commits(self):
        seen = []
        captured = []
        adapter = GeminiAdapter(client=_client(STREAM_CHUNKS, captured), api_key="k")
        service = LLMService(adapters={"gemini": adapter}, observer=seen.append)
        conv = _history()

        result, return_calls, _ = service.exec(conv, _para())

        assert result == "do"
        assert return_calls == [{"call_id": "g_2", "cmd_type": "memory_write",
                                 "args": {"ref": "$MEM.out"}}]
        parts = conv.messages[-1].parts
        assert {"type": "reasoning", "interface": "gemini", "text": "plan",
                "opaque": "sig-2"} in parts
        assert {"type": "text", "text": "do"} in parts
        assert {"type": "tool_call", "call_id": "g_2", "name": "memory_write",
                "args": '{"ref": "$MEM.out"}', "id": "g_2"} in parts
        assert [event["type"] for event in seen][:2] == ["start", "part_start"]
        assert seen[-1]["type"] == "finish"
        assert service.last_call["usage"]["input_tokens"] == 5
        assert service.last_call["usage"]["reasoning_tokens"] == 2
        assert service.last_call["stop_reason"] == "stop"
        call = captured[0]
        assert "alt=sse" in call["url"]
        assert "key=k" in call["url"]

    def test_http_error_maps_to_api_error(self):
        captured = []
        adapter = GeminiAdapter(client=_client(
            [], captured, status=429,
            body={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                            "message": "quota exceeded"}}), api_key="k")
        service = LLMService(adapters={"gemini": adapter})
        conv = _history()
        before = list(conv.messages)
        with pytest.raises(APIError) as excinfo:
            service.exec(conv, _para())
        assert excinfo.value.status_code == 429
        assert conv.messages == before

    def test_mid_stream_error_chunk_commits_nothing(self):
        chunks = [
            {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]},
            {"error": {"code": 500, "message": "boom"}},
        ]
        captured = []
        adapter = GeminiAdapter(client=_client(chunks, captured), api_key="k")
        service = LLMService(adapters={"gemini": adapter})
        conv = _history()
        before = list(conv.messages)
        with pytest.raises(APIError, match="boom"):
            service.exec(conv, _para())
        assert conv.messages == before

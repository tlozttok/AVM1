"""Anthropic 适配器：请求渲染、SSE 事件映射与错误（计划 I4）。

用 httpx.MockTransport，不调真实 API。
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from avm.exceptions import APIError
from avm.llm import LLMService
from avm.llm_anthropic import AnthropicAdapter
from avm.types import (
    AssistantMessage, Conversation, MetaDict, SystemMessage, ToolMessage,
    UserMessage, reasoning_part, text_part, tool_call_part,
)


def _sse(events):
    chunks = []
    for name, data in events:
        chunks.append(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")
    return "".join(chunks).encode("utf-8")


def _client(events, captured, status=200, body=None):
    def handler(request):
        captured.append(json.loads(request.content))
        if status >= 400:
            return httpx.Response(status, json=body or {})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_sse(events))
    return httpx.Client(transport=httpx.MockTransport(handler),
                        base_url="https://anthropic.test")


def _para(**extra):
    data = {"model": "claude-test", "api": "anthropic"}
    data.update(extra)
    return MetaDict(data=data, ctrl={"type": "para"})


def _history():
    return Conversation(messages=[
        SystemMessage("sys-a"),
        UserMessage("q"),
        AssistantMessage(parts=[
            reasoning_part("chat", text="foreign"),
            reasoning_part("anthropic", text="plan", opaque="sig-1"),
            text_part("earlier"),
            tool_call_part("tu_1", "memory_read", '{"ref": "$MEM"}'),
        ]),
        ToolMessage(content="tool-result", tool_call_id="tu_1"),
    ])


STREAM_EVENTS = [
    ("message_start", {"type": "message_start",
                       "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "thinking"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "thinking_delta", "thinking": "plan"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "signature_delta", "signature": "sig-2"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {"type": "content_block_start", "index": 1,
                             "content_block": {"type": "text"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1,
                             "delta": {"type": "text_delta", "text": "done"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("content_block_start", {"type": "content_block_start", "index": 2,
                             "content_block": {"type": "tool_use", "id": "tu_2",
                                               "name": "memory_write"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 2,
                             "delta": {"type": "input_json_delta",
                                       "partial_json": '{"ref":'}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 2,
                             "delta": {"type": "input_json_delta",
                                       "partial_json": ' "$MEM.out"}'}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 2}),
    ("message_delta", {"type": "message_delta",
                       "delta": {"stop_reason": "tool_use"},
                       "usage": {"output_tokens": 5}}),
    ("message_stop", {"type": "message_stop"}),
]


class TestAnthropicRequest:
    def test_request_rendering(self):
        captured = []
        adapter = AnthropicAdapter(client=_client([], captured))
        request = adapter._build_request(_history(), _para(max_tokens="1024",
                                                          use_tool="auto"))
        assert request["model"] == "claude-test"
        assert request["max_tokens"] == 1024
        assert request["stream"] is True
        assert request["system"] == "sys-a"
        assert request["tool_choice"] == {"type": "auto"}
        assert request["tools"][0]["name"]
        assert request["tools"][0]["input_schema"]["type"] == "object"

        messages = request["messages"]
        assert messages[0] == {"role": "user",
                               "content": [{"type": "text", "text": "q"}]}
        assistant = messages[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"][0] == {"type": "thinking", "thinking": "plan",
                                           "signature": "sig-1"}
        assert {"type": "text", "text": "earlier"} in assistant["content"]
        tool_use = [block for block in assistant["content"]
                    if block.get("type") == "tool_use"]
        assert tool_use == [{"type": "tool_use", "id": "tu_1",
                             "name": "memory_read", "input": {"ref": "$MEM"}}]
        assert messages[2] == {"role": "user",
                               "content": [{"type": "tool_result",
                                            "tool_use_id": "tu_1",
                                            "content": "tool-result"}]}
        # 其它接口的 reasoning 不渲染并记账
        assert adapter.last_dropped == [{"type": "reasoning", "interface": "chat",
                                         "reason": "cross-interface"}]

    def test_no_sampling_params_sent(self):
        captured = []
        adapter = AnthropicAdapter(client=_client([], captured))
        request = adapter._build_request(_history(), _para(temperature="0.5",
                                                          top_p="0.9"))
        assert "temperature" not in request and "top_p" not in request


class TestAnthropicStream:
    def test_stream_maps_events_and_commits(self):
        seen = []
        captured = []
        adapter = AnthropicAdapter(client=_client(STREAM_EVENTS, captured))
        service = LLMService(adapters={"anthropic": adapter}, observer=seen.append)
        conv = _history()

        result, return_calls, _ = service.exec(conv, _para())

        assert result == "done"
        assert return_calls == [{"call_id": "tu_2", "cmd_type": "memory_write",
                                 "args": {"ref": "$MEM.out"}}]
        parts = conv.messages[-1].parts
        assert {"type": "reasoning", "interface": "anthropic", "text": "plan",
                "opaque": "sig-2"} in parts
        assert {"type": "text", "text": "done"} in parts
        assert {"type": "tool_call", "call_id": "tu_2", "name": "memory_write",
                "args": '{"ref": "$MEM.out"}'} in parts
        assert [e["type"] for e in seen][:2] == ["start", "part_start"]
        assert seen[-1]["type"] == "finish"
        assert service.last_call["usage"] == {"input_tokens": 7, "output_tokens": 5,
                                              "total_tokens": 12}
        assert service.last_call["stop_reason"] == "tool_calls"
        assert captured[0]["messages"][1]["content"][0]["type"] == "thinking"

    def test_http_error_maps_to_api_error(self):
        captured = []
        adapter = AnthropicAdapter(client=_client(
            [], captured, status=402,
            body={"type": "error", "error": {"type": "billing_error",
                                             "message": "credit balance low"}}))
        service = LLMService(adapters={"anthropic": adapter})
        conv = _history()
        before = list(conv.messages)
        with pytest.raises(APIError) as excinfo:
            service.exec(conv, _para())
        assert excinfo.value.category == "insufficient_balance"
        assert conv.messages == before

    def test_mid_stream_error_event_commits_nothing(self):
        events = [
            ("message_start", {"type": "message_start",
                               "message": {"usage": {"input_tokens": 1}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text"}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "partial"}}),
            ("error", {"type": "error",
                       "error": {"type": "overloaded_error", "message": "busy"}}),
        ]
        captured = []
        adapter = AnthropicAdapter(client=_client(events, captured))
        service = LLMService(adapters={"anthropic": adapter})
        conv = _history()
        before = list(conv.messages)
        with pytest.raises(APIError, match="busy"):
            service.exec(conv, _para())
        assert conv.messages == before

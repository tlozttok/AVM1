"""Responses 适配器：请求渲染、语义流事件映射与声明（计划 I3）。

全部用假 SDK 客户端；不调真实 API。
"""

from types import SimpleNamespace

import pytest

from avm.exceptions import APIError
from avm.llm import LLMService
from avm.llm_responses import ResponsesAdapter
from avm.types import (
    AssistantMessage, Conversation, MetaDict, SystemMessage, ToolMessage,
    UserMessage, provider_part, reasoning_part, text_part, tool_call_part,
)


class FakeResponses:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events)


class FakeClient:
    def __init__(self, events):
        self.responses = FakeResponses(events)


def _event(etype, **fields):
    return SimpleNamespace(type=etype, **fields)


def _service(events, observer=None):
    service = LLMService(adapters={"responses": ResponsesAdapter()}, observer=observer)
    service.adapters["responses"]._client = FakeClient(events)
    return service


def _para(**extra):
    data = {"model": "resp-model", "api": "responses"}
    data.update(extra)
    return MetaDict(data=data, ctrl={"type": "para"})


def _history():
    return Conversation(messages=[
        SystemMessage("sys"),
        UserMessage("hi"),
        AssistantMessage(parts=[
            reasoning_part("chat", text="old chat think"),
            reasoning_part("responses", text="summary", opaque="enc-1", id="rs_1"),
            text_part("earlier"),
            tool_call_part("call_1", "memory_read", '{"ref": "$MEM"}'),
        ]),
        ToolMessage(content="tool-result", tool_call_id="call_1"),
    ])


class TestResponsesRequest:
    def test_request_rendering(self):
        adapter = ResponsesAdapter()
        adapter._client = FakeClient([])
        request = adapter._build_request(_history(), _para(
            max_output_tokens="256", temperature="0.2",
            reasoning_effort="high", use_tool="auto",
        ))
        assert request["model"] == "resp-model"
        assert request["store"] is False
        assert request["stream"] is True
        assert request["instructions"] == "sys"
        assert request["max_output_tokens"] == 256
        assert request["temperature"] == 0.2
        assert request["reasoning"] == {"effort": "high"}
        assert request["tool_choice"] == "auto"
        assert request["tools"][0]["type"] == "function"
        assert request["tools"][0]["name"]
        assert request["tools"][0]["strict"] is False

        items = request["input"]
        assert {"role": "user", "content": "hi"} in items
        assert {"type": "function_call_output", "call_id": "call_1",
                "output": "tool-result"} in items
        assert {"role": "assistant",
                "content": [{"type": "output_text", "text": "earlier"}]} in items
        reasoning_items = [item for item in items if item.get("type") == "reasoning"]
        assert reasoning_items == [{"type": "reasoning", "id": "rs_1",
                                    "encrypted_content": "enc-1",
                                    "summary": [{"type": "summary_text",
                                                 "text": "summary"}]}]
        calls = [item for item in items if item.get("type") == "function_call"]
        assert calls == [{"type": "function_call", "call_id": "call_1",
                          "name": "memory_read", "arguments": '{"ref": "$MEM"}'}]
        # 其它接口的 reasoning 块不渲染，并记账供声明
        assert adapter.last_dropped == [{"type": "reasoning", "interface": "chat",
                                         "reason": "cross-interface"}]


class TestResponsesStream:
    def test_stream_maps_events_to_ir_and_commits(self):
        seen = []
        events = [
            _event("response.output_item.added",
                   item=SimpleNamespace(type="function_call", id="fc_2",
                                        call_id="call_2", name="memory_write")),
            _event("response.function_call_arguments.delta", item_id="fc_2",
                   delta='{"ref":'),
            _event("response.function_call_arguments.delta", item_id="fc_2",
                   delta=' "$MEM.out"}'),
            _event("response.output_item.done",
                   item=SimpleNamespace(type="function_call", id="fc_2",
                                        call_id="call_2", name="memory_write")),
            _event("response.output_text.delta", delta="done"),
            _event("response.output_item.done",
                   item=SimpleNamespace(type="message", id="msg_2")),
            _event("response.completed", response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
                incomplete_details=None)),
        ]
        service = _service(events, observer=seen.append)
        conv = _history()

        result, return_calls, _ = service.exec(conv, _para())

        assert result == "done"
        assert return_calls == [{"call_id": "call_2", "cmd_type": "memory_write",
                                 "args": {"ref": "$MEM.out"}}]
        assistant = conv.messages[-1]
        parts = assistant.parts
        assert {"type": "tool_call", "call_id": "call_2", "name": "memory_write",
                "args": '{"ref": "$MEM.out"}', "id": "fc_2"} in parts
        assert {"type": "text", "text": "done"} in parts
        assert [e["type"] for e in seen][:2] == ["start", "tool_call_start"]
        assert seen[-1]["type"] == "finish"
        assert service.last_call["usage"] == {"input_tokens": 10, "output_tokens": 4,
                                              "total_tokens": 14}
        assert service.last_call["dropped_parts"][0]["interface"] == "chat"
        # 请求带上了同一份历史（无状态重放）
        request = service.adapters["responses"]._client.responses.calls[0]
        assert request["store"] is False
        assert any(item.get("type") == "reasoning" for item in request["input"])

    def test_failed_event_raises_api_error_and_commits_nothing(self):
        events = [
            _event("response.created"),
            _event("response.output_text.delta", delta="partial"),
            _event("response.failed", response=SimpleNamespace(
                error=SimpleNamespace(message="boom", code="server_error"))),
        ]
        service = _service(events)
        conv = _history()
        before = list(conv.messages)

        with pytest.raises(APIError) as excinfo:
            service.exec(conv, _para())

        assert "boom" in str(excinfo.value)
        assert conv.messages == before
        assert service.last_call["error"].startswith("APIError")

    def test_incomplete_maps_to_length(self):
        events = [
            _event("response.output_text.delta", delta="cut"),
            _event("response.output_item.done", item=SimpleNamespace(type="message", id="m")),
            _event("response.incomplete", response=SimpleNamespace(
                usage=None,
                incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
        ]
        service = _service(events)
        result, _, _ = service.exec(_history(), _para())
        assert result == "cut"
        assert service.last_call["stop_reason"] == "length"

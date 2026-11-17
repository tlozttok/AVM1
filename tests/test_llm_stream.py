"""chat 适配器的流式路径与统一事件组装（计划 plan-ir-multiprotocol I2）。

全部用假客户端；不调真实 API。
"""

from types import SimpleNamespace

import pytest

from avm.exceptions import APIError
from avm.executor import assistant_from_events
from avm.llm import LLMService
from avm.types import Conversation, MetaDict, SystemMessage, UserMessage


def _chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning,
                            tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tool_delta(index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=call_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


class FakeStreamCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("stream") is True
        return iter(self.chunks)


class FakeStreamClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=FakeStreamCompletions(chunks))


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok", reasoning_content="r", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChatClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def _conv():
    return Conversation(messages=[SystemMessage("sys"), UserMessage("hi")])


def _para(**extra):
    data = {"model": "test-model"}
    data.update(extra)
    return MetaDict(data=data, ctrl={"type": "para"})


def _service(chunks=None, *, observer=None):
    service = LLMService(observer=observer)
    if chunks is not None:
        service.chat._client = FakeStreamClient(chunks)
    return service


class TestChatStreaming:
    def test_stream_assembles_parts_events_and_last_call(self):
        events = []
        service = _service([
            _chunk(reasoning="think-"),
            _chunk(reasoning="more"),
            _chunk(content="hello "),
            _chunk(content="world"),
            _chunk(tool_calls=[_tool_delta(0, call_id="c1", name="memory_read",
                                           arguments='{"ref"')]),
            _chunk(tool_calls=[_tool_delta(0, arguments=': "$MEM"}')]),
            _chunk(finish="tool_calls",
                   usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5,
                                         total_tokens=8)),
        ], observer=events.append)
        conv = _conv()

        result, return_calls, _ = service.exec(conv, _para())

        assert result == "hello world"
        assert return_calls == [{"call_id": "c1", "cmd_type": "memory_read",
                                 "args": {"ref": "$MEM"}}]
        assistant = conv.messages[-1]
        assert assistant.role == "assistant"
        assert assistant.parts == [
            {"type": "reasoning", "interface": "chat", "text": "think-more"},
            {"type": "text", "text": "hello world"},
            {"type": "tool_call", "call_id": "c1", "name": "memory_read",
             "args": '{"ref": "$MEM"}'},
        ]
        assert [e["type"] for e in events][:3] == ["start", "part_start", "reasoning_delta"]
        assert events[-1]["type"] == "finish"
        assert events[-1]["stop_reason"] == "tool_calls"
        assert service.last_call["usage"] == {"input_tokens": 3, "output_tokens": 5,
                                              "total_tokens": 8}
        assert service.last_call["result"] == "hello world"
        assert service.last_call["reasoning"] == "think-more"
        # 请求是 chat 投影，包含历史与 tools，且 stream=True
        call = service.chat._client.chat.completions.calls[0]
        assert call["messages"][0] == {"role": "system", "content": "sys"}
        assert isinstance(call["tools"], list) and call["tools"]
        assert "stream" not in call.get("stream_options", {})

    def test_stream_error_midway_commits_nothing(self):
        class Boom(Exception):
            status_code = 500

        def broken_chunks():
            yield _chunk(content="partial")
            raise Boom("stream broken")

        service = _service(None, observer=lambda event: None)
        service.chat._client = FakeStreamClient(broken_chunks())
        conv = _conv()
        before = list(conv.messages)

        with pytest.raises(APIError) as excinfo:
            service.exec(conv, _para())

        assert "Boom" in str(excinfo.value)
        assert conv.messages == before  # 零提交
        assert "Boom" in service.last_call["error"]
        assert service.last_call["result"] is None

    def test_malformed_tool_args_become_json_error(self):
        service = _service([
            _chunk(tool_calls=[_tool_delta(0, call_id="c1", name="memory_read",
                                           arguments="{bad")]),
            _chunk(finish="tool_calls"),
        ], observer=lambda event: None)
        conv = _conv()
        result, return_calls, _ = service.exec(conv, _para())
        assert result == ""
        assert return_calls[0]["cmd_type"] == "json_error"
        assert return_calls[0]["call_id"] == "c1"
        # 历史仍提交 assistant（保持 tool_calls 与后续占位响应的配对）
        assert conv.messages[-1].role == "assistant"

    def test_reasoning_opaque_attached_at_part_end(self):
        events = [
            {"type": "part_start", "index": 0, "part": "reasoning", "interface": "anthropic"},
            {"type": "reasoning_delta", "index": 0, "text": "t"},
            {"type": "part_end", "index": 0, "opaque": "sig"},
            {"type": "part_start", "index": 1, "part": "text"},
            {"type": "text_delta", "index": 1, "text": "x"},
            {"type": "unknown_event", "index": 99},
            {"type": "finish", "stop_reason": "stop", "usage": {}},
        ]
        assistant, usage, stop_reason = assistant_from_events(events)
        assert assistant.parts == [
            {"type": "reasoning", "interface": "anthropic", "text": "t", "opaque": "sig"},
            {"type": "text", "text": "x"},
        ]
        assert stop_reason == "stop"


class TestServiceCompat:
    def test_non_stream_path_unchanged(self):
        service = LLMService()
        service.chat._client = FakeChatClient()
        conv = _conv()
        result, return_calls, _ = service.exec(conv, _para())
        assert result == "ok"
        assert return_calls == []
        assert service.last_call is service.chat.last_call
        assert service.last_call["reasoning"] == "r"

    def test_unconfigured_interface_reports_clearly(self):
        service = LLMService()
        with pytest.raises(APIError, match="未配置"):
            service.exec(_conv(), _para(api="anthropic"))

    def test_stream_param_removed_from_chat_whitelist(self):
        service = LLMService()
        assert "stream" not in service.api_param_names()
        assert "stream" not in service.chat._API_PARAM_KEYS

    def test_client_compat_property_sets_chat_adapter(self):
        service = LLMService()
        client = FakeChatClient()
        service._client = client
        assert service.chat._client is client

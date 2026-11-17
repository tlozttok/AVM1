"""IR v1（中立消息格式）的往返、投影与兼容边界。

计划 plan-ir-multiprotocol-2026-11-17 批次 I1：存储形状是 role+parts；
content/tool_calls/reasoning_content/tool_call_id 只是 chat 投影视图。
"""

import json

import pytest

from avm.types import (
    CHECKPOINT_VERSION, AssistantMessage, Conversation, Message, SystemMessage,
    ToolMessage, UserMessage, message_from_dict, message_to_dict,
    provider_part, reasoning_part, text_part,
)


def _conv(messages):
    return Conversation(messages=messages, cid=0)


def _roundtrip(conv):
    return Conversation.from_checkpoint(json.loads(json.dumps(conv.to_checkpoint())))


class TestIRStorage:
    def test_parts_are_storage_and_content_is_projection(self):
        msg = AssistantMessage(content="hi", reasoning_content="think")
        assert msg.parts == [
            {"type": "text", "text": "hi"},
            {"type": "reasoning", "interface": "chat", "text": "think"},
        ]
        assert msg.content == "hi"
        assert msg.reasoning_content == "think"

    def test_tool_call_args_kept_as_raw_string(self):
        raw = '{"ref": "$MEM.inputs.next"}'
        msg = AssistantMessage(tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "memory_read", "arguments": raw},
        }])
        assert msg.parts == [{"type": "tool_call", "call_id": "c1",
                              "name": "memory_read", "args": raw}]
        assert msg.tool_calls[0]["function"]["arguments"] == raw

    def test_tool_result_lives_in_tool_role(self):
        conv = _conv([
            SystemMessage("s"),
            AssistantMessage(tool_calls=[{"id": "c1", "type": "function",
                                          "function": {"name": "memory_read",
                                                       "arguments": "{}"}}]),
            ToolMessage(content="result", tool_call_id="c1"),
        ])
        stored = conv.messages[-1]
        assert stored.role == "tool"
        assert stored.parts == [{"type": "tool_result", "call_id": "c1",
                                 "content": "result"}]
        assert stored.content == "result"

    def test_multiple_system_messages_kept_but_chat_projection_merges_leading(self):
        conv = _conv([
            SystemMessage("a"), SystemMessage("b"), UserMessage("u"),
            AssistantMessage("ok"),
        ])
        assert [m.role for m in conv.messages] == ["system", "system", "user", "assistant"]
        api = conv.to_api_messages()
        assert api[0] == {"role": "system", "content": "a\n\nb"}
        assert [m["role"] for m in api] == ["system", "user", "assistant"]


class TestIRRoundtrip:
    def test_checkpoint_roundtrip_parts_and_chat_projection(self):
        conv = _conv([
            SystemMessage("s"),
            UserMessage("u"),
            AssistantMessage("", reasoning_content="think",
                             tool_calls=[{"id": "c1", "type": "function",
                                          "function": {"name": "memory_read",
                                                       "arguments": '{"ref": "$MEM"}'}}]),
            ToolMessage(content="{}", tool_call_id="c1"),
            AssistantMessage("done", reasoning_content=None),
        ])
        restored = _roundtrip(conv)
        assert [m.role for m in restored.messages] == [m.role for m in conv.messages]
        assert [m.parts for m in restored.messages] == [m.parts for m in conv.messages]
        assert restored.to_api_messages() == conv.to_api_messages()

    def test_reasoning_none_and_empty_string_survive_roundtrip(self):
        conv = _conv([AssistantMessage("a", reasoning_content=None),
                      AssistantMessage("b", reasoning_content="")])
        restored = _roundtrip(conv)
        assert restored.messages[0].reasoning_content is None
        assert restored.messages[1].reasoning_content == ""
        assert restored.to_api_messages() == conv.to_api_messages()

    def test_provider_and_foreign_reasoning_roundtrip_but_not_projected(self):
        msg = AssistantMessage(parts=[
            text_part("answer"),
            reasoning_part("anthropic", text="", opaque="sig-123"),
            provider_part("openai", "web_search_call", {"id": "ws_1"}),
        ])
        conv = _conv([SystemMessage("s"), UserMessage("u"), msg])
        restored = _roundtrip(conv)
        assert restored.messages[-1].parts == msg.parts
        projected = restored.to_api_messages()[-1]
        assert projected == {"role": "assistant", "content": "answer",
                             "reasoning_content": ""}

    def test_old_chat_shape_rejected(self):
        with pytest.raises(TypeError, match="parts"):
            message_from_dict({"role": "system", "content": "s"})

    def test_serialization_is_deterministic(self):
        msg = AssistantMessage(parts=[
            {"type": "reasoning", "text": "t", "interface": "chat", "opaque": "o"},
            text_part("x"),
            {"type": "tool_call", "name": "n", "id_extra": 1,
             "call_id": "c1", "args": "{}"},
        ])
        a = message_to_dict(msg)
        b = message_to_dict(msg)
        assert a == b
        assert a["parts"][0] == {"type": "reasoning", "interface": "chat",
                                 "text": "t", "opaque": "o"}
        assert list(a["parts"][2])[:4] == ["type", "call_id", "name", "args"]

    def test_checkpoint_version_is_ir_version(self):
        assert CHECKPOINT_VERSION == 3

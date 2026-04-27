"""messages 模块单元测试"""

import pytest
from avm.messages import (
    Role, ToolCall, ToolCallResponse,
    Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage,
    ConversationMessage, Conversation, UserMessageBatch,
)


class TestConversationMessage:
    def test_from_any_dict(self):
        msg = ConversationMessage.from_any({"role": "user", "content": "hello"})
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_from_any_dict_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        msg = ConversationMessage.from_any({"role": "assistant", "content": "", "tool_calls": tc})
        assert msg.tool_calls == tc

    def test_from_any_tuple(self):
        msg = ConversationMessage.from_any(("system", "sys"))
        assert msg.role == "system"
        assert msg.content == "sys"

    def test_from_any_message_object(self):
        msg = ConversationMessage.from_any(UserMessage(content="hi"))
        assert msg.role == "user"
        assert msg.content == "hi"

    def test_to_dict_basic(self):
        msg = ConversationMessage(role="user", content="hi")
        assert msg.to_dict() == {"role": "user", "content": "hi"}

    def test_to_dict_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        msg = ConversationMessage(role="assistant", content="ok", tool_calls=tc)
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "ok"
        assert d["tool_calls"] == tc


class TestConversation:
    def test_from_any_list(self):
        conv = Conversation.from_any_list([
            ("system", "sys"),
            ("user", "usr"),
        ])
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"

    def test_from_any_list_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        conv = Conversation.from_any_list([
            {"role": "assistant", "content": "", "tool_calls": tc},
        ])
        assert conv.messages[0].tool_calls == tc

    def test_validate_empty_raises(self):
        conv = Conversation()
        with pytest.raises(ValueError):
            conv.validate(require_last_assistant=True)

    def test_validate_empty_ok(self):
        conv = Conversation()
        conv.validate(require_last_assistant=False)  # 不抛异常

    def test_validate_last_not_assistant_raises(self):
        conv = Conversation.from_any_list([("system", "sys"), ("user", "usr")])
        with pytest.raises(ValueError):
            conv.validate()

    def test_merge_system_messages(self):
        conv = Conversation.from_any_list([
            ("system", "s1"),
            ("system", "s2"),
            ("user", "u"),
            ("assistant", "a"),
        ])
        merged = conv.merge_system_messages()
        assert len(merged) == 3
        assert merged[0].role == "system"
        assert merged[0].content == "s1\n\ns2"

    def test_to_api_messages(self):
        conv = Conversation.from_any_list([
            ("system", "s1"),
            ("system", "s2"),
            ("user", "u"),
            ("assistant", "a"),
        ])
        msgs = conv.to_api_messages()
        assert len(msgs) == 3
        assert msgs[0] == {"role": "system", "content": "s1\n\ns2"}

    def test_to_api_messages_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        conv = Conversation.from_any_list([
            ("user", "u"),
            {"role": "assistant", "content": "", "tool_calls": tc},
        ])
        msgs = conv.to_api_messages()
        assert len(msgs) == 2
        assert msgs[1]["tool_calls"] == tc

    def test_append_user_message(self):
        conv = Conversation()
        conv.append_user_message("hi")
        assert conv.messages[-1].role == "user"
        assert conv.messages[-1].content == "hi"

    def test_append_assistant_message_with_tool_calls(self):
        conv = Conversation()
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        conv.append_assistant_message("ok", tool_calls=tc)
        assert conv.messages[-1].role == "assistant"
        assert conv.messages[-1].tool_calls == tc

    def test_get_last_messages(self):
        conv = Conversation.from_any_list([("user", "u1"), ("user", "u2")])
        last = conv.get_last_messages(1)
        assert len(last) == 1
        assert last[0].content == "u2"


class TestUserMessageBatch:
    def test_add_tool_response(self):
        batch = UserMessageBatch()
        batch.add_tool_response("result", "tc1")
        assert len(batch.tool_responses) == 1
        assert batch.tool_responses[0].content == "result"
        assert batch.tool_responses[0].tool_call_id == "tc1"

    def test_add_user_content(self):
        batch = UserMessageBatch()
        batch.add_user_content("hello")
        assert batch.user_contents == ["hello"]

    def test_clear(self):
        batch = UserMessageBatch()
        batch.add_tool_response("r", "tc1")
        batch.add_user_content("u")
        batch.clear()
        assert batch.tool_responses == []
        assert batch.user_contents == []

    def test_to_tool_messages(self):
        batch = UserMessageBatch()
        batch.add_tool_response("r1", "tc1")
        batch.add_tool_response("r2", "tc2")
        msgs = batch.to_tool_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "tool", "content": "r1", "tool_call_id": "tc1"}

    def test_get_user_content(self):
        batch = UserMessageBatch()
        batch.add_user_content("a")
        batch.add_user_content("b")
        assert batch.get_user_content() == "a\n\nb"

    def test_from_any_list(self):
        batch = UserMessageBatch.from_any_list([("r1", "tc1"), "u1", ("r2", "tc2")])
        assert len(batch.tool_responses) == 2
        assert batch.tool_responses[0].content == "r1"
        assert batch.user_contents == ["u1"]


class TestAssistantMessage:
    def test_to_dict_no_tool_calls(self):
        msg = AssistantMessage(content="hi")
        assert msg.to_dict() == {"role": "assistant", "content": "hi"}

    def test_to_dict_with_tool_calls(self):
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="1", type="function", name="read", arguments={"ref": "x"})]
        )
        d = msg.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["id"] == "1"

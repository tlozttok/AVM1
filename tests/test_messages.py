"""types 模块单元测试"""

import pytest
from avm.types import (
    Role, Message,
    SystemMessage, UserMessage, AssistantMessage, ToolMessage,
    message_to_api_dict,
    Conversation, UserMessageBatch,
)


class TestMessageToApiDict:
    def test_system_message(self):
        d = message_to_api_dict(SystemMessage(content="sys"))
        assert d == {"role": "system", "content": "sys"}

    def test_user_message(self):
        d = message_to_api_dict(UserMessage(content="hi"))
        assert d == {"role": "user", "content": "hi"}

    def test_assistant_no_tool_calls(self):
        d = message_to_api_dict(AssistantMessage(content="ok"))
        assert d == {"role": "assistant", "content": "ok"}

    def test_assistant_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        d = message_to_api_dict(AssistantMessage(content="", tool_calls=tc))
        assert d["role"] == "assistant"
        assert d["content"] == ""
        assert d["tool_calls"] == tc

    def test_tool_message(self):
        d = message_to_api_dict(ToolMessage(content="result", tool_call_id="tc1"))
        assert d == {"role": "tool", "content": "result", "tool_call_id": "tc1"}


class TestConversation:
    def test_from_any_list_tuples(self):
        conv = Conversation.from_any_list([
            ("system", "sys"),
            ("user", "usr"),
        ])
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"

    def test_from_any_list_dict_with_tool_calls(self):
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
        conv.validate(require_last_assistant=False)

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

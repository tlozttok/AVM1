"""类型系统测试"""

import pytest
from avm.types import (
    MetaList, MetaDict,
    Message, SystemMessage, UserMessage, AssistantMessage, ToolMessage,
    message_to_api_dict,
    Conversation, UserMessageBatch,
    Role,
)


class TestMetaList:
    def test_basic(self):
        ml = MetaList(data=[1, 2, 3], metadata="test")
        assert len(ml) == 3
        assert ml[0] == 1
        assert 2 in ml
        assert ml.get_metadata() == "test"

    def test_append(self):
        ml = MetaList()
        ml.append("a")
        assert ml[0] == "a"

    def test_iter(self):
        ml = MetaList(data=[1, 2])
        assert list(ml) == [1, 2]

    def test_to_llm_string(self):
        ml = MetaList(data=[1, 2], metadata="desc")
        assert 'list[len=2' in ml.to_llm_string()
        assert 'desc' in ml.to_llm_string()

    def test_to_list(self):
        inner = MetaDict(data={"k": "v"})
        ml = MetaList(data=[1, inner])
        result = ml.to_list()
        assert result == [1, {"k": "v"}]

    def test_eq(self):
        a = MetaList(data=[1, 2])
        b = MetaList(data=[1, 2])
        assert a == b


class TestMetaDict:
    def test_basic(self):
        md = MetaDict(data={"a": 1, "b": 2}, metadata="desc")
        assert md["a"] == 1
        assert "a" in md

    def test_set_and_delete(self):
        md = MetaDict()
        md["x"] = "y"
        assert md["x"] == "y"
        del md["x"]
        assert "x" not in md

    def test_copy(self):
        md = MetaDict(data={"k": "v"})
        cp = md.copy()
        cp["k"] = "changed"
        assert md["k"] == "v"

    def test_to_llm_string(self):
        md = MetaDict(data={"a": 1}, metadata="info")
        assert 'dict[' in md.to_llm_string()
        assert 'info' in md.to_llm_string()

    def test_to_dict(self):
        inner = MetaList(data=[MetaDict(data={"nested": True})])
        md = MetaDict(data={"outer": inner})
        result = md.to_dict()
        assert result == {"outer": [{"nested": True}]}


class TestMessageToApiDict:
    def test_system(self):
        d = message_to_api_dict(SystemMessage(content="sys"))
        assert d == {"role": "system", "content": "sys"}

    def test_user(self):
        d = message_to_api_dict(UserMessage(content="hi"))
        assert d == {"role": "user", "content": "hi"}

    def test_assistant_no_tool_calls(self):
        d = message_to_api_dict(AssistantMessage(content="ok"))
        assert d == {"role": "assistant", "content": "ok"}

    def test_assistant_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        d = message_to_api_dict(AssistantMessage(content="", tool_calls=tc))
        assert d["tool_calls"] == tc

    def test_tool_message(self):
        d = message_to_api_dict(ToolMessage(content="result", tool_call_id="tc1"))
        assert d == {"role": "tool", "content": "result", "tool_call_id": "tc1"}


class TestConversationLifecycle:
    def test_root_conversation(self):
        conv = Conversation(
            messages=[SystemMessage(content="sys"), UserMessage(content="usr"), AssistantMessage(content="a")],
            cid=0, is_sub=False, parent=None, is_root=True,
        )
        assert conv.cid == 0
        assert conv.is_root == True
        assert conv.is_sub == False
        assert conv.parent is None

    def test_sub_conversation(self):
        root = Conversation(
            messages=[SystemMessage(content="r"), UserMessage(content="u"), AssistantMessage(content="a")],
            cid=0, is_root=True,
        )
        sub = Conversation(
            messages=[SystemMessage(content="s"), UserMessage(content="u"), AssistantMessage(content="a")],
            cid=1, is_sub=True, parent=root,
        )
        assert sub.is_sub == True
        assert sub.parent is root
        assert sub.cid == 1

    def test_validate_empty_raises(self):
        conv = Conversation(messages=[], is_root=True)
        with pytest.raises(ValueError):
            conv.validate(require_last_assistant=True)

    def test_validate_last_not_assistant_raises(self):
        conv = Conversation(
            messages=[SystemMessage(content="sys"), UserMessage(content="usr")],
            is_root=True,
        )
        with pytest.raises(ValueError):
            conv.validate(require_last_assistant=True)

    def test_validate_empty_skip(self):
        conv = Conversation(messages=[], is_root=True)
        conv.validate(require_last_assistant=False)  # ok

        conv2 = Conversation(
            messages=[SystemMessage(content="s"), UserMessage(content="u"), AssistantMessage(content="a")],
            is_root=True,
        )
        conv2.validate(require_last_assistant=True)  # ok, last is assistant

    def test_user_batch_exists(self):
        conv = Conversation(
            messages=[SystemMessage(content="s"), UserMessage(content="u"), AssistantMessage(content="a")],
        )
        assert conv.user_batch is not None
        assert isinstance(conv.user_batch, UserMessageBatch)


class TestConversationFromAnyList:
    def test_tuples(self):
        conv = Conversation.from_any_list([
            ("system", "sys"),
            ("user", "usr"),
            ("assistant", "a"),
        ])
        assert len(conv.messages) == 3
        assert conv.messages[0].role == "system"

    def test_dict_with_tool_calls(self):
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        conv = Conversation.from_any_list([
            {"role": "assistant", "content": "", "tool_calls": tc},
        ])
        assert conv.messages[0].tool_calls == tc

    def test_tool_message_from_dict(self):
        conv = Conversation.from_any_list([
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "a"},
        ])
        tool_msg = conv.messages[1]
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "tc1"


class TestConversationApi:
    def test_merge_system_messages(self):
        conv = Conversation.from_any_list([
            ("system", "s1"),
            ("system", "s2"),
            ("user", "u"),
            ("assistant", "a"),
        ])
        merged = conv.merge_system_messages()
        # 两个 system 合并，加上 user + assistant
        assert len(merged) == 3
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

    def test_append_user_message(self):
        conv = Conversation.from_any_list([
            ("system", "s"), ("user", "u"), ("assistant", "a"),
        ])
        conv.append_user_message("hello")
        assert conv.messages[-1].role == "user"
        assert conv.messages[-1].content == "hello"

    def test_append_assistant_with_tool_calls(self):
        conv = Conversation.from_any_list([
            ("system", "s"), ("user", "u"), ("assistant", "a"),
        ])
        tc = [{"id": "1", "type": "function", "function": {"name": "read"}}]
        conv.append_assistant_message("ok", tool_calls=tc)
        assert conv.messages[-1].role == "assistant"
        assert conv.messages[-1].tool_calls == tc

    def test_append_tool_message(self):
        conv = Conversation.from_any_list([
            ("system", "s"), ("user", "u"), ("assistant", "a"),
        ])
        conv.append_tool_message("result", "tc1")
        assert conv.messages[-1].role == "tool"
        assert conv.messages[-1].tool_call_id == "tc1"

    def test_get_last_messages(self):
        conv = Conversation.from_any_list([
            ("user", "u1"), ("user", "u2"), ("assistant", "a"),
        ])
        last = conv.get_last_messages(2)
        assert len(last) == 2
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

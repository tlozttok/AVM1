"""API 错误分类与对话设备 context 字段的纯离线用例。"""

import json

import pytest

from avm.api_errors import classify_api_error
from avm.core import Core
from avm.exceptions import APIError, IllegalOperationError


def _error(message="", status_code=None):
    return APIError(message, status_code=status_code)


class TestClassifyApiError:
    def test_http_402_is_insufficient_balance(self):
        category, detail = classify_api_error(_error("Payment Required"), status_code=402)
        assert category == "insufficient_balance"
        assert detail["category"] == "insufficient_balance"
        assert detail["status_code"] == 402
        assert detail["message"] == "Payment Required"
        assert detail["context_limit"] is None

    @pytest.mark.parametrize("message", [
        "insufficient_quota",
        "Insufficient Balance: top up your account",
        "账户余额不足",
        "billing hard limit reached",
        "payment required",
        "no credit remaining",
    ])
    def test_balance_markers_in_message(self, message):
        category, detail = classify_api_error(_error(message))
        assert category == "insufficient_balance", message
        assert detail["category"] == "insufficient_balance"

    def test_429_rate_limit_is_other(self):
        """限流不是余额不足：429 且没有余额标记时归 other。"""
        category, detail = classify_api_error(
            _error("Rate limit reached for requests"), status_code=429
        )
        assert category == "other"
        assert detail["status_code"] == 429

    def test_429_insufficient_quota_still_balance(self):
        """OpenAI 用 429 报配额耗尽：消息标记优先于状态码。"""
        category, _ = classify_api_error(
            _error("You exceeded your current quota: insufficient_quota"), status_code=429
        )
        assert category == "insufficient_balance"

    @pytest.mark.parametrize("status", [400, 413, 422])
    @pytest.mark.parametrize("message", [
        "context_length_exceeded",
        "This model's maximum context length is 128000 tokens",
        "the context window is full",
        "too many tokens in request",
        "input exceeds the maximum allowed size",
        "上下文超过限制",
        "token limit reached",
    ])
    def test_context_limit_markers(self, status, message):
        category, detail = classify_api_error(_error(message), status_code=status)
        assert category == "context_limit", (status, message)
        assert detail["category"] == "context_limit"

    def test_context_limit_parses_limit(self):
        category, detail = classify_api_error(
            _error("This model's maximum context length is 128000 tokens. "
                   "However, your messages resulted in 130001 tokens."),
            status_code=400,
        )
        assert category == "context_limit"
        assert detail["context_limit"] == 128000

    def test_context_limit_parses_commas(self):
        _category, detail = classify_api_error(
            _error("maximum context length is 1,048,576 tokens"), status_code=400
        )
        assert detail["context_limit"] == 1048576

    def test_context_limit_without_parsable_number(self):
        category, detail = classify_api_error(
            _error("context_length_exceeded: please reduce the length"), status_code=400
        )
        assert category == "context_limit"
        assert detail["context_limit"] is None

    def test_context_marker_needs_context_status(self):
        """500/429 上的 context 字样不算上下文超限：状态码是判定的一部分。"""
        for status in (429, 500):
            category, _ = classify_api_error(
                _error("context_length_exceeded"), status_code=status
            )
            assert category == "other", status

    def test_generic_error_is_other(self):
        category, detail = classify_api_error(
            _error("ConnectionError: network down"), status_code=503
        )
        assert category == "other"
        assert detail == {
            "category": "other",
            "message": "ConnectionError: network down",
            "status_code": 503,
            "context_limit": None,
        }

    def test_status_code_from_exception_attributes(self):
        exc = APIError("Payment Required", status_code=402)
        category, detail = classify_api_error(exc)
        assert category == "insufficient_balance"
        assert detail["status_code"] == 402

    def test_status_code_from_response_attribute(self):
        class Response:
            status_code = 422

        class Error(Exception):
            response = Response()

            def __str__(self):
                return "maximum context length is 8192 tokens"

        category, detail = classify_api_error(Error())
        assert category == "context_limit"
        assert detail["status_code"] == 422
        assert detail["context_limit"] == 8192


class TestLMUClassification:
    def test_lmu_exec_attaches_category_and_keeps_text(self):
        """真实 LMU.exec：捕获外部异常时保留 status_code/category，文本格式不变。"""
        from avm.core import LMU
        from avm.types import MetaDict

        class ProviderError(Exception):
            status_code = 400

            def __str__(self):
                return "maximum context length is 8192 tokens"

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                raise ProviderError()

        class FakeChat:
            completions = FakeCompletions

        class FakeClient:
            chat = FakeChat

        lmu = LMU()
        lmu._client = FakeClient
        core = Core()
        conv = core.start("sys", "hi")
        para = MetaDict(data={"model": "m"}, ctrl={"type": "para"})

        with pytest.raises(APIError) as excinfo:
            lmu.exec(conv, para)
        error = excinfo.value
        assert str(error) == "ProviderError: maximum context length is 8192 tokens"
        assert error.category == "context_limit"
        assert error.status_code == 400
        assert error.detail["context_limit"] == 8192
        assert lmu.last_call["error"] == str(error)


class TestConversationsContextField:
    def test_normal_conversation_has_empty_context_error(self):
        core = Core()
        core.start("system", "hello")
        data = json.loads(core.unwrap("$MEM.conversations.0.context", for_llm=False))
        assert data == {
            "messages": 2,
            "chars": 11,
            "tool_messages": 0,
            "last_error": None,
            "context_limit": None,
        }

    def test_context_counts_tool_messages_and_last_error(self):
        core = Core()
        conv = core.start("system", "hello")
        conv.append_tool_message("ok", "c1")
        conv.last_api_error = {
            "category": "context_limit",
            "message": "maximum context length is 8192 tokens",
            "status_code": 400,
            "context_limit": 8192,
            "at": "2026-11-12T12:00:00+0800",
        }
        data = json.loads(core.unwrap("$MEM.conversations.0.context", for_llm=False))
        assert data["messages"] == 3
        assert data["chars"] == 13
        assert data["tool_messages"] == 1
        assert data["context_limit"] == 8192
        assert data["last_error"] == conv.last_api_error

    def test_context_is_read_only(self):
        core = Core()
        core.start("system", "hello")
        with pytest.raises(IllegalOperationError):
            core.mem.set("$MEM.conversations.0.context", "{}")


class TestConversationsMessagesField:
    def test_transcript_is_readable(self):
        core = Core()
        conv = core.start("system", "hello")
        conv.append_assistant_message("answer")
        data = json.loads(
            core.unwrap("$MEM.conversations.0.messages", for_llm=False)
        )
        assert data == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer", "reasoning_content": ""},
        ]

    def test_transcript_keeps_tool_call_shapes(self):
        core = Core()
        conv = core.start("system", "hello")
        conv.append_assistant_message("", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "memory_read", "arguments": "{}"},
        }])
        conv.append_tool_message("ok", "c1")
        data = json.loads(
            core.unwrap("$MEM.conversations.0.messages", for_llm=False)
        )
        assert data[-2]["tool_calls"][0]["id"] == "c1"
        assert data[-1] == {"role": "tool", "content": "ok", "tool_call_id": "c1"}

    def test_transcript_is_read_only(self):
        core = Core()
        core.start("system", "hello")
        with pytest.raises(IllegalOperationError):
            core.mem.set("$MEM.conversations.0.messages", "[]")

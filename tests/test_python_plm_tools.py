"""plm.python 的生成器工具通道：yield from 挂起、从断点继续、不重放。

不调真实 API：Python 对话不经过模型；需要 LLM 子对话的用例装 MockLMU。
"""

import json

import pytest

from avm.core import Core, _instruction_registry
from avm.image import load_image
from avm.python_server import PYTHON_TOOL_SPECS, create_plm
from avm.sandbox import SecurityError


class MockLMU:
    """够用的假执行器：返回固定文本，不产生工具调用；历史提交与真 LMU 一致。"""

    def __init__(self, text="好的"):
        self.text = text
        self.calls = []

    def exec(self, conversation, para):
        self.calls.append(para)
        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            conversation.append_user_message(user_content)
        conversation.append_assistant_message(self.text)
        return self.text, [], conversation


def _write(tmp_path, image):
    path = tmp_path / "image.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _plm_image(program, user_code="yield from pack()", extra=None):
    mem = {
        "packer": {
            "kind": "dict",
            "ctrl": {"type": "python"},
            "value": {
                "content": {"kind": "str", "value": program},
                "para_ref": {"kind": "str", "value": "$MEM.packer_params"},
            },
        },
        "packer_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "plm.python"}},
        },
        "src": {"kind": "str", "value": "你好"},
        "dst": {"kind": "str", "value": ""},
        "user": {"kind": "str", "value": user_code},
        "model_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
    }
    mem.update(extra or {})
    return {
        "meta": {"version": 1},
        "mem": mem,
        "para_ref": "$MEM.model_params",
        "init": {"name": "packer", "system_ref": "$MEM.packer", "user_ref": "$MEM.user"},
    }


class TestToolChannel:
    def _plm(self, setup):
        plm = create_plm("plm.python")
        plm.handle_messages([{"role": "system", "content": setup}], {})
        return plm

    def test_tool_set_matches_instruction_registry(self):
        assert set(PYTHON_TOOL_SPECS) == set(_instruction_registry())

    def test_single_call_suspends_then_resumes(self):
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "text = yield from memory_read('$MEM.a')\n"
                "print('得到:' + str(text))"
            ),
        }]
        first = plm.handle_messages(list(messages), {})
        assert first["content"] == ""
        (call,) = first["tool_calls"]
        assert call["function"]["name"] == "memory_read"
        assert json.loads(call["function"]["arguments"]) == {"ref": "$MEM.a"}

        messages.append({"role": "assistant", "content": "", "tool_calls": first["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "一条内容"})
        assert plm.handle_messages(messages, {}) == {"content": "得到:一条内容\n"}

    def test_sequential_calls_resume_in_order(self):
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "a = yield from memory_read('$MEM.a')\n"
                "b = yield from memory_read('$MEM.b')\n"
                "print(str(a) + '/' + str(b))"
            ),
        }]
        refs = []
        for response in ["A", "B"]:
            out = plm.handle_messages(list(messages), {})
            (call,) = out["tool_calls"]
            refs.append(json.loads(call["function"]["arguments"])["ref"])
            messages.append({"role": "assistant", "content": out["content"],
                             "tool_calls": out["tool_calls"]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": response})
        assert refs == ["$MEM.a", "$MEM.b"]
        assert plm.handle_messages(messages, {}) == {"content": "A/B\n"}

    def test_prefix_is_not_replayed(self):
        """前缀只执行一次：k 次工具调用不会把前缀副作用重复 k 次。"""
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "log = []\n"
                "for i in range(3):\n"
                "    log.append(i)\n"
                "    yield from memory_read('$MEM.a')\n"
                "print(len(log))"
            ),
        }]
        for _ in range(3):
            out = plm.handle_messages(list(messages), {})
            (call,) = out["tool_calls"]
            messages.append({"role": "assistant", "content": out["content"],
                             "tool_calls": out["tool_calls"]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "x"})
        assert plm.handle_messages(messages, {}) == {"content": "3\n"}

    def test_stdout_before_suspension_is_returned_once(self):
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "print('前缀')\n"
                "text = yield from memory_read('$MEM.a')\n"
                "print('后缀:' + str(text))"
            ),
        }]
        first = plm.handle_messages(list(messages), {})
        assert first["content"] == "前缀\n"
        (call,) = first["tool_calls"]
        messages.append({"role": "assistant", "content": first["content"],
                         "tool_calls": first["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "X"})
        assert plm.handle_messages(messages, {}) == {"content": "后缀:X\n"}

    def test_error_response_reaches_program(self):
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "text = yield from memory_read('$MEM.ghost')\n"
                "print('响应:' + str(text))"
            ),
        }]
        out = plm.handle_messages(list(messages), {})
        (call,) = out["tool_calls"]
        messages.append({"role": "assistant", "content": out["content"],
                         "tool_calls": out["tool_calls"]})
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "Error: 键 'ghost' 不存在",
        })
        assert plm.handle_messages(messages, {}) == {"content": "响应:Error: 键 'ghost' 不存在\n"}

    def test_return_value_is_appended_to_output(self):
        plm = self._plm("")
        messages = [{
            "role": "user",
            "content": (
                "text = yield from memory_read('$MEM.a')\n"
                "print('收到')\n"
                "return '值:' + str(text)"
            ),
        }]
        out = plm.handle_messages(list(messages), {})
        (call,) = out["tool_calls"]
        messages.append({"role": "assistant", "content": out["content"],
                         "tool_calls": out["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "A"})
        assert plm.handle_messages(messages, {}) == {"content": "收到\n值:A"}

    def test_new_user_message_abandons_suspended_program(self):
        plm = self._plm("")
        first = plm.handle_messages(
            [{"role": "user", "content": "yield from memory_read('$MEM.a')"}], {}
        )
        assert first["tool_calls"]
        assert plm.handle_messages([{"role": "user", "content": "print('新程序')"}], {}) == {
            "content": "新程序\n"
        }

    def test_tool_response_mismatch_is_error(self):
        plm = self._plm("")
        out = plm.handle_messages(
            [{"role": "user", "content": "yield from memory_read('$MEM.a')"}], {}
        )
        (call,) = out["tool_calls"]
        messages = [
            {"role": "user", "content": "yield from memory_read('$MEM.a')"},
            {"role": "assistant", "content": out["content"], "tool_calls": out["tool_calls"]},
            {"role": "tool", "tool_call_id": "wrong-id", "content": "x"},
        ]
        result = plm.handle_messages(messages, {})
        assert result["content"].startswith("错误: PythonPLM: 工具响应与挂起的调用不匹配")

    def test_setup_code_cannot_call_tools(self):
        with pytest.raises(SecurityError, match="setup"):
            create_plm("plm.python").handle_messages(
                [{"role": "system", "content": "memory_read('$MEM.a')"}], {}
            )

    def test_direct_tool_call_without_yield_from_is_rejected(self):
        plm = self._plm("")
        out = plm.handle_messages(
            [{"role": "user", "content": "memory_read('$MEM.a')"}], {}
        )
        assert out["content"].startswith("错误: SecurityError")
        assert "yield from" in out["content"]

    def test_bad_arguments_are_program_errors(self):
        plm = self._plm("")
        out = plm.handle_messages(
            [{"role": "user", "content": "yield from memory_read()"}], {}
        )
        assert out["content"].startswith("错误: TypeError")

    def test_unauthorized_tool_is_rejected(self):
        plm = create_plm("plm.python")
        tools = [{"type": "function", "function": {"name": "memory_read"}}]
        out = plm.handle_messages(
            [{"role": "user", "content": "yield from memory_write('$MEM.a', 'x')"}],
            {"tools": tools},
        )
        assert out["content"].startswith("错误: SecurityError")
        assert "未授权" in out["content"]


class TestThroughCore:
    def test_init_plm_reads_and_writes_memory(self, tmp_path):
        program = (
            "def pack():\n"
            "    text = yield from memory_read('$MEM.src')\n"
            "    yield from memory_write('$MEM.dst', '抄写:' + str(text))\n"
            "    print('完成:' + str(text))"
        )
        core = load_image(Core(), _write(tmp_path, _plm_image(program)))
        core.run()

        assert core.mem["dst"] == "抄写:你好"
        conv = core.get_conversation(0)
        assert conv.messages[-1].content == "完成:你好\n"
        assert 0 in core._dormant_cids

    def test_plm_creates_conversation_and_closes_itself(self, tmp_path):
        """create_cmd / send_instruction / close_conversation 对 Python 程序也可用。"""
        program = (
            "def pack():\n"
            "    resp = yield from create_cmd('$MEM.tester', '$MEM.model_params')\n"
            "    cid = int(str(resp).rsplit('=', 1)[1])\n"
            "    start = yield from memory_read('$MEM.user_text')\n"
            "    yield from send_instruction(cid, str(start))\n"
            "    yield from close_conversation(0)"
        )
        image = _plm_image(
            program,
            extra={
                "tester": {
                    "kind": "dict",
                    "ctrl": {"type": "settingup"},
                    "value": {
                        "name": {"kind": "str", "value": "tester"},
                        "content": {"kind": "str", "value": "你是测试者。"},
                    },
                },
                "user_text": {"kind": "str", "value": "开始测试"},
            },
        )
        core = load_image(Core(), _write(tmp_path, image))
        core.lmu = MockLMU("收到")
        core.run()

        assert 0 in core._finished
        tester = next(c for c in core._conv_by_cid.values() if c.name == "tester")
        assert tester.is_python is False
        assert tester.messages[0].content == "你是测试者。"
        assert tester.messages[1].content == "开始测试"

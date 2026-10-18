"""plm.python 的工具通道：9 个注入工具、调用即挂起、响应回来重放。

不调真实 API：Python 对话不经过模型；需要 LLM 子对话的用例装 MockLMU。
"""

import json

import pytest

from avm.core import Core, _instruction_registry
from avm.image import load_image
from avm.python_server import PYTHON_TOOL_SPECS, create_plm


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


def _plm_image(program, user_code="pack()", extra=None):
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

    def test_call_suspends_then_replays(self):
        plm = self._plm(
            "def go():\n"
            "    text = memory_read('$MEM.a')\n"
            "    print('得到:' + str(text))"
        )
        messages = [{"role": "user", "content": "go()"}]
        first = plm.handle_messages(list(messages), {})
        assert first["content"] == ""
        (call,) = first["tool_calls"]
        assert call["function"]["name"] == "memory_read"
        assert json.loads(call["function"]["arguments"]) == {"ref": "$MEM.a"}

        messages.append({"role": "assistant", "content": "", "tool_calls": first["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "一条内容"})
        assert plm.handle_messages(messages, {}) == {"content": "得到:一条内容\n"}

    def test_sequential_calls_are_not_repeated(self):
        plm = self._plm(
            "def go():\n"
            "    a = memory_read('$MEM.a')\n"
            "    b = memory_read('$MEM.b')\n"
            "    print(str(a) + '/' + str(b))"
        )
        messages = [{"role": "user", "content": "go()"}]
        calls = []
        for index, response in enumerate(["A", "B"]):
            out = plm.handle_messages(list(messages), {})
            (call,) = out["tool_calls"]
            calls.append(json.loads(call["function"]["arguments"])["ref"])
            messages.append({"role": "assistant", "content": "", "tool_calls": out["tool_calls"]})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": response})
        assert calls == ["$MEM.a", "$MEM.b"]
        assert plm.handle_messages(messages, {}) == {"content": "A/B\n"}

    def test_error_response_reaches_program(self):
        plm = self._plm(
            "def go():\n"
            "    text = memory_read('$MEM.ghost')\n"
            "    print('响应:' + str(text))"
        )
        messages = [{"role": "user", "content": "go()"}]
        out = plm.handle_messages(list(messages), {})
        (call,) = out["tool_calls"]
        messages.append({"role": "assistant", "content": "", "tool_calls": out["tool_calls"]})
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "Error: 键 'ghost' 不存在",
        })
        assert plm.handle_messages(messages, {}) == {"content": "响应:Error: 键 'ghost' 不存在\n"}

    def test_replayed_stdout_is_not_repeated(self):
        """重放会重跑前缀代码，但重跑产生的 stdout 不重复返回。"""
        plm = self._plm(
            "def go():\n"
            "    print('前缀')\n"
            "    text = memory_read('$MEM.a')\n"
            "    print('后缀:' + str(text))"
        )
        messages = [{"role": "user", "content": "go()"}]
        out = plm.handle_messages(list(messages), {})
        (call,) = out["tool_calls"]
        messages.append({"role": "assistant", "content": "", "tool_calls": out["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "X"})
        assert plm.handle_messages(messages, {}) == {"content": "后缀:X\n"}

    def test_write_call_returned_to_core_once(self):
        """写入类调用只发出一次：响应回来后重放按记录返回，不再发第二次。"""
        plm = self._plm(
            "def go():\n"
            "    r = memory_write('$MEM.b', 'X')\n"
            "    print('写响应:' + str(r))"
        )
        messages = [{"role": "user", "content": "go()"}]
        first = plm.handle_messages(list(messages), {})
        (call,) = first["tool_calls"]
        assert call["function"]["name"] == "memory_write"
        assert json.loads(call["function"]["arguments"]) == {"ref": "$MEM.b", "content": "X"}
        messages.append({"role": "assistant", "content": "", "tool_calls": first["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Success"})
        assert plm.handle_messages(messages, {}) == {"content": "写响应:Success\n"}

    def test_setup_code_cannot_call_tools(self):
        with pytest.raises(RuntimeError, match="setup"):
            create_plm("plm.python").handle_messages(
                [{"role": "system", "content": "memory_read('$MEM.a')"}], {}
            )

    def test_bad_arguments_are_program_errors(self):
        plm = self._plm("def go():\n    memory_read()")
        out = plm.handle_messages([{"role": "user", "content": "go()"}], {})
        assert out["content"].startswith("错误: TypeError")
        assert "tool_calls" not in out or not out["tool_calls"]


class TestThroughCore:
    def test_init_plm_reads_and_writes_memory(self, tmp_path):
        program = (
            "def pack():\n"
            "    text = memory_read('$MEM.src')\n"
            "    memory_write('$MEM.dst', '抄写:' + str(text))\n"
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
            "    resp = create_cmd('$MEM.tester', '$MEM.model_params')\n"
            "    cid = int(str(resp).rsplit('=', 1)[1])\n"
            "    start = memory_read('$MEM.user_text')\n"
            "    send_instruction(cid, str(start))\n"
            "    close_conversation(0)"
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

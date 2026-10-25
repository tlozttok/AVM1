from avm.exceptions import MemoryKeyNotFoundError
"""plm.command 测试：字符串 IO 命令行、生成器工具调用往返、与 Core 集成。"""

import json

from avm.core import Core
from avm.python_server import create_plm, PLM_REGISTRY, CommandPLM
from avm.types import MetaDict


def _seed(mem, ref, value, owner_cid=None):
    """测试夹具：先 make 建链再写（memory_write 不再建节点）。"""
    try:
        return mem.set(ref, value, owner_cid=owner_cid)
    except MemoryKeyNotFoundError:
        parent, _, key = ref.rpartition(".")
        mem.make(parent, key, "str", owner_cid=owner_cid)
        return mem.set(ref, value, owner_cid=owner_cid)


def _params(*tool_names):
    return {"tools": [{"type": "function", "function": {"name": name}} for name in tool_names]}


def _history(*messages):
    return [{"role": role, "content": content} for role, content in messages]


def _read_x_program():
    return (
        "def handle(cmd):\n"
        "    content = yield from memory_read('$MEM.x')\n"
        "    return 'got:' + content\n"
    )


class TestCommandPLMStringIO:
    def test_plain_string_in_string_out(self):
        plm = create_plm("plm.command")
        h = _history(("system", "def handle(cmd):\n    return cmd.upper()"))
        assert plm.handle_messages(list(h), {}) == {"content": ""}
        h.append({"role": "user", "content": "put a 1"})
        assert plm.handle_messages(list(h), {}) == {"content": "PUT A 1"}

    def test_program_parses_command_and_args(self):
        setup = (
            "def handle(cmd):\n"
            "    op, _, rest = cmd.partition(' ')\n"
            "    return op + '|' + rest\n"
        )
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", setup), ("user", "PUT a 1")), {})
        assert r == {"content": "PUT|a 1"}

    def test_no_cross_command_state(self):
        """每条命令全新命名空间：跨命令状态必须走 AVM 内存。"""
        setup = (
            "n = 0\n"
            "def handle(cmd):\n"
            "    global n\n"
            "    n += 1\n"
            "    return str(n)\n"
        )
        plm = create_plm("plm.command")
        h = _history(("system", setup), ("user", "bump"))
        assert plm.handle_messages(list(h), {})["content"] == "1"
        h.append({"role": "assistant", "content": "1"})
        h.append({"role": "user", "content": "bump"})
        assert plm.handle_messages(list(h), {})["content"] == "1"

    def test_none_return_is_empty_output(self):
        plm = create_plm("plm.command")
        r = plm.handle_messages(
            _history(("system", "def handle(cmd):\n    pass"), ("user", "x")), {}
        )
        assert r == {"content": ""}

    def test_non_string_return_is_error(self):
        plm = create_plm("plm.command")
        r = plm.handle_messages(
            _history(("system", "def handle(cmd):\n    return 42"), ("user", "x")), {}
        )
        assert "错误: CommandPLM" in r["content"]
        assert "必须返回字符串" in r["content"]

    def test_print_discarded(self):
        setup = "def handle(cmd):\n    print('secret')\n    return 'ok'"
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", setup), ("user", "x")), {})
        assert r == {"content": "ok"}

    def test_sandbox_blocks_import(self):
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", "import os"), ("user", "x")), {})
        assert "错误: CommandPLM" in r["content"]
        assert "程序编译失败" in r["content"]

    def test_missing_handle_is_error(self):
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", "x = 1"), ("user", "x")), {})
        assert "handle" in r["content"]

    def test_runtime_error_in_handler(self):
        plm = create_plm("plm.command")
        r = plm.handle_messages(
            _history(("system", "def handle(cmd):\n    return 1/0"), ("user", "x")), {}
        )
        assert "ZeroDivisionError" in r["content"]


class TestCommandPLMToolCalls:
    def test_single_tool_roundtrip(self):
        plm = create_plm("plm.command")
        tools = _params("memory_read")
        h = _history(("system", _read_x_program()))
        plm.handle_messages(list(h), tools)

        h.append({"role": "user", "content": "read x"})
        r = plm.handle_messages(list(h), tools)
        assert r["content"] == ""
        tool_call = r["tool_calls"][0]
        assert tool_call["id"] == "cmd-0"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "memory_read"
        assert json.loads(tool_call["function"]["arguments"]) == {"ref": "$MEM.x"}

        h.append({"role": "tool", "content": "42", "tool_call_id": "cmd-0"})
        assert plm.handle_messages(list(h), tools) == {"content": "got:42"}

    def test_multi_step_read_then_write(self):
        setup = (
            "def handle(cmd):\n"
            "    cur = yield from memory_read('$MEM.n')\n"
            "    yield from memory_write('$MEM.n', cur + '!')\n"
            "    return 'done:' + cur\n"
        )
        plm = create_plm("plm.command")
        tools = _params("memory_read", "memory_write")
        h = _history(("system", setup))
        plm.handle_messages(list(h), tools)

        h.append({"role": "user", "content": "bump"})
        r1 = plm.handle_messages(list(h), tools)
        assert r1["tool_calls"][0]["id"] == "cmd-0"
        assert r1["tool_calls"][0]["function"]["name"] == "memory_read"

        h.append({"role": "tool", "content": "7", "tool_call_id": "cmd-0"})
        r2 = plm.handle_messages(list(h), tools)
        assert r2["tool_calls"][0]["id"] == "cmd-1"
        assert r2["tool_calls"][0]["function"]["name"] == "memory_write"
        assert json.loads(r2["tool_calls"][0]["function"]["arguments"]) == {
            "ref": "$MEM.n",
            "content": "7!",
        }

        h.append({"role": "tool", "content": "Success set: $MEM.n", "tool_call_id": "cmd-1"})
        assert plm.handle_messages(list(h), tools) == {"content": "done:7"}

    def test_tool_error_text_returned_to_program(self):
        setup = (
            "def handle(cmd):\n"
            "    content = yield from memory_read('$MEM.missing')\n"
            "    return 'err=' + content\n"
        )
        plm = create_plm("plm.command")
        tools = _params("memory_read")
        h = _history(("system", setup), ("user", "go"))
        r = plm.handle_messages(list(h), tools)
        h.append({"role": "tool", "content": "Error: 路径不存在", "tool_call_id": r["tool_calls"][0]["id"]})
        assert plm.handle_messages(list(h), tools) == {"content": "err=Error: 路径不存在"}

    def test_all_five_tools_yield_expected_arguments(self):
        cases = {
            "memory_write": ("memory_write('$MEM.a', 'v')", {"ref": "$MEM.a", "content": "v"}),
            "memory_make": ("memory_make('$MEM.a', 'k', 'str')", {"ref": "$MEM.a", "key": "k", "mem_type": "str"}),
            "send_instruction": ("send_instruction(2, 'hi')", {"cid": 2, "content": "hi"}),
            "close_conversation": ("close_conversation(3)", {"cid": 3}),
        }
        for name, (call, expected) in cases.items():
            plm = create_plm("plm.command")
            setup = f"def handle(cmd):\n    return (yield from {call})\n"
            r = plm.handle_messages(_history(("system", setup), ("user", "go")), _params(name))
            tool_call = r["tool_calls"][0]
            assert tool_call["function"]["name"] == name
            assert json.loads(tool_call["function"]["arguments"]) == expected

    def test_unknown_tool_yield_rejected(self):
        setup = "def handle(cmd):\n    return (yield ('create_cmd', {}))\n"
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", setup), ("user", "go")), {})
        assert "非法工具调用" in r["content"]
        assert r.get("tool_calls") is None

    def test_malformed_yield_rejected(self):
        setup = "def handle(cmd):\n    return (yield 'memory_read')\n"
        plm = create_plm("plm.command")
        r = plm.handle_messages(_history(("system", setup), ("user", "go")), {})
        assert "非法工具调用" in r["content"]

    def test_error_after_resume(self):
        setup = (
            "def handle(cmd):\n"
            "    yield from memory_read('$MEM.x')\n"
            "    return 1/0\n"
        )
        plm = create_plm("plm.command")
        tools = _params("memory_read")
        h = _history(("system", setup), ("user", "go"))
        r = plm.handle_messages(list(h), tools)
        h.append({"role": "tool", "content": "1", "tool_call_id": r["tool_calls"][0]["id"]})
        out = plm.handle_messages(list(h), tools)
        assert "ZeroDivisionError" in out["content"]

    def test_tool_message_without_pending_generator_ignored(self):
        plm = create_plm("plm.command")
        h = _history(("system", _read_x_program()))
        plm.handle_messages(list(h), {})
        h.append({"role": "tool", "content": "1", "tool_call_id": "cmd-0"})
        assert plm.handle_messages(list(h), {}) == {"content": ""}

    def test_mismatched_tool_call_id_is_error(self):
        plm = create_plm("plm.command")
        tools = _params("memory_read")
        h = _history(("system", _read_x_program()), ("user", "go"))
        plm.handle_messages(list(h), tools)
        h.append({"role": "tool", "content": "1", "tool_call_id": "other-id"})
        out = plm.handle_messages(list(h), tools)
        assert "不匹配" in out["content"]

    def test_new_command_abandons_pending_generator(self):
        plm = create_plm("plm.command")
        tools = _params("memory_read")
        h = _history(("system", _read_x_program()))
        plm.handle_messages(list(h), tools)

        h.append({"role": "user", "content": "first"})
        r1 = plm.handle_messages(list(h), tools)
        assert r1["tool_calls"][0]["id"] == "cmd-0"

        # 未回应工具调用就投递新命令：旧生成器丢弃，新命令从 cmd-1 开始
        h.append({"role": "assistant", "content": ""})
        h.append({"role": "user", "content": "second"})
        r2 = plm.handle_messages(list(h), tools)
        assert r2["tool_calls"][0]["id"] == "cmd-1"

        h.append({"role": "tool", "content": "9", "tool_call_id": "cmd-1"})
        assert plm.handle_messages(list(h), tools) == {"content": "got:9"}


class TestCommandPLMRegistryAndIntegration:
    def test_registry_contains_command_plm(self):
        assert PLM_REGISTRY["plm.command"] is CommandPLM
        assert create_plm("plm.command").__class__ is CommandPLM


class MockLMU:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_index = 0

    def exec(self, conversation, para):
        if self.call_index >= len(self.responses):
            raise RuntimeError("MockLMU 没有更多响应")
        r = self.responses[self.call_index]
        self.call_index += 1
        return r


def _setup_core(program_content, model):
    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "gpt"}, ctrl={"type": "para"})
    core.mem["program"] = MetaDict(data={"content": program_content}, ctrl={"type": "python"})
    core.mem["params"] = MetaDict(data={"model": model}, ctrl={"type": "para"})
    root = core.start("sys", "hello", "$MEM.model_params")
    root.name = "init"
    core._ready_cids.clear()
    core._active_cid = root.cid
    return core, root


class TestCommandPLMViaCore:
    def test_command_plm_reads_and_writes_memory(self):
        program = (
            "def handle(cmd):\n"
            "    value = yield from memory_read('$MEM.seed')\n"
            "    yield from memory_write('$MEM.out', value + '!')\n"
            "    return 'ok:' + value\n"
        )
        core, root = _setup_core(program, "plm.command")
        core.mem["seed"] = "42"
        _seed(core.mem, "$MEM.out", "")
        core.lmu = MockLMU([
            (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.program", "para_ref": "$MEM.params"}}], None),
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": "run"}}], None),
            ("done", [], None),
        ])

        core.advance_conversation()  # root: create child
        core.advance_conversation()  # root: send command
        core.advance_conversation()  # root: finish, child becomes active
        core.advance_conversation()  # child: memory_read tool call executed
        core.advance_conversation()  # child: memory_write tool call executed
        core.advance_conversation()  # child: returns output, interaction ends

        child = core.get_conversation(1)
        assert child.is_python is True
        assert core.mem["out"] == "42!"
        assert child.messages[-1].content == "ok:42"

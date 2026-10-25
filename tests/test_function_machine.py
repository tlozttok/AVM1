from avm.exceptions import MemoryKeyNotFoundError
"""plm.function / plm.machine 测试：纯函数、状态机 DSL、工具与 Core 集成。"""

import json

import pytest

from avm.core import Core
from avm.python_server import create_plm, PLM_REGISTRY
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


# ------------------------------------------------------------------
# plm.function
# ------------------------------------------------------------------

class TestFunctionPLM:
    def test_basic_json_in_json_out(self):
        plm = create_plm("plm.function")
        setup = "def handle(input):\n    return {'sum': input['a'] + input['b']}"
        h = _history(("system", setup))
        assert plm.handle_messages(list(h), {}) == {"content": ""}
        h.append({"role": "user", "content": json.dumps({"a": 1, "b": 2})})
        r = plm.handle_messages(list(h), {})
        assert json.loads(r["content"]) == {"sum": 3}

    def test_no_cross_call_state(self):
        plm = create_plm("plm.function")
        setup = (
            "def handle(input):\n"
            "    global counter\n"
            "    try:\n"
            "        counter += 1\n"
            "    except NameError:\n"
            "        counter = 1\n"
            "    return counter"
        )
        h = _history(("system", setup))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        first = plm.handle_messages(list(h), {})["content"]
        assert first == "1"
        # 第二次调用仍是新命名空间（历史继续增长，模拟真实调用）
        h.append({"role": "assistant", "content": first})
        h.append({"role": "user", "content": "null"})
        assert plm.handle_messages(list(h), {})["content"] == "1"

    def test_print_discarded(self):
        plm = create_plm("plm.function")
        setup = "def handle(input):\n    print('secret')\n    return 'ok'"
        h = _history(("system", setup))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        content = plm.handle_messages(list(h), {})["content"]
        assert json.loads(content) == "ok"
        assert "secret" not in content

    def test_errors(self):
        plm = create_plm("plm.function")
        h = _history(("system", "x = 1"))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        assert "handle" in plm.handle_messages(list(h), {})["content"]

        plm = create_plm("plm.function")
        h = _history(("system", "def handle(input):\n    return 1/0"))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        assert "ZeroDivisionError" in plm.handle_messages(list(h), {})["content"]

        plm = create_plm("plm.function")
        h = _history(("system", "def handle(input):\n    return {1, 2}"))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        assert "TypeError" in plm.handle_messages(list(h), {})["content"]

        plm = create_plm("plm.function")
        h = _history(("system", "def handle(input):\n    return 1"))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "not-json"})
        assert "JSONDecodeError" in plm.handle_messages(list(h), {})["content"]

        plm = create_plm("plm.function")
        h = _history(("system", "import os"))
        plm.handle_messages(list(h), {})
        h.append({"role": "user", "content": "null"})
        assert "SecurityError" in plm.handle_messages(list(h), {})["content"]

    def test_tool_message_returns_empty(self):
        plm = create_plm("plm.function")
        r = plm.handle_messages(_history(("system", "def handle(input):\n    return input")), {})
        assert r == {"content": ""}
        r = plm.handle_messages([
            {"role": "system", "content": "def handle(input):\n    return input"},
            {"role": "tool", "content": "ignored", "tool_call_id": "t1"},
        ], {})
        assert r == {"content": ""}


# ------------------------------------------------------------------
# plm.machine
# ------------------------------------------------------------------

class TestStateMachinePLM:
    def _plm(self, dsl, tools=None):
        plm = create_plm("plm.machine")
        h = _history(("system", dsl))
        plm.handle_messages(list(h), tools or {})
        return plm, h

    def test_basic_transitions_guard_set(self):
        dsl = """initial: idle

state idle:
  on start { set counter = 0 } -> running

state running:
  on tick if counter < 3 { set counter = counter + 1 } -> running
  on stop -> idle | {"content": "stopped"}
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "start"})})
        assert plm.handle_messages(list(h), {})["content"] == ""
        assert plm._state == "running"
        assert plm._vars == {"counter": 0}

        for expected in (1, 2, 3):
            h.append({"role": "user", "content": json.dumps({"event": "tick"})})
            r = plm.handle_messages(list(h), {})
            assert r["content"] == ""
            assert plm._vars["counter"] == expected

        h.append({"role": "user", "content": json.dumps({"event": "stop"})})
        r = plm.handle_messages(list(h), {})
        assert r["content"] == "stopped"
        assert "tool_calls" not in r
        assert plm._state == "idle"

    def test_guard_false_and_default_star(self):
        dsl = """initial: s

state s:
  on go if 1 == 0 -> bad
  on * -> ok

state ok:
  on * -> ok

state bad:
  on * -> bad
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "other"})})
        assert plm.handle_messages(list(h), {})["content"] == ""
        assert plm._state == "ok"

    def test_actions_become_tool_calls(self):
        dsl = """initial: s

state s:
  on go -> s | {"content": "working", "tools": [
    {"name": "memory_write", "arguments": {"ref": "$MEM.x", "content": "1"}},
    {"name": "memory_make", "arguments": {"ref": "$MEM", "key": "a", "mem_type": "str"}}
  ]}
"""
        plm, h = self._plm(dsl, _params("memory_write", "memory_make"))
        h.append({"role": "user", "content": json.dumps({"event": "go"})})
        r = plm.handle_messages(list(h), _params("memory_write", "memory_make"))
        assert r["content"] == "working"
        names = [tc["function"]["name"] for tc in r["tool_calls"]]
        assert names == ["memory_write", "memory_make"]
        assert [tc["id"] for tc in r["tool_calls"]] == ["fsm-0", "fsm-1"]

    def test_unknown_tool_is_parse_error(self):
        dsl = """initial: s

state s:
  on go -> s | {"tools": [{"name": "not_a_tool", "arguments": {}}]}
"""
        plm, h = self._plm(dsl, _params("memory_write"))
        h.append({"role": "user", "content": json.dumps({"event": "go"})})
        r = plm.handle_messages(list(h), _params("memory_write"))
        assert "定义解析失败" in r["content"]
        assert "not_a_tool" in r["content"]

    def test_last_tool_output_guard(self):
        dsl = """initial: s

state s:
  on first -> s | {"content": "first", "tools": [{"name": "memory_write", "arguments": {"ref": "$MEM.x", "content": "a"}}]}
  on second if last == "ok" -> done | {"content": "accepted"}
  on * -> s | {"content": "rejected"}

state done:
  on * -> done
"""
        tools = _params("memory_write")
        plm, h = self._plm(dsl, tools)
        h.append({"role": "user", "content": json.dumps({"event": "first"})})
        assert plm.handle_messages(list(h), tools)["content"] == "first"
        h.append({"role": "tool", "content": "ok", "tool_call_id": "fsm-0"})
        assert plm.handle_messages(list(h), tools)["content"] == ""
        h.append({"role": "user", "content": json.dumps({"event": "second"})})
        assert plm.handle_messages(list(h), tools)["content"] == "accepted"
        assert plm._state == "done"

    def test_no_match_keeps_state_and_vars(self):
        dsl = """initial: s

state s:
  on go { set x = 1 } -> t

state t:
  on * -> t
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "other"})})
        r = plm.handle_messages(list(h), {})
        assert "无匹配事件 other" in r["content"]
        assert plm._state == "s"
        assert plm._vars == {}

    def test_guard_error_transaction(self):
        dsl = """initial: s

state s:
  on go if 1/0 -> t

state t:
  on * -> t
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "go"})})
        r = plm.handle_messages(list(h), {})
        assert "ZeroDivisionError" in r["content"]
        assert plm._state == "s"
        assert plm._vars == {}

    def test_set_error_transaction(self):
        dsl = """initial: s

state s:
  on go { set x = 1; set y = 1/0 } -> t

state t:
  on * -> t
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "go"})})
        r = plm.handle_messages(list(h), {})
        assert "ZeroDivisionError" in r["content"]
        assert plm._state == "s"
        assert plm._vars == {}

    def test_syntax_error_returns_text_not_exception(self):
        plm = create_plm("plm.machine")
        r = plm.handle_messages(_history(("system", "not a valid dsl")), {})
        assert r == {"content": ""}
        r = plm.handle_messages(_history(("system", "not a valid dsl"), ("user", '{"event":"x"}')), {})
        assert "定义解析失败" in r["content"]

    def test_content_only_response_ends_interaction(self):
        dsl = """initial: s

state s:
  on go -> s | {"content": "done"}
"""
        plm, h = self._plm(dsl)
        h.append({"role": "user", "content": json.dumps({"event": "go"})})
        r = plm.handle_messages(list(h), {})
        assert r == {"content": "done"}


# ------------------------------------------------------------------
# 注册表与 Core 集成
# ------------------------------------------------------------------

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


class TestRegistryAndIntegration:
    def test_registry_contains_new_plms(self):
        assert PLM_REGISTRY["plm.function"] is not None
        assert PLM_REGISTRY["plm.machine"] is not None

    def test_function_plm_via_core(self):
        program = "def handle(input):\n    return input['x'] + 1"
        core, root = _setup_core(program, "plm.function")
        core.lmu = MockLMU([
            (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.program", "para_ref": "$MEM.params"}}], None),
            (None, [{"call_id": "si", "cmd_type": "send_instruction", "args": {"cid": 1, "content": json.dumps({"x": 41})}}], None),
            ("done", [], None),
        ])

        core.advance_conversation()  # root: create child
        core.advance_conversation()  # root: send JSON input
        core.advance_conversation()  # root: finish, child becomes active
        core.advance_conversation()  # child: plm.function executes

        child = core.get_conversation(1)
        assert child.is_python is True
        assert child.messages[-1].content == "42"

    def test_machine_plm_via_core_writes_memory(self):
        dsl = """initial: idle

state idle:
  on start { set counter = 0 } -> running | {"content": "started"}

state running:
  on tick { set counter = counter + 1 } -> running | {"content": "ticked"}
  on stop -> idle | {"content": "stopped", "tools": [{"name": "memory_write", "arguments": {"ref": "$MEM.log", "content": "stopped"}}]}
"""
        core, root = _setup_core(dsl, "plm.machine")
        _seed(core.mem, "$MEM.log", "")
        core.lmu = MockLMU([
            (None, [{"call_id": "cc", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.program", "para_ref": "$MEM.params"}}], None),
            (None, [{"call_id": "s1", "cmd_type": "send_instruction", "args": {"cid": 1, "content": json.dumps({"event": "start"})}}], None),
            ("done", [], None),
            (None, [{"call_id": "s2", "cmd_type": "send_instruction", "args": {"cid": 1, "content": json.dumps({"event": "tick"})}}], None),
            ("done", [], None),
            (None, [{"call_id": "s3", "cmd_type": "send_instruction", "args": {"cid": 1, "content": json.dumps({"event": "stop"})}}], None),
            ("done", [], None),
        ])

        core.advance_conversation()  # root: create child
        core.advance_conversation()  # root: send start
        core.advance_conversation()  # root: finish, child active
        core.advance_conversation()  # child: start event -> running, no tools

        child = core.get_conversation(1)
        assert child.is_python is True
        executor = core._executors[child.cid]
        assert executor.plm._state == "running"
        assert executor.plm._vars == {"counter": 0}

        # 手工唤醒 root 继续投递 tick / stop，验证 machine action 经 Core 执行。
        core._resume(root.cid)
        core.advance_conversation()  # root: send tick
        core.advance_conversation()  # root: finish, child active
        core.advance_conversation()  # child: tick -> counter=1
        assert executor.plm._vars == {"counter": 1}

        core._resume(root.cid)
        core.advance_conversation()  # root: send stop
        core.advance_conversation()  # root: finish, child active
        core.advance_conversation()  # child: stop -> memory_write tool
        core.advance_conversation()  # child: tool message -> finish

        assert core.mem["log"] == "stopped"
        assert executor.plm._state == "idle"

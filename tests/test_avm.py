"""AVM 测试套件

本测试使用 MockLMU 替代真实 LLM 调用，确保测试：
1. 完全确定性（不依赖外部 API）
2. 可复现
3. 快速执行

运行方式: python -m pytest test_avm.py -v
"""

import pytest
from avm.core import (
    Core, parse_instruction, LMU,
    CreateInstruction, ExecInstruction,
    MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    CRT,
)
from avm.memory import Memory
from avm.messages import Conversation, UserMessageBatch
from avm.exceptions import VMSyntaxError, VMMemoryError
from avm.memory_device import StringDevice


# ---------------------------------------------------------------------------
# MockLMU
# ---------------------------------------------------------------------------

class MockLMU:
    """确定性 LLM 模拟器
    
    用法：
        mock = MockLMU([
            ("hello", [], None),           # 第一次调用返回 "hello"
            ("world", ["memory_read ..."], None),  # 第二次调用返回 "world" + 一个工具调用
        ])
    """

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_index = 0
        self.calls = []  # 记录所有调用，用于断言

    def _next(self, call_type, *args):
        self.calls.append((call_type, *args))
        if self.call_index >= len(self.responses):
            raise RuntimeError(f"MockLMU: 第 {self.call_index} 次调用没有预设响应")
        resp = self.responses[self.call_index]
        self.call_index += 1
        if callable(resp):
            return resp(call_type, *args)
        return resp

    def exec_crt(self, system_prompt, user_prompt, para):
        return self._next("exec_crt", system_prompt, user_prompt, para)

    def exec(self, conversation, user_msg_batch, para):
        return self._next("exec", conversation, user_msg_batch, para)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def make_core(mock_responses=None):
    """创建一个带有 MockLMU 的 Core 实例"""
    core = Core()
    if mock_responses is not None:
        core.lmu = MockLMU(mock_responses)
    return core


# ---------------------------------------------------------------------------
# parse_instruction 测试
# ---------------------------------------------------------------------------

class TestParseInstruction:
    def test_create(self):
        instr = parse_instruction("create cid_1 0 $MEM.sys $MEM.usr $MEM.para")
        assert isinstance(instr, CreateInstruction)
        assert instr.call_id == "cid_1"
        assert instr.utr_index == 0
        assert instr.system_ref == "$MEM.sys"
        assert instr.user_ref == "$MEM.usr"
        assert instr.para_ref == "$MEM.para"

    def test_exec(self):
        instr = parse_instruction("exec cid_1 0 $last_msg_reg.0 $usr_tool_reg.0 $MEM.para")
        assert isinstance(instr, ExecInstruction)
        assert instr.call_id == "cid_1"
        assert instr.utr_index == 0
        assert instr.last_msg_ref == "$last_msg_reg.0"
        assert instr.user_msg_ref == "$usr_tool_reg.0"
        assert instr.para_ref == "$MEM.para"

    def test_memory_read(self):
        instr = parse_instruction("memory_read cid_1 0 $MEM.user_input")
        assert isinstance(instr, MemoryReadInstruction)
        assert instr.ref == "$MEM.user_input"

    def test_memory_write(self):
        instr = parse_instruction("memory_write cid_1 0 $MEM.out hello")
        assert isinstance(instr, MemoryWriteInstruction)
        assert instr.ref == "$MEM.out"
        assert instr.content == "hello"

    def test_memory_make(self):
        instr = parse_instruction("memory_make cid_1 0 $MEM.data new_key dict")
        assert isinstance(instr, MemoryMakeInstruction)
        assert instr.ref == "$MEM.data"
        assert instr.key == "new_key"
        assert instr.mem_type == "dict"

    def test_empty_raises(self):
        with pytest.raises(VMSyntaxError):
            parse_instruction("")

    def test_unknown_raises(self):
        with pytest.raises(VMSyntaxError):
            parse_instruction("foobar 0 1")


# ---------------------------------------------------------------------------
# Memory 测试
# ---------------------------------------------------------------------------

class TestMemory:
    def test_basic_set_get(self):
        mem = Memory()
        mem["key"] = "value"
        assert mem["key"] == "value"

    def test_nested_set_get(self):
        mem = Memory()
        mem["a"] = {}
        mem.set("$MEM.a.b", "nested")
        assert mem["a"]["b"] == "nested"

    def test_dollar_unwrap(self):
        mem = Memory()
        mem["a"] = "hello"
        # Core 传入的 value 格式: ['$', 'MEM', 'a']
        result = mem.unwrap(["$", "MEM", "a"], for_llm=True)
        assert result == "hello"

    def test_dollar_unwrap_without_mem_prefix(self):
        mem = Memory()
        mem["a"] = "hello"
        result = mem.unwrap(["$", "a"], for_llm=True)
        assert result == "hello"

    def test_recursive_dereference(self):
        mem = Memory()
        mem["a"] = "$MEM.b"
        mem["b"] = "final"
        result = mem.unwrap(["$", "MEM", "a"], for_llm=True)
        assert result == "final"

    def test_ampersand_one_level(self):
        mem = Memory()
        mem["a"] = "$MEM.b"
        mem["b"] = "final"
        # & 直接返回原始值，不做额外解引用
        result = mem.unwrap(["&", "MEM", "a"], for_llm=True)
        assert result == "$MEM.b"

    def test_device_mount_and_read(self):
        mem = Memory()
        dev = StringDevice("device_value")
        mem.mount("io.test", dev)
        result = mem.unwrap(["$", "MEM", "io", "test"], for_llm=True)
        assert result == "device_value"

    def test_device_write(self):
        mem = Memory()
        dev = StringDevice("old")
        mem.mount("io.test", dev)
        mem.set("$MEM.io.test", "new")
        assert dev.get_value() == "new"

    def test_make_dict(self):
        mem = Memory()
        mem["base"] = {}
        mem.make("$MEM.base", "child", "dict")
        assert mem["base"]["child"] == {}

    def test_make_list(self):
        mem = Memory()
        mem["base"] = [None]
        mem.make("$MEM.base", "0", "str")
        assert mem["base"][0] == ""


# ---------------------------------------------------------------------------
# Core 状态测试（无 LLM）
# ---------------------------------------------------------------------------

class TestCoreState:
    def test_core_init(self):
        core = make_core()
        assert core.command_stack == []
        assert core.last_msg_reg == []
        assert core.usr_tool_reg == []
        assert isinstance(core.mem, Memory)

    def test_unwrap_register(self):
        core = make_core()
        conv = Conversation.from_any_list([("system", "sys"), ("user", "usr")])
        batch = UserMessageBatch()
        core.last_msg_reg.append(conv)
        core.usr_tool_reg.append(batch)

        assert core.unwrap("$last_msg_reg.0") is conv
        assert core.unwrap("$usr_tool_reg.0") is batch

    def test_unwrap_mem(self):
        core = make_core()
        core.mem["x"] = "hello"
        assert core.unwrap("$MEM.x") == "hello"


# ---------------------------------------------------------------------------
# 指令执行测试（使用 MockLMU）
# ---------------------------------------------------------------------------

class TestInstructionExecution:
    def test_memory_read(self):
        core = make_core()
        core.mem["input"] = "user_says_hi"
        core.usr_tool_reg.append(UserMessageBatch())

        instr = MemoryReadInstruction("cid", 0, "$MEM.input")
        rt = instr.execute(core)

        assert rt == CRT.EXIT
        batch = core.usr_tool_reg[0]
        assert len(batch.tool_responses) == 1
        assert batch.tool_responses[0].content == "user_says_hi"
        assert batch.tool_responses[0].tool_call_id == "cid"

    def test_memory_write(self):
        core = make_core()
        core.mem["out"] = ""
        core.usr_tool_reg.append(UserMessageBatch())

        instr = MemoryWriteInstruction("cid", 0, "$MEM.out", "hello")
        rt = instr.execute(core)

        assert rt == CRT.EXIT
        assert core.mem["out"] == "hello"
        # 工具响应也写入了 batch
        assert len(core.usr_tool_reg[0].tool_responses) == 1

    def test_create_no_tool_calls(self):
        """create 指令，LLM 无工具调用 -> EXIT，结果写回父 batch"""
        core = make_core([("hello", [], None)])
        core.mem["sys"] = "system_prompt"
        core.mem["usr"] = "user_prompt"
        core.mem["para"] = {"model": "test"}
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create cid_1 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        assert len(core.command_stack) == 0
        assert len(core.usr_tool_reg) == 1
        assert core.usr_tool_reg[0].tool_responses[0].content == "hello"
        assert core.usr_tool_reg[0].tool_responses[0].tool_call_id == "cid_1"

    def test_create_with_tool_calls(self):
        """create 指令，LLM 返回一个工具调用 -> CONTINUE，栈顶替换为 exec + 子指令"""
        core = make_core([
            (
                None,
                [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.input"}}],
                Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")]),
            ),
            ("done", [], None),  # exec 的响应
        ])
        core.mem["sys"] = "s"
        core.mem["usr"] = "u"
        core.mem["para"] = {"model": "test", "use_tool": True}
        core.mem["input"] = "hello"
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create cid_1 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        # 执行过程：
        # 1. create 执行 -> 生成 exec + memory_read
        # 2. memory_read 执行 -> EXIT -> 弹出
        # 3. exec -> 消费 mock[1]，无更多工具调用 -> EXIT
        assert core.lmu.call_index == 2
        assert len(core.command_stack) == 0

    def test_create_then_exec_chain(self):
        """完整的 create -> exec 链条，无更多工具调用"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.input"}}], conv),  # create 的响应
            ("final_answer", [], None),  # exec 的响应
        ])
        core.mem["sys"] = "s"
        core.mem["usr"] = "u"
        core.mem["para"] = {"model": "test", "use_tool": True}
        core.mem["input"] = "hello"
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create cid_1 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        # 执行流程：
        # 1. create -> 消费 mock[0]，生成 exec + memory_read
        #    栈: [exec, memory_read] (memory_read 后压入，先执行)
        # 2. memory_read -> 读取 $MEM.input，写入 batch[1]
        #    栈: [exec]
        # 3. exec -> 消费 mock[1]，conv + batch[1] -> "final_answer"
        #    无更多 return_calls -> EXIT，弹出寄存器
        #    栈: []

        assert len(core.command_stack) == 0
        assert len(core.last_msg_reg) == 0
        assert len(core.usr_tool_reg) == 1  # 只剩父 batch
        # exec 的 result 写回父 batch
        assert core.usr_tool_reg[0].tool_responses[-1].content == "final_answer"

    def test_nested_create(self):
        """exec 中 LLM 触发 create_cmd -> 产生新的 create 指令"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            (None, [{"call_id": "tc2", "cmd_type": "create", "args": {"system_ref": "$MEM.sys2", "user_ref": "$MEM.usr2", "para_ref": "$MEM.para2"}}], conv),  # create
            (None, [], None),  # exec（第一次）
            ("nested_done", [], None),  # 新的 create 的 exec_crt
        ])
        core.mem["sys"] = "s"
        core.mem["usr"] = "u"
        core.mem["para"] = {"model": "test", "use_tool": True}
        core.mem["sys2"] = "s2"
        core.mem["usr2"] = "u2"
        core.mem["para2"] = {"model": "test2"}
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create cid_1 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        # create -> 产生 exec + create
        # create 指令在栈顶，先执行 -> 创建第二层对话，消费 mock[2]
        # 然后 exec 执行 -> 消费 mock[1]
        
        assert core.lmu.call_index == 3
        assert len(core.command_stack) == 0
        # 第二层对话的寄存器已被弹出，只剩父 batch
        assert len(core.usr_tool_reg) == 1


# ---------------------------------------------------------------------------
# 集成测试：完整运行流程
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_run_with_memory_ops(self):
        """模拟一个完整的工作流：
        - create 指令启动对话
        - LLM 先触发 memory_read 读取输入
        - 然后 LLM 再次调用（exec），无更多工具调用，返回结果
        """
        conv = Conversation.from_any_list([("system", "sys"), ("user", "usr"), ("assistant", "")])
        core = make_core([
            # 第一次 LLM 调用（create）: 要求读取输入
            (None, [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.data.input"}}], conv),
            # 第二次 LLM 调用（exec）: 处理输入后返回结果
            ("processed", [], None),
        ])

        core.mem["sys"] = "system"
        core.mem["usr"] = "user"
        core.mem["para"] = {"model": "test", "use_tool": True}
        core.mem["data"] = {"input": "raw_data"}
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create root 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        assert len(core.command_stack) == 0
        assert len(core.last_msg_reg) == 0
        assert len(core.usr_tool_reg) == 1
        # 最终 result 回到父 batch
        assert core.usr_tool_reg[0].tool_responses[-1].content == "processed"

    def test_multi_tool_calls_order(self):
        """验证多个工具调用的压栈顺序（反序）"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            (None, [
                {"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.a", "content": "1"}},
                {"call_id": "tc2", "cmd_type": "memory_write", "args": {"ref": "$MEM.b", "content": "2"}},
            ], conv),
            ("ok1", [], None),
            ("ok2", [], None),
        ])
        core.mem["s"] = "s"
        core.mem["u"] = "u"
        core.mem["para"] = {"model": "test", "use_tool": True}
        core.mem["a"] = ""
        core.mem["b"] = ""
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create root 0 $MEM.s $MEM.u $MEM.para")
        core.run()

        # return_calls 被反序压栈：
        # 原始: [write_a, write_b]
        # 压栈后: [exec, write_b, write_a]（exec 替换栈顶，然后 extend [write_b, write_a] 的反转 [write_a, write_b]）
        # 等等，原始 return_calls = ["memory_write tc1 ...", "memory_write tc2 ..."]
        # core.command_stack.extend(return_calls[::-1])
        # 所以压入栈的顺序是："memory_write tc2 ...", "memory_write tc1 ..."
        # 栈顶是最后压入的，所以先执行 "memory_write tc1 ..."
        
        # 但实际上 create 先替换栈顶为 exec，然后 extend return_calls[::-1]
        # 所以栈变成：[..., exec, write_b, write_a]
        # 等等不对，return_calls[::-1] 会把 return_calls 反转后 extend
        # return_calls = [write_a, write_b]
        # return_calls[::-1] = [write_b, write_a]
        # extend 后栈顶是 write_a（最后压入）
        # 所以先执行 write_a，然后 write_b，然后 exec
        
        # mock[0] 被 create 消费
        # mock[1] 被 exec 消费（在 write_a 和 write_b 之后）
        # 但 exec 之后没有更多工具调用，所以只消费了 2 条 mock
        
        assert core.mem["a"] == "1"
        assert core.mem["b"] == "2"
        assert core.lmu.call_index == 2  # create + exec 各消费一条


# ---------------------------------------------------------------------------
# main 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# 补充测试：parse_instruction 边界情况
# ---------------------------------------------------------------------------

class TestParseInstructionEdgeCases:
    def test_memory_read_missing_args(self):
        instr = parse_instruction("memory_read cid 0")
        assert isinstance(instr, MemoryReadInstruction)
        assert instr.ref == ""

    def test_memory_write_missing_content(self):
        instr = parse_instruction("memory_write cid 0 $MEM.out")
        assert isinstance(instr, MemoryWriteInstruction)
        assert instr.content == ""

    def test_utr_index_defaults_to_minus_one(self):
        instr = parse_instruction("memory_read cid")
        assert instr.utr_index == -1


# ---------------------------------------------------------------------------
# 补充测试：MemoryMakeInstruction 执行
# ---------------------------------------------------------------------------

class TestMemoryMakeExecution:
    def test_memory_make_execute_dict(self):
        core = make_core()
        core.mem["data"] = {}
        core.usr_tool_reg.append(UserMessageBatch())

        instr = MemoryMakeInstruction("cid", 0, "$MEM.data", "new_key", "dict")
        rt = instr.execute(core)

        assert rt == CRT.EXIT
        assert core.mem["data"]["new_key"] == {}
        assert len(core.usr_tool_reg[0].tool_responses) == 1
        assert "Success created dict" in core.usr_tool_reg[0].tool_responses[0].content

    def test_memory_make_execute_error(self):
        core = make_core()
        core.mem["data"] = "string"
        core.usr_tool_reg.append(UserMessageBatch())

        instr = MemoryMakeInstruction("cid", 0, "$MEM.data", "key", "dict")
        rt = instr.execute(core)

        assert rt == CRT.EXIT
        assert "Error" in core.usr_tool_reg[0].tool_responses[0].content


# ---------------------------------------------------------------------------
# 补充测试：ExecInstruction 直接执行
# ---------------------------------------------------------------------------

class TestExecInstructionDirect:
    def test_exec_exit_no_subcalls(self):
        """ExecInstruction EXIT，弹出寄存器，result 写回父 batch"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            ("result", [], None),
        ])
        core.mem["para"] = {"model": "test"}
        core.last_msg_reg.append(conv)
        core.usr_tool_reg.append(UserMessageBatch())   # idx 0: exec 使用的
        core.usr_tool_reg.append(UserMessageBatch())   # idx 1: 父 batch

        instr = ExecInstruction("cid", 1, "$last_msg_reg.0", "$usr_tool_reg.0", "$MEM.para")
        core.command_stack.append(instr)
        core.run()

        # EXIT 后弹出寄存器
        assert len(core.last_msg_reg) == 0
        assert len(core.usr_tool_reg) == 1  # 只剩父 batch
        assert core.usr_tool_reg[0].tool_responses[0].content == "result"
        assert core.usr_tool_reg[0].tool_responses[0].tool_call_id == "cid"

    def test_exec_continue_with_tool_calls(self):
        """ExecInstruction CONTINUE，压入子指令，之后继续执行"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_read", "args": {"ref": "$MEM.x"}}], None),
            ("done", [], None),
        ])
        core.mem["x"] = "val"
        core.mem["para"] = {"model": "test"}
        core.last_msg_reg.append(conv)
        core.usr_tool_reg.append(UserMessageBatch())   # idx 0: exec 使用的
        core.usr_tool_reg.append(UserMessageBatch())   # idx 1: 父 batch

        instr = ExecInstruction("cid", 1, "$last_msg_reg.0", "$usr_tool_reg.0", "$MEM.para")
        core.command_stack.append(instr)
        core.run()

        assert len(core.command_stack) == 0
        assert len(core.last_msg_reg) == 0
        assert len(core.usr_tool_reg) == 1
        # memory_read 的结果 + exec 的 result
        assert core.usr_tool_reg[0].tool_responses[-1].content == "done"

    def test_exec_with_user_content(self):
        """ExecInstruction 消费 user_msg_batch 中的 user_content"""
        conv = Conversation.from_any_list([("system", "s"), ("user", "u"), ("assistant", "")])
        core = make_core([
            ("got_it", [], None),
        ])
        core.mem["para"] = {"model": "test"}
        core.last_msg_reg.append(conv)
        batch = UserMessageBatch()
        batch.add_user_content("additional input")
        core.usr_tool_reg.append(batch)
        core.usr_tool_reg.append(UserMessageBatch())  # 父 batch

        instr = ExecInstruction("cid", 1, "$last_msg_reg.0", "$usr_tool_reg.0", "$MEM.para")
        core.command_stack.append(instr)
        core.run()

        assert core.usr_tool_reg[0].tool_responses[0].content == "got_it"


# ---------------------------------------------------------------------------
# 补充测试：Core.run 流程
# ---------------------------------------------------------------------------

class TestCoreRun:
    def test_run_empty_stack(self):
        core = make_core()
        core.run()  # 空栈，不抛异常

    def test_run_string_instruction(self):
        core = make_core([("hello", [], None)])
        core.mem["sys"] = "s"
        core.mem["usr"] = "u"
        core.mem["para"] = {"model": "test"}
        core.usr_tool_reg.append(UserMessageBatch())

        core.command_stack.append("create cid 0 $MEM.sys $MEM.usr $MEM.para")
        core.run()

        assert len(core.command_stack) == 0
        assert core.usr_tool_reg[0].tool_responses[0].content == "hello"


# ---------------------------------------------------------------------------
# 补充测试：tool_calls 在 Conversation 中的保存和序列化
# ---------------------------------------------------------------------------

class TestToolCallsAndConversation:
    """测试 tool_calls 在 Conversation 中的保存和序列化（修复 API 报错的关键）"""

    def test_conversation_preserves_tool_calls(self):
        tc = [{"id": "call_1", "type": "function", "function": {"name": "memory_read", "arguments": "{}"}}]
        conv = Conversation.from_any_list([
            ("system", "sys"),
            ("user", "usr"),
            {"role": "assistant", "content": "", "tool_calls": tc},
        ])
        msgs = conv.to_api_messages()
        assert len(msgs) == 3
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["tool_calls"] == tc

    def test_append_assistant_with_tool_calls(self):
        conv = Conversation.from_any_list([("system", "sys")])
        tc = [{"id": "c1", "type": "function", "function": {"name": "read"}}]
        conv.append_assistant_message("", tool_calls=tc)
        msgs = conv.to_api_messages()
        assert msgs[1]["tool_calls"] == tc

    def test_tool_calls_followed_by_tool_message(self):
        """模拟 API 消息列表：assistant(tool_calls) + tool"""
        conv = Conversation.from_any_list([
            ("user", "hi"),
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "memory_read", "arguments": "{}"}}
            ]},
        ])
        batch = UserMessageBatch()
        batch.add_tool_response("result", "tc1")

        msgs = conv.to_api_messages()
        msgs.extend(batch.to_tool_messages())

        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "tool_calls" in msgs[1]
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "tc1"


# ---------------------------------------------------------------------------
# 补充测试：_parse_index 边界
# ---------------------------------------------------------------------------

class TestParseIndex:
    def test_parse_index_last_msg_reg(self):
        instr = ExecInstruction("c", 0, "$last_msg_reg.5", "$usr_tool_reg.0", "$MEM.para")
        assert instr._parse_index("$last_msg_reg.5") == 5

    def test_parse_index_usr_tool_reg(self):
        instr = ExecInstruction("c", 0, "$last_msg_reg.0", "$usr_tool_reg.3", "$MEM.para")
        assert instr._parse_index("$usr_tool_reg.3") == 3

    def test_parse_index_mem_returns_none(self):
        instr = ExecInstruction("c", 0, "$last_msg_reg.0", "$usr_tool_reg.0", "$MEM.para")
        assert instr._parse_index("$MEM.key") is None

    def test_parse_index_invalid_returns_none(self):
        instr = ExecInstruction("c", 0, "$last_msg_reg.0", "$usr_tool_reg.0", "$MEM.para")
        assert instr._parse_index("not_a_ref") is None

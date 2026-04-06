

import enum

from openai import OpenAI

from avm_types import MetaList, MetaDict
from exceptions import VMSyntaxError, VMMemoryError
from messages import SystemMessage, UserMessage, Conversation, UserMessageBatch
from memory import Memory

class CommandReturnType(enum):
    """指令返回类型"""
    EXIT=0
    CONTINUE=1

type CRT=CommandReturnType

class Instruction:
    """指令基类"""
    def __init__(self,call_id: str, utr_index: int, **kargs):
        self.call_id = call_id
        self.utr_index = utr_index
        for k, v in kargs.items():
            setattr(self, k, v)
        
    def execute(self, core: 'Core') -> None:
        raise NotImplementedError
class MemoryReadInstruction(Instruction):
    """memory_read 指令：从内存中读取数据"""
    call_id: str
    utr_index: int
    ref: str
    def __init__(self, call_id: str, utr_index: int, ref: str, **kargs):
        super().__init__(call_id, utr_index, ref=ref, **kargs)
    
    def execute(self, core: 'Core') -> None:
        user_batch = core.usr_tool_reg[self.utr_index]
        content=core.unwrap(self.ref, for_llm=True)
        user_batch.add_user_message(content)
        return CRT.EXIT

class MemoryWriteInstruction(Instruction):
    """memory_write 指令：写入内存"""
    call_id: str
    utr_index: int
    ref: str
    content: str
    def __init__(self, call_id: str, utr_index: int, ref: str, content: str, **kargs):
        super().__init__(call_id, utr_index, ref=ref, content=content, **kargs)
        self.content = content
    
    def execute(self, core: 'Core') -> None:
        user_batch = core.usr_tool_reg[self.utr_index]
        try:
            core.mem.set(self.ref, self.content)
        except VMMemoryError as e:
            user_batch.add_tool_response(f"Error: {e}", self.call_id)
            return CRT.EXIT
        user_batch.add_tool_response(f"Success set: {self.ref}", self.call_id)
        return CRT.EXIT

class MemoryMakeInstruction(Instruction):
    """memory_make 指令：创建内存地址"""
    call_id: str
    utr_index: int
    ref: str
    key: str
    mem_type: str

    def __init__(self, call_id: str, utr_index: int, ref: str, key: str, mem_type: str, **kargs):
        super().__init__(call_id, utr_index, ref=ref, key=key, mem_type=mem_type, **kargs)

    def execute(self, core: 'Core') -> CRT:
        user_batch = core.usr_tool_reg[self.utr_index]
        try:
            core.mem.make(self.ref, self.key, self.mem_type)
            user_batch.add_tool_response(f"Success created {self.mem_type} at {self.ref}.{self.key}", self.call_id)
        except VMMemoryError as e:
            user_batch.add_tool_response(f"Error: {e}", self.call_id)
        return CRT.EXIT


class CreateInstruction(Instruction):
    """create 指令：发起新的对话"""
    call_id: str
    utr_index: int
    system_ref: str
    user_ref: str
    para_ref: str

    def __init__(self, call_id: str, utr_index: int, system_ref: str, user_ref: str, para_ref: str, **kargs):
        super().__init__(call_id, utr_index, system_ref=system_ref, user_ref=user_ref, para_ref=para_ref, **kargs)

    def execute(self, core: 'Core') -> int:
        system = core.unwrap(self.system_ref)
        user = core.unwrap(self.user_ref)
        para = core.unwrap(self.para_ref, for_llm=False)
        result, return_calls, conversation = core.lmu.exec_crt(system, user, para, self.utr_index)

        # 处理 result：存入 usr_tool_reg[utr_index]
        if result:
            user_batch = core.usr_tool_reg[self.utr_index]
            # 追加工具响应
            user_batch.add_tool_response(result, self.call_id)

        # 处理 return_calls
        if return_calls:
            # 分配新的寄存器槽位
            last_msg_idx = len(core.last_msg_reg)
            user_msg_idx = len(core.usr_tool_reg)

            # 存入 Conversation
            core.last_msg_reg.append(conversation)
            core.usr_tool_reg.append(UserMessageBatch())

            # 替换栈顶为 exec 指令
            core.command_stack[-1] = f"exec $last_msg_reg.{last_msg_idx} $user_tool_reg.{user_msg_idx} {self.para_ref} {self.utr_index} {self.call_id}"
            core.command_stack.extend(return_calls[::-1])
            return CRT.CONTINUE
        else:
            # 没有 return_calls 才 pop
            return CRT.EXIT


class ExecInstruction(Instruction):
    """exec 指令：继续对话"""
    call_id: str
    utr_index: int
    last_msg_ref: str
    user_msg_ref: str
    para_ref: str

    def __init__(self, call_id: str, utr_index: int, last_msg_ref: str, user_msg_ref: str, para_ref: str, **kargs):
        super().__init__(call_id, utr_index, last_msg_ref=last_msg_ref, user_msg_ref=user_msg_ref, para_ref=para_ref, **kargs)

    def execute(self, core: 'Core') -> None:
        # 解析索引（格式：$last_msg_reg.0 或 $MEM.key）
        last_msg_idx = self._parse_index(self.last_msg_ref)
        user_msg_idx = self._parse_index(self.user_msg_ref)

        # 直接从寄存器获取类型化对象
        conversation: Conversation = core.unwrap(self.last_msg_ref)
        user_batch: UserMessageBatch = core.unwrap(self.user_msg_ref)
        para = core.unwrap(self.para_ref, for_llm=False)

        # 调用 LMU.exec，传入 utr_index 用于工具调用
        result, return_calls, _ = core.lmu.exec(conversation, user_batch, para, self.utr_index)
        user_batch.clear()

        # 处理 result
        if result:
            # 将 (result, call_id) 存入 usr_tool_reg[utr_index]
            call_id = self.call_id
            user_batch = core.usr_tool_reg[self.utr_index]
            user_batch.add_tool_response(result, call_id)

        # 处理 return_calls：不弹出当前指令，直接压栈
        if return_calls:
            core.command_stack.extend(return_calls[::-1])
            return CRT.CONTINUE
        else:
            # pop 对应的寄存器（通过索引）
            if last_msg_idx is not None and last_msg_idx < len(core.last_msg_reg):
                core.last_msg_reg.pop(last_msg_idx)
            if user_msg_idx is not None and user_msg_idx < len(core.usr_tool_reg):
                core.usr_tool_reg.pop(user_msg_idx)
            return CRT.EXIT

    def _parse_index(self, ref: str):
        """解析引用中的索引（如 $last_msg_reg.0 返回 0）"""
        if ref.startswith('$'):
            parts = ref[1:].split('.')
            if len(parts) >= 2 and parts[0] in ('last_msg_reg', 'usr_tool_reg'):
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return None




def parse_instruction(raw: str) -> Instruction:
    """解析指令字符串为指令对象"""
    pass


class LMU:
    """LLM 调用模块"""

    client: OpenAI

    # 两个工具定义
    command_tool = {
        "type": "function",
        "function": {
            "name": "command",
            "description": "执行命令。如果提示词中没有命令格式，不要使用该工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "命令字符串",
                    },
                },
                "required": ["command"]
            }
        }
    }

    create_cmd_tool = {
        "type": "function",
        "function": {
            "name": "create_cmd",
            "description": "创建新的对话上下文",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_ref": {
                        "type": "string",
                        "description": "系统提示词引用（如 $MEM.sys）",
                    },
                    "user_ref": {
                        "type": "string",
                        "description": "用户消息引用（如 $MEM.usr）",
                    },
                    "para_ref": {
                        "type": "string",
                        "description": "参数引用（如 $MEM.para）",
                    },
                    "mode": {
                        "type": "string",
                        "description": "模式：a(追加) 或 w(写入)",
                        "enum": ["a", "w"]
                    }
                },
                "required": ["system_ref", "user_ref", "para_ref", "mode"]
            }
        }
    }

    tools = [command_tool, create_cmd_tool]

    def exec_crt(self, system_prompt: str, user_prompt: str, para: dict, current_utr_index: int = 0):
        """处理字符串输入的 create 模式
        system_prompt: 字符串，系统提示词
        user_prompt: 字符串，用户提示词
        para: 参数字典
        current_utr_index: 当前 utr 索引，用于工具调用
        """
        messages = [
            SystemMessage(content=system_prompt).to_dict(),
            UserMessage(content=user_prompt).to_dict()
        ]

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=self.tools if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    args = tool_call.function.arguments
                    command = args.get("command", "")
                    call_id = tool_call.id
                    # 将 call_id 追加到指令末尾
                    return_calls.append(f"{command} {call_id}")
                elif tool_call.function.name == "create_cmd":
                    args = tool_call.function.arguments
                    # 构造 create 指令，自动注入 current_utr_index
                    cmd = f"create {args['system_ref']} {args['user_ref']} {args['para_ref']} {args['mode']} {current_utr_index} {tool_call.id}"
                    return_calls.append(cmd)

        # 创建 Conversation 对象返回
        conversation = Conversation.from_any_list([
            (SYSTEM, system_prompt),
            (USER, user_prompt),
            (ASSISTANT, result or "")
        ])

        return result, return_calls, conversation

    def exec(self, conversation: Conversation, user_msg_batch: UserMessageBatch, para: dict, current_utr_index: int = 0):
        """执行对话
        conversation: 对话历史（已验证并封装）
        user_msg_batch: 用户消息批量输入
        para: 参数字典
        current_utr_index: 当前 utr 索引，用于工具调用
        """
        # 使用 Conversation 类型处理消息转换
        messages = conversation.to_api_messages()
        messages.extend(user_msg_batch.to_tool_messages())

        user_content = user_msg_batch.get_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=self.tools if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    args = tool_call.function.arguments
                    command = args.get("command", "")
                    call_id = tool_call.id
                    return_calls.append(f"{command} {call_id}")
                elif tool_call.function.name == "create_cmd":
                    args = tool_call.function.arguments
                    cmd = f"create {args['system_ref']} {args['user_ref']} {args['para_ref']} {args['mode']} {current_utr_index} {tool_call.id}"
                    return_calls.append(cmd)

        # 更新对话历史
        conversation.append_user_message(user_content or "")
        conversation.append_assistant_message(result or "")

        return result, return_calls, conversation


SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Core:
    def __init__(self):
        self.command_stack = []
        self.last_msg_reg = []  # 存储 Conversation 对象
        self.usr_tool_reg = []  # 存储 UserMessageBatch 对象
        self.mem = Memory()
        self.lmu = LMU()

    @property
    def MEM(self):
        """兼容旧代码的 MEM 访问"""
        return self.mem._data

    def run(self):
        while self.command_stack:
            raw = self.command_stack[-1]
            instruction = parse_instruction(raw)
            instruction.execute(self)

    def unwrap(self, value, for_llm=True):
        """解引用值
        只处理 $last_msg_reg 和 $usr_tool_reg 的寄存器访问
        其他情况调用 self.mem.unwrap
        """
        value = [value[0], *value[1:].split(".")]
        if value[1] == "last_msg_reg":
            assert value[0] == "$"
            return self.last_msg_reg[int(value[2])]
        elif value[1] == "usr_tool_reg":
            assert value[0] == "$"
            return self.usr_tool_reg[int(value[2])]
        else:
            return self.mem.unwrap(value, for_llm=for_llm)

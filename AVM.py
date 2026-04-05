

from collections import defaultdict

from openai import OpenAI

from avm_types import MetaList, MetaDict
from exceptions import VMSyntaxError, VMMemoryError
from messages import SystemMessage, UserMessage, Conversation, UserMessageBatch


class Instruction:
    """指令基类"""
    def execute(self, core: 'Core') -> None:
        raise NotImplementedError


class CreateInstruction(Instruction):
    """create 指令：发起新的对话"""
    def __init__(self, system_ref: str, user_ref: str, para_ref: str, mode: str, return_key_ref: str):
        self.system_ref = system_ref
        self.user_ref = user_ref
        self.para_ref = para_ref
        self.mode = mode  # 'a' 或 'w'
        self.return_key_ref = return_key_ref

    def execute(self, core: 'Core') -> None:
        system = core.unwrap(self.system_ref, for_llm=False)
        user = core.unwrap(self.user_ref, for_llm=False)
        para = core.unwrap(self.para_ref, for_llm=False)
        result, return_calls = core.lmu.exec_crt(system, user, para)
        if result:
            key = core.unwrap(self.return_key_ref, for_llm=False)
            if self.mode == 'a':
                if not isinstance(core.MEM[key], (MetaList, list)):
                    core.MEM[key] = MetaList()
                core.MEM[key].append(result)
            else:
                core.MEM[key] = result
        if return_calls:
            # 创建类型化的 Conversation 和 UserMessageBatch
            conversation = Conversation.from_any_list([(SYSTEM, system), (USER, user)])
            user_batch = UserMessageBatch()  # 空的工具响应批量
            # 寄存明确存储类型化对象
            core.last_msg_reg.append(conversation)
            core.usr_tool_reg.append(user_batch)
            this_command_context_id = len(core.last_msg_reg)
            this_command_user_reg_id = len(core.usr_tool_reg)
            # 替换栈顶为 exec 指令
            core.command_stack[-1] = f"exec $last_msg_reg.{this_command_context_id} $user_tool_reg.{this_command_user_reg_id} {self.para_ref} {self.mode} {self.return_key_ref}"
            core.command_stack.extend(return_calls[::-1])
        else:
            # 没有 return_calls 才 pop
            core.command_stack.pop()


class ExecInstruction(Instruction):
    """exec 指令：继续对话"""
    def __init__(self, last_msg_ref: str, user_msg_ref: str, para_ref: str, mode: str, return_key_ref: str, call_id_ref: str):
        self.last_msg_ref = last_msg_ref
        self.user_msg_ref = user_msg_ref
        self.para_ref = para_ref
        self.mode = mode
        self.return_key_ref = return_key_ref
        self.call_id_ref = call_id_ref

    def execute(self, core: 'Core') -> None:
        # 解析索引（格式：$last_msg_reg.0 或 $MEM.key）
        last_msg_idx = self._parse_index(self.last_msg_ref)
        user_msg_idx = self._parse_index(self.user_msg_ref)

        # 直接从寄存器获取类型化对象（不再需要转换）
        conversation: Conversation = core.unwrap(self.last_msg_ref, for_llm=False)
        user_batch: UserMessageBatch = core.unwrap(self.user_msg_ref, for_llm=False)
        para = core.unwrap(self.para_ref, for_llm=False)

        result, return_calls, updated_conversation = core.lmu.exec(conversation, user_batch, para)

        # 只有有 result 时才 pop 和写入内存
        if result:
            core.command_stack.pop()
            # pop 对应的寄存器（通过索引）
            if last_msg_idx is not None and last_msg_idx < len(core.last_msg_reg):
                core.last_msg_reg.pop(last_msg_idx)
            if user_msg_idx is not None and user_msg_idx < len(core.usr_tool_reg):
                core.usr_tool_reg.pop(user_msg_idx)
            # 写入内存（存储更新后的 conversation）
            if self.mode == 'a':
                core.MEM[self.return_key_ref].append((result, self.call_id_ref))
            else:
                core.MEM[self.return_key_ref] = updated_conversation

        # 有 return_calls 时压栈
        if return_calls:
            core.command_stack.extend(return_calls[::-1])

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


class SetMetadataInstruction(Instruction):
    """set_metadata 指令：设置元数据的元数据"""
    def __init__(self, target_ref: str, metadata: str):
        self.target_ref = target_ref
        self.metadata = metadata

    def execute(self, core: 'Core') -> None:
        target = core.unwrap(self.target_ref, for_llm=False)
        if isinstance(target, MetaList):
            target.set_metadata(self.metadata)
        elif isinstance(target, MetaDict):
            target.set_metadata(self.metadata)
        else:
            raise VMMemoryError(f"set_metadata 目标必须是 MetaList 或 MetaDict，得到 {type(target)}")
        core.command_stack.pop()


def parse_instruction(raw: str) -> Instruction:
    """解析指令字符串为指令对象"""
    parts = raw.split()
    if not parts:
        raise VMSyntaxError("空指令")
    op = parts[0]
    if op == "create":
        if len(parts) != 6:
            raise VMSyntaxError(f"create 需要 6 个参数，得到 {len(parts)}")
        return CreateInstruction(parts[1], parts[2], parts[3], parts[4], parts[5])
    elif op == "exec":
        if len(parts) != 7:
            raise VMSyntaxError(f"exec 需要 7 个参数，得到 {len(parts)}")
        return ExecInstruction(parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
    elif op == "set_metadata":
        if len(parts) < 3:
            raise VMSyntaxError(f"set_metadata 至少需要 2 个参数，得到 {len(parts)}")
        metadata = " ".join(parts[2:])
        return SetMetadataInstruction(parts[1], metadata)
    else:
        raise VMSyntaxError(f"未知指令：{op}")

class LMU:
    """LLM 调用模块"""

    client: OpenAI

    tools = {
        "type": "function",
        "function": {
            "name": "command",
            "description": "执行命令。如果提示词中没有命令格式，不要使用该工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "命令。如果提示词中没有命令格式，请忽略此字段",
                    }
                },
                "required": ["command"]
            }
        }
    }

    def exec_crt(self, system_prompt: str, user_prompt: str, para: dict):
        """处理字符串输入的 create 模式
        system_prompt: 字符串，系统提示词
        user_prompt: 字符串，用户提示词
        para: 参数字典
        """
        messages = [
            SystemMessage(content=system_prompt).to_dict(),
            UserMessage(content=user_prompt).to_dict()
        ]

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=[self.tools] if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    return_calls.append(f"{tool_call.function.arguments}")

        return result, return_calls

    def exec(self, conversation: Conversation, user_msg_batch: UserMessageBatch, para: dict):
        """执行对话
        conversation: 对话历史（已验证并封装）
        user_msg_batch: 用户消息批量输入
        para: 参数字典
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
            tools=[self.tools] if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    return_calls.append(f"{tool_call.function.arguments}")

        # 更新对话历史
        conversation.append_user_message(user_content)
        conversation.append_assistant_message(result or "")

        return result, return_calls, conversation

    @staticmethod
    def build_last_msg(last_msgs, _result, _return_calls):
        # 构建用于后续调用的消息历史
        return last_msgs

MEM=defaultdict(list)
SYSTEM="system"
USER="user"
ASSISTANT="assistant"


class Core:
    command_stack=[]
    last_msg_reg=[]
    usr_tool_reg=[]

    def run(self):
        while self.command_stack:
            raw = self.command_stack[-1]
            instruction = parse_instruction(raw)
            instruction.execute(self)

    def unwrap(self, value, for_llm=False):
        """解引用值
        for_llm: 如果为 True，对 MetaList/MetaDict 返回元数据字符串
        """
        value=[value[0],*value[1:].split(".")]
        if value[1]=="last_msg_reg":
            assert value[0]=="$"
            return self.last_msg_reg[int(value[2])]
        elif value[1]=="usr_tool_reg":
            assert value[0]=="$"
            return self.usr_tool_reg[int(value[2])]
        else:
            return self._mem_unwrap(value, for_llm=for_llm)

    def _mem_unwrap(self, value, for_llm=False):
        """从 MEM 中解引用值
        for_llm: 如果为 True，对 MetaList/MetaDict 返回 to_llm_string()
        """
        if value[0]=="$":
            temp=MEM
            for i in range(len(value)-1):
                temp=temp[value[i+1]]
            if isinstance(temp,str):
                if temp.startswith("$"):
                    temp_value=[temp[0],*temp[1:].split(".")]
                    temp=self._mem_unwrap(temp_value, for_llm=for_llm)
            # 循环结束后再处理类型
            if not isinstance(temp,str):
                if isinstance(temp, MetaList):
                    return temp.to_llm_string() if for_llm else temp
                if isinstance(temp, MetaDict):
                    return temp.to_llm_string() if for_llm else temp
                if isinstance(temp,list):
                    return temp[0] #期望约定：列表中第一个元素是列表元数据
                if isinstance(temp,dict):
                    return f"dict.keys:{temp.keys()}"
            return temp
        if value[0]=="&":
            temp=MEM
            for i in range(len(value)-1):
                temp=temp[value[i+1]]
            if isinstance(temp,str):
                return temp
            if not isinstance(temp,str):
                if isinstance(temp, MetaList):
                    return temp.to_llm_string() if for_llm else temp
                if isinstance(temp, MetaDict):
                    return temp.to_llm_string() if for_llm else temp
                if isinstance(temp,list):
                    return temp[0]
                if isinstance(temp,dict):
                    return f"dict.keys:{temp.keys()}"

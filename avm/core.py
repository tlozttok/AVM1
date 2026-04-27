

from enum import Enum

from openai import OpenAI
from dotenv import load_dotenv
import json
import logging
import os
import socket
import threading
import time

from .types import MetaList, MetaDict
from .exceptions import VMSyntaxError, VMMemoryError
from .messages import SystemMessage, UserMessage, Conversation, UserMessageBatch
from .memory import Memory

logger = logging.getLogger(__name__)

class CommandReturnType(Enum):
    """指令返回类型"""
    EXIT = 0
    CONTINUE = 1

CRT = CommandReturnType

class Instruction:
    """指令基类"""
    def __init__(self,call_id: str, utr_index: int, **kargs):
        self.call_id = call_id
        self.utr_index = utr_index
        for k, v in kargs.items():
            setattr(self, k, v)
        
    def execute(self, core: 'Core') -> CRT:
        raise NotImplementedError
    
class MemoryReadInstruction(Instruction):
    """memory_read 指令：从内存中读取数据"""
    call_id: str
    utr_index: int
    ref: str
    def __init__(self, call_id: str, utr_index: int, ref: str, **kargs):
        super().__init__(call_id, utr_index, ref=ref, **kargs)
    
    def execute(self, core: 'Core') -> CRT:
        logger.info("[memory_read] call_id=%s ref=%s", self.call_id, self.ref)
        user_batch = core.usr_tool_reg[self.utr_index]
        try:
            content = core.unwrap(self.ref, for_llm=True)
            logger.debug("[memory_read] content=%r", content)
            user_batch.add_tool_response(content, self.call_id)
        except (VMMemoryError, KeyError, IndexError, TypeError, ValueError) as e:
            logger.error("[memory_read] error: %s", e)
            user_batch.add_tool_response(f"Error: {e}", self.call_id)
        except Exception as e:
            logger.exception("[memory_read] unexpected error")
            user_batch.add_tool_response(f"Internal Error: {type(e).__name__}: {e}", self.call_id)
        logger.info("[memory_read] done call_id=%s", self.call_id)
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
    
    def execute(self, core: 'Core') -> CRT:
        logger.info("[memory_write] call_id=%s ref=%s", self.call_id, self.ref)
        user_batch = core.usr_tool_reg[self.utr_index]
        try:
            core.mem.set(self.ref, self.content)
            user_batch.add_tool_response(f"Success set: {self.ref}", self.call_id)
            logger.info("[memory_write] done call_id=%s", self.call_id)
        except VMMemoryError as e:
            logger.error("[memory_write] error: %s", e)
            user_batch.add_tool_response(f"Error: {e}", self.call_id)
            return CRT.EXIT
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
        logger.info("[memory_make] call_id=%s ref=%s key=%s type=%s", self.call_id, self.ref, self.key, self.mem_type)
        user_batch = core.usr_tool_reg[self.utr_index]
        try:
            core.mem.make(self.ref, self.key, self.mem_type)
            user_batch.add_tool_response(f"Success created {self.mem_type} at {self.ref}.{self.key}", self.call_id)
            logger.info("[memory_make] done call_id=%s", self.call_id)
        except VMMemoryError as e:
            logger.error("[memory_make] error: %s", e)
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

    def execute(self, core: 'Core') -> CRT:
        logger.info("[create] call_id=%s utr=%s", self.call_id, self.utr_index)
        system = core.unwrap(self.system_ref)
        user = core.unwrap(self.user_ref)
        para = core.unwrap(self.para_ref, for_llm=False)
        logger.debug("[create] system_ref=%s user_ref=%s para=%s", self.system_ref, self.user_ref, para)
        result, return_calls, conversation = core.lmu.exec_crt(system, user, para)
        logger.debug("[create] result=%r return_calls=%s", result, len(return_calls))

        # 处理 return_calls
        if return_calls:
            logger.info("[create] return_calls=%d", len(return_calls))
            # 分配新的寄存器槽位
            last_msg_idx = len(core.last_msg_reg)
            user_msg_idx = len(core.usr_tool_reg)

            # 存入 Conversation
            core.last_msg_reg.append(conversation)
            core.usr_tool_reg.append(UserMessageBatch())

            # 替换栈顶为 exec 指令
            core.command_stack[-1] = ExecInstruction(
                self.call_id, self.utr_index,
                f"$last_msg_reg.{last_msg_idx}",
                f"$usr_tool_reg.{user_msg_idx}",
                self.para_ref
            )
            # 压入子指令（反序）
            for rc in reversed(return_calls):
                instr = _make_instruction(rc, user_msg_idx)
                if instr is not None:
                    core.command_stack.append(instr)
            return CRT.CONTINUE
        else:
            if result and self.utr_index != -1:
                user_batch = core.usr_tool_reg[self.utr_index]
                user_batch.add_tool_response(result, self.call_id)
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

    def execute(self, core: 'Core') -> CRT:
        logger.info("[exec] call_id=%s utr=%s", self.call_id, self.utr_index)
        # 解析索引（格式：$last_msg_reg.0 或 $MEM.key）
        last_msg_idx = self._parse_index(self.last_msg_ref)
        user_msg_idx = self._parse_index(self.user_msg_ref)

        # 直接从寄存器获取类型化对象
        conversation: Conversation = core.unwrap(self.last_msg_ref)
        user_batch: UserMessageBatch = core.unwrap(self.user_msg_ref)
        para = core.unwrap(self.para_ref, for_llm=False)

        # 调用 LMU.exec
        result, return_calls, _ = core.lmu.exec(conversation, user_batch, para)
        logger.debug("[exec] result=%r return_calls=%d", result, len(return_calls))
        user_batch.clear()

        # 处理 return_calls：不弹出当前指令，直接压栈
        if return_calls:
            logger.info("[exec] pushing %d sub-instructions", len(return_calls))
            for rc in reversed(return_calls):
                instr = _make_instruction(rc, user_msg_idx)
                if instr is not None:
                    core.command_stack.append(instr)
            return CRT.CONTINUE
        else:
            if result and self.utr_index != -1:
                parent_batch = core.usr_tool_reg[self.utr_index]
                parent_batch.add_tool_response(result, self.call_id)
            # pop 对应的寄存器
            if user_msg_idx is not None and user_msg_idx < len(core.usr_tool_reg):
                core.usr_tool_reg.pop(user_msg_idx)
            if last_msg_idx is not None and last_msg_idx < len(core.last_msg_reg):
                core.last_msg_reg.pop(last_msg_idx)
            logger.info("[exec] done call_id=%s", self.call_id)
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
    """解析指令字符串为指令对象
    统一格式: <cmd_type> <call_id> <utr_index> <...args>
    """
    from .exceptions import VMSyntaxError
    logger.debug("[parse_instruction] raw=%r", raw)
    parts = raw.strip().split()
    if not parts:
        raise VMSyntaxError("空指令")
    
    cmd_type = parts[0]
    call_id = parts[1] if len(parts) > 1 else ""
    utr_index = int(parts[2]) if len(parts) > 2 else -1
    
    if cmd_type == "create":
        system_ref = parts[3] if len(parts) > 3 else ""
        user_ref = parts[4] if len(parts) > 4 else ""
        para_ref = parts[5] if len(parts) > 5 else ""
        return CreateInstruction(call_id, utr_index, system_ref, user_ref, para_ref)
    
    elif cmd_type == "exec":
        last_msg_ref = parts[3] if len(parts) > 3 else ""
        user_msg_ref = parts[4] if len(parts) > 4 else ""
        para_ref = parts[5] if len(parts) > 5 else ""
        return ExecInstruction(call_id, utr_index, last_msg_ref, user_msg_ref, para_ref)
    
    elif cmd_type == "memory_read":
        ref = parts[3] if len(parts) > 3 else ""
        return MemoryReadInstruction(call_id, utr_index, ref)
    
    elif cmd_type == "memory_write":
        ref = parts[3] if len(parts) > 3 else ""
        content = parts[4] if len(parts) > 4 else ""
        return MemoryWriteInstruction(call_id, utr_index, ref, content)
    
    elif cmd_type == "memory_make":
        ref = parts[3] if len(parts) > 3 else ""
        key = parts[4] if len(parts) > 4 else ""
        mem_type = parts[5] if len(parts) > 5 else ""
        return MemoryMakeInstruction(call_id, utr_index, ref, key, mem_type)
    
    else:
        raise VMSyntaxError(f"未知指令类型：{cmd_type}")


def _make_instruction(rc: dict, utr_index: int) -> Instruction:
    """根据 LMU 返回的半成品对象构造完整指令
    rc 格式: {"call_id": str, "cmd_type": str, "args": dict}
    """
    cmd_type = rc.get("cmd_type", "")
    call_id = rc.get("call_id", "")
    args = rc.get("args", {})
    
    if cmd_type == "create":
        return CreateInstruction(
            call_id, utr_index,
            args.get("system_ref", ""),
            args.get("user_ref", ""),
            args.get("para_ref", "")
        )
    elif cmd_type == "memory_read":
        return MemoryReadInstruction(call_id, utr_index, args.get("ref", ""))
    elif cmd_type == "memory_write":
        return MemoryWriteInstruction(call_id, utr_index, args.get("ref", ""), args.get("content", ""))
    elif cmd_type == "memory_make":
        return MemoryMakeInstruction(call_id, utr_index, args.get("ref", ""), args.get("key", ""), args.get("mem_type", ""))
    elif cmd_type == "command":
        # placeholder: command 工具尚未设计
        return None
    else:
        return None


class LMU:
    """LLM 调用模块"""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

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
                        "description": "系统提示词引用（如 $MEM.sys），也可以是字面值（不以$或&开头）",
                    },
                    "user_ref": {
                        "type": "string",
                        "description": "用户消息引用（如 $MEM.usr），也可以是字面值（不以$或&开头）",
                    },
                    "para_ref": {
                        "type": "string",
                        "description": "调用模型时候的参数的引用（如 $MEM.para），请从内存中寻找格式正确的",
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

    memory_read_tool = {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "从内存中读取数据，通过引用获取值",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "内存引用，如 $MEM.input",
                    }
                },
                "required": ["ref"]
            }
        }
    }

    memory_write_tool = {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "向内存写入数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "内存引用，如 $MEM.output",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的内容",
                    }
                },
                "required": ["ref", "content"]
            }
        }
    }

    memory_make_tool = {
        "type": "function",
        "function": {
            "name": "memory_make",
            "description": "在指定内存路径下创建新的子地址",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "父级内存引用，如 $MEM.data",
                    },
                    "key": {
                        "type": "string",
                        "description": "新地址的键名",
                    },
                    "mem_type": {
                        "type": "string",
                        "description": "要创建的类型",
                        "enum": ["str", "dict", "list"]
                    }
                },
                "required": ["ref", "key", "mem_type"]
            }
        }
    }

    tools = [command_tool, create_cmd_tool, memory_read_tool, memory_write_tool, memory_make_tool]

    # OpenAI API 允许作为 kwargs 传入的参数白名单
    _API_PARAM_KEYS = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "extra_body",
        "seed", "logit_bias", "logprobs", "top_logprobs",
        "n", "response_format", "timeout",
    }

    def _filter_api_params(self, para: dict) -> dict:
        """只保留 OpenAI API 支持的参数，过滤掉业务数据"""
        return {k: v for k, v in para.items() if k in self._API_PARAM_KEYS}


    def exec_crt(self, system_prompt: str, user_prompt: str, para: dict):
        """处理字符串输入的 create 模式
        system_prompt: 字符串，系统提示词
        user_prompt: 字符串，用户提示词
        para: 参数字典
        """
        logger.info("[LMU.exec_crt] model=%s use_tool=%s", para.get("model"), para.get("use_tool"))
        messages = [
            SystemMessage(content=system_prompt).to_dict(),
            UserMessage(content=user_prompt).to_dict()
        ]
        logger.debug("[LMU.exec_crt] messages=%s", messages)

        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=self.tools,
            tool_choice=use_tool,
            **extra_para
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []
        logger.debug("[LMU.exec] response content=%r tool_calls=%s", result, bool(message.tool_calls))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                call_id = tool_call.id
                if tool_call.function.name == "command":
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command", "")
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "command",
                        "raw": command
                    })
                elif tool_call.function.name == "create_cmd":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "create",
                        "args": {
                            "system_ref": args.get("system_ref", ""),
                            "user_ref": args.get("user_ref", ""),
                            "para_ref": args.get("para_ref", "")
                        }
                    })
                elif tool_call.function.name == "memory_read":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_read",
                        "args": {
                            "ref": args.get("ref", "")
                        }
                    })
                elif tool_call.function.name == "memory_write":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_write",
                        "args": {
                            "ref": args.get("ref", ""),
                            "content": args.get("content", "")
                        }
                    })
                elif tool_call.function.name == "memory_make":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_make",
                        "args": {
                            "ref": args.get("ref", ""),
                            "key": args.get("key", ""),
                            "mem_type": args.get("mem_type", "")
                        }
                    })

        # 创建 Conversation 对象返回，assistant 消息必须保留 tool_calls
        assistant_msg = {"role": ASSISTANT, "content": result or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        conversation = Conversation.from_any_list([
            (SYSTEM, system_prompt),
            (USER, user_prompt),
            assistant_msg
        ])

        return result, return_calls, conversation

    def exec(self, conversation: Conversation, user_msg_batch: UserMessageBatch, para: MetaDict):
        """执行对话
        conversation: 对话历史（已验证并封装）
        user_msg_batch: 用户消息批量输入
        para: 参数字典
        """
        logger.info("[LMU.exec] model=%s use_tool=%s", para.get("model"), para.get("use_tool", False))
        # 使用 Conversation 类型处理消息转换
        messages = conversation.to_api_messages()
        messages.extend(user_msg_batch.to_tool_messages())
        logger.debug("[LMU.exec] messages_count=%d", len(messages))

        user_content = user_msg_batch.get_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})

        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=self.tools,
            tool_choice=use_tool,
            **extra_para
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []
        logger.debug("[LMU.exec] response content=%r tool_calls=%s", result, bool(message.tool_calls))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                call_id = tool_call.id
                if tool_call.function.name == "command":
                    args = json.loads(tool_call.function.arguments)
                    command = args.get("command", "")
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "command",
                        "raw": command
                    })
                elif tool_call.function.name == "create_cmd":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "create",
                        "args": {
                            "system_ref": args.get("system_ref", ""),
                            "user_ref": args.get("user_ref", ""),
                            "para_ref": args.get("para_ref", "")
                        }
                    })
                elif tool_call.function.name == "memory_read":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_read",
                        "args": {
                            "ref": args.get("ref", "")
                        }
                    })
                elif tool_call.function.name == "memory_write":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_write",
                        "args": {
                            "ref": args.get("ref", ""),
                            "content": args.get("content", "")
                        }
                    })
                elif tool_call.function.name == "memory_make":
                    args = json.loads(tool_call.function.arguments)
                    return_calls.append({
                        "call_id": call_id,
                        "cmd_type": "memory_make",
                        "args": {
                            "ref": args.get("ref", ""),
                            "key": args.get("key", ""),
                            "mem_type": args.get("mem_type", "")
                        }
                    })

        # 更新对话历史：tool 响应 -> user 输入(如有) -> assistant 回复
        # 必须先把 tool 响应保存到 conversation，否则下次 exec 时 conversation
        # 中缺少 tool 消息，API 会报 "tool_calls 没有对应的 tool 响应"
        for resp in user_msg_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)

        if user_content:
            conversation.append_user_message(user_content)

        tc_list = None
        if message.tool_calls:
            tc_list = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        conversation.append_assistant_message(result or "", tool_calls=tc_list)

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
        self.debug = False
        self._monitor_thread = None
        self._monitor_running = False

    def start_memory_monitor(self, output_file: str, interval: float = 0.3,
                             socket_path: str | None = None):
        """启动后台线程，定期将内存树写入文件，可选开启 Unix socket 交互查询"""
        self._monitor_running = True

        def _monitor_loop():
            while self._monitor_running:
                try:
                    dump = self.mem.dump_tree()
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(dump)
                except Exception:
                    pass
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[Core] memory monitor started, output=%s interval=%s", output_file, interval)

        if socket_path:
            self._start_mem_socket_server(socket_path)

    def _start_mem_socket_server(self, socket_path: str):
        """在独立线程中启动 Unix socket server，接受路径查询返回结果"""

        def _server():
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(socket_path)
            sock.listen(1)
            sock.settimeout(0.5)
            logger.info("[Core] memory socket server listening on %s", socket_path)
            while self._monitor_running:
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
                except Exception:
                    break
                with conn:
                    try:
                        data = conn.recv(4096).decode("utf-8").strip()
                        if data:
                            result = self.mem.query_path(data)
                            payload = result.encode("utf-8")
                            header = format(len(payload), "08x").encode()
                            conn.sendall(header + payload)
                    except Exception:
                        pass
            sock.close()
            try:
                os.unlink(socket_path)
            except OSError:
                pass

        threading.Thread(target=_server, daemon=True).start()



    def run(self):
        logger.info("[Core.run] start, stack_size=%d", len(self.command_stack))
        while self.command_stack:
            instruction = self.command_stack[-1]
            if isinstance(instruction, str):
                instruction = parse_instruction(instruction)
            logger.debug("[Core.run] executing %s(call_id=%s)", type(instruction).__name__, getattr(instruction, 'call_id', 'N/A'))
            return_type = instruction.execute(self)
            if self.debug:
                debug_event = getattr(self, '_debug_event', None)
                if debug_event is not None:
                    debug_event.clear()
                    debug_event.wait()
                else:
                    input("[核心循环] Press Enter to continue...")
            if return_type == CRT.EXIT:
                self.command_stack.pop()
                logger.debug("[Core.run] EXIT, stack_size=%d", len(self.command_stack))
            elif return_type == CRT.CONTINUE:
                logger.debug("[Core.run] CONTINUE, stack_size=%d", len(self.command_stack))
                continue
        logger.info("[Core.run] end")

    def unwrap(self, value, for_llm=True):
        """解引用值
        只处理 $last_msg_reg 和 $usr_tool_reg 的寄存器访问
        其他情况调用 self.mem.unwrap
        不以 $ 开头则视为字面值直接返回
        """
        if not value.startswith("$"):
            return value
        value = [value[0], *value[1:].split(".")]
        if value[1] == "last_msg_reg":
            assert value[0] == "$"
            return self.last_msg_reg[int(value[2])]
        elif value[1] == "usr_tool_reg":
            assert value[0] == "$"
            return self.usr_tool_reg[int(value[2])]
        else:
            return self.mem.unwrap(value, for_llm=for_llm)

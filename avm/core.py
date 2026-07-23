

from enum import Enum
from typing import Optional, List, Dict, Any, Callable

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
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict
from .memory import Memory

logger = logging.getLogger(__name__)

class CommandReturnType(Enum):
    """指令返回类型"""
    EXIT = 0
    CONTINUE = 1

CRT = CommandReturnType

class Instruction:
    """指令基类"""
    call_id: str
    caller_id: int

    def __init__(self,call_id: str, caller_id: int, **kargs):
        self.call_id = call_id
        self.caller_id = caller_id
        for k, v in kargs.items():
            setattr(self, k, v)
        
    def execute(self, core: 'Core') -> CRT:
        raise NotImplementedError
    
class MemoryReadInstruction(Instruction):
    """memory_read 指令：从内存中读取数据"""
    call_id: str
    caller_id: int
    ref: str
    def __init__(self, call_id: str, caller_id: int, ref: str, **kargs):
        super().__init__(call_id, caller_id, ref=ref, **kargs)
    
    def execute(self, core: 'Core') -> CRT:
        logger.info("[memory_read] call_id=%s ref=%s", self.call_id, self.ref)
        user_batch = core._conv_by_cid[self.caller_id].user_batch
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
    caller_id: int
    ref: str
    content: str
    def __init__(self, call_id: str, caller_id: int, ref: str, content: str, **kargs):
        super().__init__(call_id, caller_id, ref=ref, content=content, **kargs)
        self.content = content
    
    def execute(self, core: 'Core') -> CRT:
        logger.info("[memory_write] call_id=%s ref=%s", self.call_id, self.ref)
        user_batch = core._conv_by_cid[self.caller_id].user_batch
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
    caller_id: int
    ref: str
    key: str
    mem_type: str

    def __init__(self, call_id: str, caller_id: int, ref: str, key: str, mem_type: str, **kargs):
        super().__init__(call_id, caller_id, ref=ref, key=key, mem_type=mem_type, **kargs)

    def execute(self, core: 'Core') -> CRT:
        logger.info("[memory_make] call_id=%s ref=%s key=%s type=%s", self.call_id, self.ref, self.key, self.mem_type)
        user_batch = core._conv_by_cid[self.caller_id].user_batch
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
    system_ref: str
    user_ref: str
    para_ref: str

    def __init__(self, call_id: str, system_ref: str, user_ref: str, para_ref: str, **kargs):
        super().__init__(call_id, system_ref=system_ref, user_ref=user_ref, para_ref=para_ref, **kargs)

    def execute(self, core: 'Core') -> CRT:
        logger.info("[create] call_id=%s caller=%s", self.call_id, self.caller_id)
        try:
            system = core.unwrap(self.system_ref)
            user = core.unwrap(self.user_ref)
            para = core.unwrap(self.para_ref, for_llm=False)
        except VMMemoryError as e:
            logger.error("[create] error: %s", e)
            if self.caller_id != -1:
                core._conv_by_cid[self.caller_id].user_batch.add_tool_response(
                    f"[参数错误] {e}。请检查 create 指令的 memory 引用是否正确。"
                    f"确认引用路径是否存在，必要时先用 memory_make 创建。",
                    self.call_id
                )
            return CRT.EXIT
        logger.debug("[create] system_ref=%s user_ref=%s para=%s", self.system_ref, self.user_ref, para)
        result, return_calls, messages_list = core.lmu.exec_crt(system, user, para)
        conversation = Conversation(
            messages=messages_list,
            cid=0,  # 由 core._register 覆盖
            is_sub=(self.caller_id != -1),
            parent=core._conv_by_cid[self.caller_id] if self.caller_id != -1 else None,
        )
        core._register(conversation)
        logger.debug("[create] result=%r return_calls=%s cid=%s", result, len(return_calls), conversation.cid)

        # 处理 return_calls
        if return_calls:
            logger.info("[create] return_calls=%d", len(return_calls))
            core._notify("tool_calls_detected", {
                "source": "create",
                "call_id": self.call_id,
                "caller_id": self.caller_id,
                "tool_calls": return_calls,
            })

            core._notify("conversation_created", {
                "call_id": self.call_id,
                "cid": conversation.cid,
                "messages": [message_to_api_dict(m) for m in conversation.messages],
            })

            core.command_stack[-1] = ExecInstruction(
                self.call_id, self.caller_id,
                f"$conv.{conversation.cid}",
                f"$batch.{conversation.cid}",
                self.para_ref
            )
            # 压入子指令（反序）
            for rc in reversed(return_calls):
                instr = _make_instruction(rc, conversation.cid)
                if instr is not None:
                    core.command_stack.append(instr)
                else:
                    batch = conversation.user_batch
                    args = rc.get("args", {})
                    detail = args.get("error", "")
                    name = args.get("name", rc.get("cmd_type", "?"))
                    msg = f"Error: {name} 执行失败"
                    if detail:
                        msg += f" — {detail}"
                    batch.add_tool_response(msg, rc.get("call_id", ""))
            return CRT.CONTINUE
        else:
            if result and self.caller_id != -1:
                core._conv_by_cid[self.caller_id].user_batch.add_tool_response(result, self.call_id)
            # --- monitor: conversation 完成（无子调用）---
            if conversation is not None:
                core._notify("conversation_completed", {
                    "call_id": self.call_id,
                    "messages": [message_to_api_dict(m) for m in conversation.messages],
                })
            return CRT.EXIT


class ExecInstruction(Instruction):
    """exec 指令：继续对话"""
    call_id: str
    caller_id: int
    conv_ref: str
    batch_ref: str
    para_ref: str

    def __init__(self, call_id: str, caller_id: int, conv_ref: str, batch_ref: str, para_ref: str, **kargs):
        super().__init__(call_id, caller_id, conv_ref=conv_ref, batch_ref=batch_ref, para_ref=para_ref, **kargs)

    def execute(self, core: 'Core') -> CRT:
        logger.info("[exec] call_id=%s caller=%s", self.call_id, self.caller_id)

        try:
            conversation: Conversation = core.unwrap(self.conv_ref)
            para = core.unwrap(self.para_ref, for_llm=False)
        except VMMemoryError as e:
            logger.error("[exec] error: %s", e)
            if self.caller_id != -1:
                core._conv_by_cid[self.caller_id].user_batch.add_tool_response(
                    f"[参数错误] {e}。请检查 exec 指令的 memory 引用是否正确。"
                    f"确认引用路径是否存在，必要时先用 memory_make 创建。",
                    self.call_id
                )
            return CRT.EXIT

        # 调用 LMU.exec
        result, return_calls, _ = core.lmu.exec(conversation, para)
        logger.debug("[exec] result=%r return_calls=%d", result, len(return_calls))
        conversation.user_batch.clear()

        if return_calls:
            logger.info("[exec] pushing %d sub-instructions", len(return_calls))
            core._notify("tool_calls_detected", {
                "source": "exec",
                "call_id": self.call_id,
                "caller_id": self.caller_id,
                "cid": conversation.cid,
                "tool_calls": return_calls,
            })
            for rc in reversed(return_calls):
                instr = _make_instruction(rc, conversation.cid)
                if instr is not None:
                    core.command_stack.append(instr)
                else:
                    args = rc.get("args", {})
                    detail = args.get("error", "")
                    name = args.get("name", rc.get("cmd_type", "?"))
                    msg = f"Error: {name} 执行失败"
                    if detail:
                        msg += f" — {detail}"
                    conversation.user_batch.add_tool_response(msg, rc.get("call_id", ""))
            return CRT.CONTINUE
        else:
            if result and self.caller_id != -1:
                core._conv_by_cid[self.caller_id].user_batch.add_tool_response(result, self.call_id)
            core._notify("conversation_updated", {
                "call_id": self.call_id,
                "cid": conversation.cid,
                "messages": [message_to_api_dict(m) for m in conversation.messages],
                "closed": True,
            })
            logger.info("[exec] done call_id=%s cid=%s", self.call_id, conversation.cid)
            return CRT.EXIT


def parse_instruction(raw: str) -> Instruction:
    """解析指令字符串为指令对象
    统一格式: <cmd_type> <call_id> <caller_id> <...args>
    """
    from .exceptions import VMSyntaxError
    logger.debug("[parse_instruction] raw=%r", raw)
    parts = raw.strip().split()
    if not parts:
        raise VMSyntaxError("空指令")
    
    cmd_type = parts[0]
    call_id = parts[1] if len(parts) > 1 else ""
    caller_id = int(parts[2]) if len(parts) > 2 else -1
    
    if cmd_type == "create":
        system_ref = parts[3] if len(parts) > 3 else ""
        user_ref = parts[4] if len(parts) > 4 else ""
        para_ref = parts[5] if len(parts) > 5 else ""
        return CreateInstruction(call_id, caller_id, system_ref, user_ref, para_ref)
    
    elif cmd_type == "exec":
        conv_ref = parts[3] if len(parts) > 3 else ""
        batch_ref = parts[4] if len(parts) > 4 else ""
        para_ref = parts[5] if len(parts) > 5 else ""
        return ExecInstruction(call_id, caller_id, conv_ref, batch_ref, para_ref)
    
    elif cmd_type == "memory_read":
        ref = parts[3] if len(parts) > 3 else ""
        return MemoryReadInstruction(call_id, caller_id, ref)
    
    elif cmd_type == "memory_write":
        ref = parts[3] if len(parts) > 3 else ""
        content = parts[4] if len(parts) > 4 else ""
        return MemoryWriteInstruction(call_id, caller_id, ref, content)
    
    elif cmd_type == "memory_make":
        ref = parts[3] if len(parts) > 3 else ""
        key = parts[4] if len(parts) > 4 else ""
        mem_type = parts[5] if len(parts) > 5 else ""
        return MemoryMakeInstruction(call_id, caller_id, ref, key, mem_type)
    
    else:
        raise VMSyntaxError(f"未知指令类型：{cmd_type}")


def _make_instruction(rc: dict, caller_id: int) -> Instruction:
    """根据 LMU 返回的半成品对象构造完整指令
    rc 格式: {"call_id": str, "cmd_type": str, "args": dict}
    """
    cmd_type = rc.get("cmd_type", "")
    call_id = rc.get("call_id", "")
    args = rc.get("args", {})
    
    if cmd_type == "create":
        return CreateInstruction(
            call_id, caller_id,
            args.get("system_ref", ""),
            args.get("user_ref", ""),
            args.get("para_ref", "")
        )
    elif cmd_type == "memory_read":
        return MemoryReadInstruction(call_id, caller_id, args.get("ref", ""))
    elif cmd_type == "memory_write":
        return MemoryWriteInstruction(call_id, caller_id, args.get("ref", ""), args.get("content", ""))
    elif cmd_type == "memory_make":
        return MemoryMakeInstruction(call_id, caller_id, args.get("ref", ""), args.get("key", ""), args.get("mem_type", ""))
    elif cmd_type == "command":
        return None
    elif cmd_type == "json_error":
        return None
    else:
        return None


class LMU:
    """LLM 调用模块"""

    _client: Optional[OpenAI] = None

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    # 两个工具定义
    command_tool: dict = {
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

    create_cmd_tool: dict = {
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

    memory_read_tool: dict = {
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

    memory_write_tool: dict = {
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

    memory_make_tool: dict = {
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

    tools: list = [command_tool, create_cmd_tool, memory_read_tool, memory_write_tool, memory_make_tool]

    # OpenAI API 允许作为 kwargs 传入的参数白名单
    _API_PARAM_KEYS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "extra_body",
        "seed", "logit_bias", "logprobs", "top_logprobs",
        "n", "response_format", "timeout",
    }

    def _filter_api_params(self, para: dict) -> dict:
        """只保留 OpenAI API 支持的参数，过滤掉业务数据"""
        return {k: v for k, v in para.items() if k in self._API_PARAM_KEYS}


    def _parse_tool_args(self, tool_call, return_calls: list):
        """安全解析 tool call 的 arguments JSON，失败时生成错误 return_call"""
        call_id = tool_call.id
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error("[LMU] JSON decode error for %s (%s): %s", tool_call.function.name, call_id, e)
            return_calls.append({
                "call_id": call_id,
                "cmd_type": "json_error",
                "args": {"error": str(e), "name": tool_call.function.name},
            })
            return None

    def exec_crt(self, system_prompt: str, user_prompt: str, para: dict):
        """处理字符串输入的 create 模式
        返回 (result, return_calls, messages_list)
        """
        logger.info("[LMU.exec_crt] model=%s use_tool=%s", para.get("model"), para.get("use_tool"))
        messages_list: list = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]
        messages = [message_to_api_dict(m) for m in messages_list]
        logger.debug("[LMU.exec_crt] model=%s, msg_count=%d system_preview=%r user_preview=%r",
            para.get("model"), 2,
            system_prompt[:100], user_prompt[:100])

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
        logger.debug("[LMU.exec] response content=%r tool_calls=%s", result[:200] if result else result, bool(message.tool_calls))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                call_id = tool_call.id
                name = tool_call.function.name
                if name == "command":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    command = args.get("command", "")
                    return_calls.append({"call_id": call_id, "cmd_type": "command", "raw": command})
                elif name == "create_cmd":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "create", "args": {
                        "system_ref": args.get("system_ref", ""),
                        "user_ref": args.get("user_ref", ""),
                        "para_ref": args.get("para_ref", "")
                    }})
                elif name == "memory_read":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_read", "args": {"ref": args.get("ref", "")}})
                elif name == "memory_write":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_write", "args": {
                        "ref": args.get("ref", ""), "content": args.get("content", "")
                    }})
                elif name == "memory_make":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_make", "args": {
                        "ref": args.get("ref", ""), "key": args.get("key", ""), "mem_type": args.get("mem_type", "")
                    }})

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
        messages_list.append(AssistantMessage(content=result or "", tool_calls=tc_list))

        return result, return_calls, messages_list

    def exec(self, conversation: Conversation, para: MetaDict):
        """执行对话
        conversation: 对话历史（含内嵌的 user_batch）
        para: 参数字典
        """
        logger.info("[LMU.exec] model=%s use_tool=%s", para.get("model"), para.get("use_tool", False))
        # 使用 Conversation 类型处理消息转换
        messages = conversation.to_api_messages()
        messages.extend(conversation.user_batch.to_tool_messages())
        logger.debug("[LMU.exec] messages_count=%d", len(messages))

        user_content = conversation.user_batch.get_user_content()
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
        logger.debug("[LMU.exec] response content=%r tool_calls=%s", result[:200] if result else result, bool(message.tool_calls))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                call_id = tool_call.id
                name = tool_call.function.name
                if name == "command":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    command = args.get("command", "")
                    return_calls.append({"call_id": call_id, "cmd_type": "command", "raw": command})
                elif name == "create_cmd":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "create", "args": {
                        "system_ref": args.get("system_ref", ""),
                        "user_ref": args.get("user_ref", ""),
                        "para_ref": args.get("para_ref", "")
                    }})
                elif name == "memory_read":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_read", "args": {"ref": args.get("ref", "")}})
                elif name == "memory_write":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_write", "args": {
                        "ref": args.get("ref", ""), "content": args.get("content", "")
                    }})
                elif name == "memory_make":
                    args = self._parse_tool_args(tool_call, return_calls)
                    if args is None: continue
                    return_calls.append({"call_id": call_id, "cmd_type": "memory_make", "args": {
                        "ref": args.get("ref", ""), "key": args.get("key", ""), "mem_type": args.get("mem_type", "")
                    }})

        # 更新对话历史：tool 响应 -> user 输入(如有) -> assistant 回复
        # 必须先把 tool 响应保存到 conversation，否则下次 exec 时 conversation
        # 中缺少 tool 消息，API 会报 "tool_calls 没有对应的 tool 响应"
        for resp in conversation.user_batch.tool_responses:
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
        self.command_stack: list = []
        self._conv_by_cid: Dict[int, Conversation] = {}
        self._ready_covn:List[int]=[]
        self._dormant_convs: List[int] = []
        self._next_cid: int = 0
        self.mem: Memory = Memory()
        self.lmu: LMU = LMU()
        self.debug: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running: bool = False
        self._state_observers: List[Callable] = []
        self._observer_lock: threading.Lock = threading.Lock()

    def _register(self, conv: Conversation) -> Conversation:
        conv.cid = self._next_cid
        self._conv_by_cid[conv.cid] = conv
        self._next_cid += 1
        return conv
    
    def _unregister(self, cid: int) -> None:
        if cid in self._conv_by_cid:
            del self._conv_by_cid[cid]
    
    def get_conversation(self, cid: int) -> Optional[Conversation]:
        return self._conv_by_cid.get(cid)


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

    # ------------------------------------------------------------------
    # 状态观察器（供外部实时监控使用）
    # ------------------------------------------------------------------

    def add_state_observer(self, fn):
        """注册状态观察回调。fn 签名: (event_type: str, payload: dict) -> None"""
        with self._observer_lock:
            if fn not in self._state_observers:
                self._state_observers.append(fn)

    def remove_state_observer(self, fn):
        """注销状态观察回调"""
        with self._observer_lock:
            if fn in self._state_observers:
                self._state_observers.remove(fn)

    def _notify(self, event_type: str, payload: dict):
        """通知所有观察者（在 Core.run 线程中调用，观察者需自行保证线程安全）"""
        with self._observer_lock:
            observers = list(self._state_observers)
        for fn in observers:
            try:
                fn(event_type, payload)
            except Exception:
                pass

    @staticmethod
    def _instruction_to_dict(instr) -> dict:
        """将指令对象序列化为可 JSON 的字典"""
        base = {
            "type": type(instr).__name__.replace("Instruction", "").lower(),
            "call_id": getattr(instr, "call_id", ""),
            "caller_id": getattr(instr, "caller_id", -1),
        }
        if isinstance(instr, CreateInstruction):
            base.update({
                "system_ref": getattr(instr, "system_ref", ""),
                "user_ref": getattr(instr, "user_ref", ""),
                "para_ref": getattr(instr, "para_ref", ""),
            })
        elif isinstance(instr, ExecInstruction):
            base.update({
                "conv_ref": getattr(instr, "conv_ref", ""),
                "batch_ref": getattr(instr, "batch_ref", ""),
                "para_ref": getattr(instr, "para_ref", ""),
            })
        elif isinstance(instr, MemoryReadInstruction):
            base.update({"ref": getattr(instr, "ref", "")})
        elif isinstance(instr, MemoryWriteInstruction):
            base.update({"ref": getattr(instr, "ref", ""), "content": getattr(instr, "content", "")})
        elif isinstance(instr, MemoryMakeInstruction):
            base.update({
                "ref": getattr(instr, "ref", ""),
                "key": getattr(instr, "key", ""),
                "mem_type": getattr(instr, "mem_type", ""),
            })
        return base

    def run(self):
        logger.info("[Core.run] start, stack_size=%d", len(self.command_stack))
        persist_path = getattr(self, 'persist_path', None)
        persist_level = getattr(self, 'persist_level', 'off')
        if persist_path and persist_level != 'off':
            logger.info("[Core.run] persistence enabled, level=%s path=%s", persist_level, persist_path)
        while self.command_stack:
            instruction = self.command_stack[-1]
            if isinstance(instruction, str):
                instruction = parse_instruction(instruction)
            logger.debug("[Core.run] executing %s(call_id=%s)", type(instruction).__name__, getattr(instruction, 'call_id', 'N/A'))

            is_mem_op = isinstance(instruction, (MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction))
            # 高级持久化：每条指令前
            if persist_path and persist_level == "high":
                self.mem.save(persist_path)

            # --- monitor: 指令执行前 ---
            self._notify("instruction_start", {
                "instruction": self._instruction_to_dict(instruction),
            })

            return_type = instruction.execute(self)

            # --- monitor: 指令执行后 ---
            self._notify("instruction_end", {
                "instruction": self._instruction_to_dict(instruction),
                "return_type": return_type.name,
            })

            # 中级持久化：内存操作后
            if persist_path and persist_level == "medium" and is_mem_op:
                self.mem.save(persist_path)
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
        self._notify("run_finished", {})

    def unwrap(self, value, for_llm=True):
        """解引用值
        $conv.{cid} → Conversation 对象
        $batch.{cid} → Conversation 的 user_batch
        其他以 $ 开头走 self.mem.unwrap
        不以 $ 开头视为字面值
        """
        if not value.startswith("$"):
            return value
        value = [value[0], *value[1:].split(".")]
        if value[1] == "conv":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])]
        elif value[1] == "batch":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])].user_batch
        else:
            return self.mem.unwrap(value, for_llm=for_llm)

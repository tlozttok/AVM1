from typing import Optional, List, Dict, Type

from openai import OpenAI
from dotenv import load_dotenv
import json
import time

from .types import MetaList, MetaDict
from .exceptions import VMSyntaxError, VMMemoryError
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict
from .memory import Memory
from .memory_device import MemoryDevice
from .monitor import Monitor


class Instruction:
    tool_name: str = ""
    tool_def: dict = {}

    call_id: str
    caller_id: int

    def __init__(self, call_id: str, caller_id: int, args: dict):
        self.call_id = call_id
        self.caller_id = caller_id
        for k, v in args.items():
            setattr(self, k, v)

    def execute(self, core: 'Core', conv: Conversation):
        raise NotImplementedError


class MemoryReadInstruction(Instruction):
    tool_name = "memory_read"
    tool_def = {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "从内存读取数据",
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "string", "description": "内存引用如 $MEM.input"}},
                "required": ["ref"],
            }
        }
    }
    ref: str

    def execute(self, core: 'Core', conv: Conversation):
        try:
            content = core.unwrap(self.ref, for_llm=True)
            conv.user_batch.add_tool_response(content, self.call_id)
        except (VMMemoryError, KeyError, IndexError, TypeError, ValueError) as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)
        except Exception as e:
            conv.user_batch.add_tool_response(f"Internal Error: {type(e).__name__}: {e}", self.call_id)


class MemoryWriteInstruction(Instruction):
    tool_name = "memory_write"
    tool_def = {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "向内存写入数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["ref", "content"],
            }
        }
    }
    ref: str
    content: str

    def execute(self, core: 'Core', conv: Conversation):
        try:
            core.mem.set(self.ref, self.content)
            conv.user_batch.add_tool_response(f"Success set: {self.ref}", self.call_id)
        except VMMemoryError as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)


class MemoryMakeInstruction(Instruction):
    tool_name = "memory_make"
    tool_def = {
        "type": "function",
        "function": {
            "name": "memory_make",
            "description": "创建新内存地址",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "key": {"type": "string"},
                    "mem_type": {"type": "string", "enum": ["str", "dict", "list"]},
                },
                "required": ["ref", "key", "mem_type"],
            }
        }
    }
    ref: str
    key: str
    mem_type: str

    def execute(self, core: 'Core', conv: Conversation):
        try:
            core.mem.make(self.ref, self.key, self.mem_type)
            conv.user_batch.add_tool_response(f"Success created {self.mem_type} at {self.ref}.{self.key}", self.call_id)
        except VMMemoryError as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)


class CreateInstruction(Instruction):
    tool_name = "create_cmd"
    tool_def = {
        "type": "function",
        "function": {
            "name": "create_cmd",
            "description": "创建子对话",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_ref": {"type": "string", "description": "系统提示词引用"},
                    "user_ref": {"type": "string", "description": "用户消息引用"},
                    "para_ref": {"type": "string", "description": "参数引用"},
                },
                "required": ["system_ref", "user_ref", "para_ref"],
            }
        }
    }
    system_ref: str
    user_ref: str
    para_ref: str

    def execute(self, core: 'Core', conv: Conversation):
        child = _build_conversation(core, self.system_ref, self.user_ref, self.para_ref, parent=conv, is_sub=False)
        child.metadata["call_id"] = self.call_id
        core._ready_cids.append(child.cid)


class CreateSubInstruction(Instruction):
    tool_name = "create_sub"
    tool_def = {
        "type": "function",
        "function": {
            "name": "create_sub",
            "description": "创建亚对话",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_ref": {"type": "string", "description": "系统提示词引用"},
                    "user_ref": {"type": "string", "description": "用户消息引用"},
                    "para_ref": {"type": "string", "description": "参数引用"},
                },
                "required": ["system_ref", "user_ref", "para_ref"],
            }
        }
    }
    system_ref: str
    user_ref: str
    para_ref: str

    def execute(self, core: 'Core', conv: Conversation):
        sub = _build_conversation(core, self.system_ref, self.user_ref, self.para_ref, parent=conv, is_sub=True)
        sub.metadata["call_id"] = self.call_id
        core._ready_cids.insert(0, conv.cid)
        core._ready_cids.insert(0, sub.cid)


class RegisterServiceInstruction(Instruction):
    tool_name = "register_service"
    tool_def = {
        "type": "function",
        "function": {
            "name": "register_service",
            "description": "将当前对话注册为服务，其他对话可通过 call_service 调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "服务名"},
                    "what": {"type": "string", "description": "服务做什么"},
                    "needs": {"type": "string", "description": "需要什么输入"},
                    "returns": {"type": "string", "description": "返回什么"},
                },
                "required": ["name", "what", "needs", "returns"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        core._services[self.name] = conv.cid
        conv.service_desc = {
            "name": self.name, "what": self.what,
            "needs": self.needs, "returns": self.returns,
        }
        conv.user_batch.add_tool_response(f"服务 {self.name} 注册成功", self.call_id)


class CallServiceInstruction(Instruction):
    tool_name = "call_service"
    tool_def = {
        "type": "function",
        "function": {
            "name": "call_service",
            "description": "调用一个已注册服务，等待返回后继续",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "要调用的服务名"},
                    "input": {"type": "string", "description": "传给服务的输入"},
                },
                "required": ["service_name", "input"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        target_cid = core._services.get(self.service_name)
        if target_cid is None:
            conv.user_batch.add_tool_response(f"Error: 服务 {self.service_name} 不存在", self.call_id)
            return
        target = core._conv_by_cid[target_cid]
        target.user_batch.add_user_content(self.input)
        # 目标结束时回调 caller
        target.metadata["caller_cid"] = str(conv.cid)
        target.metadata["caller_call_id"] = self.call_id
        core._ready_cids.insert(0, conv.cid)
        core._ready_cids.insert(0, target.cid)


class TransferServiceInstruction(Instruction):
    tool_name = "transfer_service"
    tool_def = {
        "type": "function",
        "function": {
            "name": "transfer_service",
            "description": "移交控制权给一个服务，自身休眠，不等待返回",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "要调用的服务名"},
                    "input": {"type": "string", "description": "传给服务的输入"},
                },
                "required": ["service_name", "input"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        target_cid = core._services.get(self.service_name)
        if target_cid is None:
            conv.user_batch.add_tool_response(f"Error: 服务 {self.service_name} 不存在", self.call_id)
            return
        target = core._conv_by_cid[target_cid]
        target.user_batch.add_user_content(self.input)
        core._dormant_cids.append(conv.cid)
        core._ready_cids.append(target.cid)


def _instruction_registry() -> Dict[str, Type[Instruction]]:
    return {
        cls.tool_name: cls
        for cls in [
            MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
            CreateInstruction, CreateSubInstruction,
            RegisterServiceInstruction, CallServiceInstruction, TransferServiceInstruction,
        ]
    }


def _build_tools() -> list:
    return [cls.tool_def for cls in _instruction_registry().values()]


class _ServicesDevice(MemoryDevice):
    def __init__(self, core: 'Core'):
        self._core = core

    def pretend_as_type(self) -> str:
        return "str"

    def to_llm_string(self) -> str:
        if not self._core._services:
            return "暂无已注册服务"
        lines = []
        for name, cid in self._core._services.items():
            conv = self._core._conv_by_cid.get(cid)
            if conv and conv.service_desc:
                d = conv.service_desc
                lines.append(f"  {name}: {d.get('what','')} → {d.get('returns','')}")
        return "已注册服务:\n" + "\n".join(lines) if lines else "暂无已注册服务"

    def resolve_path(self, path: list):
        from avm.exceptions import VMMemoryError
        raise VMMemoryError("$services 只支持读（列出服务），不支持子路径")


def _build_conversation(core: 'Core', system_ref: str, user_ref: str, para_ref: str, parent: Conversation, is_sub: bool) -> Conversation:
    system = core.unwrap(system_ref)
    user = core.unwrap(user_ref)
    core.unwrap(para_ref, for_llm=False)
    conv = Conversation(
        messages=[SystemMessage(content=system), UserMessage(content=user)],
        cid=0,
        is_sub=is_sub,
        parent=parent,
        is_root=False,
    )
    core._register(conv)
    return conv


class LMU:
    _client: Optional[OpenAI] = None

    def __init__(self):
        self._client = None
        self.last_call: Optional[dict] = None  # 最近一次 API 调用全文（供监测器写全文文件）

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    _API_PARAM_KEYS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "extra_body",
        "seed", "logit_bias", "logprobs", "top_logprobs",
        "n", "response_format", "timeout",
    }

    def _filter_api_params(self, para: dict) -> dict:
        return {k: v for k, v in para.items() if k in self._API_PARAM_KEYS}

    def _parse_tool_args(self, tool_call, return_calls: list):
        call_id = tool_call.id
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return_calls.append({"call_id": call_id, "cmd_type": "json_error", "args": {"error": str(e), "name": tool_call.function.name}})
            return None

    def _return_calls_from_message(self, message) -> list:
        return_calls = []
        if not message.tool_calls:
            return return_calls

        registry = _instruction_registry()
        for tool_call in message.tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            args = self._parse_tool_args(tool_call, return_calls)
            if args is None:
                continue

            if name in registry:
                return_calls.append({"call_id": call_id, "cmd_type": name, "args": args})
            elif name == "command":
                return_calls.append({"call_id": call_id, "cmd_type": "command", "raw": args.get("command", "")})

        return return_calls

    def exec(self, conversation: Conversation, para: MetaDict):
        messages = conversation.to_api_messages()
        messages.extend(conversation.user_batch.to_tool_messages())

        user_content = conversation.user_batch.get_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})

        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")
        tools = _build_tools()
        model = para.get("model", "gpt-4")

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=use_tool,
                **extra_para
            )
        except Exception as e:
            self.last_call = {
                "model": model,
                "messages": messages,
                "result": None,
                "tool_calls": [],
                "elapsed_ms": (time.time() - start) * 1000,
                "error": f"{type(e).__name__}: {e}",
            }
            raise

        choice = response.choices[0]
        message = choice.message
        result = message.content
        return_calls = self._return_calls_from_message(message)

        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)

        if user_content:
            conversation.append_user_message(user_content)

        tc_list = None
        if message.tool_calls:
            tc_list = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
        conversation.append_assistant_message(result or "", tool_calls=tc_list)

        self.last_call = {
            "model": model,
            "messages": messages,
            "result": result,
            "tool_calls": return_calls,
            "elapsed_ms": (time.time() - start) * 1000,
            "error": None,
        }
        return result, return_calls, conversation


class Core:
    def __init__(self):
        self._conv_by_cid: Dict[int, Conversation] = {}
        self._services: Dict[str, int] = {}  # name → cid
        self._ready_cids: List[int] = []
        self._dormant_cids: List[int] = []
        self._active_cid: Optional[int] = None
        self._next_cid: int = 0
        self.mem: Memory = Memory()
        self.lmu: LMU = LMU()
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.mem.mount("services", _ServicesDevice(self))

    def _register(self, conv: Conversation) -> Conversation:
        conv.cid = self._next_cid
        self._conv_by_cid[conv.cid] = conv
        self._next_cid += 1
        return conv

    def get_conversation(self, cid: int) -> Optional[Conversation]:
        return self._conv_by_cid.get(cid)

    def _process_return_calls(self, return_calls: list, conv: Conversation):
        registry = _instruction_registry()
        for rc in return_calls:
            cmd_type = rc.get("cmd_type", "")
            if cmd_type not in registry:
                continue
            instr_cls = registry[cmd_type]
            instr = instr_cls(rc.get("call_id", ""), conv.cid, rc.get("args", {}))
            instr.execute(self, conv)

    def _on_conversation_finished(self, conv: Conversation, result: str):
        # 服务回调：如果 caller_cid 存在，写回结果
        caller_cid = conv.metadata.get("caller_cid")
        if caller_cid is not None:
            caller_cid = int(caller_cid)
            call_id = conv.metadata.get("caller_call_id", "direct_output")
            if caller_cid in self._conv_by_cid:
                self._conv_by_cid[caller_cid].user_batch.add_tool_response(result, call_id)
            self._active_cid = None
            self._pick_next_active()
            return

        if conv.is_sub:
            parent = conv.parent
            if parent is not None:
                call_id = conv.metadata.get("call_id", "direct_output")
                parent.user_batch.add_tool_response(result, call_id)
                if parent.cid in self._ready_cids:
                    self._ready_cids.remove(parent.cid)
                self._active_cid = parent.cid
        else:
            self._active_cid = None
            self._pick_next_active()

    def _pick_next_active(self):
        if self._ready_cids:
            self._active_cid = self._ready_cids.pop(0)
        else:
            self._active_cid = None

    def advance_conversation(self):
        conv = self._conv_by_cid[self._active_cid]
        para = self.mem.get("model_params", MetaDict(data={"model": "gpt-4"}))
        try:
            result, return_calls, _ = self.lmu.exec(conv, para)
            conv.user_batch.clear()

            if return_calls:
                has_create = any(rc["cmd_type"] in ("create_cmd", "create_sub", "call_service", "transfer_service") for rc in return_calls)
                self._process_return_calls(return_calls, conv)
                if has_create:
                    if conv.cid not in self._dormant_cids:
                        self._dormant_cids.append(conv.cid)
                    self._pick_next_active()
            else:
                self._on_conversation_finished(conv, result)
        except Exception as e:
            self.monitor.record(self, error=e)
            raise
        else:
            self.monitor.record(self)

    def start(self, system: str, user: str, para_ref: str = "$MEM.model_params") -> Conversation:
        conv = Conversation(
            messages=[SystemMessage(content=system), UserMessage(content=user)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=True,
        )
        self._register(conv)
        self._ready_cids.append(conv.cid)
        return conv

    def run(self):
        if self._ready_cids:
            self._active_cid = self._ready_cids.pop(0)

        self.monitor.record_baseline(self)

        while self._active_cid is not None:
            self.advance_conversation()

    def unwrap(self, value, for_llm=True):
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

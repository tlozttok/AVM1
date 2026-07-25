from typing import Optional, List, Dict, Callable, Type

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


def _instruction_registry() -> Dict[str, Type[Instruction]]:
    return {
        cls.tool_name: cls
        for cls in [
            MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
            CreateInstruction, CreateSubInstruction,
        ]
    }


def _build_tools() -> list:
    return [cls.tool_def for cls in _instruction_registry().values()]


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
            logger.error("[LMU] JSON decode error for %s (%s): %s", tool_call.function.name, call_id, e)
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
        logger.info("[LMU.exec] model=%s use_tool=%s", para.get("model"), para.get("use_tool", False))
        messages = conversation.to_api_messages()
        messages.extend(conversation.user_batch.to_tool_messages())

        user_content = conversation.user_batch.get_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})

        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")
        tools = _build_tools()

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=tools,
            tool_choice=use_tool,
            **extra_para
        )

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

        return result, return_calls, conversation


class Core:
    def __init__(self):
        self._conv_by_cid: Dict[int, Conversation] = {}
        self._ready_cids: List[int] = []
        self._dormant_cids: List[int] = []
        self._active_cid: Optional[int] = None
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
        result, return_calls, _ = self.lmu.exec(conv, para)
        conv.user_batch.clear()

        if return_calls:
            has_create = any(rc["cmd_type"] in ("create_cmd", "create_sub") for rc in return_calls)
            self._process_return_calls(return_calls, conv)
            if has_create:
                if conv.cid not in self._dormant_cids:
                    self._dormant_cids.append(conv.cid)
                self._pick_next_active()
        else:
            self._on_conversation_finished(conv, result)

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
        logger.info("[Core.run] start")

        if self._ready_cids:
            self._active_cid = self._ready_cids.pop(0)

        while self._active_cid is not None:
            self.advance_conversation()

        logger.info("[Core.run] end")
        self._notify("run_finished", {})

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

    def add_state_observer(self, fn):
        with self._observer_lock:
            if fn not in self._state_observers:
                self._state_observers.append(fn)

    def remove_state_observer(self, fn):
        with self._observer_lock:
            if fn in self._state_observers:
                self._state_observers.remove(fn)

    def _notify(self, event_type: str, payload: dict):
        with self._observer_lock:
            observers = list(self._state_observers)
        for fn in observers:
            try:
                fn(event_type, payload)
            except Exception:
                pass

    def start_memory_monitor(self, output_file: str, interval: float = 0.3, socket_path: str | None = None):
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

from typing import Optional, List, Dict, Type, Any

from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import sys
import time

from .types import MetaList, MetaDict
from .exceptions import VMSyntaxError, VMMemoryError, APIError, RunnerStop, KernelPanic
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict, LinkNode
from .memory import Memory
from .memory_device import MemoryDevice
from .monitor import Monitor
from .python_server import PLM_REGISTRY, create_plm
from .events import EventSystem, build_event, EVENT_TYPES
from .info_devices import (
    ConversationsDevice, SchedulerDevice,
    ListenersDevice, MemorySummaryDevice, MonitorDevice,
)
from .interrupts import InterruptTable


# 操作员对话的系统消息：说明这条对话的执行器是操作员而不是模型。
_OPERATOR_SYSTEM_PROMPT = (
    "这是操作员对话。执行器是控制台的操作员，不是模型：本对话不产生推理结果，"
    "每轮由操作员提交一次工具调用；操作员选择继续时本对话结束本轮交互。"
)


def _report_tool_error(conv: Conversation, message: str):
    """工具调用错误面向运行者（stderr）输出，不喂回给 LLM。"""
    print(f"[AVM] 对话 {conv.cid}: {message}", file=sys.stderr)


def _publish_memory_after(
    core: 'Core',
    event_type: str,
    conv: Conversation,
    target: str,
    payload: dict,
    result: Optional[dict] = None,
) -> None:
    """读取/写入成功后发布 memory/device after 事实事件。

    memory 目标由 Core 直接以 source=memory 发布；device 目标询问 device.after_event，
    设备返回纯 payload，Core 统一加事件字段并强制 actor_cid/ref。
    """
    follow = core.mem.follow_link_target(target)
    if follow[0] == "memory":
        core.events.publish_after_event(
            event_type, conv, target, payload, source_kind="memory"
        )
        return
    _, mount_key, remainder = follow
    device = core.mem._devices[mount_key]
    extra = device.after_event(event_type, remainder, result or {})
    if extra is None:
        return
    merged = dict(extra)
    merged["ref"] = target
    merged["actor_cid"] = conv.cid
    core.events.publish_after_event(
        event_type, conv, target, merged, device_path=mount_key
    )


# 指令执行体约定的错误类：可判定、可返回给对话的一类。九条指令共用同一套，
# 不在这套之内的异常是内核缺陷，由 advance_conversation 记帧后向上抛（不在这里吞）。
_INSTRUCTION_ERRORS = (VMMemoryError, KeyError, IndexError, TypeError, ValueError)


# 启动自带的内存节点：模型目录。普通内存节点，不是设备——不产生事件、不参与调度。
# 名字里的点号是禁的（$MEM 寻址按点切段），所以 plm.* 与带点的模型 id 一律装进列表项的字段。
MODEL_CATALOG_KEY = "model_catalog"
_CATALOG_META = "模型目录：可用的 LLM 模型与本地 PLM；两侧都写明各自接受的参数"
_PLM_PARAMS_NOTE = "只有 model 有效；tools 由内核注入；LLM 参数无效"
_PLM_OPERATOR_NOTE = "只能由宿主创建（plm.operator 需要注入操作员通道）；不能由镜像声明"


def _first_line(text: str, limit: int = 72) -> str:
    """取文档串首行，超长截断——目录里的说明要能一眼看完。"""
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


# 协议层拒绝（JSON 解析失败、未知工具名、缺少可配对的 call_id）时回给对话的占位文本。
# 内容不携带错误细节：错误面向运行者（_report_tool_error），这里只为保住
# 「assistant 的 tool_calls 与 tool 消息一一配对」这条协议不变量。
_UNEXECUTED_TOOL_RESPONSE = "（该工具调用未执行）"


def _parameters_of(instr_cls) -> dict:
    """指令的参数规格。tool_def 是唯一来源，仲裁层不维护第二份 schema。"""
    return (instr_cls.tool_def.get("function") or {}).get("parameters") or {}


def _value_matches(value, spec: dict) -> bool:
    """按 tool_def 实际用到的 JSON Schema 子集校验一个值。未声明的形状不做假设。"""
    if not isinstance(spec, dict):
        return True
    if "anyOf" in spec:
        return any(_value_matches(value, sub) for sub in spec["anyOf"])
    if "enum" in spec:
        return value in spec["enum"]
    kind = spec.get("type")
    if kind == "integer":
        # bool 是 int 的子类，但 JSON 里是两种类型
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        if not isinstance(value, list):
            return False
        items = spec.get("items")
        return all(_value_matches(v, items) for v in value) if items else True
    if kind == "object":
        return isinstance(value, dict)
    if kind == "string":
        return isinstance(value, str)
    return True


def _validate_instruction_args(instr_cls, args) -> Optional[str]:
    """指令参数仲裁：返回错误说明，None 表示通过。

    可判定的参数错误在进入指令对象之前拦下（非法操作），由调用方回一条工具响应。
    未知键不在这里拒绝：它们不进指令对象（见 Instruction.__init__ 的白名单），
    白名单之外的键不构成错误，避免把提示词里常见的多余字段变成硬失败。
    """
    if not isinstance(args, dict):
        return f"参数必须是对象，得到 {type(args).__name__}"
    params = _parameters_of(instr_cls)
    properties = params.get("properties") or {}
    for name in params.get("required") or []:
        if name not in args or args[name] is None:
            return f"缺少必填参数 {name}"
    for name, value in args.items():
        spec = properties.get(name)
        if spec is None:
            continue
        if not _value_matches(value, spec):
            return f"参数 {name} 不合法：{value!r}"
    tool = instr_cls.tool_name
    if tool == "memory_write" and not args["ref"].startswith("$"):
        return f"ref 必须是 $ 开头的内存引用，得到 {args['ref']!r}"
    if tool == "edit_metadata":
        op = args.get("type")
        if op not in ("set", "get", "del"):
            return f"type 必须是 set/get/del，得到 {op!r}"
        if op == "set" and ("key" not in args or "value" not in args):
            return "type=set 时 key 和 value 都必需"
        if op == "del" and "key" not in args:
            return "type=del 时 key 必需"
    if tool == "listen_event":
        action = args.get("action")
        if action == "listen" and any(
            args.get(k) is None for k in ("event_type", "target", "handler_ref")
        ):
            return "action=listen 时必须提供 event_type、target、handler_ref"
        if action == "modify" and any(
            args.get(k) is None for k in ("listener_id", "handler_ref")
        ):
            return "action=modify 时必须提供 listener_id、handler_ref"
        if action == "cancel" and args.get("listener_id") is None:
            return "action=cancel 时必须提供 listener_id"
    return None


class Instruction:
    tool_name: str = ""
    tool_def: dict = {}

    call_id: str
    caller_id: int

    def __init__(self, call_id: str, caller_id: int, args: dict):
        self.call_id = call_id
        self.caller_id = caller_id
        # 键名白名单：只接受 tool_def 声明过的键。参数里的 call_id / execute 之类
        # 名字不能改写指令对象自己的属性（否则 tool_call_id 可被伪造、方法可被遮蔽）。
        allowed = set((_parameters_of(type(self)).get("properties") or {}))
        for k, v in args.items():
            if k in allowed:
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
        event = build_event(core, "memory_read", "before", conv=conv, target=self.ref, payload={"ref": self.ref})
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        try:
            content = core.unwrap(self.ref, for_llm=True)
            conv.user_batch.add_tool_response(content, self.call_id)
            _publish_memory_after(
                core,
                "memory_read",
                conv,
                self.ref,
                {"ref": self.ref, "actor_cid": conv.cid, "result": content},
                result={"result": content},
            )
        except _INSTRUCTION_ERRORS as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)


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
        event = build_event(
            core,
            "memory_write",
            "before",
            conv=conv,
            target=self.ref,
            payload={"ref": self.ref, "content": self.content},
        )
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        content = event["payload"]["content"]
        try:
            core.mem.set(self.ref, content, owner_cid=conv.cid)
            conv.user_batch.add_tool_response(f"Success set: {self.ref}", self.call_id)
            _publish_memory_after(
                core,
                "memory_write",
                conv,
                self.ref,
                {"ref": self.ref, "content": content, "actor_cid": conv.cid},
                result={"content": content},
            )
        except _INSTRUCTION_ERRORS as e:
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
        target = f"{self.ref}.{self.key}" if self.ref != "$MEM" else f"$MEM.{self.key}"
        event = build_event(
            core,
            "memory_make",
            "before",
            conv=conv,
            target=target,
            payload={"ref": self.ref, "key": self.key, "mem_type": self.mem_type},
        )
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        try:
            core.mem.make(self.ref, self.key, self.mem_type, owner_cid=conv.cid)
            conv.user_batch.add_tool_response(f"Success created {self.mem_type} at {self.ref}.{self.key}", self.call_id)
            _publish_memory_after(
                core,
                "memory_make",
                conv,
                target,
                {
                    "ref": self.ref,
                    "key": self.key,
                    "mem_type": self.mem_type,
                    "path": target,
                    "actor_cid": conv.cid,
                },
                result={"key": self.key, "mem_type": self.mem_type, "path": target},
            )
        except _INSTRUCTION_ERRORS as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)


class EditMetadataInstruction(Instruction):
    tool_name = "edit_metadata"
    tool_def = {
        "type": "function",
        "function": {
            "name": "edit_metadata",
            "description": "编辑内存节点的 ctrl 元数据（元数据二，dict[str,str]）。type=set：用 value 写入 ctrl[key]（key/value 都必需，ctrl 不存在时创建）；type=get：key 省略时返回整个 ctrl（JSON），带 key 时返回 ctrl[key]；type=del：删除 ctrl[key]（key 必需）。非法参数以 Error 工具响应返回，对话保持活跃可自纠",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "内存节点引用，如 $MEM.a.b"},
                    "type": {"type": "string", "enum": ["set", "get", "del"], "description": "操作类型"},
                    "key": {"type": "string", "description": "ctrl 键名；type=set/del 时必需"},
                    "value": {"type": "string", "description": "ctrl 值（字符串）；仅 type=set 时使用且必需"},
                },
                "required": ["ref", "type"],
            }
        }
    }
    ref: str

    def execute(self, core: 'Core', conv: Conversation):
        op = getattr(self, "type", None)
        key = getattr(self, "key", None)
        value = getattr(self, "value", None)
        # 在参数合法性校验之前先分发 before 事件；处理器可以 replace 短路。
        event = build_event(
            core,
            "edit_metadata",
            "before",
            conv=conv,
            target=self.ref,
            payload={"ref": self.ref, "type": op, "key": key, "value": value},
        )
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        value = event["payload"].get("value")
        if op not in ("set", "get", "del"):
            conv.user_batch.add_tool_response(
                f"Error: edit_metadata 的 type 必须是 set/get/del，得到 {op!r}", self.call_id
            )
            return
        try:
            node = core.unwrap(self.ref, for_llm=False)
        except _INSTRUCTION_ERRORS as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)
            return
        if not isinstance(node, (MetaDict, MetaList)):
            conv.user_batch.add_tool_response(
                f"Error: {self.ref} 不是 MetaDict/MetaList 节点，无法编辑 ctrl", self.call_id
            )
            return
        ctrl = dict(node.get_ctrl() or {})
        if op == "set":
            if not key or value is None:
                conv.user_batch.add_tool_response(
                    "Error: edit_metadata type=set 时 key 和 value 都必需", self.call_id
                )
                return
            if key == "owner":
                conv.user_batch.add_tool_response(
                    "Error: owner 由 Core 维护，不允许通过 edit_metadata 修改", self.call_id
                )
                return
            ctrl[key] = value
            node.set_ctrl(ctrl)
            conv.user_batch.add_tool_response(f"ctrl.{key} = {value!r} 已写入", self.call_id)
            core.events.publish_after_event(
                "edit_metadata",
                conv,
                self.ref,
                {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                source_kind="memory",
            )
        elif op == "get":
            if not key:
                conv.user_batch.add_tool_response(
                    json.dumps(ctrl, ensure_ascii=False) if ctrl else "{}", self.call_id
                )
                core.events.publish_after_event(
                    "edit_metadata",
                    conv,
                    self.ref,
                    {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                    source_kind="memory",
                )
            elif key in ctrl:
                conv.user_batch.add_tool_response(str(ctrl[key]), self.call_id)
                core.events.publish_after_event(
                    "edit_metadata",
                    conv,
                    self.ref,
                    {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                    source_kind="memory",
                )
            else:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
        else:  # del
            if not key:
                conv.user_batch.add_tool_response(
                    "Error: edit_metadata type=del 时 key 必需", self.call_id
                )
                return
            if key == "owner":
                conv.user_batch.add_tool_response(
                    "Error: owner 由 Core 维护，不允许通过 edit_metadata 修改", self.call_id
                )
                return
            if key not in ctrl:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
                return
            del ctrl[key]
            node.set_ctrl(ctrl)
            conv.user_batch.add_tool_response(f"ctrl.{key} 已删除", self.call_id)
            core.events.publish_after_event(
                "edit_metadata",
                conv,
                self.ref,
                {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                source_kind="memory",
            )


class CreateInstruction(Instruction):
    tool_name = "create_cmd"
    tool_def = {
        "type": "function",
        "function": {
            "name": "create_cmd",
            "description": "创建子对话（system_ref 必须指向 ctrl.type='settingup' 的 LLM 程序节点或 ctrl.type='python' 的 Python 程序节点；只创建并返回 cid，子对话进入休眠等待指令；本对话保持活跃）",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_ref": {"type": "string", "description": "程序节点引用（settingup 或 python）"},
                    "para_ref": {"type": "string", "description": "参数引用"},
                },
                "required": ["system_ref", "para_ref"],
            }
        }
    }
    system_ref: str
    para_ref: str

    def execute(self, core: 'Core', conv: Conversation):
        try:
            node = core.unwrap(self.system_ref, for_llm=False)
            child = _build_conversation(core, self.system_ref, self.para_ref, parent=conv, is_sub=False)
        except (VMMemoryError, ValueError) as e:
            # system_ref 不是程序节点或引用不存在：错误作为工具响应返回，不创建对话，本对话保持活跃可自纠
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)
            return
        # 带名字的对话必然来自 settingup 文件：name 是节点数据字段，不是元信息
        if isinstance(node, MetaDict):
            name = node.get("name")
            if isinstance(name, str) and name:
                child.name = name
        # 子对话创建后进入休眠，等待 send_instruction 投递指令
        if child.cid not in core._dormant_cids:
            core._dormant_cids.append(child.cid)
        core.events.publish_after_event(
            "create_cmd",
            conv,
            f"cid:{child.cid}",
            {"cid": child.cid, "actor_cid": conv.cid},
            source_kind="core",
        )
        conv.user_batch.add_tool_response(f"Success created: cid={child.cid}", self.call_id)


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
        try:
            user_content = core.unwrap(self.user_ref)
            sub = _build_conversation(core, self.system_ref, self.para_ref, parent=conv, is_sub=True)
        except (VMMemoryError, ValueError) as e:
            # system_ref 不是程序节点或引用不存在：错误作为工具响应返回，不创建对话，本对话保持活跃可自纠
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)
            return
        sub.append_user_message(user_content)  # 亚对话的任务在创建时直接给出
        sub.metadata["call_id"] = self.call_id
        # 父进入休眠等待亚对话完成，完成时由 core 唤醒（_on_conversation_finished）
        core._ready_cids.insert(0, sub.cid)
        if conv.cid not in core._dormant_cids:
            core._dormant_cids.append(conv.cid)
        core.events.publish_after_event(
            "create_sub",
            conv,
            f"cid:{sub.cid}",
            {"cid": sub.cid, "actor_cid": conv.cid},
            source_kind="core",
        )


class SendInstruction(Instruction):
    tool_name = "send_instruction"
    tool_def = {
        "type": "function",
        "function": {
            "name": "send_instruction",
            "description": "向指定 cid 的对话投递指令内容（不保证对方返回消息，只确认投递成功）。不允许向自己投递。cid 可为整数或整数列表（多播，每个目标各收一条；多播中单个目标被事件拦截/处理失败不阻断其他目标）。投递后本对话继续执行，不等待返回；是否休眠由本对话自己决定",
            "parameters": {
                "type": "object",
                "properties": {
                    "cid": {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}], "description": "目标对话的 cid 或 cid 列表"},
                    "content": {"type": "string", "description": "投递的指令内容"},
                },
                "required": ["cid", "content"],
            }
        }
    }
    cid: int
    content: str

    def execute(self, core: 'Core', conv: Conversation):
        targets = self.cid if isinstance(self.cid, list) else [self.cid]
        if not targets:
            msg = "send_instruction 失败：cid 列表为空"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        for cid in targets:
            if cid == conv.cid:
                msg = f"send_instruction 不允许自我投递（cid={cid} 是当前对话）"
                _report_tool_error(conv, msg)
                conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
                return
        # 先解析目标，任一不存在或已关闭即整体失败
        resolved = []
        for cid in targets:
            target = core._conv_by_cid.get(cid)
            if target is None:
                msg = f"send_instruction 失败：对话 {cid} 不存在"
                _report_tool_error(conv, msg)
                conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
                return
            if cid in core._finished:
                msg = f"send_instruction 失败：对话 {cid} 已关闭"
                _report_tool_error(conv, msg)
                conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
                return
            resolved.append(target)

        delivered = 0
        blocked = []
        failed = []
        for target in resolved:
            event = build_event(
                core,
                "send_instruction",
                "before",
                conv=conv,
                target=f"cid:{target.cid}",
                payload={"cid": target.cid, "content": self.content},
            )
            event = core.events.dispatch(event)
            core.events.deliver_notifies(event)
            if event["action"] == "handler_error":
                failed.append(event["response"])
                _report_tool_error(conv, event["response"])
                continue
            if event["action"] == "replace":
                blocked.append(event["response"])
                continue
            content = event["payload"]["content"]
            # 简单投递：内容原样放入目标对话的输入批次，不包装 JSON 信封
            target.user_batch.add_user_content(content)
            if target.cid in core._dormant_cids:
                core._dormant_cids.remove(target.cid)
            if target.cid in core._ready_cids:
                core._ready_cids.remove(target.cid)
            core._ready_cids.append(target.cid)  # 排队调度：被唤醒者进队列末尾
            # TODO(issue): send_instruction 事实事件的语义需进一步研究
            core.events.publish_after_event(
                "send_instruction",
                conv,
                f"cid:{target.cid}",
                {"cid": target.cid, "content": content, "actor_cid": conv.cid},
                source_kind="core",
            )
            delivered += 1

        if delivered > 0 and (blocked or failed):
            conv.user_batch.add_tool_response(
                f"已投递到 {delivered}，{len(blocked)} 个被拦截，{len(failed)} 个处理失败",
                self.call_id,
            )
            return
        if delivered == 0:
            if failed:
                conv.user_batch.add_tool_response(failed[0], self.call_id)
                return
            if blocked:
                conv.user_batch.add_tool_response(blocked[0], self.call_id)
                return
        label = len(resolved) if len(resolved) > 1 else resolved[0].identity
        conv.user_batch.add_tool_response(f"已投递到 {label}", self.call_id)


class CloseInstruction(Instruction):
    tool_name = "close_conversation"
    tool_def = {
        "type": "function",
        "function": {
            "name": "close_conversation",
            "description": "内核级关闭指令：关闭指定 cid 的对话（可关闭自身或其他对话），不需要被关闭对话的回应。关闭后该对话不可再调度、不可再被调用，记录为 finished；其注册的事件监听自动撤销、PLM 状态释放",
            "parameters": {
                "type": "object",
                "properties": {
                    "cid": {"type": "integer", "description": "要关闭的对话 cid（自身 cid 可从 create_cmd 的工具响应或 $MEM.conversations 获取）"},
                },
                "required": ["cid"],
            }
        }
    }
    cid: int

    def execute(self, core: 'Core', conv: Conversation):
        target = core._conv_by_cid.get(self.cid)
        if target is None:
            conv.user_batch.add_tool_response(f"Error: 对话 {self.cid} 不存在", self.call_id)
            return
        if self.cid in core._finished:
            conv.user_batch.add_tool_response(f"Error: 对话 {self.cid} 已是 finished", self.call_id)
            return
        core._close_conversation(self.cid)
        core.events.publish_after_event(
            "close_conversation",
            conv,
            f"cid:{self.cid}",
            {"cid": self.cid, "actor_cid": conv.cid},
            source_kind="core",
        )
        conv.user_batch.add_tool_response(f"对话 {self.cid} 已关闭（finished）", self.call_id)


class ListenEventInstruction(Instruction):
    tool_name = "listen_event"
    tool_def = {
        "type": "function",
        "function": {
            "name": "listen_event",
            "description": "注册/修改/取消事件监听。action=listen：按 event_type/target/handler_ref 注册（handler_ref 必须指向 ctrl.type='python' 的 Python 节点，其 content 定义 handle(event)->dict）；phase 可选 before/after/all，缺省 all；action=modify：用新的 handler_ref 替换已注册监听器的处理器（event_type/target/phase 不可变）；action=cancel：取消监听器。注册/修改/取消只允许所有者执行。处理器返回 notify 时不允许投递给当前对话（禁止自我投递）",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["listen", "modify", "cancel"]},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "target": {"type": "string"},
                    "handler_ref": {"type": "string"},
                    "listener_id": {"type": "integer"},
                    "phase": {"type": "string", "enum": ["before", "after", "all"]},
                },
                "required": ["action"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        action = getattr(self, "action", None)
        if action not in ("listen", "modify", "cancel"):
            conv.user_batch.add_tool_response(
                f"Error: listen_event 的 action 必须是 listen/modify/cancel，得到 {action!r}",
                self.call_id,
            )
            return
        try:
            if action == "listen":
                event_type = getattr(self, "event_type", None)
                target = getattr(self, "target", None)
                handler_ref = getattr(self, "handler_ref", None)
                if event_type is None or target is None or handler_ref is None:
                    raise ValueError("action=listen 时必须提供 event_type、target、handler_ref")
                phase = getattr(self, "phase", None) or "all"
                if phase not in ("before", "after", "all"):
                    raise ValueError("phase 必须是 before/after/all")
                listener_id = core.events.register(conv.cid, event_type, target, handler_ref, phase=phase)
                response = f"listener_id={listener_id} 已注册"
                after_payload = {"listener_id": listener_id, "action": "listen", "actor_cid": conv.cid}
            elif action == "modify":
                listener_id = getattr(self, "listener_id", None)
                handler_ref = getattr(self, "handler_ref", None)
                if listener_id is None or handler_ref is None:
                    raise ValueError("action=modify 时必须提供 listener_id、handler_ref")
                if (
                    getattr(self, "event_type", None) is not None
                    or getattr(self, "target", None) is not None
                    or getattr(self, "phase", None) is not None
                ):
                    raise ValueError("action=modify 只允许修改 handler_ref，不能改 event_type/target/phase")
                core.events.modify(listener_id, conv.cid, handler_ref)
                response = f"listener_id={listener_id} 已更新"
                after_payload = {"listener_id": listener_id, "action": "modify", "actor_cid": conv.cid}
            else:  # cancel
                listener_id = getattr(self, "listener_id", None)
                if listener_id is None:
                    raise ValueError("action=cancel 时必须提供 listener_id")
                core.events.cancel(listener_id, conv.cid)
                response = f"listener_id={listener_id} 已取消"
                after_payload = {"listener_id": listener_id, "action": "cancel", "actor_cid": conv.cid}

            core.events.publish_after_event(
                "listen_event",
                conv,
                f"listener:{listener_id}",
                after_payload,
                source_kind="core",
            )
            conv.user_batch.add_tool_response(response, self.call_id)
        except ValueError as e:
            conv.user_batch.add_tool_response(f"Error: {e}", self.call_id)


def _instruction_registry() -> Dict[str, Type[Instruction]]:
    # TODO(issue): 内存节点有四个操作（创建 / 读写 / 编辑元数据 / 删除），指令集只覆盖前三个：
    # 没有删除内存节点的工具，Memory.__delitem__ 现在只有 Python 侧与调试原语在用。
    # 删除指令的形态未定——删节点还是删列表元素、删链接节点要不要连带处理目标、
    # owner 检查与 before/after 事件如何参与，都需要先想清。
    return {
        cls.tool_name: cls
        for cls in [
            MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction, EditMetadataInstruction,
            CreateInstruction, CreateSubInstruction,
            SendInstruction, CloseInstruction, ListenEventInstruction,
        ]
    }


def _build_tools() -> list:
    return [cls.tool_def for cls in _instruction_registry().values()]


class PLMExecutor:
    """把 PLMServer 适配成与 LMU 相同的 exec 接口，供 Core 统一调度。

    输入：Conversation + para；内部构建 OpenAI 格式消息调用 PLM，
    输出：(result, return_calls, conversation)，与 LMU.exec 形状一致。
    """

    def __init__(self, plm):
        self.plm = plm
        self.last_call: Optional[dict] = None

    def exec(self, conversation: Conversation, para: MetaDict):
        messages = conversation.to_api_messages()
        messages.extend(conversation.user_batch.to_tool_messages())
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})

        params = para.to_dict()
        params["tools"] = _build_tools()
        response = self.plm.handle_messages(messages, params)
        result = response.get("content", "") or ""
        return_calls = self._parse_tool_calls(response.get("tool_calls") or [])
        self.last_call = {
            "messages": messages,
            "result": result,
            "tool_calls": return_calls,
        }

        # 提交历史（与 LMU.exec 一致）：工具响应 → 用户内容 → 新的 assistant 消息
        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        if user_content:
            conversation.append_user_message(user_content)
        tc_list = [
            {"id": t.get("id"), "type": t.get("type", "function"), "function": t.get("function")}
            for t in (response.get("tool_calls") or [])
        ] or None
        conversation.append_assistant_message(result, tool_calls=tc_list)
        return result, return_calls, conversation

    @staticmethod
    def _parse_tool_calls(tool_calls: list) -> list:
        out = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except ValueError as e:
                out.append({
                    "call_id": tc.get("id"), "cmd_type": "json_error",
                    "args": {"error": str(e), "name": fn.get("name"), "raw": args_raw},
                })
                continue
            out.append({"call_id": tc.get("id"), "cmd_type": fn.get("name"), "args": args})
        return out


def _check_model_namespace(core: 'Core', program_type: str, para: MetaDict, para_ref: str) -> None:
    """输入边界的命名空间交叉校验：程序类型与 model 必须配套。

    python 程序 → model 必须是 PLM 注册名；settingup 程序 → model 必须是上游 LLM 模型名。
    上游清单没取到时只警告不拦：`/models` 只是参考，取不到不该让镜像起不来（讨论案第五节）。
    """
    catalog = core.mem._data.get(MODEL_CATALOG_KEY)
    if not isinstance(catalog, MetaDict):
        return
    model = para.get("model")
    if program_type == "python":
        names = sorted(
            e.get("name") for e in (catalog.get("plm") or []) if isinstance(e, MetaDict)
        )
        if not isinstance(model, str) or not model:
            raise VMMemoryError(
                f"{para_ref} 缺少 model：python 程序的 model 必须是 PLM 注册名（可用：{names}）"
            )
        if model not in names:
            raise VMMemoryError(
                f"{para_ref} 的 model {model!r} 不是 PLM 注册名（可用：{names}）；"
                f"LLM 模型名只能配 settingup 程序"
            )
        return
    llm = catalog.get("llm")
    if not isinstance(llm, MetaDict) or llm.get("source") != "api":
        print(
            f"[AVM] 模型目录未取到上游清单，跳过 LLM 模型名校验：{para_ref} 的 model {model!r}",
            file=sys.stderr,
        )
        return
    listed = sorted(
        e.get("id") for e in (llm.get("models") or []) if isinstance(e, MetaDict)
    )
    if isinstance(model, str) and model and model not in listed:
        raise VMMemoryError(
            f"{para_ref} 的 model {model!r} 不在上游模型清单里（可用：{listed}）"
        )


def _build_conversation(core: 'Core', system_ref: str, para_ref: str, parent: Conversation, is_sub: bool) -> Conversation:
    node = core.unwrap(system_ref, for_llm=False)
    # 程序节点校验：与 Python 对话要求 ctrl.type='python' 对等，LLM 对话要求 ctrl.type='settingup'。
    # str 节点、无类型 dict 节点（如参考手册）都不是程序，create_cmd/create_sub 不接受。
    node_type = (node.get_ctrl() or {}).get("type") if isinstance(node, MetaDict) else None
    if node_type not in ("settingup", "python"):
        raise VMMemoryError(
            f"{system_ref} 不是程序节点：system_ref 必须指向 ctrl.type='settingup'（LLM 程序）"
            f"或 'python'（Python 程序）的 dict 节点"
        )
    content = node.get("content")
    system = content if isinstance(content, str) else node.to_llm_string()
    is_python = (node_type == "python")
    messages = [SystemMessage(content=system)]
    para_node = core.unwrap(para_ref, for_llm=False)
    para_type = (para_node.get_ctrl() or {}).get("type") if isinstance(para_node, MetaDict) else None
    if para_type != "para":
        raise VMMemoryError(
            f"{para_ref} 不是参数节点：para_ref 必须指向 ctrl.type='para' 的 dict 节点"
        )
    _check_model_namespace(core, node_type, para_node, para_ref)
    conv = Conversation(
        messages=messages,
        cid=0,
        is_sub=is_sub,
        parent=parent,
        is_root=False,
    )
    core._register(conv)
    conv.para_ref = para_ref
    conv.is_python = is_python
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

    def reset_client(self) -> None:
        """丢弃缓存的客户端，按当前环境重建。

        运行者选"更新环境"后调用：光重建客户端不够——load_dotenv 默认不覆盖已存在的
        环境变量，第二次读 .env 不会生效，这里显式 override。
        """
        self._client = None
        load_dotenv(override=True)

    _API_PARAM_KEYS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "extra_body",
        "seed", "logit_bias", "logprobs", "top_logprobs",
        "n", "response_format", "timeout", "reasoning_effort",
    }
    _NUMERIC_API_PARAMS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "n", "seed", "timeout", "logprobs", "top_logprobs",
    }

    def _filter_api_params(self, para: dict) -> dict:
        out = {}
        for k, v in para.items():
            if k not in self._API_PARAM_KEYS:
                continue
            # para 以 str 存储在内存中，数值型参数需要还原为数值
            if k in self._NUMERIC_API_PARAMS and isinstance(v, str):
                try:
                    v = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    pass
            out[k] = v
        return out

    def _parse_tool_args(self, tool_call, return_calls: list):
        call_id = tool_call.id
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return_calls.append({
                "call_id": call_id,
                "cmd_type": "json_error",
                "args": {"error": str(e), "name": tool_call.function.name, "raw": tool_call.function.arguments},
            })
            return None

    def _return_calls_from_message(self, message) -> list:
        return_calls = []
        if not message.tool_calls:
            return return_calls

        for tool_call in message.tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            args = self._parse_tool_args(tool_call, return_calls)
            if args is None:
                continue

            # 解析器不查指令集：名字原样进 cmd_type。认不认识这个名字由
            # _process_return_calls 在处理指令时判断一次——查表只在一处发生，
            # 解析器因此不必知道指令集，也不会把未知名字改写掉。
            return_calls.append({"call_id": call_id, "cmd_type": name, "args": args})

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
            # 失败时历史与输入批次都没动过（追加在下面、clear 在 exec 返回之后），
            # 所以从这里重试同一轮是干净的，不需要检查点或回退。
            raise APIError(f"{type(e).__name__}: {e}") from e

        choice = response.choices[0]
        message = choice.message
        result = message.content
        # 思维链内容：防御性读取（openai SDK 对非标准响应字段的支持依赖版本）
        reasoning_content = getattr(message, "reasoning_content", None)
        return_calls = self._return_calls_from_message(message)

        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)

        if user_content:
            conversation.append_user_message(user_content)

        tc_list = None
        if message.tool_calls:
            tc_list = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
        conversation.append_assistant_message(
            result or "",
            tool_calls=tc_list,
            reasoning_content=reasoning_content,
        )

        self.last_call = {
            "model": model,
            "messages": messages,
            "result": result,
            "reasoning": reasoning_content,
            "tool_calls": return_calls,
            "elapsed_ms": (time.time() - start) * 1000,
            "error": None,
        }
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
        self._executors: Dict[int, PLMExecutor] = {}  # cid → PLM 执行器（Python 对话；PLM 状态跨轮次保留）
        self._finished: set = set()  # 已关闭（个体结束）的对话 cid；记录供监测/信息设备展示
        self._instruction_budget: int = 50  # 每次激活的指令预算（核心中断阈值）
        self._activation_instructions: int = 0
        self.image_dir: Optional[str] = None  # 调试用：镜像目录（检查点默认路径）
        self.image_name: Optional[str] = None  # 调试用：镜像名（不含扩展名，检查点默认路径）
        # 启动标志：本次运行由哪个宿主托管（如 {"console": True}）。宿主事实，不属于镜像内容；
        # 由 load_image 在挂载设备之前写入，核心启动后不再变化。设备在 attach_core 里读，
        # 核心不读、不校验、不监督。
        self.host_flags: Dict[str, Any] = {}
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.para_ref: str = "$MEM.model_params"
        # API 失败时的运行者策略：callable(error, attempt) -> "retry" | "reload" | "exit"。
        # 宿主安装（终端提示见 avm/runner_prompt.py）；没装时按退出处理。
        self.api_error_policy = None
        self.events: EventSystem = EventSystem(self)
        self.interrupts: InterruptTable = InterruptTable(self)  # 中断向量表：外部（控制台）发起的控制权转移
        self._turn_events: List[dict] = []
        self.mem.mount("conversations", ConversationsDevice(self))
        self.mem.mount("scheduler", SchedulerDevice(self))
        self.mem.mount("memory", MemorySummaryDevice(self))
        self.mem.mount("monitor", MonitorDevice(self))
        self.mem.mount("listeners", ListenersDevice(self))
        # 启动自带的内存节点：模型目录（静态部分；LLM 清单由宿主 refresh_model_catalog 填）
        self.install_model_catalog()

    def _register(self, conv: Conversation) -> Conversation:
        conv.cid = self._next_cid
        self._conv_by_cid[conv.cid] = conv
        self._next_cid += 1
        return conv

    def get_conversation(self, cid: int) -> Optional[Conversation]:
        return self._conv_by_cid.get(cid)

    def checkpoint_path(self, cid: int) -> str:
        """调试用：单对话检查点的默认路径（镜像目录/out/<镜像名>.conv.json；无镜像上下文时用 checkpoints/conv-<cid>.json）"""
        if self.image_dir and self.image_name:
            return os.path.join(self.image_dir, "out", f"{self.image_name}.conv.json")
        return os.path.join("checkpoints", f"conv-{cid}.json")

    def save_conversation(self, cid: int, path: str) -> None:
        """调试用：保存单对话检查点（独立 JSON 文件，无损、确定性）。

        若历史末尾是未配对的 assistant 工具调用（如等待输入响应时），截掉该条，
        保证恢复后的历史是"截至最后一条完整交换"的干净前缀，to_api_messages()
        与保存时的请求前缀逐字节一致（前缀缓存命中）。v1 不支持 Python 对话。
        """
        conv = self._conv_by_cid.get(cid)
        if conv is None:
            raise VMMemoryError(f"对话 {cid} 不存在")
        if conv.is_python:
            raise ValueError("v1 检查点不支持 Python 对话")
        cp = conv.to_checkpoint()
        msgs = cp["messages"]
        if msgs and msgs[-1].get("role") == "assistant" and "tool_calls" in msgs[-1]:
            msgs = msgs[:-1]  # 末尾未配对工具调用：截断到最后一个完整交换
            cp["messages"] = msgs
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cp, f, ensure_ascii=False, indent=2)

    def restore_conversation(self, path: str, schedule: str = "dormant") -> Conversation:
        """调试用：从检查点恢复对话（注册新 cid；v1 不支持 Python 对话）。

        schedule: "dormant"（默认，等待被调用）或 "ready"（排队等待调度）。
        """
        with open(path, encoding="utf-8") as f:
            cp = json.load(f)
        conv = Conversation.from_checkpoint(cp)
        if conv.is_python:
            raise ValueError("v1 检查点不支持 Python 对话")
        self._register(conv)
        if schedule == "ready":
            self._ready_cids.append(conv.cid)
        else:
            self._dormant_cids.append(conv.cid)
        return conv

    @staticmethod
    def _respond_unexecuted(conv: Conversation, call_id):
        """协议层拒绝：写一条中性工具响应保住配对；call_id 不可用时无法配对，只记 stderr。"""
        if isinstance(call_id, str) and call_id:
            conv.user_batch.add_tool_response(_UNEXECUTED_TOOL_RESPONSE, call_id)

    def _process_return_calls(self, return_calls: list, conv: Conversation):
        registry = _instruction_registry()
        for rc in return_calls:
            cmd_type = rc.get("cmd_type", "")
            call_id = rc.get("call_id", "")
            args = rc.get("args", {})
            if cmd_type == "json_error":
                detail = args if isinstance(args, dict) else {}
                _report_tool_error(
                    conv,
                    f"工具调用 {detail.get('name', '')} 参数不是合法 JSON: {detail.get('error', '')}，原始参数: {detail.get('raw', '')}",
                )
                self._respond_unexecuted(conv, call_id)
                continue
            if cmd_type not in registry:
                # 未知工具名的唯一出口。两条解析路径（LMU 的 _return_calls_from_message、
                # PLM 的 _parse_tool_calls）都只把真名放进 cmd_type，查表在这里做一次。
                # 处置与 json_error 一致：报给运行者，回一条中性占位保住配对。
                _report_tool_error(
                    conv,
                    f"未知工具 {cmd_type}，参数: {json.dumps(args, ensure_ascii=False)}",
                )
                self._respond_unexecuted(conv, call_id)
                continue
            if not isinstance(call_id, str) or not call_id:
                # 没有 call_id 就无法写配对响应，也不能伪造一个（assistant 那侧的 id 不是它）
                _report_tool_error(conv, f"工具 {cmd_type} 缺少可配对的 call_id，跳过")
                continue
            instr_cls = registry[cmd_type]
            error = _validate_instruction_args(instr_cls, args)
            if error is not None:
                _report_tool_error(conv, f"工具 {cmd_type} 参数不合法：{error}")
                conv.user_batch.add_tool_response(f"Error: {error}", call_id)
                continue
            instr = instr_cls(call_id, conv.cid, args)
            instr.execute(self, conv)

    def _on_conversation_finished(self, conv: Conversation, result: str):
        if conv.is_sub:
            parent = conv.parent
            if parent is not None and parent.cid not in self._finished:
                # 亚对话紧耦合：完成即自动写回并唤醒父
                call_id = conv.metadata.get("call_id", "direct_output")
                parent.user_batch.add_tool_response(result, call_id)
                if conv.cid not in self._dormant_cids:
                    self._dormant_cids.append(conv.cid)
                self._resume(parent.cid)
                return
            # 父已关闭：无处写回，亚对话按普通交互结束进入休眠
        # 交互结束：对话进入休眠（个体保留，等待事件/再次被调用）。
        # "finished" 是概念态（被内核级关闭指令关闭的对话），Core 调度状态只有活跃/休眠/就绪。
        if conv.cid not in self._dormant_cids:
            self._dormant_cids.append(conv.cid)
        self._active_cid = None
        self._pick_next_active()

    def _resume(self, cid: int):
        """将等待中的对话唤醒为活跃对话（亚对话完成唤醒父：同一调用链，延续预算不重置）。"""
        if cid in self._finished:
            return  # 已关闭的对话不可再被激活
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        self._active_cid = cid

    def _close_conversation(self, cid: int) -> None:
        """内核级关闭：对话个体结束，不需要被关闭对话的回应。

        效果：标记 finished、移出调度队列、释放 PLM 执行器（变量空间）。
        关闭后对话不可再调度、不可再被调用（send_instruction 会以 Error 拒绝）。
        """
        if cid in self._finished:
            return
        self._finished.add(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        self.events.cancel_all_for_owner(cid)
        self._executors.pop(cid, None)  # 释放 PLM 状态（命名空间等）

    def _pick_next_active(self):
        if self._ready_cids:
            picked = self._ready_cids.pop(0)
            if picked in self._dormant_cids:
                self._dormant_cids.remove(picked)
            if not self._same_chain(self._active_cid, picked):
                self._activation_instructions = 0  # 调用链断开：新链重新给满预算
            self._active_cid = picked
        else:
            self._active_cid = None

    def _same_chain(self, prev_cid, new_cid) -> bool:
        """亚对话与父对话是同一调用链（预算共享）：
        亚对话激活时续用父的剩余预算；亚对话完成、父恢复时也续用。
        create_cmd 子对话（is_sub=False）独立预算，不在此列。"""
        prev = self._conv_by_cid.get(prev_cid)
        new = self._conv_by_cid.get(new_cid)
        if prev is None or new is None:
            return False
        if new.is_sub and new.parent is not None and new.parent.cid == prev.cid:
            return True
        if prev.is_sub and prev.parent is not None and prev.parent.cid == new.cid:
            return True
        return False

    def advance_conversation(self):
        self._turn_events.clear()
        conv = self._conv_by_cid[self._active_cid]
        executor = None
        attempt = 0
        while True:
            try:
                # 取 para 与选执行器也放进 try：它们会因为配置错配抛错，
                # 原先在 try 之外，逃出去连监测帧都没有（问题 1）。
                para = self._get_para(conv)
                executor = self._get_executor(conv, para)
                result, return_calls, _ = executor.exec(conv, para)
                conv.user_batch.clear()

                if return_calls:
                    self._process_return_calls(return_calls, conv)
                    self._activation_instructions += len(return_calls)
                    if conv.cid in self._finished:
                        # 内核级关闭（个体结束）：本轮结束后选择下一个活跃对话
                        self._active_cid = None
                        self._pick_next_active()
                    elif conv.cid in self._dormant_cids:
                        # 对话自己让出（等待返回/子完成）
                        self._pick_next_active()
                    elif self._activation_instructions >= self._instruction_budget:
                        # 核心中断：工具返回已加入 batch、状态完整；让出执行权，进就绪队列末尾
                        self._activation_instructions = 0  # 链被中断，重新计数
                        if conv.cid not in self._ready_cids:
                            self._ready_cids.append(conv.cid)
                        self._pick_next_active()
                else:
                    self._on_conversation_finished(conv, result)
            except APIError as e:
                # 外部失败：记帧，然后停下问运行者。失败时历史与输入批次都没动过，
                # 选 retry/reload 就重跑本轮（attempt 只用于提示，不自动放弃）。
                attempt += 1
                self.monitor.record(self, error=e, last_call=self._last_call_for(executor))
                if self._resolve_api_error(e, conv, attempt, executor) == "exit":
                    raise RunnerStop(f"API 调用失败，运行者选择退出：{e}") from e
                continue
            except Exception as e:
                self.monitor.record(self, error=e, last_call=self._last_call_for(executor))
                raise
            else:
                self.monitor.record(self, last_call=self._last_call_for(executor))
                return

    def _last_call_for(self, executor):
        """最近一次 API 调用的全文；本轮事件挂在它上面给监测帧。"""
        last_call = getattr(executor, "last_call", None)
        if last_call is not None and self._turn_events:
            last_call = {**last_call, "events": list(self._turn_events)}
        return last_call

    def _resolve_api_error(self, error: APIError, conv: Conversation, attempt: int, executor) -> str:
        """问运行者 API 失败怎么办，返回 retry / reload / exit。

        没装回调、回调自己出错、返回未知动作，一律按退出处理：停下等一个不会来的
        选择比退出更糟。策略的落点是宿主（终端提示见 avm/runner_prompt.py）。
        """
        policy = self.api_error_policy
        if policy is None:
            print(f"[AVM] API 调用失败且没有安装 api_error_policy，按退出处理：{error}", file=sys.stderr)
            return "exit"
        try:
            action = policy(error, attempt)
        except Exception as e:
            print(
                f"[AVM] api_error_policy 自身异常（{type(e).__name__}: {e}），按退出处理",
                file=sys.stderr,
            )
            return "exit"
        if action not in ("retry", "reload", "exit"):
            print(f"[AVM] api_error_policy 返回未知动作 {action!r}，按退出处理", file=sys.stderr)
            return "exit"
        if action == "reload":
            reset = getattr(executor, "reset_client", None)
            if callable(reset):
                reset()
        return action

    def _get_executor(self, conv: Conversation, para: MetaDict):
        """按对话的程序类型选择执行器（由 Core 管理，不挂在对话上）。
        Python 对话：model 必须在 PLM_REGISTRY 中，否则报错（不静默回退）；
        LLM 对话：用 core.lmu（model 交给 API 校验）。"""
        if conv.cid in self._executors:
            return self._executors[conv.cid]
        model = para.get("model")
        if conv.is_python:
            if not isinstance(model, str) or model not in PLM_REGISTRY:
                raise ValueError(
                    f"Python 对话 {conv.cid} 的 model {model!r} 不在 PLM_REGISTRY 中（可用: {list(PLM_REGISTRY)}）"
                )
            executor = PLMExecutor(create_plm(model))
            self._executors[conv.cid] = executor
            return executor
        return self.lmu

    def _get_para(self, conv: Conversation):
        """按对话自己的 para_ref 读取模型调用参数（缺省用 Core.para_ref）"""
        para_ref = getattr(conv, "para_ref", None) or self.para_ref
        try:
            para = self.unwrap(para_ref, for_llm=False)
            if not isinstance(para, MetaDict) or (para.get_ctrl() or {}).get("type") != "para":
                # 类型检查补全：指向的节点不是 para 节点时不再静默使用；回退默认参数并提示运行者
                _report_tool_error(
                    conv,
                    f"para_ref {para_ref} 不是 ctrl.type='para' 的 dict 节点，回退默认参数（model=gpt-4）",
                )
                para = MetaDict(data={"model": "gpt-4"})
        except VMMemoryError:
            para = MetaDict(data={"model": "gpt-4"})
        return para

    def start(self, system: str, user: str, para_ref: str = "$MEM.model_params") -> Conversation:
        conv = Conversation(
            messages=[SystemMessage(content=system), UserMessage(content=user)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=True,
        )
        self._register(conv)
        conv.para_ref = para_ref
        self._ready_cids.append(conv.cid)
        return conv

    def start_operator_conversation(
        self,
        channel,
        para_ref: str = "$MEM.model_params",
    ) -> Conversation:
        """调试用：建一条操作员对话，执行器是控制台持有的操作员。

        - cid 真实，走正常调度，出现在 $MEM.conversations 里，不做特殊标记；
        - 初始进休眠队列、不进就绪队列：它不主动抢占，只有中断能激活它；
        - 执行器在这里建好并预置，使 _get_executor 直接命中缓存——plm.operator
          需要注入通道，注册表无参构造出来的实例不能用；
        - 不可存检查点（save_conversation 拒绝 Python 对话），不需要额外处理。
        """
        conv = Conversation(
            messages=[SystemMessage(content=_OPERATOR_SYSTEM_PROMPT)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=False,
        )
        self._register(conv)
        conv.para_ref = para_ref
        conv.is_python = True
        self._executors[conv.cid] = PLMExecutor(create_plm("plm.operator", channel=channel))
        self._dormant_cids.append(conv.cid)
        return conv

    # ------------------------------------------------------------------
    # 安全点自检（六条具名不变量）
    # ------------------------------------------------------------------

    def _check_invariants(self) -> None:
        """六条具名不变量的结构自检。违反即 KernelPanic：出现说明某条边界少了一条判定。

        只在安全点调用（run 的每轮推理之前）。成本是常数级遍历，加上活跃对话的一小段历史。
        """
        # 1 调度互斥：就绪与休眠不相交，就绪队列无重复
        both = set(self._ready_cids) & set(self._dormant_cids)
        if both:
            raise KernelPanic(f"调度互斥被破坏：cid {sorted(both)} 同时在就绪与休眠队列")
        if len(set(self._ready_cids)) != len(self._ready_cids):
            raise KernelPanic(f"就绪队列有重复 cid：{self._ready_cids}")
        # 6 预算计数：非负整数
        if not isinstance(self._activation_instructions, int) or self._activation_instructions < 0:
            raise KernelPanic(f"指令预算计数不合法：{self._activation_instructions!r}")
        # 2 活跃有来源
        if self._active_cid is not None:
            conv = self._conv_by_cid.get(self._active_cid)
            if conv is None:
                raise KernelPanic(f"活跃 cid {self._active_cid} 不在对话表里")
            if self._active_cid in self._finished:
                raise KernelPanic(f"活跃 cid {self._active_cid} 已 finished")
            # 3 历史配对、4 值域：只查正要发出去的那条对话
            self._check_pairing(conv)
            self._check_value_domain(conv)
        # 5 设备表一致
        self._check_device_table()

    @staticmethod
    def _check_pairing(conv: Conversation) -> None:
        """调用 API 前，带 tool_calls 的 assistant 消息要与 tool 消息一一配对。

        待发批次（user_batch）也算数：工具响应先入批次，下一轮推理才写进历史。
        没有 id 的工具调用两边都配不上，不在这里判定。
        """
        messages = conv.messages
        last = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and getattr(messages[i], "tool_calls", None):
                last = i
                break
        if last is None:
            return
        wanted = []
        for tc in messages[last].tool_calls or []:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if isinstance(call_id, str) and call_id:
                wanted.append(call_id)
        if not wanted:
            return
        got = [m.tool_call_id for m in messages[last + 1:] if m.role == "tool"]
        got += [r.tool_call_id for r in conv.user_batch.tool_responses]
        if sorted(wanted) != sorted(got):
            raise KernelPanic(
                f"对话 {conv.cid} 的历史配对被破坏：assistant 声明 {sorted(wanted)}，"
                f"现存 tool 消息 {sorted(got)}"
            )

    @staticmethod
    def _check_value_domain(conv: Conversation) -> None:
        """进入 LLM 可见通道的值必须是字符串。查待发批次与最后一条消息。"""
        for resp in conv.user_batch.tool_responses:
            if not isinstance(resp.content, str) or not isinstance(resp.tool_call_id, str):
                raise KernelPanic(
                    f"对话 {conv.cid} 的工具响应值域不合法："
                    f"content={type(resp.content).__name__} tool_call_id={type(resp.tool_call_id).__name__}"
                )
        for content in conv.user_batch.user_contents:
            if not isinstance(content, str):
                raise KernelPanic(f"对话 {conv.cid} 的待发用户内容不是字符串：{type(content).__name__}")
        if conv.messages:
            content = conv.messages[-1].content
            if content is not None and not isinstance(content, str):
                raise KernelPanic(
                    f"对话 {conv.cid} 最后一条消息的内容不是字符串：{type(content).__name__}"
                )

    def _check_device_table(self) -> None:
        """设备表与挂载点一致：每个挂载地址有链接节点；每个标识符至少有一个挂载地址。"""
        mounted_ids = set()
        for address, device in list(self.mem._devices.items()):
            mounted_ids.add(getattr(device, "device_id", None) or address)
            node = self.mem._data
            for seg in address.split("."):
                node = node.get(seg) if isinstance(node, (dict, MetaDict)) else None
                if node is None:
                    break
            if not isinstance(node, LinkNode):
                raise KernelPanic(f"设备表不一致：{address} 没有挂载点链接节点")
        for device_id in list(self.mem._device_ids):
            if device_id not in mounted_ids:
                raise KernelPanic(f"设备表不一致：标识符 {device_id!r} 没有挂载地址")

    # ------------------------------------------------------------------
    # 模型目录（$MEM.model_catalog）
    # ------------------------------------------------------------------

    def install_model_catalog(self) -> None:
        """安装/补齐模型目录的静态部分。幂等；镜像或检查点里已有的字段不覆盖。

        整树替换（load_image）之后必须重装一次，否则镜像会把启动节点冲掉。
        静态部分不依赖网络：PLM 注册表、para 能用的 API 参数、必填字段。
        """
        root = self.mem._data
        catalog = root.get(MODEL_CATALOG_KEY)
        if not isinstance(catalog, MetaDict):
            catalog = MetaDict(data={}, metadata=_CATALOG_META)
            root[MODEL_CATALOG_KEY] = catalog
        llm = catalog.get("llm")
        if not isinstance(llm, MetaDict):
            llm = MetaDict(data={}, metadata="可用的 LLM 模型；source 说明这份清单的来源")
            catalog["llm"] = llm
        llm.setdefault("source", "unavailable")
        llm.setdefault("required", "model")
        params = llm.get("params")
        if not isinstance(params, MetaDict):
            params = MetaDict(data={}, metadata="可以写进 para 的 API 参数（model 之外全部可选）")
            llm["params"] = params
        for name in sorted(LMU._API_PARAM_KEYS):
            if name not in params:
                params[name] = "数值型" if name in LMU._NUMERIC_API_PARAMS else "字符串/原样"
        if not isinstance(llm.get("models"), MetaList):
            llm["models"] = MetaList(data=[], metadata="上游模型清单；每项一个 id")
        if not isinstance(catalog.get("plm"), MetaList):
            catalog["plm"] = MetaList(
                data=[
                    Core._plm_catalog_entry(name, cls)
                    for name, cls in PLM_REGISTRY.items()
                ],
                metadata="PLM 注册表：本地实现，不需要网络",
            )

    @staticmethod
    def _plm_catalog_entry(name: str, cls) -> MetaDict:
        entry = MetaDict(
            data={"name": name, "required": "model"},
            metadata=_first_line(getattr(cls, "__doc__", "")) or name,
        )
        entry["params"] = _PLM_OPERATOR_NOTE if name == "plm.operator" else _PLM_PARAMS_NOTE
        return entry

    def refresh_model_catalog(self, timeout: float = 10.0) -> dict:
        """拉一次上游 `/models` 写进目录。任何失败都收敛成 source="unavailable"，不抛。

        防御按讨论案第五节：自己校验信封、除 id 外全部按可选字段读、忽略未知字段。
        清单只当参考——取不到不拦启动，也不拦模型名（后者见 _check_model_namespace）。
        """
        self.install_model_catalog()
        llm = self.mem._data[MODEL_CATALOG_KEY]["llm"]

        def _unavailable(reason: str) -> dict:
            llm["source"] = "unavailable"
            llm["reason"] = reason
            print(
                f"[AVM] 模型目录：没能取到上游清单（{reason}），本轮不校验 LLM 模型名",
                file=sys.stderr,
            )
            return {"source": "unavailable", "reason": reason}

        try:
            # 用原始响应自己解析：SDK 默认模式对非 OpenAI 信封抛的是 AttributeError/TypeError，
            # 分不清"这家不是 OpenAI 形状"与"SDK 有 bug"；自己校验信封才有确定的诊断。
            raw = self.lmu.client.models.with_raw_response.list(timeout=timeout)
            payload = raw.json()
        except Exception as e:
            return _unavailable(f"{type(e).__name__}: {e}")
        if (
            not isinstance(payload, dict)
            or payload.get("object") != "list"
            or not isinstance(payload.get("data"), list)
        ):
            return _unavailable("上游 /models 的信封不是 OpenAI 形状（object=list 且 data 是数组）")
        entries = []
        skipped = 0
        for item in payload["data"]:
            if not isinstance(item, dict):
                skipped += 1
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                skipped += 1
                continue
            entry = MetaDict(data={"id": model_id}, metadata="上游模型 id")
            for key in ("context_window", "max_output_tokens", "input_modalities", "output_modalities"):
                value = item.get(key)
                if value is None:
                    continue
                # 内存约定：para 与目录里的值都以 str 存储
                entry[key] = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            entries.append(entry)
        llm["models"] = MetaList(data=entries, metadata="上游模型清单；每项一个 id")
        llm["source"] = "api"
        if "reason" in llm:
            del llm["reason"]
        print(
            f"[AVM] 模型目录：从上游取到 {len(entries)} 个模型"
            + (f"（跳过 {skipped} 条形状不合法的条目）" if skipped else ""),
            file=sys.stderr,
        )
        self._warn_unlisted_para_model(llm)
        return {"source": "api", "count": len(entries), "skipped": skipped}

    def _warn_unlisted_para_model(self, llm: MetaDict) -> None:
        """默认 para 的 model 不在清单里时提醒运行者（例如用着已退役的别名）。"""
        try:
            para = self.unwrap(self.para_ref, for_llm=False)
        except Exception:
            return
        model = para.get("model") if isinstance(para, MetaDict) else None
        if not isinstance(model, str) or not model:
            return
        listed = {e.get("id") for e in (llm.get("models") or []) if isinstance(e, MetaDict)}
        if model not in listed:
            print(
                f"[AVM] 模型目录：para {self.para_ref} 的 model {model!r} 不在上游清单里"
                f"（可能是拼写错误或已退役的别名）",
                file=sys.stderr,
            )

    def run(self):
        self._pick_next_active()

        self.monitor.record_baseline(self)

        while self._active_cid is not None:
            # 安全点：上一轮推理（含该轮产生的全部指令）已执行完，是状态完整、可恢复
            # 的边界——预算中断用的是同一个边界。中断在这里夺取执行权。
            # 中断表为空时这一行没有副作用。
            self.interrupts.dispatch()
            try:
                self._check_invariants()
            except KernelPanic as e:
                self.monitor.record(self, error=e)
                raise
            self.advance_conversation()

    def emit_device_event(self, device_path: str, target: str, payload: dict) -> dict:
        """设备自改事件入口（v1）。

        device_path 是内存设备挂载路径（不含 $MEM 前缀），例如 "io"。
        只支持 memory_write before 事件；target 必须位于该设备路径下。
        走正常 before 管道，并返回处理后的 event，设备据此决定是否执行自改。
        """
        if device_path not in self.mem._devices:
            raise VMMemoryError(f"设备 {device_path} 未挂载")
        split = self.mem.split_path(target)
        if len(split) != 3 or split[0] != "device" or split[1] != device_path:
            raise VMMemoryError(
                f"设备自改事件 target 必须属于 $MEM.{device_path}：{target}"
            )
        event = build_event(
            self,
            "memory_write",
            "before",
            target=target,
            payload=payload,
            source={"kind": "device", "path": "$MEM." + device_path},
        )
        event = self.events.dispatch(event)
        self.events.deliver_notifies(event)
        return event

    def unwrap(self, value, for_llm=True):
        if not value.startswith("$"):
            return value
        # TODO(issue): 这里一次把整条引用按点号切分，设备拿到的是散开的段
        # （$MEM.AVMRepo.read."avm/core.py".0 → ["read", '"avm/core', 'py"', "0"]），
        # AVMRepo 因此要用引号把区间还原。是否改成逐步切分——逐段走内存树，命中设备时把
        # 剩余的原字符串整段交给设备、由设备自己定义子路径语法——现在不能确定；
        # 它与 split_path 的最长挂载键匹配、设备授权是同一件事，要一起想。
        value = [value[0], *value[1:].split(".")]
        if value[1] == "conv":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])]
        elif value[1] == "batch":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])].user_batch
        else:
            return self.mem.unwrap(value, for_llm=for_llm)

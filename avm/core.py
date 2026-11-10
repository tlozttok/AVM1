import dataclasses
import traceback
from typing import Optional, List, Dict, Type, Any

from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import sys
import time

from .types import MetaList, MetaDict
from .capabilities import (
    CONTROL_DEVICE_ROOTS, MEM_ROOT_NAME, MODE_KERNEL, MODE_USER, Capability, caps_to_dict,
    check_root_name, default_kernel_caps, grant_target_segments, path_is_prefix,
    targets_prefix_free,
)
from .exceptions import IllegalOperationError, VMSyntaxError, VMMemoryError, APIError, RunnerStop, KernelPanic
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict, LinkNode
from .types import message_to_dict, batch_to_dict
from .memory import Memory, check_new_key
from .memory_device import MemoryDevice
from .monitor import Monitor
from .python_server import HOST_ONLY_PLMS, PLM_REGISTRY, create_plm
from .events import EventSystem, build_event, EVENT_TYPES
from .journal import (
    Journal, JournalError, apply as apply_journal_record,
    conversation_create_payload, scheduler_to_snapshot,
)
from .info_devices import (
    CapsDevice, ConversationsDevice, RootsDevice, SchedulerDevice,
    ListenersDevice, MemorySummaryDevice, MonitorDevice,
)
from .interrupts import InterruptTable


def _effect_space(conv: Conversation) -> dict:
    """把对话的地址空间写成日志效果可重放形状（根名 → 真实段）。"""
    return {name: list(segments) for name, segments in conv.caps.space().as_roots().items()}


def _note_memory_write(core: 'Core', conv: Conversation, ref: str, content: str) -> None:
    """内存写（含追加）的日志效果。

    目标是设备（或 legacy LinkDevice 透明挂载）时只记审计条目：设备内部状态不在
    `$MEM` 里，重放不访问设备。LinkDevice 把写委托进内存树的路径因此不会被重放——
    它是"不要用于新设计"的兼容设备，日志不承诺覆盖（见 docs/runtime.md）。
    """
    if not core._journaling():
        return
    try:
        split = core.mem.split_path(ref, space=conv.caps.space())
    except (VMMemoryError, IllegalOperationError, ValueError):
        split = None
    if split is not None and split[0] == "device":
        core.note_effect("device_write", ref=ref, mount=split[1])
        return
    core.note_effect(
        "memory_set", ref=ref, value=content,
        space=_effect_space(conv),
    )


# 操作员对话的系统消息：说明这条对话的执行器是操作员而不是模型。
_OPERATOR_SYSTEM_PROMPT = (
    "这是操作员对话。执行器是控制台的操作员，不是模型：本对话不产生推理结果，"
    "每轮由操作员提交一次工具调用；操作员选择继续时本对话结束本轮交互。"
)


def _report_tool_error(conv: Conversation, message: str):
    """工具调用错误面向运行者（stderr）输出，不喂回给 LLM。"""
    print(f"[AVM] 对话 {conv.cid}: {message}", file=sys.stderr)


def _canonical_ref(core: 'Core', target, space=None, follow_final: bool = True):
    """把请求别名转成事件内部 canonical target_ref；无法解析时返回 None。

    follow_final=False 用于删除链接：最终段不跟随，canonical 指向链接节点本身。
    """
    return core.canonical_ref(target, space=space, follow_final=follow_final)


def _condition_before_target(condition: dict) -> tuple:
    """listen_event before 事件的目标：condition 目标的 canonical ref。

    无 target（全局条件）时用 "*"；返回 (target 别名, canonical target_ref)。
    """
    target = (condition or {}).get("target") or {}
    kind = target.get("kind")
    ref = None
    if kind in ("exact", "subtree"):
        ref = (target.get("refs") or {}).get("normal")
    elif kind in ("cid_owner", "self_lifecycle"):
        cid = target.get("cid")
        if isinstance(cid, int):
            ref = ("cid", cid)
    if not ref:
        return "*", ("*",)
    ref = tuple(ref)
    tag = ref[0]
    if tag == "memory":
        return "$MEM" + ("." + ".".join(ref[1]) if ref[1] else ""), ref
    if tag == "device":
        return "$MEM." + ref[1] + ("." + ".".join(ref[2]) if ref[2] else ""), ref
    if tag == "cid":
        return f"cid:{ref[1]}", ref
    if tag == "listener":
        return f"listener:{ref[1]}", ref
    return "*", ("*",)


def _dispatch_before_event(core: 'Core', conv: Conversation, event_type: str, target: str,
                           payload: dict, target_ref=None):
    """构造并派发一条 before 事件（10 条指令共用）；返回处理后的 event。"""
    event = build_event(
        core, event_type, "before", conv=conv, target=target,
        payload=payload, target_ref=target_ref,
    )
    event = core.events.dispatch(event)
    core.events.deliver_notifies(event)
    if event["action"] == "handler_error":
        _report_tool_error(conv, event["response"])
    return event


def _publish_memory_after(
    core: 'Core',
    event_type: str,
    conv: Conversation,
    target: str,
    payload: dict,
    result: Optional[dict] = None,
    follow_target: Optional[str] = None,
    space=None,
) -> None:
    """读取/写入成功后发布 memory/device after 事实事件。

    事件 target 用调用者给的地址（append 时保持原 ref，带 -1）；追设备能力时要
    一个可解析的地址，追加写入由调用方通过 follow_target 传真实下标地址。
    `space` 是发起对话的地址上下文（根表）。
    """
    follow = core.mem.follow_link_target(follow_target or target, space=space)
    if follow[0] == "memory":
        core.events.publish_after_event(
            event_type, conv, target, payload, source_kind="memory",
            target_ref=("memory", tuple(follow[1])),
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
        event_type, conv, target, merged, device_path=mount_key,
        target_ref=("device", mount_key, tuple(remainder)),
    )


# 指令执行体约定的错误类：可判定、可返回给对话的一类。九条指令共用同一套，
# 不在这套之内的异常是内核缺陷，由 advance_conversation 记帧后向上抛（不在这里吞）。
_INSTRUCTION_ERRORS = (VMMemoryError, KeyError, IndexError, TypeError, ValueError)

# 保留对话 cid（作者规定）：宿主侧操作员对话的身份，不参与正常分配，也不会撞上
# `_next_cid` 的正数。它对普通对话隐藏：不出现在 $MEM.conversations / $MEM.scheduler
# 里，也不可用 send_instruction / close_conversation 或 $conv.<cid> / $batch.<cid>
# 寻址。隐藏只是显示/寻址策略：对话本身随快照/日志正常恢复，日志里它的回合按普通
# 回合重放。
PANEL_CID = -1


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
    if tool in ("memory_read", "memory_write", "memory_delete") and not args["ref"].startswith("$"):
        return f"ref 必须是 $ 开头的内存引用，得到 {args['ref']!r}"
    if tool == "memory_make":
        try:
            check_new_key(args["key"])
        except VMMemoryError as e:
            return str(e)
    if tool == "edit_metadata":
        op = args.get("type")
        if op not in ("set", "get", "del"):
            return f"type 必须是 set/get/del，得到 {op!r}"
        if op == "set":
            if not args.get("key"):
                return "type=set 时 key 必须是非空字符串"
            if "value" not in args:
                return "type=set 时 value 必需（可以是空字符串）"
        if op == "del" and "key" not in args:
            return "type=del 时 key 必需"
    if tool == "listen_event":
        action = args.get("action")
        if action == "listen":
            if args.get("handler_ref") is None:
                return "action=listen 时必须提供 handler_ref"
        if action == "modify":
            if args.get("listener_id") is None:
                return "action=modify 时必须提供 listener_id"
            if args.get("handler_ref") is None and args.get("priority") is None:
                return "action=modify 时必须提供 handler_ref 或 priority"
        if action == "cancel" and args.get("listener_id") is None:
            return "action=cancel 时必须提供 listener_id"
    return None


def _respond_error(conv: Conversation, call_id: str, exc: BaseException) -> None:
    """错误响应统一出口：输入边界非法用 Illegal（改参数可重试），世界不配合用 Error。"""
    prefix = "Illegal" if isinstance(exc, IllegalOperationError) else "Error"
    conv.user_batch.add_tool_response(f"{prefix}: {exc}", call_id)


def _publish_after_safely(conv: Conversation, label: str, publish) -> None:
    """执行一次 after 事实事件发布，失败只记 stderr（fail-open）。

    发布属于观测面：它失败不该改调用者看到的工具响应，更不该让同一个工具调用出现
    第二条响应——那会破坏 assistant 的 tool_calls 与 tool 消息一一配对（配对不变量）。
    """
    try:
        publish()
    except Exception as e:
        _report_tool_error(conv, f"{label} 的 after 事件发布失败（{type(e).__name__}: {e}），已忽略")


def _publish_core_after_safely(core: 'Core', event_type: str, conv: Conversation, target: str,
                               payload: dict, source_kind: str = "core",
                               target_ref=None) -> None:
    """发布一条 Core / 内存侧 after 事实事件，失败只记 stderr（fail-open）。

    与 `_publish_after_safely` 同一件事，只是把 `core.events.publish_after_event`
    的参数原样接过来，省掉每处一层 lambda。
    """
    _publish_after_safely(
        conv,
        event_type,
        lambda: core.events.publish_after_event(
            event_type, conv, target, payload, source_kind=source_kind,
            target_ref=target_ref,
        ),
    )


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
        event = build_event(core, "memory_read", "before", conv=conv, target=self.ref,
                            payload={"ref": self.ref},
                            target_ref=_canonical_ref(core, self.ref, conv.caps.space()))
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        try:
            content = core.unwrap(self.ref, for_llm=True, space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return
        if not isinstance(content, str):
            # 值域：进 LLM 可见通道的必须是字符串。$conv.N 这类伪地址会返回活对象；
            # 内存里也可能留着非字符串的叶子（早期写入或镜像预置）。
            conv.user_batch.add_tool_response(
                f"Error: {self.ref} 的值是 {type(content).__name__}，不能作为工具响应", self.call_id
            )
            return
        conv.user_batch.add_tool_response(content, self.call_id)
        _publish_after_safely(
            conv,
            "memory_read",
            lambda: _publish_memory_after(
                core,
                "memory_read",
                conv,
                self.ref,
                {"ref": self.ref, "actor_cid": conv.cid, "result": content},
                result={"result": content},
                space=conv.caps.space(),
            ),
        )


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
            target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
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
            address = core.mem.set(self.ref, content, space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return
        _note_memory_write(core, conv, self.ref, content)
        after_payload = {"ref": self.ref, "content": content, "actor_cid": conv.cid}
        if address is not None:
            # 列表追加：-1 换成了真实下标，响应给出可寻址地址
            conv.user_batch.add_tool_response(f"Success appended str at {address}", self.call_id)
            after_payload["address"] = address  # 真实地址放 payload；after 的 target 仍是原 ref
        else:
            conv.user_batch.add_tool_response(f"Success set: {self.ref}", self.call_id)
        _publish_after_safely(
            conv,
            "memory_write",
            lambda: _publish_memory_after(
                core,
                "memory_write",
                conv,
                self.ref,
                after_payload,
                result={"content": content},
                follow_target=address,
                space=conv.caps.space(),
            ),
        )


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
            target_ref=_canonical_ref(core, target, conv.caps.space()),
        )
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        try:
            address = core.mem.make(self.ref, self.key, self.mem_type,
                                     space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return
        if core._journaling():
            core.note_effect(
                "memory_make", ref=self.ref, key=self.key, mem_type=self.mem_type,
                space=_effect_space(conv),
            )
        # 真实地址：链上新建的段与 -1 追加段都带真实下标，调用者据此可寻址
        if self.key == "-1":
            conv.user_batch.add_tool_response(f"Success created {self.mem_type} at {address}", self.call_id)
        else:
            conv.user_batch.add_tool_response(
                f"Success created {self.mem_type} at {address}, key={self.key!r}", self.call_id
            )
        _publish_after_safely(
            conv,
            "memory_make",
            lambda: _publish_memory_after(
                core,
                "memory_make",
                conv,
                address,
                {
                    "ref": self.ref,
                    "key": self.key,
                    "mem_type": self.mem_type,
                    "path": address,
                    "actor_cid": conv.cid,
                },
                result={"key": self.key, "mem_type": self.mem_type, "path": address},
                space=conv.caps.space(),
            ),
        )


class MemoryDeleteInstruction(Instruction):
    tool_name = "memory_delete"
    tool_def = {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "删除内存节点（只删叶子与空容器；挂载点与根不能删；整棵子树删除保留给内核态对话）",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "要删除的内存地址，如 $MEM.a.b"},
                },
                "required": ["ref"],
            }
        }
    }
    ref: str

    def execute(self, core: 'Core', conv: Conversation):
        event = build_event(
            core,
            "memory_delete",
            "before",
            conv=conv,
            target=self.ref,
            payload={"ref": self.ref},
            target_ref=_canonical_ref(core, self.ref, conv.caps.space(), follow_final=False),
        )
        event = core.events.dispatch(event)
        core.events.deliver_notifies(event)
        if event["action"] == "handler_error":
            _report_tool_error(conv, event["response"])
        if event["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(event["response"], self.call_id)
            return
        try:
            deleted_type, address = core.mem.delete(self.ref, space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return
        if core._journaling():
            core.note_effect("memory_delete", ref=self.ref, space=_effect_space(conv))
        conv.user_batch.add_tool_response(f"Success deleted {deleted_type} at {address}", self.call_id)
        _publish_core_after_safely(
            core,
            "memory_delete",
            conv,
            address,
            {"ref": address, "deleted_type": deleted_type, "actor_cid": conv.cid},
            source_kind="memory",
            target_ref=_canonical_ref(core, address, conv.caps.space(), follow_final=False),
        )


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
            target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
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
                f"Illegal: edit_metadata 的 type 必须是 set/get/del，得到 {op!r}", self.call_id
            )
            return
        try:
            node = core.unwrap(self.ref, for_llm=False, space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return
        if not hasattr(node, "get_ctrl"):
            conv.user_batch.add_tool_response(
                f"Error: {self.ref} 不是带控制数据的节点（dict/list/str），无法编辑 ctrl",
                self.call_id,
            )
            return
        ctrl = dict(node.get_ctrl() or {})
        if op == "set":
            if not key:
                conv.user_batch.add_tool_response(
                    "Illegal: edit_metadata type=set 时 key 必须是非空字符串", self.call_id
                )
                return
            if value is None:
                conv.user_batch.add_tool_response(
                    "Illegal: edit_metadata type=set 时 value 必需（可以是空字符串）", self.call_id
                )
                return
            ctrl[key] = value
            node.set_ctrl(ctrl)
            if core._journaling():
                core.note_effect(
                    "edit_metadata", ref=self.ref, type="set", key=key, value=value,
                    space=_effect_space(conv),
                )
            conv.user_batch.add_tool_response(f"ctrl.{key} = {value!r} 已写入", self.call_id)
            _publish_core_after_safely(core, 
                "edit_metadata",
                conv,
                self.ref,
                {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                source_kind="memory",
                target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
            )
        elif op == "get":
            if not key:
                conv.user_batch.add_tool_response(
                    json.dumps(ctrl, ensure_ascii=False) if ctrl else "{}", self.call_id
                )
                _publish_core_after_safely(core, 
                    "edit_metadata",
                    conv,
                    self.ref,
                    {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                    source_kind="memory",
                target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
                )
            elif key in ctrl:
                conv.user_batch.add_tool_response(str(ctrl[key]), self.call_id)
                _publish_core_after_safely(core, 
                    "edit_metadata",
                    conv,
                    self.ref,
                    {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                    source_kind="memory",
                target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
                )
            else:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
        else:  # del
            if not key:
                conv.user_batch.add_tool_response(
                    "Illegal: edit_metadata type=del 时 key 必需", self.call_id
                )
                return
            if key not in ctrl:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
                return
            del ctrl[key]
            node.set_ctrl(ctrl)
            if core._journaling():
                core.note_effect(
                    "edit_metadata", ref=self.ref, type="del", key=key,
                    space=_effect_space(conv),
                )
            conv.user_batch.add_tool_response(f"ctrl.{key} 已删除", self.call_id)
            _publish_core_after_safely(core, 
                "edit_metadata",
                conv,
                self.ref,
                {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
                source_kind="memory",
                target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
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
        before = _dispatch_before_event(
            core, conv, "create_cmd", self.system_ref,
            {"system_ref": self.system_ref, "para_ref": self.para_ref},
            target_ref=_canonical_ref(core, self.system_ref, conv.caps.space()),
        )
        if before["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(before["response"], self.call_id)
            return
        try:
            node = core.unwrap(self.system_ref, for_llm=False, space=conv.caps.space())
            child = _build_conversation(core, self.system_ref, self.para_ref, parent=conv, is_sub=False,
                                        space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            # system_ref 不是程序节点或引用不存在：错误作为工具响应返回，不创建对话，本对话保持活跃可自纠
            _respond_error(conv, self.call_id, e)
            return
        # 带名字的对话必然来自 settingup 文件：name 是节点数据字段，不是元信息
        if isinstance(node, MetaDict):
            name = node.get("name")
            if isinstance(name, str) and name:
                child.name = name
        # 子对话创建后进入休眠，等待 send_instruction 投递指令
        if child.cid not in core._dormant_cids:
            core._dormant_cids.append(child.cid)
        if core._journaling():
            core.note_effect("conversation_create", **conversation_create_payload(core, child))
        _publish_core_after_safely(core,
            "create_cmd",
            conv,
            f"cid:{child.cid}",
            {"cid": child.cid, "actor_cid": conv.cid},
            source_kind="core",
            target_ref=("cid", child.cid),
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
        before = _dispatch_before_event(
            core, conv, "create_sub", self.system_ref,
            {"system_ref": self.system_ref, "user_ref": self.user_ref, "para_ref": self.para_ref},
            target_ref=_canonical_ref(core, self.system_ref, conv.caps.space()),
        )
        if before["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(before["response"], self.call_id)
            return
        try:
            user_content = core.unwrap(self.user_ref, for_llm=False, space=conv.caps.space())
            if not isinstance(user_content, str):
                raise VMMemoryError(
                    f"{self.user_ref} 不是字符串节点：user_ref 必须指向 str 节点"
                    f"（得到 {type(user_content).__name__}）"
                )
            sub = _build_conversation(core, self.system_ref, self.para_ref, parent=conv, is_sub=True,
                                      space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            # system_ref 不是程序节点或引用不存在：错误作为工具响应返回，不创建对话，本对话保持活跃可自纠
            _respond_error(conv, self.call_id, e)
            return
        sub.append_user_message(user_content)  # 亚对话的任务在创建时直接给出
        sub.metadata["call_id"] = self.call_id
        # 父进入休眠等待亚对话完成，完成时由 core 唤醒（_on_conversation_finished）
        core._ready_cids.insert(0, sub.cid)
        if conv.cid not in core._dormant_cids:
            core._dormant_cids.append(conv.cid)
        if core._journaling():
            core.note_effect("conversation_create", **conversation_create_payload(core, sub))
        _publish_core_after_safely(core,
            "create_sub",
            conv,
            f"cid:{sub.cid}",
            {"cid": sub.cid, "actor_cid": conv.cid},
            source_kind="core",
            target_ref=("cid", sub.cid),
        )


class SendInstruction(Instruction):
    tool_name = "send_instruction"
    tool_def = {
        "type": "function",
        "function": {
            "name": "send_instruction",
            "description": "向指定 cid 的对话投递指令内容（不保证对方返回消息，只确认投递成功）。不允许向自己投递。cid 可为整数或整数列表（多播，每个目标各收一条）；多播先整体校验所有目标的存在性与开启状态，任一不满足则整体不投递；校验通过后，单个目标被事件拦截/处理失败不阻断其他目标。投递后本对话继续执行，不等待返回；是否休眠由本对话自己决定",
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
        if conv.is_sub:
            ancestor = conv.parent
            while ancestor is not None:
                if ancestor.cid in targets:
                    msg = ("send_instruction 失败：亚对话不能向主对话发送消息，"
                           "只能用 create_sub 的返回通道")
                    _report_tool_error(conv, msg)
                    conv.user_batch.add_tool_response(f"Illegal: {msg}", self.call_id)
                    return
                ancestor = ancestor.parent
        for cid in targets:
            if cid == conv.cid:
                msg = f"send_instruction 不允许自我投递（cid={cid} 是当前对话）"
                _report_tool_error(conv, msg)
                conv.user_batch.add_tool_response(f"Illegal: {msg}", self.call_id)
                return
        # 先解析目标，任一不存在或已关闭即整体失败
        resolved = []
        for cid in targets:
            target = core._conv_by_cid.get(cid)
            if target is None or core.is_hidden_cid(cid):
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
        delivered_cids = []
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
                target_ref=("cid", target.cid),
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
            core.note_effect("delivery", to_cid=target.cid, user_contents=[content])
            delivered_cids.append(target.cid)
            delivered += 1
            # 事实事件按"消息已被接受"发布：等待返回的目标只入批次、也照样有 after。
            # TODO(issue): send_instruction 事实事件的语义需进一步研究
            _publish_core_after_safely(core,
                "send_instruction",
                conv,
                f"cid:{target.cid}",
                {"cid": target.cid, "content": content, "actor_cid": conv.cid},
                source_kind="core",
                target_ref=("cid", target.cid),
            )
            if core._is_waiting_for_return(target):
                # 等亚对话返回的对话：消息只入批次，不唤醒；亚对话结束后父对话先跑。
                continue
            if target.cid in core._dormant_cids:
                core._dormant_cids.remove(target.cid)
            if target.cid in core._ready_cids:
                core._ready_cids.remove(target.cid)
            core._ready_cids.append(target.cid)  # 排队调度：被唤醒者进队列末尾

        if delivered > 0 and (blocked or failed):
            conv.user_batch.add_tool_response(
                f"已投递到 cid={delivered_cids}（{len(blocked)} 个被拦截，{len(failed)} 个处理失败）",
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
        if len(resolved) > 1:
            conv.user_batch.add_tool_response(f"已投递到 cid={delivered_cids}", self.call_id)
        else:
            conv.user_batch.add_tool_response(
                f"已投递到 {resolved[0].identity}（cid={resolved[0].cid}）", self.call_id
            )


class CloseInstruction(Instruction):
    tool_name = "close_conversation"
    tool_def = {
        "type": "function",
        "function": {
            "name": "close_conversation",
            "description": "内核级关闭指令：关闭指定 cid 的对话（可关闭自身或其他对话），不需要被关闭对话的回应。关闭后该对话不可再调度、不可再被调用，记录为 finished；其注册的事件监听自动撤销、PLM 状态释放。若目标还有待执行批次，关闭会一并丢弃，工具响应里会说明",
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
        if conv.is_sub:
            ancestor = conv.parent
            while ancestor is not None:
                if ancestor.cid == self.cid:
                    msg = "close_conversation 失败：亚对话不能关闭主对话，只能通过返回通道结束"
                    _report_tool_error(conv, msg)
                    conv.user_batch.add_tool_response(f"Illegal: {msg}", self.call_id)
                    return
                ancestor = ancestor.parent
        target = core._conv_by_cid.get(self.cid)
        if target is None or core.is_hidden_cid(self.cid):
            conv.user_batch.add_tool_response(f"Error: 对话 {self.cid} 不存在", self.call_id)
            return
        if self.cid in core._finished:
            conv.user_batch.add_tool_response(f"Error: 对话 {self.cid} 已关闭", self.call_id)
            return
        if conv.caps.mode != MODE_KERNEL and target.caps.mode == MODE_KERNEL:
            msg = f"用户态对话不能关闭内核态对话（目标 cid={self.cid} mode=kernel）"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Illegal: {msg}", self.call_id)
            return
        before = _dispatch_before_event(
            core, conv, "close_conversation", f"cid:{self.cid}",
            {"cid": self.cid},
            target_ref=("cid", self.cid),
        )
        if before["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(before["response"], self.call_id)
            return
        dropped = bool(target.user_batch.tool_responses or target.user_batch.user_contents)
        dropped = dropped or self.cid in core._ready_cids
        core._close_conversation(self.cid)
        _publish_core_after_safely(core,
            "close_conversation",
            conv,
            f"cid:{self.cid}",
            {"cid": self.cid, "actor_cid": conv.cid},
            source_kind="core",
            target_ref=("cid", self.cid),
        )
        hint = "，其待执行批次已丢弃" if dropped else ""
        conv.user_batch.add_tool_response(f"对话 {self.cid} 已关闭（finished）{hint}", self.call_id)


class ListenEventInstruction(Instruction):
    tool_name = "listen_event"
    tool_def = {
        "type": "function",
        "function": {
            "name": "listen_event",
            "description": "注册/修改/取消事件监听。监听条件在 handler 节点的 condition 数据叶子中声明（events/phase/target/source；用户态只能用命名条件子集 preset）。action=listen：只提供 handler_ref，注册时解析 handler 的 condition；priority 可选（0 起，降序执行；>0 只有内核态对话可设置）；tier 可选 normal/kernel（kernel 级载荷含目标完整数据与来源上下文，只有内核态对话可注册）；action=modify：用新的 handler_ref 和/或 priority 更新已注册监听器（重新解析 handler 的 condition；event_type/target/phase/tier 不可变）；action=cancel：取消监听器。注册/修改/取消只允许所有者执行。处理器返回 notify 时不允许投递给当前对话（禁止自我投递）",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["listen", "modify", "cancel"]},
                    "handler_ref": {"type": "string"},
                    "listener_id": {"type": "integer"},
                    "priority": {"type": "integer", "description": "监听优先级，非负整数，缺省 0；大的先执行"},
                    "tier": {"type": "string", "enum": ["normal", "kernel"], "description": "载荷级别，缺省 normal"},
                },
                "required": ["action"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        action = getattr(self, "action", None)
        if action not in ("listen", "modify", "cancel"):
            conv.user_batch.add_tool_response(
                f"Illegal: listen_event 的 action 必须是 listen/modify/cancel，得到 {action!r}",
                self.call_id,
            )
            return
        # before 事件：监听注册/修改/取消本身也受事件层检查与软件 Guard 裁决。
        # action=listen 先解析 handler 的 condition，before 载荷带解析后的条件；
        # 注册在 before 通过后进行，不再重复解析。
        listener_id = getattr(self, "listener_id", None)
        parsed_condition = None
        if action == "listen":
            handler_ref = getattr(self, "handler_ref", None)
            if handler_ref is None:
                conv.user_batch.add_tool_response(
                    "Illegal: action=listen 时必须提供 handler_ref", self.call_id
                )
                return
            try:
                parsed_condition = core.events.parse_condition(
                    conv.cid, handler_ref, space=conv.caps.space(),
                    kernel_mode=(conv.caps.mode == "kernel"),
                )
            except _INSTRUCTION_ERRORS as e:
                _respond_error(conv, self.call_id, e)
                return
            b_target, b_ref = _condition_before_target(parsed_condition)
            b_payload = {
                "action": action,
                "event_type": getattr(self, "event_type", None),
                "condition": parsed_condition,
                "handler_ref": handler_ref,
                "priority": getattr(self, "priority", None),
                "tier": getattr(self, "tier", None),
            }
        else:
            b_target = f"listener:{listener_id}"
            b_payload = {"action": action, "listener_id": listener_id,
                         "handler_ref": getattr(self, "handler_ref", None),
                         "priority": getattr(self, "priority", None)}
            b_ref = ("listener", listener_id) if isinstance(listener_id, int) else None
        before = _dispatch_before_event(core, conv, "listen_event", b_target, b_payload, target_ref=b_ref)
        if before["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(before["response"], self.call_id)
            return
        try:
            if action == "listen":
                priority = getattr(self, "priority", 0)
                tier = getattr(self, "tier", None) or "normal"
                listener_id = core.events.register_condition(
                    conv.cid, handler_ref, space=conv.caps.space(),
                    tier=tier, priority=0 if priority is None else priority,
                    kernel_mode=(conv.caps.mode == "kernel"),
                    condition=parsed_condition,
                )
                response = f"listener_id={listener_id} 已注册"
                after_payload = {"listener_id": listener_id, "action": "listen", "actor_cid": conv.cid}
            elif action == "modify":
                listener_id = getattr(self, "listener_id", None)
                handler_ref = getattr(self, "handler_ref", None)
                if listener_id is None or (handler_ref is None and getattr(self, "priority", None) is None):
                    raise IllegalOperationError(
                        "action=modify 时必须提供 listener_id，以及 handler_ref 或 priority 至少一个"
                    )
                if (
                    getattr(self, "event_type", None) is not None
                    or getattr(self, "target", None) is not None
                    or getattr(self, "phase", None) is not None
                    or getattr(self, "tier", None) is not None
                ):
                    raise IllegalOperationError(
                        "action=modify 只允许修改 handler_ref/priority，不能改 event_type/target/phase/tier"
                    )
                core.events.modify(listener_id, conv.cid, handler_ref, space=conv.caps.space(),
                                   priority=getattr(self, "priority", None))
                response = f"listener_id={listener_id} 已更新"
                after_payload = {"listener_id": listener_id, "action": "modify", "actor_cid": conv.cid}
            else:  # cancel
                listener_id = getattr(self, "listener_id", None)
                if listener_id is None:
                    raise IllegalOperationError("action=cancel 时必须提供 listener_id")
                core.events.cancel(listener_id, conv.cid)
                response = f"listener_id={listener_id} 已取消"
                after_payload = {"listener_id": listener_id, "action": "cancel", "actor_cid": conv.cid}

            _publish_core_after_safely(core,
                "listen_event",
                conv,
                f"listener:{listener_id}",
                after_payload,
                source_kind="core",
                target_ref=("listener", listener_id),
            )
            conv.user_batch.add_tool_response(response, self.call_id)
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)


def _instruction_registry() -> Dict[str, Type[Instruction]]:
    # 删除默认只删叶子与空容器；整棵子树删除留给内核态对话（尚未实现），见 Memory.delete。
    return {
        cls.tool_name: cls
        for cls in [
            MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
            MemoryDeleteInstruction, EditMetadataInstruction,
            CreateInstruction, CreateSubInstruction,
            SendInstruction, CloseInstruction, ListenEventInstruction,
        ]
    }


def _build_tools(conv: Optional[Conversation] = None) -> list:
    """全部工具定义；给了对话时只返回该对话工具集里的工具（宿主/控制台传 None 拿全量）。"""
    tools = [cls.tool_def for cls in _instruction_registry().values()]
    if conv is None:
        return tools
    return [t for t in tools if conv.caps.tool_allowed((t.get("function") or {}).get("name"))]


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
        params["tools"] = _build_tools(conversation)
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
        if isinstance(model, str) and model in HOST_ONLY_PLMS:
            # 宿主独占 PLM：拒绝理由涉及宿主/调试实现，不返回给对话
            raise IllegalOperationError("操作被外部原因拒绝")
        names = sorted(
            e.get("name") for e in (catalog.get("plm") or []) if isinstance(e, MetaDict)
        )
        if not isinstance(model, str) or not model:
            raise IllegalOperationError(
                f"{para_ref} 缺少 model：python 程序的 model 必须是 PLM 注册名（可用：{names}）"
            )
        if model not in names:
            raise IllegalOperationError(
                f"{para_ref} 的 model {model!r} 不是 PLM 注册名（可用：{names}）；"
                f"LLM 模型名只能配 settingup 程序"
            )
        return
    if not isinstance(model, str) or not model:
        # 缺 model 与上游清单无关，输入边界就能判定
        raise IllegalOperationError(
            f"{para_ref} 缺少 model：settingup 程序的 model 必须是 LLM 模型名"
        )
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
    if model not in listed:
        raise IllegalOperationError(
            f"{para_ref} 的 model {model!r} 不在上游模型清单里（可用：{listed}）"
        )


_CAPS_SPEC_KEYS = frozenset({"mode", "groups", "roots", "tools"})


def _read_caps_value(caps_node, key: str):
    """把 value.caps 下的一个数据节点读成 Python 值。"""
    value = caps_node.get(key)
    if key in ("groups", "roots", "tools"):
        if not isinstance(value, MetaList):
            raise IllegalOperationError(
                f"程序节点的 value.caps.{key} 必须是 list 数据节点"
            )
        items = []
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise IllegalOperationError(
                    f"程序节点的 value.caps.{key}[{i}] 必须是 str 数据节点"
                )
            items.append(str(item))
        return items
    if not isinstance(value, str):
        raise IllegalOperationError(
            f"程序节点的 value.caps.{key} 必须是 str 数据节点"
        )
    return str(value)


def _program_caps_spec(node) -> dict:
    """读取程序节点 value.caps 数据；缺省空规格（继承父能力）。

    能力声明是节点数据，不是控制数据：ctrl 只描述类型与数据组织格式。
    ctrl.caps 一律拒绝（镜像装载与运行期创建两处入口）。
    """
    if not isinstance(node, MetaDict):
        return {}
    ctrl = node.get_ctrl() or {}
    if "caps" in ctrl:
        raise IllegalOperationError(
            "程序节点的能力声明必须放在 value.caps 数据节点里；ctrl.caps 不再支持"
        )
    caps_node = node.get("caps")
    if caps_node is None:
        return {}
    if not isinstance(caps_node, MetaDict):
        raise IllegalOperationError("程序节点的 value.caps 必须是 dict 数据节点")
    unknown = sorted(set(caps_node.keys()) - _CAPS_SPEC_KEYS)
    if "workspace" in unknown:
        raise IllegalOperationError(
            "程序节点的 value.caps.workspace 已移除：工作目录不再由内核分配。"
            "改用软件建目录（memory_make）后写 $MEM.conversations.<cid>.roots 动态加根"
        )
    if unknown:
        raise IllegalOperationError(
            f"程序节点的 value.caps 含未知字段：{unknown}（允许：{sorted(_CAPS_SPEC_KEYS)}）"
        )
    return {key: _read_caps_value(caps_node, key) for key in caps_node.keys()}


def _derive_child_caps(core: 'Core', parent: Optional[Conversation], node, is_sub: bool,
                       is_init: bool = False) -> Capability:
    """按程序 ctrl.caps 与父能力计算子对话能力。

    规则（批次一）：
    - 模式：缺省继承；用户态不能创建内核态；亚对话不能跨模式；init 强制内核态。
    - 工具集：缺省继承；显式声明时必须 ⊆ 父工具集。
    - 组：缺省继承；用户态父对话只能给子对话父已有的组（内核态父可指定）。
    - 根：内核态子对话固定 `{MEM: ()}`；用户态子对话的授予根来自 caps.roots 的
      `根名 -> $MEM 路径`（同域创建时只能沿用父根，不能新增）。用户态对话的工作
      目录不再由内核分配：目录由软件建，根由 `$MEM.conversations.<cid>.roots` 动态授予。
    """
    spec = _program_caps_spec(node)
    parent_caps = parent.caps if parent is not None else default_kernel_caps()
    parent_mode = parent_caps.mode

    mode = spec.get("mode", parent_mode)
    if mode not in (MODE_KERNEL, MODE_USER):
        raise IllegalOperationError(f"程序 caps.mode 必须是 kernel/user，得到 {mode!r}")
    if is_init and mode != MODE_KERNEL:
        raise VMMemoryError("init 程序必须是内核态（没有内核态就无法管理权限）")
    if parent is not None:
        if parent_mode == MODE_USER and mode != MODE_USER:
            raise VMMemoryError("模式不可提权：用户态对话只能创建用户态对话")
        if is_sub and mode != parent_mode:
            raise VMMemoryError("亚对话不能跨模式创建")

    tools_spec = spec.get("tools")
    if tools_spec is None:
        tools = parent_caps.tools
    else:
        if not isinstance(tools_spec, list) or not all(isinstance(x, str) for x in tools_spec):
            raise IllegalOperationError("程序 caps.tools 必须是字符串列表")
        tools = frozenset(tools_spec)
        extra = tools - parent_caps.tools
        if extra:
            raise VMMemoryError(f"工具集不可超出父对话：{sorted(extra)}")

    groups_spec = spec.get("groups")
    if groups_spec is None:
        groups = parent_caps.groups
    else:
        if not isinstance(groups_spec, list) or not all(isinstance(x, str) and x for x in groups_spec):
            raise IllegalOperationError("程序 caps.groups 必须是非空字符串列表")
        groups = tuple(dict.fromkeys(groups_spec))
        if parent is not None and parent_mode == MODE_USER and not set(groups) <= set(parent_caps.groups):
            raise VMMemoryError("组不可提权：用户态对话只能创建同组或更少组的子对话")

    if mode == MODE_KERNEL:
        roots = {MEM_ROOT_NAME: ()}
    else:
        grants = {}
        roots_spec = spec.get("roots") or []
        if not isinstance(roots_spec, list) or not all(isinstance(x, str) for x in roots_spec):
            raise IllegalOperationError(
                "程序 caps.roots 必须是根名列表（名字来自镜像 security.roots 注册表）"
            )
        for name in roots_spec:
            check_root_name(name)
            if name == MEM_ROOT_NAME:
                raise IllegalOperationError("程序 caps.roots 不能声明 MEM 根")
            target = core.root_registry.get(name)
            if target is None:
                raise VMMemoryError(f"程序 caps.roots 引用了未注册的根：{name!r}")
            grants[name] = tuple(target)
        same_domain = is_sub or (parent is not None and parent_mode == MODE_USER)
        if same_domain:
            inherited = {n: t for n, t in parent_caps.roots.items() if n != MEM_ROOT_NAME}
            for name, target in grants.items():
                if inherited.get(name) != target:
                    raise VMMemoryError(f"不能向子对话授予父对话没有的根：{name!r}")
            roots = inherited
        else:
            roots = grants

    return Capability(mode=mode, groups=groups, roots=roots, tools=tools)


def _build_conversation(core: 'Core', system_ref: str, para_ref: str, parent: Conversation, is_sub: bool,
                        space=None, is_init: bool = False) -> Conversation:
    node = core.unwrap(system_ref, for_llm=False, space=space)
    # 程序节点校验：与 Python 对话要求 ctrl.type='python' 对等，LLM 对话要求 ctrl.type='settingup'。
    # str 节点、无类型 dict 节点（如参考手册）都不是程序，create_cmd/create_sub 不接受。
    node_type = (node.get_ctrl() or {}).get("type") if isinstance(node, MetaDict) else None
    if node_type not in ("settingup", "python"):
        raise VMMemoryError(
            f"{system_ref} 不是程序节点：system_ref 必须指向 ctrl.type='settingup'（LLM 程序）"
            f"或 'python'（Python 程序）的 dict 节点"
        )
    content = node.get("content")
    if not isinstance(content, str):
        raise VMMemoryError(
            f"{system_ref} 缺 content：ctrl.type='{node_type}' 节点的 value 必须含字符串 content"
        )
    system = content
    is_python = (node_type == "python")
    messages = [SystemMessage(content=system)]
    para_node = core.unwrap(para_ref, for_llm=False, space=space)
    para_type = (para_node.get_ctrl() or {}).get("type") if isinstance(para_node, MetaDict) else None
    if para_type != "para":
        raise VMMemoryError(
            f"{para_ref} 不是参数节点：para_ref 必须指向 ctrl.type='para' 的 dict 节点"
        )
    _check_model_namespace(core, node_type, para_node, para_ref)
    caps = _derive_child_caps(core, parent, node, is_sub, is_init=is_init)
    conv = Conversation(
        messages=messages,
        cid=0,
        is_sub=is_sub,
        parent=parent,
        is_root=False,
        caps=caps,
    )
    core._register(conv)
    if conv.caps.mode == MODE_USER:
        # 能力自述：保留根 caps 绑定到 $MEM.caps.<cid>，由 Core 拥有，不参与授予
        roots = dict(conv.caps.roots)
        roots["caps"] = ("caps", str(conv.cid))
        conv.caps = conv.caps.copy(roots=roots)
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
        tools = _build_tools(conversation)
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


def _raw_response_json(raw):
    """从 SDK 的"原始响应"对象里取回 JSON 对象，兼容 openai 各版本。

    - openai 2.x 的 `LegacyAPIResponse`：没有 `.json()`，有 `.http_response`（httpx 响应）与 `.text`；
    - openai 1.x：有 `.json()`；
    - 未来主版本把 `.text`/`.content` 改成方法——两者都按可调用或属性处理。
    """
    http = getattr(raw, "http_response", None)
    json_fn = getattr(http, "json", None) if http is not None else None
    if callable(json_fn):
        return json_fn()
    json_fn = getattr(raw, "json", None)
    if callable(json_fn):
        return json_fn()
    text = getattr(raw, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return json.loads(text)
    content = getattr(raw, "content", None)
    if callable(content):
        content = content()
    if isinstance(content, (bytes, bytearray)):
        return json.loads(bytes(content).decode("utf-8"))
    raise TypeError(f"无法从 {type(raw).__name__} 取原始 JSON 响应体")


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
        # 宿主侧 IO 总线：由 console 宿主安装。与 host_flags 一样是宿主事实，
        # 不属于镜像内容、不进 $MEM；旧 $MEM.inputs / $MEM.outputs 设备不读它，
        # 操作面板宿主的 IO 设备由 Console 单独挂载（见 avm/io_devices.py）。
        self.io_bus = None
        # 操作面板的生命周期：宿主请求停止（关机的安全点标志）。
        self._stop_requested: bool = False
        # 本次启动的恢复摘要（boot 填写；/api/state 暴露给面板首屏）。
        self.recovery: Dict[str, Any] = {
            "source": "cold", "snapshot_seq": None, "replayed": 0,
            "restored": {}, "warnings": [],
        }
        # 镜像声明的 PLM 实现：model 名 → 类，由 load_image 按镜像的 plms 段装载。
        # 与宿主内建的 PLM_REGISTRY 并列，内建优先（镜像不得同名覆盖，装载期拒绝）。
        # 放在这里是因为本函数末尾的 install_model_catalog 要枚举它。
        self.plms: Dict[str, type] = {}
        # 安全配置（批次一 S4）：根注册表（名字 -> 真实段）。
        # 用户态对话的根表由软件动态授予（$MEM.conversations.<cid>.roots），内核不分配目录。
        self.root_registry: Dict[str, tuple] = {}
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.para_ref: str = "$MEM.model_params"
        # API 失败时的运行者策略：callable(error, attempt) -> "retry" | "reload" | "exit"。
        # 宿主安装（终端提示见 avm/runner_prompt.py）；没装时按退出处理。
        self.api_error_policy = None
        # 内核转储：镜像可用 core_dump_to / core_dump 覆盖（见 load_image）
        self.core_dump_to: Optional[str] = None
        self.core_dump_enabled: bool = True
        self.events: EventSystem = EventSystem(self)
        self.interrupts: InterruptTable = InterruptTable(self)  # 中断向量表：外部（控制台）发起的控制权转移
        self._turn_events: List[dict] = []
        # 持久运行：回合级效果日志（journal）与重启快照（snapshot）。
        # journal 为 None 时所有记录点是空操作——无日志镜像行为不变。
        self.journal: Optional[Journal] = None
        self.journal_path: Optional[str] = None   # 镜像 journal_to（load_image 解析后写入）
        self.snapshot_path: Optional[str] = None  # 镜像 snapshot_to（load_image 解析后写入）
        self._journal_seq: int = 0                # 已提交/已重放到的日志位置
        # 日志代际：resume 续当前代际；reboot/reset 由宿主归档旧日志并开新代际（seq 归零）。
        self.journal_generation: int = 1
        self._journal_batch: Dict[int, dict] = {}  # cid → 上次提交的待发批次（增量记录用）
        self._scheduler_committed: Optional[dict] = None  # 上次提交后的调度状态（增量基线）
        self.snapshot_every: Optional[int] = None  # 每 N 个已提交回合写一次定时快照；None 关闭
        self._last_snapshot_seq: int = 0           # 上次写快照对应的日志 seq
        self._turn_effects: List[dict] = []       # 本轮指令产生的语义效果（回合结束提交）
        self._replaying: bool = False             # 重放期间禁止再记录
        self._recording: bool = False             # 只在回合执行体内记录
        # 调度严格模式：镜像声明了 meta.guest_programs 时打开。打开后"活跃为空、
        # 就绪为空且无待派发中断"不再视为跑完，而是向 idle 角色挂中断；处理程序
        # 不走开空闲就 KernelPanic（空闲 = 严重错误）。没有声明的旧镜像保持旧语义：
        # run() 在队列空了时正常返回。
        self.strict_scheduler: bool = False
        self.guest_roles: Dict[str, dict] = {}      # 角色 → {"cid", "entry", "kind"}（宿主中断向量来源）
        self.guest_programs_decl: Dict[str, str] = {}  # 镜像 meta.guest_programs（角色 → 程序节点引用）
        self._image_init = None                     # load_image(restore=True) 暂存的 init 段（冷启动补跑）
        self._power_state_seen = None               # 断电判定的边沿基线（见 power_off_requested）
        self._power_request_seen = None             # 关机请求的边沿基线（见 shutdown_requested）
        # 恢复出来的对话（快照/日志）：Python 执行器首次创建时要预热一次
        self._warmup_needed: set = set()
        # 设备来源表：挂载地址 → 设备 spec（文件/类/参数/标识符；内建设备 builtin）。
        # 快照要覆盖热装设备，而设备表里只有实例，没有"这是从哪来的"。
        self.device_specs: Dict[str, dict] = {}
        self.mem.mount("conversations", ConversationsDevice(self))
        self.mem.mount("scheduler", SchedulerDevice(self))
        self.mem.mount("memory", MemorySummaryDevice(self))
        self.mem.mount("monitor", MonitorDevice(self))
        self.mem.mount("listeners", ListenersDevice(self))
        self.mem.mount("roots", RootsDevice(self))
        self.mem.mount("caps", CapsDevice(self))
        for mount in ("conversations", "scheduler", "memory", "monitor",
                      "listeners", "roots", "caps"):
            self.device_specs[mount] = {
                "builtin": True,
                "class": type(self.mem._devices[mount]).__name__,
            }
        # 启动自带的内存节点：模型目录（静态部分；LLM 清单由宿主 refresh_model_catalog 填）
        self.install_model_catalog()

    def is_hidden_cid(self, cid) -> bool:
        """对话是否对普通对话隐藏、不可寻址（显示/寻址策略，不影响恢复）。

        判定读对话自己的 `attrs["hidden"]`（宿主在保留对话创建时写 "1"）；
        `cid == PANEL_CID` 作为兼容判定保留：旧快照/旧日志里恢复出来的保留对话
        没有 attrs，不能因为缺标记就变成可见。
        """
        conv = self._conv_by_cid.get(cid)
        if conv is not None and (conv.attrs or {}).get("hidden") == "1":
            return True
        return cid == PANEL_CID

    # ------------------------------------------------------------------
    # 安全配置：根注册表与动态授予根（批次一 S4）
    # ------------------------------------------------------------------

    def _validate_root_specs(self, roots_spec: dict, check_live: bool = True) -> dict:
        """校验根注册表：名字、目标语法、前缀无关、保留区、存活对话根表。

        镜像装载（check_live=False）与运行期替换（check_live=True）共用。
        """
        checked = {}
        for name, target in roots_spec.items():
            check_root_name(name)
            target = tuple(target)
            if name == MEM_ROOT_NAME:
                raise VMMemoryError("security.roots 不能注册 MEM 根")
            if not target:
                raise VMMemoryError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if target and target[0] in CONTROL_DEVICE_ROOTS:
                raise VMMemoryError(
                    f"根 {name!r} 的目标落在内核控制面设备路径 $MEM.{target[0]}"
                )
            if not targets_prefix_free(list(checked.values()), target):
                raise VMMemoryError(f"根 {name!r} 的目标与已注册根前缀冲突")
            checked[name] = target
        if check_live:
            others = []
            for conv in self._conv_by_cid.values():
                for root_name, root_target in conv.caps.roots.items():
                    if root_name == MEM_ROOT_NAME:
                        continue
                    others.append((conv.cid, tuple(root_target)))
            for name, target in checked.items():
                for cid, other_target in others:
                    if target == other_target:
                        continue
                    if path_is_prefix(target, other_target) or path_is_prefix(other_target, target):
                        raise VMMemoryError(
                            f"根 {name!r} 的目标与存活对话 {cid} 的根表前缀冲突"
                        )
        return checked

    def install_security_config(self, roots_spec: dict) -> None:
        """装载镜像 security 段：校验并安装根注册表。

        校验项：根名规则、目标前缀无关、保留区。失败抛 VMMemoryError
        （镜像装载器翻成 ImageError）。工作目录不再由内核分配：目录由软件用
        memory_make 建，根由 `$MEM.conversations.<cid>.roots` 动态授予。
        """
        checked = self._validate_root_specs(roots_spec, check_live=False)
        self.root_registry = checked

    def replace_root_registry(self, roots_spec: dict) -> None:
        """运行期整体替换根注册表（$MEM.roots 控制面），先校验后提交。"""
        checked = self._validate_root_specs(roots_spec, check_live=True)
        self.root_registry = checked

    def _core_roots(self, conv) -> dict:
        """Core 拥有的根（目前只有能力自述根 caps）：控制面不可覆盖、重放时保留。"""
        target = conv.caps.roots.get("caps")
        return {"caps": tuple(target)} if target is not None else {}

    def _validate_grants(self, conv, grants: dict) -> None:
        """授予根的全量校验：保留区、设备路径、新根之间与其他存活根/根注册表的前缀关系。

        `grants` 是"根名 -> 目标段序列"，不含 Core 自述根；先全量校验后提交。
        """
        for name, target in grants.items():
            if not target:
                raise IllegalOperationError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if target[0] in CONTROL_DEVICE_ROOTS:
                raise IllegalOperationError(
                    f"根 {name!r} 的目标落在内核控制面设备路径 $MEM.{target[0]}"
                )
            for mount in self.mem._devices:
                mount_segments = tuple(mount.split("."))
                if target == mount_segments:
                    continue  # 显式授予设备本身是允许的
                if path_is_prefix(target, mount_segments) or path_is_prefix(mount_segments, target):
                    raise IllegalOperationError(
                        f"根 {name!r} 的目标落在保留的内核设备路径 $MEM.{mount}"
                    )
        # 新根之间前缀无关（相等目标允许共享）
        items = list(grants.items())
        for i, (n1, t1) in enumerate(items):
            for n2, t2 in items[i + 1:]:
                if t1 == t2:
                    continue
                if path_is_prefix(t1, t2) or path_is_prefix(t2, t1):
                    raise IllegalOperationError(f"根 {n1!r} 与 {n2!r} 的目标前缀冲突")
        # 与其他存活对话的根表、根注册表比较
        others = []
        for cid, other in self._conv_by_cid.items():
            if cid == conv.cid:
                continue
            for other_name, other_target in other.caps.roots.items():
                if other_name == MEM_ROOT_NAME:
                    continue
                others.append((cid, other_name, tuple(other_target)))
        for name, target in self.root_registry.items():
            others.append((None, name, tuple(target)))
        for name, target in items:
            for cid, other_name, other_target in others:
                if target == other_target:
                    continue  # 相同目标允许共享
                if path_is_prefix(target, other_target) or path_is_prefix(other_target, target):
                    raise IllegalOperationError(
                        f"根 {name!r} 与已有根 {other_name!r} 的目标前缀冲突"
                    )

    def set_conversation_roots(self, cid: int, roots_map: dict) -> None:
        """整体替换用户态对话的非 core 根表（控制面 `$MEM.conversations.<cid>.roots`）。

        `roots_map` 是"根名 -> 目标段序列（list/tuple，不含 MEM 根）"的完整映射；
        控制面语义是替换，不是合并。Core 自述根（caps）不可覆盖。
        校验通过后先记 conversation_roots 效果、再写 caps（日志与内存一致）。
        """
        conv = self._conv_by_cid.get(cid)
        if conv is None:
            raise VMMemoryError(f"对话 {cid} 不存在")
        if cid in self._finished:
            raise VMMemoryError(f"对话 {cid} 已关闭")
        if conv.caps.mode == MODE_KERNEL:
            raise IllegalOperationError(
                "内核态对话的根表固定含 MEM，不能通过控制面改写（用户态对话才可授予根）"
            )
        if not isinstance(roots_map, dict):
            raise IllegalOperationError("roots 必须是映射（根名 -> 目标段序列）")
        core_roots = self._core_roots(conv)
        grants: Dict[str, tuple] = {}
        for name, segments in roots_map.items():
            if not isinstance(name, str) or not name:
                raise IllegalOperationError(f"根名必须是非空字符串，得到 {name!r}")
            if name in core_roots:
                raise IllegalOperationError(f"根名 {name!r} 由 Core 分配，不能通过控制面覆盖")
            check_root_name(name)
            if isinstance(segments, str) or not isinstance(segments, (list, tuple)):
                raise IllegalOperationError(
                    f"根 {name!r} 的目标必须是段序列（list/tuple），得到 {type(segments).__name__}"
                )
            target = tuple(segments)
            if not target:
                raise IllegalOperationError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if any(not isinstance(seg, str) or not seg for seg in target):
                raise IllegalOperationError(f"根 {name!r} 的目标段必须是非空字符串：{target!r}")
            grants[name] = target
        self._validate_grants(conv, grants)
        new_roots = dict(core_roots)
        new_roots.update(grants)
        self.note_effect(
            "conversation_roots", cid=cid,
            roots={name: list(target) for name, target in grants.items()},
        )
        conv.caps = conv.caps.copy(roots=new_roots)

    def _register(self, conv: Conversation, cid: Optional[int] = None) -> Conversation:
        conv.cid = self._next_cid if cid is None else cid
        assert conv.cid not in self._conv_by_cid, f"cid {conv.cid} 已存在"
        self._conv_by_cid[conv.cid] = conv
        if cid is None:
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

    # ------------------------------------------------------------------
    # 回合级效果日志（journal，批次 A）
    # ------------------------------------------------------------------

    def attach_journal(self, path: str) -> "Journal":
        """挂上回合日志（后续回合从这里追加）。重放用 `replay_journal`。"""
        self.detach_journal()
        self.journal = Journal(path)
        # 调度增量的基线：从装载后的状态开始记，第一条记录覆盖 run 前的选择与回合内迁移。
        self._scheduler_committed = self._scheduler_state()
        self._refresh_journal_batch()
        return self.journal

    def detach_journal(self) -> None:
        """摘掉日志（之后的回合不再记录；已提交的记录留在文件里）。"""
        self.journal = None

    def _journaling(self) -> bool:
        """当前是否正在为回合日志记录（无日志/重放/非回合执行期都是假）。

        记录点先问这一句再构造载荷：无日志镜像不该为日志做序列化工作。
        """
        return self.journal is not None and not self._replaying and self._recording

    def note_effect(self, kind: str, **payload) -> None:
        """记一条语义效果：无日志、重放期间、非回合执行期都是空操作。"""
        if not self._journaling():
            return
        self._turn_effects.append({"kind": kind, **payload})

    def note_effects(self, effects: List[dict]) -> None:
        """批量记效果（程序包安装/卸载等一步产生多条状态变化的操作用）。"""
        if not self._journaling():
            return
        self._turn_effects.extend(effects)

    def _scheduler_state(self) -> dict:
        """当前调度状态的只读快照（回合开始/提交时各取一次，做增量用）。"""
        return {
            "active": self._active_cid,
            "ready": list(self._ready_cids),
            "dormant": list(self._dormant_cids),
            "finished": set(self._finished),
            "activation_instructions": int(self._activation_instructions),
            "next_cid": int(self._next_cid),
        }

    @staticmethod
    def _scheduler_delta(before: dict, after: dict):
        """把回合前后的调度状态压成一条增量变化；没有变化返回 None。"""
        delta = {}
        if before["active"] != after["active"]:
            delta["active"] = after["active"]
        for key in ("ready", "dormant"):
            before_list = before[key]
            after_list = after[key]
            before_set = set(before_list)
            after_set = set(after_list)
            added = [cid for cid in after_list if cid not in before_set]
            removed = [cid for cid in before_list if cid not in after_set]
            if added:
                delta[key + "_add"] = added
            if removed:
                delta[key + "_remove"] = removed
        added_finished = sorted(after["finished"] - before["finished"])
        removed_finished = sorted(before["finished"] - after["finished"])
        if added_finished:
            delta["finished_add"] = added_finished
        if removed_finished:
            delta["finished_remove"] = removed_finished
        if before["activation_instructions"] != after["activation_instructions"]:
            delta["activation_instructions"] = after["activation_instructions"]
        if before["next_cid"] != after["next_cid"]:
            delta["next_cid"] = after["next_cid"]
        if not delta:
            return None
        return {"kind": "scheduler", **delta}

    def _refresh_journal_batch(self) -> None:
        """重放/恢复完成后，把"上次提交的批次"对齐到当前对话状态。"""
        self._journal_batch = {
            cid: batch_to_dict(conv.user_batch) for cid, conv in self._conv_by_cid.items()
        }

    def _refresh_journal_scheduler(self) -> None:
        """重放/恢复完成后，把调度增量基线对齐到当前调度状态。"""
        self._scheduler_committed = self._scheduler_state()

    def _commit_journal_record(self, conv: Conversation, history_delta: List[dict]) -> None:
        """回合结束提交一条记录（安全点调用；异常回合不提交）。

        记录是**有序变化序列**：本轮追加的消息、指令产生的效果、待发批次的变化、
        调度队列的变化。顺序即执行顺序；重放只按顺序应用，不再按字段分阶段重组。
        """
        effects = self._turn_effects
        self._turn_effects = []
        if self.journal is None:
            return
        changes: List[dict] = []
        for item in history_delta:
            changes.append({"kind": "message", "message": item})
        for effect in effects:
            changes.append({"kind": "effect", "effect": effect})
        cid = conv.cid
        empty_batch = {"tool_responses": [], "user_contents": []}
        # 批次变化可能发生在别的对话上（delivery 效果给目标加输入），所以对所有对话
        # 取差量；数量与对话数同级，单条改动通常为空。
        for other_cid, other_conv in self._conv_by_cid.items():
            current_batch = batch_to_dict(other_conv.user_batch)
            previous = self._journal_batch.get(other_cid, empty_batch)
            if current_batch != previous:
                changes.append(
                    {"kind": "batch", "cid": other_cid, "batch": current_batch}
                )
            self._journal_batch[other_cid] = current_batch
        scheduler_before = self._scheduler_committed
        if scheduler_before is None:
            scheduler_before = self._scheduler_state()
        delta = self._scheduler_delta(scheduler_before, self._scheduler_state())
        if delta is not None:
            changes.append(delta)
        self._scheduler_committed = self._scheduler_state()
        record = {
            "v": 2,
            "generation": int(self.journal_generation),
            "cid": cid,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "changes": changes,
        }
        self._journal_seq = self.journal.append(record)

    def replay_journal(self, records: Optional[List[dict]] = None) -> int:
        """按 seq 顺序重放日志（默认从日志文件续在已应用位置之后）。

        幂等：应用位置由 `_journal_seq` 推进，同一段日志重放两次第二次是空操作。
        返回本次实际应用的记录数。日志不连续（缺记录）直接报错。
        """
        if self.journal is None:
            raise JournalError("没有挂载日志：先 attach_journal")
        pending = (
            self.journal.iter_records(after_seq=self._journal_seq)
            if records is None else list(records)
        )
        applied = 0
        for record in pending:
            seq = record.get("seq")
            if not isinstance(seq, int):
                raise JournalError(f"日志记录缺少合法 seq：{record!r}")
            if seq <= self._journal_seq:
                continue
            if seq != self._journal_seq + 1:
                raise JournalError(
                    f"日志不连续：当前已应用到 seq={self._journal_seq}，"
                    f"下一条是 seq={seq}（缺记录，拒绝重放）"
                )
            apply_journal_record(self, record)
            self._journal_seq = seq
            applied += 1
        if applied:
            self._refresh_journal_batch()
            self._refresh_journal_scheduler()
        return applied

    # ------------------------------------------------------------------
    # 重启快照（snapshot，批次 A）
    # ------------------------------------------------------------------

    def snapshot_to(self, path: Optional[str] = None) -> str:
        """写重启快照（默认镜像声明的 snapshot_to）；返回落点。"""
        from .snapshot import snapshot, write_atomic

        target = path or self.snapshot_path
        if not target:
            raise ValueError("没有快照路径：传 path 或让镜像声明 snapshot_to")
        write_atomic(target, snapshot(self))
        self._last_snapshot_seq = self._journal_seq
        return target

    def restore_from(self, path: str) -> dict:
        """从重启快照恢复（替换调度/对话/内存/监听器/设备表）；返回快照文档。"""
        from .snapshot import read_snapshot, restore

        doc = read_snapshot(path)
        restore(self, doc)
        return doc

    def unmount_device(self, mount: str) -> None:
        """卸载设备：设备表与设备来源表一起删（包卸载、日志重放、宿主共用）。"""
        self.device_specs.pop(mount, None)
        self.mem.unmount(mount)

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
            if not conv.caps.tool_allowed(cmd_type):
                # 能力边界：schema 与 PLM 注入已裁剪，这里再兜一道实际调度
                _report_tool_error(
                    conv,
                    f"工具 {cmd_type} 未授权给本对话（mode={conv.caps.mode}），参数: "
                    f"{json.dumps(args, ensure_ascii=False)}",
                )
                conv.user_batch.add_tool_response(
                    f"Error: 工具 {cmd_type} 未授权给本对话", call_id
                )
                continue
            instr_cls = registry[cmd_type]
            error = _validate_instruction_args(instr_cls, args)
            if error is not None:
                _report_tool_error(conv, f"工具 {cmd_type} 参数不合法：{error}")
                conv.user_batch.add_tool_response(f"Illegal: {error}", call_id)
                continue
            instr = instr_cls(call_id, conv.cid, args)
            instr.execute(self, conv)

    def _on_conversation_finished(self, conv: Conversation, result: str):
        if conv.is_sub:
            parent = conv.parent
            if parent is not None and parent.cid not in self._finished:
                if conv.metadata.get("returned"):
                    _report_tool_error(conv, "亚对话已完成过一次，本次完成不再写回父对话")
                else:
                    # 亚对话紧耦合：完成即自动写回并唤醒父
                    call_id = conv.metadata.get("call_id", "direct_output")
                    parent.user_batch.add_tool_response(result, call_id)
                    self.note_effect(
                        "delivery", to_cid=parent.cid,
                        tool_responses=[{"content": result, "tool_call_id": call_id}],
                    )
                    conv.metadata["returned"] = True
                # 用完即删：返回通道只用一次，个体不再保留
                self._delete_conversation(conv.cid)
                if not self._is_waiting_for_return(parent):
                    # 同一轮可能声明多个 create_sub：全部返回后才唤醒父
                    self._resume(parent.cid)
                else:
                    # 父还在等别的亚对话：让出活跃权，继续调度（活跃指针不能留在已删个体上）
                    self._active_cid = None
                    self._pick_next_active()
                return
            # 父已关闭或不存在：没有返回通道，亚对话用完即删
            self._delete_conversation(conv.cid)
            self._active_cid = None
            self._pick_next_active()
            return
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
        关闭一个还没返回的亚对话时，给等待中的父对话补一条响应——否则父的那次
        `create_sub` 永远没有 tool 消息，父下次运行时历史配对不变量会被破坏。
        """
        if cid in self._finished:
            return
        self.note_effect("conversation_close", cid=cid)
        conv = self._conv_by_cid.get(cid)
        if conv is not None and conv.is_sub:
            # 亚对话用完即删：先给等待中的父补响应，再移除个体
            self._answer_parent_on_sub_close(conv)
            self._delete_conversation(cid)
            if self._active_cid == cid:
                self._active_cid = None
            return
        self._finished.add(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        self.events.cancel_all_for_owner(cid)
        self._executors.pop(cid, None)  # 释放 PLM 状态（命名空间等）

    def _delete_conversation(self, cid: int) -> None:
        """亚对话用完即删：移除个体与全部内核引用（调度队列、监听器、执行器、finished 标记）。

        级联：以它为父的亚对话一并删除（它们已经没有返回通道）。create_cmd 子对话
        是独立个体（is_sub=False），不在此列。
        """
        self.note_effect("conversation_delete", cid=cid)
        for child in list(self._conv_by_cid.values()):
            if child.is_sub and child.parent is not None and child.parent.cid == cid:
                self._delete_conversation(child.cid)
        conv = self._conv_by_cid.pop(cid, None)
        if conv is None:
            return
        for queue in (self._ready_cids, self._dormant_cids):
            while cid in queue:
                queue.remove(cid)
        self._finished.discard(cid)
        self.events.cancel_all_for_owner(cid)
        self._executors.pop(cid, None)

    def _answer_parent_on_sub_close(self, sub: Conversation) -> None:
        """亚对话在返回结果前被关闭：给还活着的父对话补一条 Error 响应并唤醒它。"""
        if not sub.is_sub or sub.metadata.get("returned"):
            return
        parent = sub.parent
        if parent is None or parent.cid in self._finished:
            return
        call_id = sub.metadata.get("call_id", "direct_output")
        parent.user_batch.add_tool_response(
            f"Error: 亚对话（cid={sub.cid}）在返回结果前被关闭", call_id
        )
        self.note_effect(
            "delivery", to_cid=parent.cid,
            tool_responses=[{
                "content": f"Error: 亚对话（cid={sub.cid}）在返回结果前被关闭",
                "tool_call_id": call_id,
            }],
        )
        sub.metadata["returned"] = True
        # 同一轮多个 create_sub：父还有别的待答调用时只补响应，等全部返回后再唤醒
        if parent.cid != self._active_cid and not self._is_waiting_for_return(parent):
            if parent.cid in self._dormant_cids:
                self._dormant_cids.remove(parent.cid)
            if parent.cid not in self._ready_cids:
                self._ready_cids.insert(0, parent.cid)  # 亚对话结束后父对话先跑
        _report_tool_error(sub, f"亚对话被关闭，已给父对话 cid={parent.cid} 补一条响应")

    def _pick_next_active(self):
        while self._ready_cids:
            picked = self._ready_cids.pop(0)
            conv = self._conv_by_cid.get(picked)
            if conv is not None and self._is_waiting_for_return(conv):
                # 等亚对话返回的对话不能先跑：退回休眠，响应到了再由 _resume 唤醒。
                # 它批次里的内容保留，等被唤醒时一起处理。
                if picked not in self._dormant_cids:
                    self._dormant_cids.append(picked)
                continue
            if picked in self._dormant_cids:
                self._dormant_cids.remove(picked)
            if not self._same_chain(self._active_cid, picked):
                self._activation_instructions = 0  # 调用链断开：新链重新给满预算
            self._active_cid = picked
            return
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
        self._turn_effects.clear()  # 只记录本轮；异常回合的效果随下一轮一起丢弃
        conv = self._conv_by_cid[self._active_cid]
        executor = None
        attempt = 0
        history_delta: List[dict] = []
        while True:
            try:
                # 取 para 与选执行器也放进 try：它们会因为配置错配抛错，
                # 原先在 try 之外，逃出去连监测帧都没有（问题 1）。
                para = self._get_para(conv)
                executor = self._get_executor(conv, para)
                before_count = len(conv.messages)
                result, return_calls, _ = executor.exec(conv, para)
                # 本轮写进历史的"实际消息"（消费进历史的批次 + 新的 assistant 消息）。
                # 日志按实际消息记而不是按生成它的效果推导：before 事件可以改写 payload。
                history_delta = [
                    message_to_dict(m) for m in conv.messages[before_count:]
                ]
                conv.user_batch.clear()

                self._recording = True
                try:
                    if return_calls:
                        self._process_return_calls(return_calls, conv)
                        self._activation_instructions += len(return_calls)
                        if conv.cid in self._finished or conv.cid not in self._conv_by_cid:
                            # 内核级关闭或亚对话用完即删：本轮结束后选择下一个活跃对话
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
                finally:
                    self._recording = False
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
                self._commit_journal_record(conv, history_delta)
                self._maybe_periodic_snapshot()
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
        Python 对话：model 必须在内建 PLM_REGISTRY 或本核的 self.plms 中，
        否则报错（不静默回退）；LLM 对话：用 core.lmu（model 交给 API 校验）。"""
        if conv.cid in self._executors:
            return self._executors[conv.cid]
        model = para.get("model")
        if conv.is_python:
            if not isinstance(model, str) or not (model in PLM_REGISTRY or model in self.plms):
                available = list(PLM_REGISTRY) + sorted(set(self.plms) - set(PLM_REGISTRY))
                raise ValueError(
                    f"Python 对话 {conv.cid} 的 model {model!r} 不在 PLM_REGISTRY 中（可用: {available}）"
                )
            executor = PLMExecutor(create_plm(model, extra=self.plms))
            self._executors[conv.cid] = executor
            self._warm_restored_python_executor(executor, conv)
            return executor
        return self.lmu

    def _warm_restored_python_executor(self, executor: PLMExecutor, conv: Conversation) -> None:
        """恢复出来的 Python 对话：重建执行器后重放 system 消息、把已有历史标记为已处理。

        PLM 的命名空间不序列化（计划 A5 待定夺第 1 项的默认取舍）：跨轮状态必须由
        程序自己落在 `$MEM`；执行器重建后 setup 代码要再执行一遍，而历史里的 user/tool
        消息不能再执行（否则副作用重复发生）。新对话（历史只有 system）不走这里。
        """
        if conv.cid not in self._warmup_needed:
            return
        self._warmup_needed.discard(conv.cid)
        messages = conv.to_api_messages()
        restore_state = getattr(executor.plm, "restore_state", None)
        if not callable(restore_state):
            return
        restore_state(messages, {"tools": _build_tools(conv)})

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

    def ensure_operator_conversation(self, para_ref: Optional[str] = None) -> Conversation:
        """确保保留对话（cid = PANEL_CID）存在并返回它；不绑定执行器。

        - 身份固定：cid 是作者规定的保留值，不参与正常分配，也不依赖创建顺序；
        - 初始进休眠队列、不进就绪队列：它不主动抢占，只有中断能激活它；
        - 装载时、日志重放遇到它的记录时、控制台绑定时都走这里；
        - 执行器由 bind_console_conversation 在宿主启动后注入（plm.operator 需要通道，
          注册表无参构造出来的实例不能用）。
        """
        conv = self._conv_by_cid.get(PANEL_CID)
        if conv is not None:
            return conv
        conv = Conversation(
            messages=[SystemMessage(content=_OPERATOR_SYSTEM_PROMPT)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=False,
            # 宿主标记：控制台持有的保留对话对普通对话隐藏。判定只读 attrs
            # （旧快照没有 attrs 时由 is_hidden_cid 的 cid 兼容分支兜底）。
            attrs={"host": "console", "hidden": "1"},
        )
        self._register(conv, cid=PANEL_CID)
        conv.para_ref = para_ref or self.para_ref
        conv.is_python = True
        self._dormant_cids.append(conv.cid)
        return conv

    def bind_console_conversation(
        self,
        channel,
        para_ref: Optional[str] = None,
    ) -> Conversation:
        """把控制台通道绑到保留对话；没有就先建出来，返回该对话。"""
        conv = self.ensure_operator_conversation(para_ref)
        conv.para_ref = para_ref or self.para_ref
        self._executors[conv.cid] = PLMExecutor(create_plm("plm.operator", channel=channel))
        return conv

    def start_operator_conversation(
        self,
        channel,
        para_ref: Optional[str] = None,
    ) -> Conversation:
        """兼容旧入口：绑定控制台通道；没有保留对话时建一条。"""
        return self.bind_console_conversation(channel, para_ref)

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
    def _pending_call_ids(conv: Conversation) -> List[str]:
        """还没拿到 tool 响应的调用 id（历史与待发批次合并后比对）。

        目前只有 `create_sub` 会留下待答调用：父对话在等亚对话返回。
        """
        messages = conv.messages
        last = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and getattr(messages[i], "tool_calls", None):
                last = i
                break
        if last is None:
            return []
        wanted = []
        for tc in messages[last].tool_calls or []:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if isinstance(call_id, str) and call_id:
                wanted.append(call_id)
        got = {m.tool_call_id for m in messages[last + 1:] if m.role == "tool"}
        got |= {r.tool_call_id for r in conv.user_batch.tool_responses}
        return [call_id for call_id in wanted if call_id not in got]

    def _is_waiting_for_return(self, conv: Conversation) -> bool:
        """对话是否在等工具返回（亚对话）。等返回的对话不能被外部消息唤醒运行。"""
        return bool(self._pending_call_ids(conv))

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
                    if name not in HOST_ONLY_PLMS  # 宿主独占的 PLM 不列给对话
                ] + [
                    # 镜像声明的 PLM 也列进来（同名内建优先，装载期已拒绝重名）
                    Core._plm_catalog_entry(name, cls)
                    for name, cls in self.plms.items()
                    if name not in PLM_REGISTRY
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
            payload = _raw_response_json(raw)
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
            for key in ("name", "owned_by", "context_window", "max_output_tokens",
                        "input_modalities", "output_modalities"):
                value = item.get(key)
                if value is None:
                    continue
                # 内存约定：para 与目录里的值都以 str 存储
                entry[key] = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            effort = item.get("effort")
            if isinstance(effort, dict):
                levels = effort.get("supported_levels")
                if isinstance(levels, list):
                    entry["reasoning_effort_levels"] = ", ".join(str(v) for v in levels)
                default_level = effort.get("default_level")
                if isinstance(default_level, str) and default_level:
                    entry["default_effort_level"] = default_level
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

    # ------------------------------------------------------------------
    # 内核转储（诊断快照，不是恢复输入）
    # ------------------------------------------------------------------

    def core_dump_path(self) -> str:
        """默认转储落点：<镜像目录>/out/<镜像名>.core.json；无镜像上下文时用 cwd 下的 core.<pid>.json。"""
        if self.image_dir and self.image_name:
            return os.path.join(self.image_dir, "out", f"{self.image_name}.core.json")
        return os.path.join(os.getcwd(), f"core.{os.getpid()}.json")

    def dump_core(
        self,
        path: Optional[str] = None,
        error: Optional[BaseException] = None,
        *,
        messages_per_conversation: int = 200,
        max_frames: int = 200,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> Optional[str]:
        """把死亡现场写成一个 JSON，返回写入路径；失败返回 None。

        诊断快照，**不是恢复输入**（恢复依据是回合日志，快照是它的缓存）。全程不抛：任何一段序列化失败
        都降级为占位——转储绝不能成为新的内核退出路径。
        """
        target = path or self.core_dump_to or self.core_dump_path()
        try:
            doc = self._build_core_dump(error, messages_per_conversation, max_frames)
            text = json.dumps(doc, ensure_ascii=False, indent=2)
            if len(text.encode("utf-8")) > max_bytes:
                # 内存树按作者决定不截断，先砍对话正文，再砍对话本身（旧的先砍）
                for brief in doc["conversations"]:
                    if brief.get("messages"):
                        brief["messages"] = []
                        brief["messages_truncated"] = True
                        doc["truncated"]["conversations"].append(brief["cid"])
                text = json.dumps(doc, ensure_ascii=False, indent=2)
            if len(text.encode("utf-8")) > max_bytes:
                for brief in doc["conversations"][:-1]:
                    brief.pop("messages", None)
                    if brief["cid"] not in doc["truncated"]["conversations"]:
                        doc["truncated"]["conversations"].append(brief["cid"])
                text = json.dumps(doc, ensure_ascii=False, indent=2)
            doc["truncated"]["total_bytes"] = len(text.encode("utf-8"))
            text = json.dumps(doc, ensure_ascii=False, indent=2)
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
            return target
        except Exception as e:
            print(
                f"[AVM] 内核转储写入失败（{type(e).__name__}: {e}），原始错误不受影响",
                file=sys.stderr,
            )
            return None

    def _build_core_dump(self, error: Optional[BaseException], limit: int, max_frames: int) -> dict:
        truncated = {"conversations": [], "frames": False, "total_bytes": 0}
        err = None
        if error is not None:
            try:
                tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            except Exception as e:  # traceback 都可能失败：降级
                tb = f"<traceback 不可用：{type(e).__name__}: {e}>"
            err = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": tb,
                "panicked": isinstance(error, KernelPanic),
            }
        frames = []
        try:
            all_frames = self.monitor.frames()
            if len(all_frames) > max_frames:
                truncated["frames"] = True
                all_frames = all_frames[-max_frames:]
            frames = [dataclasses.asdict(f) for f in all_frames]
        except Exception as e:
            frames = [{"error": f"帧序列化失败：{type(e).__name__}: {e}"}]
        try:
            listeners = [dict(x) for x in self.events.listeners]
        except Exception as e:
            listeners = [{"error": f"{type(e).__name__}: {e}"}]
        try:
            memory = self.mem.to_image_dict()
        except Exception as e:
            memory = {"kind": "dict", "value": {}, "meta": f"<内存树序列化失败：{type(e).__name__}: {e}>"}
        return {
            "format": "avm.core_dump",
            "version": 1,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "image": os.path.join(self.image_dir or "", self.image_name or "") or None,
            "host_flags": dict(self.host_flags or {}),
            "para_ref": self.para_ref,
            "note": "诊断快照，不是恢复输入；memory 段与 persist_to 同格式",
            "error": err,
            "scheduler": {
                "active": self._active_cid,
                "ready": list(self._ready_cids),
                "dormant": list(self._dormant_cids),
                "finished": sorted(self._finished),
                "instruction_budget": self._instruction_budget,
                "activation_instructions": self._activation_instructions,
            },
            "memory": memory,
            "security": {
                "roots": {name: list(target) for name, target in self.root_registry.items()},
            },
            "conversations": [
                self._conversation_dump(self._conv_by_cid[cid], limit)
                for cid in sorted(self._conv_by_cid)
            ],
            "listeners": listeners,
            "devices": [
                {"mount": mount, "type": type(dev).__name__, "id": getattr(dev, "device_id", None)}
                for mount, dev in list(self.mem._devices.items())
            ],
            "interrupts": {"last_dispatch": list(self.interrupts.last_dispatch)},
            "frames": frames,
            "truncated": truncated,
        }

    def _conversation_dump(self, conv: Conversation, limit: int) -> dict:
        """单个对话的现场：状态、身份、历史（保最后 limit 条）、待发批次。"""
        state = "?"
        if conv.cid == self._active_cid:
            state = "active"
        elif conv.cid in self._finished:
            state = "finished"
        elif conv.cid in self._ready_cids:
            state = "ready"
        elif conv.cid in self._dormant_cids:
            state = "dormant"
        try:
            messages = conv.to_checkpoint().get("messages", [])
        except Exception as e:
            messages = [{"error": f"历史序列化失败：{type(e).__name__}: {e}"}]
        total = len(messages)
        brief = {
            "cid": conv.cid,
            "identity": conv.identity,
            "name": getattr(conv, "name", None),
            "state": state,
            "is_python": conv.is_python,
            "is_sub": conv.is_sub,
            "parent": conv.parent.cid if conv.parent is not None else None,
            "para_ref": conv.para_ref,
            "metadata": dict(conv.metadata or {}),
            "caps": caps_to_dict(conv.caps),
            "attrs": dict(conv.attrs or {}),
            "message_count": total,
            "messages": messages[-limit:] if limit else [],
            "messages_truncated": total > limit,
            "batch": {
                "tool_responses": [
                    {"content": r.content, "tool_call_id": r.tool_call_id}
                    for r in conv.user_batch.tool_responses
                ],
                "user_contents": list(conv.user_batch.user_contents),
            },
        }
        return brief

    # -- 操作面板生命周期 ------------------------------------------------

    def _maybe_periodic_snapshot(self) -> None:
        """每 snapshot_every 个已提交回合在安全点写一次定时快照。

        快照是缓存：写失败只警告，不改变运行；日志仍是持久依据。
        """
        every = self.snapshot_every
        if not every or not self.snapshot_path or self.journal is None:
            return
        if self._journal_seq - self._last_snapshot_seq < every:
            return
        try:
            self.snapshot_to()
        except Exception as e:
            print(
                f"[AVM] 定时快照写入失败（{type(e).__name__}: {e}）",
                file=sys.stderr,
            )

    def request_stop(self) -> None:
        """宿主请求停止：run() 在下一个安全点返回，交宿主写快照。

        只置标志、不动当前回合——半截回合不能拍快照。若对话正阻塞在设备读里，
        这一轮到不了安全点；那是宿主侧的阻塞态问题，不在这里处理。
        正常断电（OS 写 `$MEM.sys.power.state == "off"`）不走这里：宿主在安全点
        看到 off 后直接退出（见 `power_off_requested`）。本入口留给强制退出路径。
        """
        self._stop_requested = True

    def read_power_state(self):
        """读 OS 约定的断电节点 `$MEM.sys.power.state`；不存在/不是字符串返回 None。"""
        try:
            value = self.unwrap("$MEM.sys.power.state", for_llm=False)
        except VMMemoryError:
            return None
        return value if isinstance(value, str) else None

    def power_off_requested(self) -> bool:
        """安全点断电判定（边沿触发）：非 off → off 的一次跳变才算一次断电。

        快照与日志重放会把上一次正常关机写下的 off 一起恢复回来；把启动时的存量
        off 当成新请求会让 VM 一上电就退出。宿主只响应本次上电之后 OS 新写入的 off。
        VM 不写 clean、不改 `$MEM.sys.boot.*`（脏关闭判定属于 OS 计划的数据面）。
        """
        state = self.read_power_state()
        previous = self._power_state_seen
        self._power_state_seen = state
        return state == "off" and previous != "off"

    def _check_journal_seq(self) -> None:
        """安全点断言：日志末 seq 不得大于内存已提交 seq。

        正常提交时两者相等（append 返回的 seq 立刻写进 `_journal_seq`）；恢复路径
        必须先把快照 seq 之后的日志尾部重放完（boot 负责）。大于说明日志里有未重放
        的记录，继续跑会把内存当成比日志新——拒绝继续。
        """
        if self.journal is None:
            return
        last = self.journal.last_seq()
        if last > self._journal_seq:
            raise JournalError(
                f"安全点 seq 断言失败：日志末 seq={last} 大于内存已提交 seq="
                f"{self._journal_seq}；先把日志重放完再运行"
            )

    def _idle_now(self) -> bool:
        """安全点空闲判定：活跃为空、就绪为空、无待派发中断。

        活跃对话（包括正阻塞在输入设备读上的面板对话）不算空闲——它的回合还没到
        安全点。空闲在严格调度模式下是严重错误（guest 活体错误），见 `run`。
        """
        return (
            self._active_cid is None
            and not self._ready_cids
            and self.interrupts.pending_count == 0
        )

    def read_power_request(self):
        """读 OS 约定的关机请求节点 `$MEM.sys.power.request`；不存在/非字符串返回 None。"""
        try:
            value = self.unwrap("$MEM.sys.power.request", for_llm=False)
        except VMMemoryError:
            return None
        return value if isinstance(value, str) else None

    def shutdown_requested(self) -> bool:
        """安全点判定（边沿触发）：OS 新写入了 `$MEM.sys.power.request == "shutdown"`。

        存量值不算：resume 会把上一次关机时写下的 "shutdown" 一起恢复回来，
        重新唤醒监督程序等于重复关机。启动基线在 run() 入口记录。
        """
        value = self.read_power_request()
        previous = self._power_request_seen
        self._power_request_seen = value
        return value == "shutdown" and previous != "shutdown"

    def _wake_guest(self, role: str) -> bool:
        """用中断把执行权交给声明的 guest 角色（宿主不代替它们发指令）。

        返回 False 表示镜像没有声明该角色：请求留在世界节点上（OS 缺席时优雅关机
        停在挂起，宿主保持运行，由操作员决定是否强制退出）。
        """
        info = self.guest_roles.get(role)
        if info is None:
            return False
        self.interrupts.raise_(role, payload=info.get("entry"))
        self.interrupts.dispatch()
        return True

    def _panic_idle(self, message: str):
        """空闲 = 严重错误：记监测帧后抛 KernelPanic（宿主记录并退出）。"""
        error = KernelPanic(message)
        self.monitor.record(self, error=error)
        raise error

    def _os_idle_waiting(self) -> bool:
        """OS 是否在空闲约定节点上声明了等待：`$MEM.sys.idle.state == "waiting"`。

        节点不存在/不是字符串时一律不等待（按空闲严重错误处理）：等待约定必须由
        镜像程序显式写出，内核不替镜像猜。
        """
        try:
            value = self.unwrap("$MEM.sys.idle.state", for_llm=False)
        except VMMemoryError:
            return False
        return isinstance(value, str) and value == "waiting"

    def _wait_for_host_or_work(self) -> None:
        """OS 声明空闲等待后，内核停在安全点等宿主动作/新就绪对话。

        不派发中断、不推进回合；只轮询三件事：宿主中断（面板输入/暂停等）、
        新就绪对话（时钟/事件 notify 可能随后续动作出现）、停止请求。等待期间
        内存与日志都是完整的，宿主可以随时安全退出。
        """
        while not (
            self._stop_requested
            or self._active_cid is not None
            or self._ready_cids
            or self.interrupts.pending_count
        ):
            time.sleep(0.01)

    def _wake_resumed_boot(self) -> None:
        """resume 后让声明的 boot 角色再跑一次，并把 last_exit 放进入口 payload。

        快照恢复不跑 init、也不自动运行 boot（冷启动由就绪队列驱动）。但 OS 的
        boot-check 需要知道这次是正常恢复还是脏关闭；VM 只提供 `recovery["last_exit"]`
        数据面，用中断把 boot 唤醒是宿主不代替 guest 发指令的既有规则。payload 的
        形状由 guest 程序类型决定（python：入口正文；LLM：角色名 + 数据行）。
        """
        mode = self.recovery.get("mode")
        resumed = mode == "resume" or (
            mode in (None, "auto") and self.recovery.get("source") == "snapshot"
        )
        if not resumed:
            return
        info = self.guest_roles.get("boot")
        if info is None:
            return
        last_exit = str(self.recovery.get("last_exit") or "unknown")
        entry = info.get("entry") or ""
        if info.get("kind") == "python":
            payload = "yield from boot(" + repr(last_exit) + ")"
        else:
            payload = entry + "\nlast_exit=" + last_exit
        self.interrupts.raise_("boot", payload=payload)

    def run(self):
        self._pick_next_active()

        self.monitor.record_baseline(self)
        # 启动时的存量断电/关机请求不是新请求：边沿基线从安全点第一次观察开始。
        self._power_state_seen = self.read_power_state()
        self._power_request_seen = self.read_power_request()
        # resume：boot 角色由中断唤醒，宿主的事实 last_exit 随 payload 进入口。
        self._wake_resumed_boot()
        idle_retry = False
        idle_cid = None

        while True:
            # 安全点：上一轮推理（含该轮产生的全部指令）已执行完，是状态完整、可恢复
            # 的边界——预算中断用的是同一个边界。中断在这里夺取执行权。
            # 中断表为空时这一行没有副作用。
            self.interrupts.dispatch()
            # 日志与内存的提交位置必须追平（恢复路径由 boot 保证）。
            self._check_journal_seq()
            try:
                self._check_invariants()
            except KernelPanic as e:
                self.monitor.record(self, error=e)
                raise
            if self._stop_requested:
                # 停在安全点：上一轮已完整提交，宿主可以拍快照后干净退出。
                return
            if self.power_off_requested():
                # 正常断电边界：OS 在回合里写下 off；宿主在安全点看到后写快照/persist
                # 并退出（宿主退出路径负责，不经 request_stop）。
                return
            if self.shutdown_requested():
                # OS 软件提交了关机请求（$MEM.sys.power.request="shutdown"）：宿主用中断
                # 给声明的 shutdown 监督程序一次执行机会。没有声明就停在挂起，不自行断电。
                self._wake_guest("shutdown")
            if self._active_cid is None and self._ready_cids:
                self._pick_next_active()
            if self._idle_now():
                if not self.strict_scheduler:
                    # 旧镜像（没有声明 guest 程序）：队列空了就是"跑完了"，保持旧语义。
                    return
                idle = self.guest_roles.get("idle")
                if idle is None:
                    self._panic_idle(
                        "系统空闲，但镜像没有声明 idle guest 程序"
                        "（meta.guest_programs.idle）：空闲是 guest 活体错误"
                    )
                if self._os_idle_waiting():
                    # OS 空闲处理程序显式声明"在等待"：不 panic、不重复挂 idle 中断，
                    # 内核停在安全点轮询宿主中断/新就绪对话。OS 再次被唤醒时自己把
                    # idle.state 写回非 waiting；写回后仍空闲却没让系统离开，才回到
                    # 下面的严重错误判定。
                    self._wait_for_host_or_work()
                    continue
                if idle_retry:
                    self._panic_idle(
                        "系统空闲：guest idle 处理程序本轮没有让系统离开空闲"
                        "（活跃为空、就绪为空、无待派发中断）"
                    )
                idle_cid = idle.get("cid")
                self.interrupts.raise_("idle", payload=idle.get("entry"))
                self.interrupts.dispatch()
                if self._active_cid is None and not self._ready_cids:
                    self._panic_idle(
                        "idle 中断已派发，但没有任何对话可运行（向量目标缺失或被关闭）"
                    )
                idle_retry = True
            elif self._active_cid is None and not self._ready_cids:
                continue  # 并发挂起的中断：回到循环顶部派发
            elif self._active_cid is not None and self._active_cid != idle_cid:
                # 空闲之后有别的对话拿到执行权：系统确实离开了空闲。
                idle_retry = False
            self.advance_conversation()

    def _canonical_ref_without_final(self, target, space=None):
        """canonical 化的变体：不跟随最终段的链接（删链接时裁决对象是链接节点）。"""
        try:
            split = self.mem.split_path(target, space=space)
        except Exception:
            return None
        if split[0] == "device":
            return ("device", split[1], tuple(split[2]))
        segments = list(split[1])
        if not segments:
            return ("memory", ())
        parent_ref = "$MEM" if len(segments) == 1 else "$MEM." + ".".join(segments[:-1])
        try:
            follow = self.mem.follow_link_target(parent_ref, space=space)
        except Exception:
            return ("memory", tuple(segments))
        if follow[0] == "memory":
            return ("memory", tuple(follow[1]) + (segments[-1],))
        return ("device", follow[1], tuple(follow[2]) + (segments[-1],))

    def canonical_ref(self, target, space=None, follow_final: bool = True):
        """把请求别名规范化为真实目标引用（跟随链接；不存在时取最长存在前缀）。

        follow_final=False：只跟随父路径上的链接，最终段保持原样（删除链接时用）。
        

        返回 ("memory", 真实段序列) / ("device", 挂载键, 剩余段)；
        无法解析返回 None。只用于事件匹配与权限判定，不展示给普通处理器。
        """
        if not isinstance(target, str) or not target.startswith("$"):
            return None
        if not follow_final:
            return self._canonical_ref_without_final(target, space=space)
        # 先尝试整条路径决议（会跟随链接与 LinkDevice 透明挂载）
        try:
            follow = self.mem.follow_link_target(target, space=space)
        except Exception:
            follow = None
        if follow is not None:
            if follow[0] == "memory":
                return ("memory", tuple(follow[1]))
            return ("device", follow[1], tuple(follow[2]))
        try:
            split = self.mem.split_path(target, space=space)
        except Exception:
            return None
        if split[0] == "device":
            return ("device", split[1], tuple(split[2]))
        segments = list(split[1])
        for i in range(len(segments), -1, -1):
            ref = "$MEM" if i == 0 else "$MEM." + ".".join(segments[:i])
            try:
                follow = self.mem.follow_link_target(ref, space=space)
            except Exception:
                continue
            suffix = tuple(segments[i:])
            if follow[0] == "memory":
                return ("memory", tuple(follow[1]) + suffix)
            return ("device", follow[1], tuple(follow[2]) + suffix)
        return ("memory", tuple(segments))

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
            target_ref=("device", split[1], tuple(split[2])),
        )
        event = self.events.dispatch(event)
        self.events.deliver_notifies(event)
        return event

    def unwrap(self, value, for_llm=True, space=None):
        if not value.startswith("$"):
            return value
        # TODO(issue): 这里一次把整条引用按点号切分，设备拿到的是散开的段
        # （$MEM.AVMRepo.read."avm/core.py".0 → ["read", '"avm/core', 'py"', "0"]），
        # AVMRepo 因此要用引号把区间还原。是否改成逐步切分——逐段走内存树，命中设备时把
        # 剩余的原字符串整段交给设备、由设备自己定义子路径语法——现在不能确定；
        # 它与 split_path 的最长挂载键匹配、设备授权是同一件事，要一起想。
        value = [value[0], *value[1:].split(".")]
        if len(value) > 1 and value[1] in ("conv", "batch"):
            assert value[0] == "$"
            if space is not None and not space.has_mem():
                raise VMMemoryError(
                    f"引用根不可用：{value[1]!r}（本对话根表不含 MEM）"
                )
            try:
                cid = int(value[2])
            except (IndexError, ValueError):
                raise VMMemoryError(f"${value[1]}.<cid> 的 cid 必须是数字：{value[2:] or '（缺）'}")
            conv = self._conv_by_cid.get(cid)
            if conv is None or self.is_hidden_cid(cid):
                raise VMMemoryError(f"对话 {cid} 不存在：${value[1]}.{cid}")
            return conv if value[1] == "conv" else conv.user_batch
        else:
            return self.mem.unwrap(value, for_llm=for_llm, space=space)

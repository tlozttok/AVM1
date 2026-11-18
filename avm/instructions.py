"""AVM 指令集：10 条内核指令、指令注册表与工具面（L3）。

从 core.py 抽出（计划 plan-service-layer-2026-11-17 批次 S1）：服务层
（LLM/PLM）与 Core 共用这里的工具定义，但不反向依赖 Core。
"""

import json
import sys
from typing import Dict, Optional, Type

from .capabilities import MODE_KERNEL
from .events import build_event
from .exceptions import IllegalOperationError, VMMemoryError
from .journal import conversation_create_payload
from .memory_nodes import check_new_key
from .modules import BASE_MODULES, grantable_module_names
from .types import Conversation, MetaDict


def _effect_space(conv: Conversation) -> dict:
    """把对话的地址空间写成日志效果可重放形状（根名 → 真实段）。"""
    return {name: list(segments) for name, segments in conv.caps.space().as_roots().items()}


def _note_memory_write(core: 'Core', conv: Conversation, ref: str, content: str) -> None:
    """内存写（含追加）的日志效果。

    目标是设备（或 legacy LinkDevice 透明挂载）时只记审计条目：设备内部状态不在
    `$MEM` 里，重放不访问设备。LinkDevice 把写委托进内存树的路径因此不会被重放——
    它是"不要用于新设计"的兼容设备，日志不承诺覆盖（见 docs/runtime.md）。
    """
    if not core.is_journaling():
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


_INSTRUCTION_ERRORS = (VMMemoryError, KeyError, IndexError, TypeError, ValueError)


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
    error = _check_declared_shape(instr_cls, args)
    if error is not None:
        return error
    return _check_tool_semantics(instr_cls.tool_name, args)


def _check_declared_shape(instr_cls, args) -> Optional[str]:
    """按 tool_def 的声明校验：必填参数齐全、给的参数值类型对得上。"""
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
    return None


def _check_tool_semantics(tool: str, args) -> Optional[str]:
    """声明管不到的语义约束：引用形态、以及各工具自己的必填组合。"""
    if tool in ("memory_read", "memory_write", "memory_delete"):
        return _check_ref_form(args["ref"])
    if tool == "memory_make":
        return _check_make_key(args["key"])
    if tool == "edit_metadata":
        return _check_edit_metadata_args(args)
    if tool == "listen_event":
        return _check_listen_event_args(args)
    return None


def _check_ref_form(ref) -> Optional[str]:
    """内存引用必须以 $ 开头（根名省略等于让内核猜，直接拒）。"""
    if not ref.startswith("$"):
        return f"ref 必须是 $ 开头的内存引用，得到 {ref!r}"
    return None


def _check_make_key(key) -> Optional[str]:
    """新建键的语法（点号/空键）由 memory 侧统一裁决，这里只翻译成工具响应。"""
    try:
        check_new_key(key)
    except VMMemoryError as e:
        return str(e)
    return None


def _check_edit_metadata_args(args) -> Optional[str]:
    """edit_metadata：type 决定 key/value 的必填组合。"""
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
    return None


def _check_listen_event_args(args) -> Optional[str]:
    """listen_event：action 决定 handler_ref / listener_id 的必填组合。"""
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
            lambda: core.publish_memory_after(
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
            lambda: core.publish_memory_after(
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
        if core.is_journaling():
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
            lambda: core.publish_memory_after(
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
        if core.is_journaling():
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
        if op == "set":
            self._ctrl_set(core, conv, node, key, value)
        elif op == "get":
            self._ctrl_get(core, conv, node, key, value)
        else:
            self._ctrl_del(core, conv, node, key, value)

    def _respond(self, conv, text: str) -> None:
        """把结果作为本次工具调用的响应写回（三条分支共用，省掉 call_id 重复）。"""
        conv.user_batch.add_tool_response(text, self.call_id)

    def _publish_after(self, core: 'Core', conv, op: str, key, value) -> None:
        """写后事件。三种操作只差 payload 里的 type，其余完全相同——
        原先五个分支各抄了一遍，缩进还抄歪了。"""
        _publish_core_after_safely(
            core,
            "edit_metadata",
            conv,
            self.ref,
            {"ref": self.ref, "type": op, "key": key, "value": value, "actor_cid": conv.cid},
            source_kind="memory",
            target_ref=_canonical_ref(core, self.ref, conv.caps.space()),
        )

    def _ctrl_set(self, core: 'Core', conv, node, key, value) -> None:
        """type=set：key/value 都必需（value 可以是空字符串，但不能缺）。"""
        if not key:
            self._respond(conv, "Illegal: edit_metadata type=set 时 key 必须是非空字符串")
            return
        if value is None:
            self._respond(conv, "Illegal: edit_metadata type=set 时 value 必需（可以是空字符串）")
            return
        ctrl = dict(node.get_ctrl() or {})
        ctrl[key] = value
        node.set_ctrl(ctrl)
        if core.is_journaling():
            core.note_effect(
                "edit_metadata", ref=self.ref, type="set", key=key, value=value,
                space=_effect_space(conv),
            )
        self._respond(conv, f"ctrl.{key} = {value!r} 已写入")
        self._publish_after(core, conv, "set", key, value)

    def _ctrl_get(self, core: 'Core', conv, node, key, value) -> None:
        """type=get：不带 key 返回整个 ctrl（JSON）；带 key 返回值。读失败不发写后事件。"""
        ctrl = dict(node.get_ctrl() or {})
        if not key:
            self._respond(conv, json.dumps(ctrl, ensure_ascii=False) if ctrl else "{}")
        elif key in ctrl:
            self._respond(conv, str(ctrl[key]))
        else:
            self._respond(conv, f"Error: ctrl 中没有键 {key}")
            return
        self._publish_after(core, conv, "get", key, value)

    def _ctrl_del(self, core: 'Core', conv, node, key, value) -> None:
        """type=del：key 必需且必须已存在。"""
        if not key:
            self._respond(conv, "Illegal: edit_metadata type=del 时 key 必需")
            return
        ctrl = dict(node.get_ctrl() or {})
        if key not in ctrl:
            self._respond(conv, f"Error: ctrl 中没有键 {key}")
            return
        del ctrl[key]
        node.set_ctrl(ctrl)
        if core.is_journaling():
            core.note_effect(
                "edit_metadata", ref=self.ref, type="del", key=key,
                space=_effect_space(conv),
            )
        self._respond(conv, f"ctrl.{key} 已删除")
        self._publish_after(core, conv, "del", key, value)


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
            child = core.build_conversation(self.system_ref, self.para_ref, parent=conv, is_sub=False,
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
        core.sleep(child.cid)
        if core.is_journaling():
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
            sub = core.build_conversation(self.system_ref, self.para_ref, parent=conv, is_sub=True,
                                      space=conv.caps.space())
        except _INSTRUCTION_ERRORS as e:
            # system_ref 不是程序节点或引用不存在：错误作为工具响应返回，不创建对话，本对话保持活跃可自纠
            _respond_error(conv, self.call_id, e)
            return
        sub.append_user_message(user_content)  # 亚对话的任务在创建时直接给出
        sub.metadata["call_id"] = self.call_id
        # 父进入休眠等待亚对话完成，完成时由 core 唤醒（_on_conversation_finished）
        core.prioritize(sub.cid)  # 亚对话插队：它的任务在创建时就已给定，先跑
        core.sleep(conv.cid)
        if core.is_journaling():
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
        if not self._check_legal(conv, targets):
            return
        # 先解析目标，任一不存在或已关闭即整体失败
        resolved = self._resolve_targets(core, conv, targets)
        if resolved is None:
            return
        delivered_cids, blocked, failed = self._deliver_all(core, conv, resolved)
        self._report(conv, resolved, delivered_cids, blocked, failed)

    def _reject(self, conv: Conversation, msg: str, kind: str = "Error") -> None:
        """整次投递不成立：记终端提示并回一条工具响应。kind=Illegal 表示越权/非法用法。"""
        _report_tool_error(conv, msg)
        conv.user_batch.add_tool_response(f"{kind}: {msg}", self.call_id)

    def _check_legal(self, conv: Conversation, targets: list) -> bool:
        """投递前置校验（空列表 / 亚对话发往主对话 / 自我投递）。False 表示已回复、不再继续。"""
        if not targets:
            self._reject(conv, "send_instruction 失败：cid 列表为空")
            return False
        if conv.is_sub:
            ancestor = conv.parent
            while ancestor is not None:
                if ancestor.cid in targets:
                    self._reject(conv, "send_instruction 失败：亚对话不能向主对话发送消息，"
                                       "只能用 create_sub 的返回通道", "Illegal")
                    return False
                ancestor = ancestor.parent
        for cid in targets:
            if cid == conv.cid:
                self._reject(conv, f"send_instruction 不允许自我投递（cid={cid} 是当前对话）",
                             "Illegal")
                return False
        return True

    def _resolve_targets(self, core: 'Core', conv: Conversation, targets: list) -> Optional[list]:
        """把 cid 列表解析成对话对象。任一不存在或已关闭即整体失败（多播的原子性）。"""
        resolved = []
        for cid in targets:
            target = core.get_conversation(cid)
            if target is None or core.is_hidden_cid(cid):
                self._reject(conv, f"send_instruction 失败：对话 {cid} 不存在")
                return None
            if core.is_finished(cid):
                self._reject(conv, f"send_instruction 失败：对话 {cid} 已关闭")
                return None
            resolved.append(target)
        return resolved

    def _deliver_all(self, core: 'Core', conv: Conversation, resolved: list) -> tuple:
        """逐个投递。单个目标被拦截/处理失败不阻断其他目标，结果分三堆收着。"""
        delivered_cids, blocked, failed = [], [], []
        for target in resolved:
            outcome, text = self._deliver_one(core, conv, target)
            if outcome == "delivered":
                delivered_cids.append(target.cid)
            elif outcome == "blocked":
                blocked.append(text)
            else:
                failed.append(text)
        return delivered_cids, blocked, failed

    def _deliver_one(self, core: 'Core', conv: Conversation, target: Conversation) -> tuple:
        """向单个目标投递一条内容，返回 (delivered / blocked / failed, 说明文本)。"""
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
            _report_tool_error(conv, event["response"])
            return "failed", event["response"]
        if event["action"] == "replace":
            return "blocked", event["response"]
        content = event["payload"]["content"]
        # 简单投递：内容原样放入目标对话的输入批次，不包装 JSON 信封
        target.user_batch.add_user_content(content)
        core.note_effect("delivery", to_cid=target.cid, user_contents=[content])
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
        # 等亚对话返回的对话：消息只入批次，不唤醒；亚对话结束后父对话先跑。
        if not core.is_waiting_for_return(target):
            core.wake(target.cid)  # 排队调度：被唤醒者进队列末尾
        return "delivered", None

    def _report(self, conv: Conversation, resolved: list,
                delivered_cids: list, blocked: list, failed: list) -> None:
        """回报投递结果：部分成功要说明拦截/失败数量，全失败时把第一条原因原样带回去。"""
        delivered = len(delivered_cids)
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
        target = core.get_conversation(self.cid)
        if target is None or core.is_hidden_cid(self.cid):
            conv.user_batch.add_tool_response(f"Error: 对话 {self.cid} 不存在", self.call_id)
            return
        if core.is_finished(self.cid):
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
        dropped = dropped or core.is_ready(self.cid)
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
        prepared = self._prepare_before(core, conv, action)
        if prepared is None:
            return
        b_target, b_payload, b_ref, parsed_condition = prepared
        before = _dispatch_before_event(core, conv, "listen_event", b_target, b_payload, target_ref=b_ref)
        if before["action"] in ("replace", "handler_error"):
            conv.user_batch.add_tool_response(before["response"], self.call_id)
            return
        try:
            response, listener_id = self._apply(core, conv, action, parsed_condition)
            # 三种动作的写后事件只差 payload 里的 action，其余相同
            _publish_core_after_safely(core,
                "listen_event",
                conv,
                f"listener:{listener_id}",
                {"listener_id": listener_id, "action": action, "actor_cid": conv.cid},
                source_kind="core",
                target_ref=("listener", listener_id),
            )
            conv.user_batch.add_tool_response(response, self.call_id)
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)

    def _prepare_before(self, core: 'Core', conv: Conversation, action: str) -> Optional[tuple]:
        """构造 before 事件的 (target, payload, target_ref, 解析后的条件)；已回复错误时返回 None。"""
        listener_id = getattr(self, "listener_id", None)
        if action != "listen":
            return (
                f"listener:{listener_id}",
                {"action": action, "listener_id": listener_id,
                 "handler_ref": getattr(self, "handler_ref", None),
                 "priority": getattr(self, "priority", None)},
                ("listener", listener_id) if isinstance(listener_id, int) else None,
                None,
            )
        handler_ref = getattr(self, "handler_ref", None)
        if handler_ref is None:
            conv.user_batch.add_tool_response(
                "Illegal: action=listen 时必须提供 handler_ref", self.call_id
            )
            return None
        try:
            parsed_condition = core.events.parse_condition(
                conv.cid, handler_ref, space=conv.caps.space(),
                kernel_mode=(conv.caps.mode == "kernel"),
            )
        except _INSTRUCTION_ERRORS as e:
            _respond_error(conv, self.call_id, e)
            return None
        b_target, b_ref = _condition_before_target(parsed_condition)
        return (
            b_target,
            {
                "action": action,
                "event_type": getattr(self, "event_type", None),
                "condition": parsed_condition,
                "handler_ref": handler_ref,
                "priority": getattr(self, "priority", None),
                "tier": getattr(self, "tier", None),
            },
            b_ref,
            parsed_condition,
        )

    def _apply(self, core: 'Core', conv: Conversation, action: str,
               parsed_condition) -> tuple:
        """执行注册/修改/取消，返回 (工具响应文本, listener_id)。"""
        if action == "listen":
            return self._register(core, conv, parsed_condition)
        if action == "modify":
            return self._modify(core, conv)
        return self._cancel(core, conv)

    def _register(self, core: 'Core', conv: Conversation, parsed_condition) -> tuple:
        priority = getattr(self, "priority", 0)
        tier = getattr(self, "tier", None) or "normal"
        listener_id = core.events.register_condition(
            conv.cid, getattr(self, "handler_ref", None), space=conv.caps.space(),
            tier=tier, priority=0 if priority is None else priority,
            kernel_mode=(conv.caps.mode == "kernel"),
            condition=parsed_condition,
        )
        return f"listener_id={listener_id} 已注册", listener_id

    def _modify(self, core: 'Core', conv: Conversation) -> tuple:
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
        return f"listener_id={listener_id} 已更新", listener_id

    def _cancel(self, core: 'Core', conv: Conversation) -> tuple:
        listener_id = getattr(self, "listener_id", None)
        if listener_id is None:
            raise IllegalOperationError("action=cancel 时必须提供 listener_id")
        core.events.cancel(listener_id, conv.cid)
        return f"listener_id={listener_id} 已取消", listener_id


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


def _build_imports(conv: Optional[Conversation] = None) -> list:
    """一次 PLM 执行可 import 的模块名 = 基线 ∪ 对话 caps.imports。

    与 `_build_tools` 同构：PLM 执行器每次调用经 params["imports"] 收到该清单；
    白名单在对话生命周期内稳定（caps 创建后只读，运行期不提供修改入口）。
    """
    if conv is None:
        return sorted(set(BASE_MODULES) | set(grantable_module_names()))
    return sorted(set(BASE_MODULES) | set(conv.caps.imports))


# 公开名（计划 S1）：宿主与控制台用公开名；旧下划线名保留为别名。
instruction_registry = _instruction_registry
build_tools = _build_tools
build_imports = _build_imports

from typing import Optional, List, Dict, Type

from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import sys
import time

from .types import MetaList, MetaDict
from .exceptions import VMSyntaxError, VMMemoryError
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict
from .memory import Memory
from .memory_device import MemoryDevice
from .monitor import Monitor
from .python_server import PLM_REGISTRY, create_plm
from .events import EventSystem, build_event, EVENT_TYPES
from .info_devices import (
    ConversationsDevice, SchedulerDevice,
    ListenersDevice, MemorySummaryDevice, MonitorDevice,
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
        except VMMemoryError as e:
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
        except (VMMemoryError, KeyError, IndexError, TypeError, ValueError) as e:
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

        registry = _instruction_registry()
        for tool_call in message.tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            args = self._parse_tool_args(tool_call, return_calls)
            if args is None:
                continue

            if name in registry:
                return_calls.append({"call_id": call_id, "cmd_type": name, "args": args})
            else:
                return_calls.append({"call_id": call_id, "cmd_type": "unknown_tool", "args": {"name": name, "raw": args}})

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
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.para_ref: str = "$MEM.model_params"
        self.events: EventSystem = EventSystem(self)
        self._turn_events: List[dict] = []
        self.mem.mount("conversations", ConversationsDevice(self))
        self.mem.mount("scheduler", SchedulerDevice(self))
        self.mem.mount("memory", MemorySummaryDevice(self))
        self.mem.mount("monitor", MonitorDevice(self))
        self.mem.mount("listeners", ListenersDevice(self))

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

    def _process_return_calls(self, return_calls: list, conv: Conversation):
        registry = _instruction_registry()
        for rc in return_calls:
            cmd_type = rc.get("cmd_type", "")
            call_id = rc.get("call_id", "")
            if cmd_type == "json_error":
                args = rc.get("args", {})
                _report_tool_error(
                    conv,
                    f"工具调用 {args.get('name', '')} 参数不是合法 JSON: {args.get('error', '')}，原始参数: {args.get('raw', '')}",
                )
                continue
            if cmd_type == "unknown_tool":
                args = rc.get("args", {})
                _report_tool_error(
                    conv,
                    f"未知工具 {args.get('name', '')}，参数: {args.get('raw', '')}",
                )
                continue
            if cmd_type not in registry:
                continue
            instr_cls = registry[cmd_type]
            instr = instr_cls(call_id, conv.cid, rc.get("args", {}))
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
        para = self._get_para(conv)
        executor = self._get_executor(conv, para)
        try:
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
        except Exception as e:
            last_call = getattr(executor, "last_call", None)
            if last_call is not None and self._turn_events:
                last_call = {**last_call, "events": list(self._turn_events)}
            self.monitor.record(self, error=e, last_call=last_call)
            raise
        else:
            last_call = getattr(executor, "last_call", None)
            if last_call is not None and self._turn_events:
                last_call = {**last_call, "events": list(self._turn_events)}
            self.monitor.record(self, last_call=last_call)

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

    def run(self):
        self._pick_next_active()

        self.monitor.record_baseline(self)

        while self._active_cid is not None:
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
        value = [value[0], *value[1:].split(".")]
        if value[1] == "conv":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])]
        elif value[1] == "batch":
            assert value[0] == "$"
            return self._conv_by_cid[int(value[2])].user_batch
        else:
            return self.mem.unwrap(value, for_llm=for_llm)

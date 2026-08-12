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
from .info_devices import (
    ConversationsDevice, SchedulerDevice, IccDevice,
    MemorySummaryDevice, MonitorDevice,
)


def _report_tool_error(conv: Conversation, message: str):
    """工具调用错误面向运行者（stderr）输出，不喂回给 LLM。"""
    print(f"[AVM] 对话 {conv.cid}: {message}", file=sys.stderr)


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
            ctrl[key] = value
            node.set_ctrl(ctrl)
            conv.user_batch.add_tool_response(f"ctrl.{key} = {value!r} 已写入", self.call_id)
        elif op == "get":
            if not key:
                conv.user_batch.add_tool_response(
                    json.dumps(ctrl, ensure_ascii=False) if ctrl else "{}", self.call_id
                )
            elif key in ctrl:
                conv.user_batch.add_tool_response(str(ctrl[key]), self.call_id)
            else:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
        else:  # del
            if not key:
                conv.user_batch.add_tool_response(
                    "Error: edit_metadata type=del 时 key 必需", self.call_id
                )
                return
            if key not in ctrl:
                conv.user_batch.add_tool_response(f"Error: ctrl 中没有键 {key}", self.call_id)
                return
            del ctrl[key]
            node.set_ctrl(ctrl)
            conv.user_batch.add_tool_response(f"ctrl.{key} 已删除", self.call_id)


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
            "description": "调用一个已注册服务，等待返回后继续。return_mode=tool 时返回为工具响应（默认）；return_mode=message 时返回为消息",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "要调用的服务名"},
                    "input": {"type": "string", "description": "传给服务的输入"},
                    "return_mode": {"type": "string", "enum": ["tool", "message"], "description": "返回方式，默认 tool"},
                },
                "required": ["service_name", "input"],
            }
        }
    }

    def execute(self, core: 'Core', conv: Conversation):
        return_mode = getattr(self, "return_mode", "tool")
        if return_mode not in ("tool", "message"):
            msg = f"call_service 失败：未知返回方式 {return_mode}"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        target_cid = core._services.get(self.service_name)
        if target_cid is None:
            conv.user_batch.add_tool_response(f"Error: 服务 {self.service_name} 不存在", self.call_id)
            return
        target = core._conv_by_cid[target_cid]
        # ICC 记录：icc_id = 本次工具调用的 call_id
        core._icc[self.call_id] = {
            "caller_cid": conv.cid,
            "caller_call_id": self.call_id,
            "mode": return_mode,
            "callee_cid": target_cid,
        }
        # 投递消息：JSON 字符串（默认格式），含 from/to/icc_id/content
        message = json.dumps({
            "from": conv.identity,
            "to": target.identity,
            "icc_id": self.call_id,
            "content": self.input,
        }, ensure_ascii=False)
        target.user_batch.add_user_content(message)
        if target.cid in core._dormant_cids:
            core._dormant_cids.remove(target.cid)
        if target.cid in core._ready_cids:
            core._ready_cids.remove(target.cid)
        core._ready_cids.append(target.cid)  # 排队调度：被唤醒者进队列末尾
        if return_mode == "message":
            # 一次工具调用只能有一个工具响应：message 模式用确认响应满足配对
            conv.user_batch.add_tool_response(f"已投递到 {target.identity}", self.call_id)
        # 调用者休眠等待服务返回（成功投递才挂起）
        if conv.cid not in core._dormant_cids:
            core._dormant_cids.append(conv.cid)


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


class ReturnResultInstruction(Instruction):
    tool_name = "return_result"
    tool_def = {
        "type": "function",
        "function": {
            "name": "return_result",
            "description": "将本对话的结论按 ICC id 返回给发起请求的对话，并确保它成为下一个被激活的对话。调用后本对话应在下一轮以无工具的结果收尾，进入休眠",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "返回给发起请求的对话的结论"},
                    "icc_id": {"type": "string", "description": "本次请求的 ICC id（从收到的指令消息中读取）"},
                },
                "required": ["content", "icc_id"],
            }
        }
    }
    content: str
    icc_id: str

    def execute(self, core: 'Core', conv: Conversation):
        record = core._icc.get(self.icc_id)
        if record is None:
            msg = f"return_result 失败：ICC id {self.icc_id} 无对应请求记录"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        caller_cid = record["caller_cid"]
        call_id = record["caller_call_id"]
        mode = record.get("mode", "tool")
        caller = core._conv_by_cid.get(caller_cid)
        if caller is None or caller_cid in core._finished:
            # 发起者不存在或已关闭：记录作废，不投递
            core._drop_icc(self.icc_id, record)
            msg = f"return_result 失败：发起者 {caller_cid} 不存在或已关闭"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return

        if mode == "message":
            # 返回以消息形式投递（确认响应已单独给出）
            message = json.dumps({
                "from": conv.identity,
                "to": caller.identity,
                "icc_id": self.icc_id,
                "content": self.content,
            }, ensure_ascii=False)
            caller.user_batch.add_user_content(message)
            wake = True
        else:
            # 结论投递给发起者，挂到原调用的 call_id；多播返回以结构化段合并（JSON 数组）
            group_key = record.get("group_key")
            if group_key is not None:
                caller.user_batch.add_tool_response(
                    self.content, call_id,
                    from_identity=conv.identity, cid=conv.cid, icc_id=self.icc_id,
                )
                core._icc_groups[group_key] -= 1
                wake = core._icc_groups[group_key] <= 0
                if wake:
                    del core._icc_groups[group_key]
            else:
                caller.user_batch.add_tool_response(self.content, call_id)
                wake = True
        if wake:
            # 排队调度：发起者进队列末尾（create_sub 是唯一前插例外）
            if caller_cid in core._dormant_cids:
                core._dormant_cids.remove(caller_cid)
            if caller_cid in core._ready_cids:
                core._ready_cids.remove(caller_cid)
            core._ready_cids.append(caller_cid)

        core._icc.pop(self.icc_id, None)  # 成功路由后移除记录
        # 该工具调用必须有一次返回（LLM API 规定）：告知本对话结果已发送
        conv.user_batch.add_tool_response("结果已发送", self.call_id)
        conv.metadata["returned"] = "1"


class SendInstruction(Instruction):
    tool_name = "send_instruction"
    tool_def = {
        "type": "function",
        "function": {
            "name": "send_instruction",
            "description": "向指定 cid 的对话投递指令（消息为 JSON 字符串，含 from/to/icc_id/content）。cid 可为整数或整数列表（多播）。return_mode=tool 时返回为工具响应（默认，多播时按 cid+名字合并成一条）；return_mode=message 时返回为消息。wait=true 时本对话休眠等待返回，wait=false 时本对话继续",
            "parameters": {
                "type": "object",
                "properties": {
                    "cid": {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}], "description": "目标对话的 cid 或 cid 列表"},
                    "content": {"type": "string", "description": "指令内容"},
                    "wait": {"type": "boolean", "description": "是否休眠等待返回"},
                    "return_mode": {"type": "string", "enum": ["tool", "message"], "description": "返回方式，默认 tool"},
                    "format": {"type": "string", "description": "投递消息格式，默认 json（预留未来格式）"},
                },
                "required": ["cid", "content", "wait"],
            }
        }
    }
    cid: int
    content: str
    wait: bool

    def execute(self, core: 'Core', conv: Conversation):
        fmt = getattr(self, "format", "json")
        if fmt != "json":
            msg = f"send_instruction 失败：不支持的格式 {fmt}（当前仅支持 json）"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        return_mode = getattr(self, "return_mode", "tool")
        if return_mode not in ("tool", "message"):
            msg = f"send_instruction 失败：未知返回方式 {return_mode}"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        targets = self.cid if isinstance(self.cid, list) else [self.cid]
        if not targets:
            msg = "send_instruction 失败：cid 列表为空"
            _report_tool_error(conv, msg)
            conv.user_batch.add_tool_response(f"Error: {msg}", self.call_id)
            return
        multicast = len(targets) > 1

        # 先解析目标，任一不存在即整体失败
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
        to_value = [t.identity for t in resolved] if multicast else resolved[0].identity

        for i, target in enumerate(resolved):
            icc_id = self.call_id if not multicast else f"{self.call_id}#{i}"
            core._icc[icc_id] = {
                "caller_cid": conv.cid,
                "caller_call_id": self.call_id,
                "mode": return_mode,
                "group_key": self.call_id if multicast else None,
                "callee_cid": target.cid,
            }
            message = json.dumps({
                "from": conv.identity,
                "to": to_value,
                "icc_id": icc_id,
                "content": self.content,
            }, ensure_ascii=False)
            target.user_batch.add_user_content(message)
            if target.cid in core._dormant_cids:
                core._dormant_cids.remove(target.cid)
            if target.cid in core._ready_cids:
                core._ready_cids.remove(target.cid)
            core._ready_cids.append(target.cid)  # 排队调度：被唤醒者进队列末尾

        if multicast and return_mode == "tool":
            # 工具返回多播：等所有目标返回后在 batch 中合并成一条工具响应，期间发起者休眠
            core._icc_groups[self.call_id] = len(targets)
            if conv.cid not in core._dormant_cids:
                core._dormant_cids.append(conv.cid)
        elif return_mode == "message":
            # 一次工具调用只能有一个工具响应：message 模式用确认响应满足配对，各目标的返回走消息
            label = len(targets) if multicast else resolved[0].identity
            conv.user_batch.add_tool_response(f"已投递到 {label}", self.call_id)
            if self.wait and conv.cid not in core._dormant_cids:
                core._dormant_cids.append(conv.cid)
        elif self.wait:
            # 单目标 wait=true：发起者休眠等待返回（成功投递才挂起）
            if conv.cid not in core._dormant_cids:
                core._dormant_cids.append(conv.cid)


class CloseInstruction(Instruction):
    tool_name = "close_conversation"
    tool_def = {
        "type": "function",
        "function": {
            "name": "close_conversation",
            "description": "内核级关闭指令：关闭指定 cid 的对话（可关闭自身或其他对话），不需要被关闭对话的回应。关闭后该对话不可再调度、不可再被调用，记录为 finished；其注册的服务注销、PLM 状态释放、未完成调用清理（发给它的调用会以 Error 通知发起者并唤醒）",
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
        conv.user_batch.add_tool_response(f"对话 {self.cid} 已关闭（finished）", self.call_id)


def _instruction_registry() -> Dict[str, Type[Instruction]]:
    return {
        cls.tool_name: cls
        for cls in [
            MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction, EditMetadataInstruction,
            CreateInstruction, CreateSubInstruction,
            RegisterServiceInstruction, CallServiceInstruction, TransferServiceInstruction,
            ReturnResultInstruction, SendInstruction, CloseInstruction,
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
        self._services: Dict[str, int] = {}  # name → cid
        self._ready_cids: List[int] = []
        self._dormant_cids: List[int] = []
        self._active_cid: Optional[int] = None
        self._next_cid: int = 0
        self.mem: Memory = Memory()
        self.lmu: LMU = LMU()
        self._icc: dict = {}  # icc_id → (发起者 cid, 发起者调用的 call_id)
        self._icc_groups: dict = {}  # 多播工具返回：call_id → 尚未返回的目标数
        self._executors: Dict[int, PLMExecutor] = {}  # cid → PLM 执行器（Python 对话；PLM 状态跨轮次保留）
        self._finished: set = set()  # 已关闭（个体结束）的对话 cid；记录供监测/信息设备展示
        self._instruction_budget: int = 50  # 每次激活的指令预算（核心中断阈值）
        self._activation_instructions: int = 0
        self.image_dir: Optional[str] = None  # 调试用：镜像目录（检查点默认路径）
        self.image_name: Optional[str] = None  # 调试用：镜像名（不含扩展名，检查点默认路径）
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.para_ref: str = "$MEM.model_params"
        self.mem.mount("services", _ServicesDevice(self))
        self.mem.mount("conversations", ConversationsDevice(self))
        self.mem.mount("scheduler", SchedulerDevice(self))
        self.mem.mount("icc", IccDevice(self))
        self.mem.mount("memory", MemorySummaryDevice(self))
        self.mem.mount("monitor", MonitorDevice(self))

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
                # 亚对话紧耦合：完成即唤醒父；已用 return_result 显式返回过则不再自动写回
                if not conv.metadata.get("returned"):
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

        效果：标记 finished、移出调度队列、释放 PLM 执行器（变量空间）、
        注销其注册的服务、清理未完成调用（其发起的调用作废；发给它的调用
        以 Error 通知发起者并按组/单目标唤醒）。
        """
        if cid in self._finished:
            return
        self._finished.add(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        self._executors.pop(cid, None)  # 释放 PLM 状态（命名空间等）
        for name in [n for n, svc_cid in self._services.items() if svc_cid == cid]:
            del self._services[name]
        closed = self._conv_by_cid.get(cid)
        closed_identity = closed.identity if closed is not None else str(cid)
        for icc_id, rec in list(self._icc.items()):
            if rec.get("caller_cid") == cid:
                # 已关闭对话发起的调用：无接收者，记录作废（多播组计数同步递减）
                self._drop_icc(icc_id, rec)
            elif rec.get("callee_cid") == cid:
                # 发给已关闭对话的调用：通知发起者（工具响应或消息）并唤醒
                caller_cid = rec["caller_cid"]
                caller = self._conv_by_cid.get(caller_cid)
                group_key = rec.get("group_key")
                mode = rec.get("mode", "tool")
                call_id = rec["caller_call_id"]
                self._drop_icc(icc_id, rec)
                if caller is None or caller_cid in self._finished:
                    continue
                if mode == "message":
                    message = json.dumps({
                        "from": closed_identity,
                        "to": caller.identity,
                        "icc_id": icc_id,
                        "content": f"Error: 对方已关闭（{cid}），请求未完成",
                    }, ensure_ascii=False)
                    caller.user_batch.add_user_content(message)
                    wake = True
                elif group_key is not None:
                    # 多播工具返回：作为一段错误并入组，组归零才唤醒发起者
                    caller.user_batch.add_tool_response(
                        f"Error: 对方已关闭（{cid}），请求未完成",
                        call_id,
                        from_identity=closed_identity, cid=cid, icc_id=icc_id,
                    )
                    wake = self._icc_groups.get(group_key, 0) <= 0
                else:
                    caller.user_batch.add_tool_response(
                        f"Error: 对方已关闭（{cid}），请求未完成", call_id
                    )
                    wake = True
                if wake:
                    if caller_cid in self._dormant_cids:
                        self._dormant_cids.remove(caller_cid)
                    if caller_cid in self._ready_cids:
                        self._ready_cids.remove(caller_cid)
                    self._ready_cids.append(caller_cid)

    def _drop_icc(self, icc_id: str, rec: dict) -> None:
        """移除一条 ICC 记录；多播组计数同步递减，组归零时清理组记录。"""
        group_key = rec.get("group_key")
        if group_key is not None and group_key in self._icc_groups:
            self._icc_groups[group_key] -= 1
            if self._icc_groups[group_key] <= 0:
                del self._icc_groups[group_key]
        self._icc.pop(icc_id, None)

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
            self.monitor.record(self, error=e, last_call=getattr(executor, "last_call", None))
            raise
        else:
            self.monitor.record(self, last_call=getattr(executor, "last_call", None))

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
            if not isinstance(para, MetaDict):
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

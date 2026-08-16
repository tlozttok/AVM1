"""AVM 事件模型 v1：事件构造、监听器注册表、权限校验、处理器执行与管道分发。

每个工具调用对应一个事件；5 种 before 事件可被监听器拦截/改写，
4 种 after 事件只产生通知。监听器是普通 Python 节点中的代码，
每次事件在全新受限命名空间执行，不保留实例状态。
"""
import copy
import json
import sys
from typing import Any, Dict, List, Optional

from .exceptions import VMMemoryError
from .sandbox import safe_globals, exec_safe
from .types import MetaDict, MetaList

EVENT_TYPES = [
    "memory_read",
    "memory_write",
    "memory_make",
    "edit_metadata",
    "send_instruction",
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
]

INTERCEPTABLE_TYPES = {
    "memory_read",
    "memory_write",
    "memory_make",
    "edit_metadata",
    "send_instruction",
}

AFTER_ONLY_TYPES = {
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
}

# 各事件类型中处理器可以修改的 payload 字段。
# 未列出的字段（身份字段等）一律不可由处理器修改。
MUTABLE_PAYLOAD_KEYS = {
    "memory_write": {"content"},
    "edit_metadata": {"value"},
    "send_instruction": {"content"},
}

DEFAULT_REPLACE_RESPONSE = "事件被监听器拦截"


class EventHandlerError(Exception):
    """before 阶段处理器失败：fail closed，原指令不执行。"""

    def __init__(self, listener_id: int, message: str):
        super().__init__(message)
        self.listener_id = listener_id
        self.message = message

    def __str__(self) -> str:
        return f"事件监听器 {self.listener_id} 处理失败: {self.message}"


def build_event(
    core: 'Core',
    event_type: str,
    phase: str,
    conv: Optional[Any] = None,
    target: str = "",
    payload: Optional[dict] = None,
    source: Optional[dict] = None,
) -> dict:
    """构造事件 dict 并分配 event_id（事件瞬时，不落盘）。"""
    return core.events.build_event(
        event_type=event_type,
        phase=phase,
        conv=conv,
        target=target,
        payload=payload,
        source=source,
    )


class EventSystem:
    def __init__(self, core: 'Core'):
        self._core = core
        # 注册顺序即优先级：先注册者先执行。
        self.listeners: List[dict] = []
        self._next_listener_id = 1
        self._next_event_id = 1

    # ------------------------------------------------------------------
    # 事件构造
    # ------------------------------------------------------------------

    def build_event(
        self,
        event_type: str,
        phase: str,
        conv: Optional[Any] = None,
        target: str = "",
        payload: Optional[dict] = None,
        source: Optional[dict] = None,
    ) -> dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型：{event_type}")
        if phase not in ("before", "after"):
            raise ValueError(f"未知 phase：{phase}")
        if source is None:
            if phase == "before" and conv is not None:
                source = {"kind": "core", "initiator_cid": conv.cid}
            else:
                raise ValueError("after 事件必须显式提供执行组件 source")
        event = {
            "event_id": f"evt-{self._next_event_id}",
            "event_type": event_type,
            "phase": phase,
            "source": source,
            "target": target,
            "payload": dict(payload or {}),
            "action": "execute",
            "response": "",
            "notify": [],
        }
        self._next_event_id += 1
        return event

    def publish_after_event(
        self,
        event_type: str,
        conv: Any,
        target: str,
        payload: dict,
        source_kind: Optional[str] = None,
        device_path: Optional[str] = None,
    ) -> dict:
        """统一构造并分发 after 事实事件（Core/内存/设备发布）。"""
        if source_kind == "memory":
            source = {"kind": "memory"}
        elif source_kind == "core":
            source = {"kind": "core"}
        elif device_path is not None:
            source = {"kind": "device", "path": "$MEM." + device_path}
        else:
            raise ValueError("publish_after_event 必须提供 source_kind=memory/core 或 device_path")
        event = self.build_event(
            event_type=event_type,
            phase="after",
            conv=conv,
            target=target,
            payload=payload,
            source=source,
        )
        event = self.dispatch(event)
        self.deliver_notifies(event)
        return event

    # ------------------------------------------------------------------
    # 注册 / 修改 / 取消
    # ------------------------------------------------------------------

    def register(self, owner_cid: int, event_type: str, target: str, handler_ref: str, phase: str = "all") -> int:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型：{event_type}，允许：{EVENT_TYPES}")
        if phase not in ("before", "after", "all"):
            raise ValueError("phase 必须是 before/after/all")
        if not isinstance(target, str) or not target:
            raise ValueError("target 必须是字符串")
        handler_code = self.resolve_handler(self._core, handler_ref)

        if event_type in AFTER_ONLY_TYPES:
            if phase == "before":
                raise ValueError(f"{event_type} 只有 after 事件，不能注册 phase=before")
            # 生命周期 after 允许 * 或精确 target，任何对话可注册
        else:
            if target == "*":
                raise ValueError("该事件类型只支持精确 target，不支持 *")
            self._check_target_permission(owner_cid, event_type, target, phase)

        listener_id = self._next_listener_id
        self._next_listener_id += 1
        self.listeners.append({
            "listener_id": listener_id,
            "owner_cid": owner_cid,
            "event_type": event_type,
            "target": target,
            "handler_ref": handler_ref,
            "handler_code": handler_code,
            "phase": phase,
        })
        return listener_id

    def modify(self, listener_id: int, owner_cid: int, handler_ref: str) -> None:
        listener = self._get_listener(listener_id)
        self._check_owner(listener, owner_cid)
        handler_code = self.resolve_handler(self._core, handler_ref)
        listener["handler_ref"] = handler_ref
        listener["handler_code"] = handler_code

    def cancel(self, listener_id: int, owner_cid: int) -> None:
        listener = self._get_listener(listener_id)
        self._check_owner(listener, owner_cid)
        self.listeners.remove(listener)

    def cancel_all_for_owner(self, owner_cid: int) -> None:
        self.listeners = [l for l in self.listeners if l.get("owner_cid") != owner_cid]

    # ------------------------------------------------------------------
    # 权限
    # ------------------------------------------------------------------

    def _get_listener(self, listener_id: int) -> dict:
        for l in self.listeners:
            if l["listener_id"] == listener_id:
                return l
        raise ValueError(f"监听器 {listener_id} 不存在")

    @staticmethod
    def _check_owner(listener: dict, owner_cid: int) -> None:
        if listener.get("owner_cid") != owner_cid:
            raise ValueError(
                f"无权修改/取消监听器 {listener['listener_id']}：只有注册者 cid={listener['owner_cid']} 可操作"
            )

    def _check_target_permission(self, owner_cid: int, event_type: str, target: str, phase: str) -> None:
        """按目标路径 kind 检查 memory/device/send 类事件的注册权限。

        memory：按 ctrl.owner；device：询问 authorize_listener；send：只允许 cid:<自己>。
        """
        if event_type in ("memory_read", "memory_write", "edit_metadata"):
            split = self._core.mem.split_path(target)
            kind = split[0]
            if kind == "memory":
                mount_key = None
                remainder = []
            else:
                _, mount_key, remainder = split
            if kind == "memory":
                owner = self._owner_of(self._core, target)
                if owner is None:
                    raise ValueError(f"不能监听无主路径：{target}")
                if owner != owner_cid:
                    raise ValueError(
                        f"无权监听 {target}：路径所有者 cid={owner}，当前 cid={owner_cid}"
                    )
            else:
                device = self._core.mem._devices[mount_key]
                if not device.authorize_listener(event_type, remainder, owner_cid):
                    raise ValueError(
                        f"设备 {mount_key} 未授权对话 {owner_cid} 监听 {target}"
                    )
        elif event_type == "memory_make":
            split = self._core.mem.split_path(target)
            kind = split[0]
            if kind == "memory":
                mount_key = None
                remainder = []
            else:
                _, mount_key, remainder = split
            if kind == "device":
                device = self._core.mem._devices[mount_key]
                if not device.authorize_listener(event_type, remainder, owner_cid):
                    raise ValueError(
                        f"设备 {mount_key} 未授权对话 {owner_cid} 监听 {target}"
                    )
            else:
                parent = self._parent_ref(target)
                owner = self._owner_of(self._core, parent)
                if owner is None:
                    raise ValueError(f"不能监听无主父路径：{parent}")
                if owner != owner_cid:
                    raise ValueError(
                        f"无权监听 {target}：父路径所有者 cid={owner}，当前 cid={owner_cid}"
                    )
        elif event_type == "send_instruction":
            if not target.startswith("cid:"):
                raise ValueError("send_instruction 事件的 target 必须是 cid:<cid>")
            try:
                cid = int(target[4:])
            except ValueError:
                raise ValueError(f"send_instruction 事件的 target cid 必须是数字：{target}")
            if cid != owner_cid:
                raise ValueError(f"对话只能监听发往自己的消息，不能监听 {target}")
        else:
            raise ValueError(f"不支持的事件注册类型：{event_type}")

    @staticmethod
    def _parent_ref(target: str) -> str:
        if target == "$MEM":
            raise ValueError("memory_make 的 target 必须是新建完整路径，不能是根")
        if not target.startswith("$MEM."):
            raise ValueError(f"memory_make 的 target 必须是 $MEM 路径：{target}")
        return target.rsplit(".", 1)[0]

    # ------------------------------------------------------------------
    # 所有者解析
    # ------------------------------------------------------------------

    @staticmethod
    def _has_device_prefix(core: 'Core', path: str) -> bool:
        """路径的任意前缀是否命中已挂载设备。

        权限检查不能触发设备 resolve_path（如 $MEM.inputs.-1 会阻塞等待输入），
        因此先做纯注册表检查；含设备前缀的路径走「最近非设备祖先」所有权规则。
        """
        if not isinstance(path, str) or not path.startswith("$MEM."):
            return False
        parts = path[1:].split(".")
        if not parts or parts[0] != "MEM":
            return False
        rest = parts[1:]
        for i in range(1, len(rest) + 1):
            if ".".join(rest[:i]) in core.mem._devices:
                return True
        return False

    @classmethod
    def _owner_of(cls, core: 'Core', path: str) -> Optional[int]:
        """按 2.6 查找 memory 路径所有者。

        只接受 memory kind；device kind 直接返回 None（设备授权走 authorize_listener）。
        """
        if not isinstance(path, str):
            return None
        try:
            kind, segments = core.mem.split_path(path)[:2]
        except Exception:
            return None
        if kind != "memory":
            return None

        # 目标路径必须真实存在；不存在时不做祖先兜底。
        ref = "$MEM" if not segments else "$MEM." + ".".join(segments)
        try:
            core.unwrap(ref, for_llm=False)
        except Exception:
            return None

        # 从完整路径向上逐级找第一个带 ctrl.owner 的 MetaDict/MetaList。
        for i in range(len(segments), -1, -1):
            prefix = segments[:i]
            ref = "$MEM" if not prefix else "$MEM." + ".".join(prefix)
            try:
                node = core.unwrap(ref, for_llm=False)
            except Exception:
                continue
            if isinstance(node, (MetaDict, MetaList)):
                owner = (node.get_ctrl() or {}).get("owner")
                if owner is not None:
                    try:
                        return int(owner)
                    except (TypeError, ValueError):
                        continue
        return None

    # ------------------------------------------------------------------
    # 处理器解析与执行
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_handler(core: 'Core', handler_ref: str) -> str:
        if not isinstance(handler_ref, str) or not handler_ref:
            raise ValueError("handler_ref 必须是字符串引用")
        try:
            node = core.unwrap(handler_ref, for_llm=False)
        except (VMMemoryError, KeyError, IndexError, TypeError, ValueError) as e:
            raise ValueError(f"handler_ref 不存在：{handler_ref}（{e}）")
        if not isinstance(node, MetaDict):
            raise ValueError("handler_ref 必须指向 ctrl.type='python' 的 dict 节点")
        if (node.get_ctrl() or {}).get("type") != "python":
            raise ValueError("handler_ref 必须指向 ctrl.type='python' 的 dict 节点")
        content = node.get("content")
        if not isinstance(content, str):
            raise ValueError("handler_ref 节点的 content 必须是字符串")
        return content

    def _run_listener(self, listener: dict, event: dict) -> dict:
        """在全新受限命名空间执行处理器，返回规范化输出。

        处理器错误统一抛 EventHandlerError（before fail closed / after fail open 由 dispatch 决定）。
        """
        namespace = safe_globals({"json": json})
        try:
            exec_safe(listener["handler_code"], namespace)
        except Exception as e:
            raise EventHandlerError(
                listener["listener_id"],
                f"{type(e).__name__}: {e}",
            )
        handle = namespace.get("handle")
        if not callable(handle):
            raise EventHandlerError(
                listener["listener_id"],
                f"处理器未定义可调用的 handle，得到 {type(handle).__name__}",
            )
        try:
            out = handle(copy.deepcopy(event))
        except Exception as e:
            raise EventHandlerError(
                listener["listener_id"],
                f"{type(e).__name__}: {e}",
            )
        if not isinstance(out, dict):
            raise EventHandlerError(
                listener["listener_id"],
                f"处理器返回值必须是 dict，得到 {type(out).__name__}",
            )
        return out

    @staticmethod
    def _merge_output(event: dict, out: dict, listener_id: Optional[int] = None) -> None:
        """把处理器允许修改的字段合并进事件。

        身份字段不可变；未知键忽略；校验失败抛 EventHandlerError。
        after 阶段遇到 notify 自我投递时丢弃该条并记入 handler_errors；
        before 阶段遇到 notify 自我投递视为处理器错误。
        """
        if "action" in out:
            action = out["action"]
            if event["phase"] == "before":
                if action not in ("execute", "replace"):
                    raise EventHandlerError(
                        0,  # 调用方随后用真实 listener_id 包装
                        f"before 阶段 action 必须是 execute/replace，得到 {action!r}",
                    )
                event["action"] = action
            # after 阶段忽略 action

        if event.get("action") == "replace":
            response = out.get("response", "")
            if response is None:
                response = ""
            if not isinstance(response, str):
                raise EventHandlerError(
                    0,
                    f"response 必须是字符串，得到 {type(response).__name__}",
                )
            event["response"] = response or DEFAULT_REPLACE_RESPONSE

        mutable = MUTABLE_PAYLOAD_KEYS.get(event["event_type"], set())
        payload_out = out.get("payload")
        if isinstance(payload_out, dict):
            for key in mutable:
                if key not in payload_out:
                    continue
                # edit_metadata 的 value 只有原操作是 set 时才可变
                if event["event_type"] == "edit_metadata" and event["payload"].get("type") != "set":
                    continue
                val = payload_out[key]
                if event["event_type"] in ("memory_write", "send_instruction") and not isinstance(val, str):
                    raise EventHandlerError(0, f"payload.{key} 必须是字符串，得到 {type(val).__name__}")
                if event["event_type"] == "edit_metadata" and not isinstance(val, str):
                    raise EventHandlerError(0, f"payload.value 必须是字符串，得到 {type(val).__name__}")
                event["payload"][key] = val

        if "notify" in out:
            notifies = out["notify"]
            if not isinstance(notifies, list):
                raise EventHandlerError(0, "notify 必须是 list")
            source_cid = None
            if event["phase"] == "before":
                source = event.get("source") or {}
                if source.get("kind") == "core":
                    source_cid = source.get("initiator_cid")
            else:
                source_cid = (event.get("payload") or {}).get("actor_cid")
            for item in notifies:
                if not isinstance(item, dict) or not isinstance(item.get("cid"), int) or not isinstance(item.get("content"), str):
                    raise EventHandlerError(
                        0,
                        f"notify 每项必须是 {{\"cid\": int, \"content\": str}}，得到 {item!r}",
                    )
                if source_cid is not None and item["cid"] == source_cid:
                    error = f"notify 不允许自我投递（cid={source_cid} 是当前对话）"
                    if event["phase"] == "before":
                        raise EventHandlerError(0, error)
                    event.setdefault("handler_errors", []).append({
                        "listener_id": listener_id,
                        "error": error,
                    })
                    print(f"[AVM] 事件监听器 {listener_id} 处理失败: {error}", file=sys.stderr)
                    continue
                event["notify"].append({"cid": item["cid"], "content": item["content"]})

    # ------------------------------------------------------------------
    # 分发
    # ------------------------------------------------------------------

    def _match(self, event: dict) -> List[dict]:
        matched = []
        for listener in self.listeners:
            if listener["event_type"] != event["event_type"]:
                continue
            listener_phase = listener.get("phase", "all")
            if listener_phase not in ("all", event["phase"]):
                continue
            if listener["target"] == "*" or listener["target"] == event["target"]:
                matched.append(listener)
        return matched

    def dispatch(self, event: dict) -> dict:
        matched = self._match(event)
        if event["phase"] == "before":
            for listener in matched:
                try:
                    out = self._run_listener(listener, event)
                    # _merge_output 中 action 为 replace 时也会校验 response/notify
                    self._merge_output(event, out, listener["listener_id"])
                except EventHandlerError as e:
                    # fail closed 改为事件结果：不抛异常、不在这里打印 stderr；
                    # 指令执行方负责向 stderr 报告一次并返回 Error 工具响应。
                    if e.listener_id == 0:
                        e = EventHandlerError(listener["listener_id"], e.message)
                    event["action"] = "handler_error"
                    event["response"] = f"Error: {e}"
                    event["error"] = str(e)
                    break
                if event["action"] == "replace":
                    break
        else:  # after
            for listener in matched:
                try:
                    out = self._run_listener(listener, event)
                    self._merge_output(event, out, listener["listener_id"])
                except EventHandlerError as e:
                    # fail open：记录 stderr，跳过该处理器的 notify，继续后续监听器
                    if e.listener_id == 0:
                        e = EventHandlerError(listener["listener_id"], e.message)
                    event.setdefault("handler_errors", []).append({
                        "listener_id": listener["listener_id"],
                        "error": str(e),
                    })
                    print(f"[AVM] {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    event.setdefault("handler_errors", []).append({
                        "listener_id": listener["listener_id"],
                        "error": err,
                    })
                    print(f"[AVM] 事件监听器 {listener['listener_id']} 处理失败: {err}", file=sys.stderr)
                    continue

        self._append_summary(event)
        return event

    def _append_summary(self, event: dict) -> None:
        summary = {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "phase": event.get("phase"),
            "target": event.get("target"),
            "source": event.get("source"),
            "action": event.get("action"),
        }
        if event.get("error") is not None:
            summary["error"] = event["error"]
        if event.get("handler_errors"):
            summary["handler_errors"] = event["handler_errors"]
        self._core._turn_events.append(summary)

    def deliver_notifies(self, event: dict) -> None:
        """按 2.7 内部投递 notify：不经事件管道、不产生 send_instruction 事件。"""
        core = self._core
        for item in event.get("notify") or []:
            cid = item["cid"]
            content = item["content"]
            target = core._conv_by_cid.get(cid)
            if target is None or cid in core._finished:
                print(
                    f"[AVM] 事件通知投递失败：对话 {cid} 不存在或已关闭",
                    file=sys.stderr,
                )
                continue
            target.user_batch.add_user_content(content)
            if cid in core._dormant_cids:
                core._dormant_cids.remove(cid)
            if cid in core._ready_cids:
                core._ready_cids.remove(cid)
            core._ready_cids.append(cid)

"""AVM 事件模型 v1：事件构造、监听器注册表、权限校验、处理器执行与管道分发。

批次二（权限体系）：10 条指令都产生 before 请求事件与 after 事实事件；
事件载荷分 normal / kernel 两级，监听器有 priority 与决策动作 allow/replace/execute；
匹配与所有者判定用 canonical 目标（target_ref），处理器看到的目标仍是请求别名。
监听器是 ctrl.type="handler" 节点中的代码，每次事件在全新受限命名空间执行，
不保留实例状态。
"""

# TODO(issue-2026-11-01): 用户态事件子集（named presets）的名单与粒度是临时方案，待作者
# 后续细化：preset 是否读写分离、是否允许用户自定义条件子集、own_nodes 依赖 owner 是否
# 合适、是否把 send_instruction/listen_event 纳入子集。设计见
# docs/private/Agents/implemented/design-unified-event-bus-2026-10-31.md §2.3、§8。
import copy
import json
import sys
from typing import Any, Dict, List, Optional

from .capabilities import AddressSpace
from .exceptions import IllegalOperationError, VMMemoryError
from .sandbox import safe_globals, exec_safe
from .types import MetaDict, MetaList

EVENT_TYPES = [
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
    "send_instruction",
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
]

# 批次二：10 条指令全部有 before 事件
INTERCEPTABLE_TYPES = set(EVENT_TYPES)

# 生命周期事件：通配监听与创建/关闭/监听注册的权限规则与内存事件不同
LIFECYCLE_TYPES = {
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
}

# 历史常量：批次二后不再有"只有 after 的事件"，保留名字给旧代码/文档引用
AFTER_ONLY_TYPES = set()

NORMAL_TIER = "normal"
KERNEL_TIER = "kernel"
LISTENER_TIERS = (NORMAL_TIER, KERNEL_TIER)

# kernel 级 source_context 的截断上限（消息条数）；超出的旧消息不放进载荷
SOURCE_CONTEXT_MAX_MESSAGES = 20

# normal 载荷祖先链的深度上限（只保留最近 MAX_ANCESTORS_DEPTH 个祖先）
MAX_ANCESTORS_DEPTH = 32

# 各事件类型中处理器可以修改的 payload 字段。
# 未列出的字段（身份字段等）一律不可由处理器修改。
MUTABLE_PAYLOAD_KEYS = {
    "memory_write": {"content"},
    "edit_metadata": {"value"},
    "send_instruction": {"content"},
}

DEFAULT_REPLACE_RESPONSE = "事件被监听器拦截"


# ----------------------------------------------------------------------
# 监听器记录的序列化（重启快照与回合日志共用）
# ----------------------------------------------------------------------

def _to_tuple_ref(value):
    """canonical 引用 / 条件引用在内存里是 tuple，JSON 里是 list；读回时换回 tuple。"""
    if isinstance(value, (list, tuple)):
        return tuple(_to_tuple_ref(item) for item in value)
    return value


def listener_to_snapshot(listener: dict) -> dict:
    """监听器记录 → JSON 安全形状。

    内存记录里 `space` 是 AddressSpace 对象（不可 JSON 化），`target_ref` 与条件里的
    引用是 tuple；快照/日志存成根表 dict 与 list，读回由 listener_from_snapshot 还原。
    """
    out = {}
    for key, value in listener.items():
        if key == "space":
            out["space"] = (
                {name: list(segments) for name, segments in value.as_roots().items()}
                if value is not None else None
            )
        else:
            out[key] = copy.deepcopy(value)
    return out


def listener_from_snapshot(data: dict) -> dict:
    """listener_to_snapshot 的逆操作（返回可放回 events.listeners 的记录）。"""
    if not isinstance(data, dict):
        raise ValueError(f"监听器快照必须是对象，得到 {type(data).__name__}")
    listener = copy.deepcopy(data)
    roots = listener.get("space")
    listener["space"] = (
        AddressSpace({name: tuple(segments) for name, segments in roots.items()})
        if isinstance(roots, dict) else None
    )
    if listener.get("target_ref") is not None:
        listener["target_ref"] = _to_tuple_ref(listener["target_ref"])
    condition = listener.get("condition")
    if isinstance(condition, dict):
        target = condition.get("target")
        if isinstance(target, dict) and isinstance(target.get("refs"), dict):
            target["refs"] = {
                key: _to_tuple_ref(ref) for key, ref in target["refs"].items()
            }
    return listener


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
    target_ref: Optional[tuple] = None,
) -> dict:
    """构造事件 dict 并分配 event_id（事件瞬时，不落盘）。"""
    return core.events.build_event(
        event_type=event_type,
        phase=phase,
        conv=conv,
        target=target,
        payload=payload,
        source=source,
        target_ref=target_ref,
    )



# ----------------------------------------------------------------------
# 声明式监听条件（R4，过渡实现：handler 节点带 condition 数据时启用）
# ----------------------------------------------------------------------

CONDITION_KEYS = frozenset({"preset", "events", "phase", "target", "source"})
MEMORY_EVENTS = ("memory_read", "memory_write", "memory_make", "memory_delete",
                 "edit_metadata")
LIFECYCLE_EVENTS = ("create_cmd", "create_sub", "close_conversation", "listen_event")

# 用户态对话可用的命名条件子集（名称是机制常量；名单待作者细化，见 issue-2026-11-01）
USER_CONDITION_PRESETS = {
    "own_work.memory": {
        "events": list(MEMORY_EVENTS),
        "phase": ["before", "after"],
        "target": {"kind": "subtree", "workspace_root": True},
    },
    "own_nodes.memory": {
        "events": list(MEMORY_EVENTS),
        "phase": ["before", "after"],
        "target": {"kind": "owned_by_owner"},
    },
    "inbox": {
        "events": ["send_instruction"],
        "phase": ["before", "after"],
        "target": {"kind": "cid_owner"},
    },
    "self.lifecycle": {
        "events": list(LIFECYCLE_EVENTS),
        "phase": ["before", "after"],
        "target": {"kind": "self_lifecycle"},
    },
}


def _condition_leaf(node):
    """读取 handler 节点的 condition 叶子（JSON 字符串）；缺省 None。"""
    if not isinstance(node, MetaDict):
        return None
    raw = node.get("condition")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise IllegalOperationError("handler 节点的 condition 必须是 str 数据叶子")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise IllegalOperationError(f"handler 节点的 condition 不是合法 JSON：{e}")

class EventSystem:
    def __init__(self, core: 'Core'):
        self._core = core
        # 注册顺序即优先级：先注册者先执行。
        self.listeners: List[dict] = []
        self._next_listener_id = 1
        self._next_event_id = 1
        self._observers: List[Any] = []  # 调试控制台用的事件观察者，见 add_observer

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
        target_ref: Optional[tuple] = None,
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
            "target_ref": target_ref,
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
        target_ref: Optional[tuple] = None,
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
            target_ref=target_ref,
        )
        event = self.dispatch(event)
        self.deliver_notifies(event)
        return event

    # ------------------------------------------------------------------
    # 注册 / 修改 / 取消
    # ------------------------------------------------------------------

    def _journaling(self) -> bool:
        """当前是否正在为回合日志记录（无日志/重放/非回合执行期为假）。"""
        check = getattr(self._core, "_journaling", None)
        return bool(check()) if callable(check) else False

    def _note(self, kind: str, **payload) -> None:
        """把监听器变化记进回合日志（无日志/重放/非回合执行期时是空操作）。"""
        note = getattr(self._core, "note_effect", None)
        if callable(note):
            note(kind, **payload)

    def register(self, owner_cid: int, event_type: str, target: str, handler_ref: str, phase: str = "all",
                 space=None, tier: str = NORMAL_TIER, priority: int = 0,
                 kernel_mode: bool = False) -> int:
        """注册监听器。

        tier/priority 由注册者模式决定：只有 kernel_mode=True（注册者对话是内核态）
        可以声明 tier=kernel 或 priority>0；任何对话都只能按 target 权限注册。
        """
        if event_type not in EVENT_TYPES:
            raise IllegalOperationError(f"未知事件类型：{event_type}，允许：{EVENT_TYPES}")
        if phase not in ("before", "after", "all"):
            raise IllegalOperationError("phase 必须是 before/after/all")
        if not isinstance(target, str) or not target:
            raise IllegalOperationError("target 必须是字符串")
        if tier not in LISTENER_TIERS:
            raise IllegalOperationError(f"tier 必须是 normal/kernel，得到 {tier!r}")
        if tier == KERNEL_TIER and not kernel_mode:
            raise IllegalOperationError("内核级监听器只有内核态对话可以注册")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise IllegalOperationError(f"priority 必须是非负整数，得到 {priority!r}")
        if priority > 0 and not kernel_mode:
            raise IllegalOperationError("priority>0 只有内核态对话可以设置")
        handler_code = self.resolve_handler(self._core, handler_ref, space=space)

        if target == "*":
            if event_type not in LIFECYCLE_TYPES:
                raise IllegalOperationError("该事件类型只支持精确 target，不支持 *")
            if not kernel_mode:
                raise IllegalOperationError("通配监听只有内核态对话可以注册")
        else:
            self._check_target_permission(owner_cid, event_type, target, phase, space=space,
                                          kernel_mode=kernel_mode)

        # TODO(issue): 监听器生命周期跟随对话、不跟随 target。目标路径被删除后监听器
        # 仍留在表里（悬空），路径重建后会“复活”（2026-10-24 事件报告 §3.4）。
        # 后续可能改为按 target 清理，需要先定前缀/设备/通配的语义。
        listener_id = self._next_listener_id
        self._next_listener_id += 1
        self.listeners.append({
            "listener_id": listener_id,
            "owner_cid": owner_cid,
            "event_type": event_type,
            "target": target,
            "target_ref": self._canonical_target_ref(
                target, space=space, follow_final=(event_type != "memory_delete")
            ),
            "handler_ref": handler_ref,
            "handler_code": handler_code,
            "phase": phase,
            "enabled": True,
            "tier": tier,
            "priority": priority,
            "space": space,
        })
        if self._journaling():
            self._note("listener_register", listener=listener_to_snapshot(self.listeners[-1]))
        return listener_id

    def modify(self, listener_id: int, owner_cid: int, handler_ref: str = None, space=None,
               priority=None) -> None:
        """对话指令的 modify：只能由注册者改 handler_ref 与 priority。"""
        listener = self._get_listener(listener_id)
        self._check_owner(listener, owner_cid)
        if handler_ref is not None:
            handler_code = self.resolve_handler(self._core, handler_ref, space=space)
            node = self._core.unwrap(handler_ref, for_llm=False, space=space)
            if _condition_leaf(node) is not None:
                owner_kernel = self._listener_owner_kernel(listener)
                condition = self._resolve_condition(
                    node, listener["owner_cid"], owner_kernel,
                    space=listener.get("space") if space is None else space,
                )
                listener["condition"] = condition
            listener["handler_ref"] = handler_ref
            listener["handler_code"] = handler_code
        if priority is not None:
            if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
                raise IllegalOperationError(f"priority 必须是非负整数，得到 {priority!r}")
            if priority > 0 and listener.get("tier") != KERNEL_TIER:
                raise IllegalOperationError("priority>0 只有内核级监听器可以设置")
            listener["priority"] = priority
        if self._journaling():
            self._note("listener_modify", listener_id=listener_id,
                       listener=listener_to_snapshot(listener))

    def cancel(self, listener_id: int, owner_cid: int) -> None:
        listener = self._get_listener(listener_id)
        self._check_owner(listener, owner_cid)
        self.listeners.remove(listener)
        if self._journaling():
            self._note("listener_cancel", listener_id=listener_id)

    @staticmethod
    def _note_listener_error(listener: dict, message: str) -> None:
        """记录监听器最近一次失败：$MEM.listeners 里可自查（B1）。"""
        listener["error_count"] = int(listener.get("error_count", 0)) + 1
        listener["last_error"] = message

    def cancel_all_for_owner(self, owner_cid: int) -> None:
        self.listeners = [l for l in self.listeners if l.get("owner_cid") != owner_cid]
        if self._journaling():
            self._note("listener_cancel_all", owner_cid=owner_cid)

    # ------------------------------------------------------------------
    # 宿主侧就地编辑（调试控制台）
    # ------------------------------------------------------------------

    EDITABLE_FIELDS = ("event_type", "target", "phase", "handler_ref", "enabled", "tier", "priority")

    def update(self, listener_id: int, **fields) -> dict:
        """就地改写一条监听器，返回改写后的记录。

        与 modify 的分工：modify 是对话指令（listen_event action=modify）走的路，
        只能改 handler_ref，且必须由注册者调用。这里是宿主侧特权 API——控制台用来
        编辑事件管线，可以改 event_type / target / phase / handler_ref / enabled，
        与 cancel_all_for_owner 同级，不做 owner 检查。

        owner_cid 不可改。无论是否给了 handler_ref，都按当前 handler_ref 重解析一次
        handler_code：处理器代码是注册时抄进记录的副本，改内存里的节点不会自动生效，
        想让新代码生效就必然要过一次重解析。
        """
        unknown = sorted(set(fields) - set(self.EDITABLE_FIELDS))
        if unknown:
            raise ValueError(
                f"不可改写的监听器字段：{unknown}；允许：{list(self.EDITABLE_FIELDS)}"
            )
        listener = self._get_listener(listener_id)

        event_type = fields.get("event_type", listener["event_type"])
        target = fields.get("target", listener["target"])
        phase = fields.get("phase", listener["phase"])
        handler_ref = fields.get("handler_ref", listener["handler_ref"])
        enabled = fields.get("enabled", listener.get("enabled", True))
        tier = fields.get("tier", listener.get("tier", NORMAL_TIER))
        priority = fields.get("priority", listener.get("priority", 0))

        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型：{event_type}，允许：{EVENT_TYPES}")
        if phase not in ("before", "after", "all"):
            raise ValueError("phase 必须是 before/after/all")
        if not isinstance(target, str) or not target:
            raise ValueError("target 必须是字符串")
        if not isinstance(enabled, bool):
            raise ValueError("enabled 必须是布尔值")
        if tier not in LISTENER_TIERS:
            raise ValueError(f"tier 必须是 normal/kernel，得到 {tier!r}")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ValueError(f"priority 必须是非负整数，得到 {priority!r}")

        if target == "*":
            if event_type not in LIFECYCLE_TYPES:
                raise ValueError("该事件类型只支持精确 target，不支持 *")
        else:
            self._check_target_permission(
                listener["owner_cid"], event_type, target, phase,
                space=listener.get("space"),
                kernel_mode=(tier == KERNEL_TIER),
            )

        handler_code = self.resolve_handler(self._core, handler_ref)

        listener["event_type"] = event_type
        listener["target"] = target
        listener["target_ref"] = self._canonical_target_ref(
            target, space=listener.get("space"),
            follow_final=(event_type != "memory_delete"),
        )
        listener["phase"] = phase
        listener["handler_ref"] = handler_ref
        listener["handler_code"] = handler_code
        listener["enabled"] = enabled
        listener["tier"] = tier
        listener["priority"] = priority
        return dict(listener)

    # ------------------------------------------------------------------
    # 事件观察（调试控制台）
    # ------------------------------------------------------------------

    def add_observer(self, observer) -> None:
        """注册事件观察者：每次派发完一条事件后，收到那条完整事件。"""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    @property
    def observers(self) -> List[Any]:
        return list(self._observers)

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

    def _check_target_permission(self, owner_cid: int, event_type: str, target: str, phase: str,
                                 space=None, kernel_mode: bool = False) -> None:
        """按目标路径 kind 检查监听注册权限。

        memory：按 ctrl.owner（内核态注册者直接放行，用于监督与安全）；
        device：由设备 authorize_listener 决定（内核态不自动越过设备策略）；
        send/close：普通对话只能监听发往自己/关闭自己的事件；
        lifecycle：通配只有内核态可注册，其余按“自己相关”判定。
        """
        if event_type in ("memory_read", "memory_write", "edit_metadata", "memory_make", "memory_delete"):
            if kernel_mode:
                return
            split = self._core.mem.split_path(target, space=space)
            kind = split[0]
            if kind == "device":
                _, mount_key, remainder = split
                device = self._core.mem._devices[mount_key]
                if not device.authorize_listener(event_type, remainder, owner_cid):
                    raise ValueError(f"设备 {mount_key} 未授权对话 {owner_cid} 监听 {target}")
                return
            if event_type in ("memory_make", "memory_delete"):
                parent = self._parent_ref(target)
                owner = self._owner_of(self._core, parent, space=space)
                if owner is None:
                    raise ValueError(f"不能监听无主父路径：{parent}")
                if owner != owner_cid:
                    raise ValueError(f"无权监听 {target}：父路径所有者 cid={owner}，当前 cid={owner_cid}")
                return
            owner = self._owner_of(self._core, target, space=space)
            if owner is None:
                raise ValueError(f"不能监听无主路径：{target}")
            if owner != owner_cid:
                raise ValueError(f"无权监听 {target}：路径所有者 cid={owner}，当前 cid={owner_cid}")
            return

        if event_type == "send_instruction":
            if not target.startswith("cid:"):
                raise IllegalOperationError("send_instruction 事件的 target 必须是 cid:<cid>")
            try:
                cid = int(target[4:])
            except ValueError:
                raise IllegalOperationError(f"send_instruction 事件的 target cid 必须是数字：{target}")
            if kernel_mode:
                return
            if cid != owner_cid:
                raise ValueError(f"对话只能监听发往自己的消息，不能监听 {target}")
            return

        if event_type in LIFECYCLE_TYPES:
            if target == "*":
                if not kernel_mode:
                    raise ValueError("通配监听只有内核态对话可以注册")
                return
            if event_type == "close_conversation":
                if not target.startswith("cid:"):
                    raise IllegalOperationError("close_conversation 事件的 target 必须是 cid:<cid>")
                try:
                    cid = int(target[4:])
                except ValueError:
                    raise IllegalOperationError(f"cid 必须是数字：{target}")
                if kernel_mode:
                    return
                if cid != owner_cid:
                    raise ValueError("普通对话只能监听关闭自己的事件")
                return
            if event_type == "listen_event":
                if target.startswith("listener:"):
                    try:
                        lid = int(target[9:])
                    except ValueError:
                        raise IllegalOperationError(f"listener 事件的 target 必须是 listener:<id>：{target}")
                    record = next((x for x in self.listeners if x["listener_id"] == lid), None)
                    if record is None:
                        raise ValueError(f"监听器 {lid} 不存在")
                    if kernel_mode:
                        return
                    if record.get("owner_cid") != owner_cid:
                        raise ValueError("普通对话只能监听自己的监听器")
                    return
                if target.startswith("cid:"):
                    try:
                        cid = int(target[4:])
                    except ValueError:
                        raise IllegalOperationError(f"cid 必须是数字：{target}")
                    if kernel_mode or cid == owner_cid:
                        return
                    raise ValueError("普通对话只能监听自己相关的事件")
                if target.startswith("$"):
                    # action=listen 的 before target 是准备注册的目标：按普通内存/设备权限检查
                    split = self._core.mem.split_path(target, space=space)
                    if split[0] == "device":
                        _, mount_key, remainder = split
                        device = self._core.mem._devices[mount_key]
                        if not device.authorize_listener(event_type, remainder, owner_cid):
                            raise ValueError(f"设备 {mount_key} 未授权对话 {owner_cid} 监听 {target}")
                        return
                    owner = self._owner_of(self._core, target, space=space)
                    if owner is None:
                        raise ValueError(f"不能监听无主路径：{target}")
                    if owner != owner_cid:
                        raise ValueError(f"无权监听 {target}：路径所有者 cid={owner}，当前 cid={owner_cid}")
                    return
                raise IllegalOperationError(
                    f"listen_event 事件的 target 不是可识别的形式：{target}"
                )
            # create_cmd / create_sub：target 是程序节点引用
            if kernel_mode:
                return
            if not isinstance(target, str) or not target.startswith("$"):
                raise IllegalOperationError(f"{event_type} 事件的 target 必须是程序节点引用")
            split = self._core.mem.split_path(target, space=space)
            if split[0] != "memory":
                raise ValueError(f"不能监听设备上的程序节点：{target}")
            owner = self._owner_of(self._core, target, space=space)
            if owner is None:
                raise ValueError(f"不能监听无主程序节点：{target}")
            if owner != owner_cid:
                raise ValueError(f"无权监听 {target}：程序节点所有者 cid={owner}，当前 cid={owner_cid}")
            return

        raise ValueError(f"不支持的事件注册类型：{event_type}")

    # ------------------------------------------------------------------
    # 目标规范化与载荷分层
    # ------------------------------------------------------------------

    def _canonical_target_ref(self, target: str, space=None, follow_final: bool = True):
        """把请求别名转成内部 canonical target_ref；无法解析返回 None。

        格式：("memory", 段序列) / ("device", 挂载键, 剩余段) / ("cid", id) /
        ("listener", id) / ("*",)。canonical 只用于匹配与权限判定，不进 normal 载荷。
        """
        if not isinstance(target, str) or not target:
            return None
        if target == "*":
            return ("*",)
        if target.startswith("cid:"):
            try:
                return ("cid", int(target[4:]))
            except ValueError:
                return None
        if target.startswith("listener:"):
            try:
                return ("listener", int(target[9:]))
            except ValueError:
                return None
        if not target.startswith("$"):
            return None
        return self._core.canonical_ref(target, space=space, follow_final=follow_final)

    def _conversation_snapshot(self, cid) -> Optional[dict]:
        if not isinstance(cid, int):
            return None
        conv = self._core._conv_by_cid.get(cid)
        if conv is None:
            return None
        if cid in self._core._finished:
            state = "finished"
        elif cid == self._core._active_cid:
            state = "active"
        elif cid in self._core._ready_cids:
            state = "ready"
        elif cid in self._core._dormant_cids:
            state = "dormant"
        else:
            state = "unknown"
        return {
            "cid": cid,
            "identity": conv.identity,
            "name": getattr(conv, "name", None),
            "mode": conv.caps.mode,
            "groups": list(conv.caps.groups),
            "is_sub": conv.is_sub,
            "is_python": conv.is_python,
            "parent_cid": conv.parent.cid if conv.parent is not None else None,
            "state": state,
            "para_ref": conv.para_ref,
        }

    def _actor_cid(self, event: dict):
        source = event.get("source") or {}
        for cid in (source.get("initiator_cid"), (event.get("payload") or {}).get("actor_cid")):
            if isinstance(cid, int):
                return cid
        return None

    @staticmethod
    def _node_kind(node) -> str:
        from .types import LinkNode
        if isinstance(node, MetaDict):
            return "dict"
        if isinstance(node, MetaList):
            return "list"
        if isinstance(node, LinkNode):
            return "link"
        if isinstance(node, str):
            return "str"
        return type(node).__name__

    @staticmethod
    def _node_meta(node):
        if hasattr(node, "get_metadata"):
            return node.get_metadata()
        return None

    def _node_ancestors(self, segments) -> List[dict]:
        """祖先链简单信息（type/meta/ctrl + depth）。

        normal 载荷不暴露真实路径；祖先从根到父排列，超过 MAX_ANCESTORS_DEPTH
        时只保留离目标最近的若干层（带 depth 便于判断截断）。
        """
        total = len(segments) - 1
        if total <= 0:
            return []
        first = max(1, total - MAX_ANCESTORS_DEPTH + 1)
        ancestors = []
        for depth in range(first, total + 1):
            prefix = list(segments[:depth])
            try:
                node = self._core.mem.resolve_path(["MEM"] + prefix, for_llm=False)
            except Exception:
                break
            ctrl = dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None
            ancestors.append({
                "depth": depth,
                "type": self._node_kind(node),
                "meta": self._node_meta(node),
                "ctrl": ctrl,
            })
        return ancestors

    @staticmethod
    def _alias_parent(alias):
        if isinstance(alias, str) and alias.startswith("$") and "." in alias:
            return alias.rsplit(".", 1)[0]
        return None

    _MISSING = object()

    def _resolve_node_no_follow(self, segments):
        """解析节点但不跟随最终段的链接（删链接时看链接节点本身）。"""
        segments = list(segments)
        if not segments:
            return self._core.mem._data
        try:
            parent = self._core.mem.resolve_path(["MEM"] + segments[:-1], for_llm=False)
        except Exception:
            return self._MISSING
        key = segments[-1]
        try:
            if isinstance(parent, (MetaDict, dict)):
                return parent[key] if key in parent else self._MISSING
        except Exception:
            return self._MISSING
        return self._MISSING

    def _node_simple(self, segments, follow_final: bool = True):
        """一个祖先节点的简单信息；节点不存在时返回 None。"""
        if follow_final:
            try:
                node = self._core.mem.resolve_path(["MEM"] + list(segments), for_llm=False)
            except Exception:
                return None
        else:
            node = self._resolve_node_no_follow(segments)
            if node is self._MISSING:
                return None
        return {
            "type": self._node_kind(node),
            "meta": self._node_meta(node),
            "ctrl": dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None,
            "ancestors": self._node_ancestors(list(segments)),
        }

    def _memory_context(self, segments, alias: str) -> dict:
        """目标不存在时的上下文：最近存在前缀 + 直接父节点（用于 make/追加裁决）。"""
        context = {}
        existing_index = None
        for i in range(len(segments) - 1, -1, -1):
            if self._node_simple(segments[:i]) is not None:
                existing_index = i
                break
            if i == 0:
                existing_index = 0
        if existing_index is not None:
            prefix = self._node_simple(segments[:existing_index]) or {}
            new_count = len(segments) - existing_index
            prefix_alias = None
            if isinstance(alias, str) and alias.startswith("$"):
                parts = alias[1:].split(".")
                keep = len(parts) - new_count
                if keep > 0:
                    prefix_alias = "$" + ".".join(parts[:keep])
                elif keep == 0:
                    prefix_alias = "$MEM"
            prefix.update({
                "alias": prefix_alias,
                "depth": existing_index,
                "ancestors": self._node_ancestors(segments[:existing_index]),
            })
            context["longest_existing_prefix"] = prefix
        parent_segments = list(segments[:-1])
        parent = self._node_simple(parent_segments) if parent_segments else None
        if parent is not None:
            parent["alias"] = self._alias_parent(alias)
        context["parent"] = parent
        return context

    def _target_info(self, target_ref, alias: str, follow_final: bool = True) -> dict:
        """normal 级目标信息：种类、别名、是否存在、类型、描述、ctrl、祖先链。"""
        info = {"kind": "unknown", "ref": alias, "exists": False, "type": None,
                "meta": None, "ctrl": None, "ancestors": []}
        if not target_ref:
            return info
        tag = target_ref[0]
        if tag == "memory":
            segments = list(target_ref[1])
            info.update({"kind": "memory"})
            if follow_final:
                try:
                    node = self._core.mem.resolve_path(["MEM"] + segments, for_llm=False)
                except Exception:
                    info.update(self._memory_context(segments, alias))
                    return info
            else:
                node = self._resolve_node_no_follow(segments)
                if node is self._MISSING:
                    info.update(self._memory_context(segments, alias))
                    return info
            info["exists"] = True
            info["type"] = self._node_kind(node)
            info["meta"] = self._node_meta(node)
            info["ctrl"] = dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None
            info["ancestors"] = self._node_ancestors(segments)
            parent = self._node_simple(segments[:-1]) if segments else None
            if parent is not None:
                parent["alias"] = self._alias_parent(alias)
            info["parent"] = parent
            return info
        if tag == "device":
            mount_key, remainder = target_ref[1], list(target_ref[2])
            device = self._core.mem._devices.get(mount_key)
            info.update({"kind": "device", "mount": mount_key, "remainder": remainder})
            if device is not None:
                info["exists"] = True
                try:
                    info["device"] = device.to_llm_string()
                except Exception:
                    info["device"] = None
            return info
        if tag == "cid":
            cid = target_ref[1]
            conv = self._core._conv_by_cid.get(cid)
            info.update({"kind": "conversation", "type": "conversation",
                         "conversation": self._conversation_snapshot(cid)})
            info["exists"] = conv is not None
            return info
        if tag == "listener":
            lid = target_ref[1]
            record = next((x for x in self.listeners if x["listener_id"] == lid), None)
            info.update({"kind": "listener", "type": "listener",
                         "listener": self._listener_summary(record) if record else None})
            info["exists"] = record is not None
            return info
        if tag == "*":
            info.update({"kind": "wildcard", "exists": True})
        return info

    def _listener_summary(self, listener: dict) -> dict:
        """监听器记录摘要（不含 handler_code，避免把安全代码交给其他处理器）。"""
        return {
            "listener_id": listener["listener_id"],
            "owner_cid": listener.get("owner_cid"),
            "event_type": listener.get("event_type"),
            "target": listener.get("target"),
            "phase": listener.get("phase"),
            "priority": listener.get("priority", 0),
            "tier": listener.get("tier", NORMAL_TIER),
            "enabled": listener.get("enabled", True),
        }

    def _target_full(self, target_ref, follow_final: bool = True):
        """kernel 级目标完整数据（JSON 安全结构）。"""
        if not target_ref or target_ref[0] != "memory":
            return None
        segments = list(target_ref[1])
        if follow_final:
            try:
                node = self._core.mem.resolve_path(["MEM"] + segments, for_llm=False)
            except Exception:
                return None
        else:
            node = self._resolve_node_no_follow(segments)
            if node is self._MISSING:
                return None
        return self._json_safe_node(node)

    def _json_safe_node(self, node):
        from .types import LinkNode, StrNode
        if isinstance(node, MetaDict):
            return {"kind": "dict", "meta": node.get_metadata(),
                    "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": {str(k): self._json_safe_node(v) for k, v in node.items()}}
        if isinstance(node, MetaList):
            return {"kind": "list", "meta": node.get_metadata(),
                    "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": [self._json_safe_node(v) for v in node]}
        if isinstance(node, LinkNode):
            return {"kind": "link", "target": node.target}
        if isinstance(node, StrNode):
            return {"kind": "str", "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": str(node)}
        if isinstance(node, str):
            return {"kind": "str", "value": str(node)}
        return {"kind": type(node).__name__, "value": str(node)}

    def _source_context(self, cid) -> Optional[dict]:
        """kernel 级来源对话上下文（消息列表，截断旧消息并标注）。"""
        if not isinstance(cid, int):
            return None
        conv = self._core._conv_by_cid.get(cid)
        if conv is None:
            return None
        messages = []
        for m in conv.messages:
            item = {"role": m.role, "content": str(m.content) if m.content is not None else ""}
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                item["tool_calls"] = copy.deepcopy(tool_calls)
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_call_id is not None:
                item["tool_call_id"] = tool_call_id
            messages.append(item)
        total = len(messages)
        truncated = total > SOURCE_CONTEXT_MAX_MESSAGES
        return {"messages": messages[-SOURCE_CONTEXT_MAX_MESSAGES:], "total": total,
                "truncated": truncated}

    def _enrich_event(self, event: dict, matched: List[dict]) -> None:
        """构造 normal 载荷；有 kernel 监听器时把重块放进内部 _kernel_payload。"""
        source = event.setdefault("source", {})
        actor = self._actor_cid(event)
        if actor is not None and "conversation" not in source:
            try:
                source["conversation"] = self._conversation_snapshot(actor)
            except Exception:
                source["conversation"] = None
        follow_final = event.get("event_type") != "memory_delete"
        try:
            event["target_info"] = self._target_info(
                event.get("target_ref"), event.get("target", ""), follow_final=follow_final,
            )
        except Exception as e:
            event["target_info"] = {"kind": "unknown", "ref": event.get("target", ""),
                                    "exists": False, "error": f"{type(e).__name__}: {e}"}
        if any(listener.get("tier") == KERNEL_TIER for listener in matched):
            kernel = {}
            try:
                ref = event.get("target_ref")
                if ref and ref[0] == "memory":
                    kernel["target_full"] = self._target_full(ref, follow_final=follow_final)
            except Exception as e:
                kernel["target_full"] = None
                kernel["target_full_error"] = f"{type(e).__name__}: {e}"
            try:
                kernel["source_context"] = self._source_context(actor)
            except Exception as e:
                kernel["source_context"] = None
                kernel["source_context_error"] = f"{type(e).__name__}: {e}"
            event["_kernel_payload"] = kernel

    @staticmethod
    def _event_view(event: dict, listener: dict) -> dict:
        """按监听器级别裁剪事件副本；kernel 块只给 kernel 级监听器。"""
        if listener.get("tier") == KERNEL_TIER:
            view = dict(event)
            kernel = view.pop("_kernel_payload", None) or {}
            view.update(kernel)
            return view
        return {k: v for k, v in event.items() if k != "_kernel_payload"}

    @staticmethod
    def _parent_ref(target: str) -> str:
        if not target.startswith("$") or "." not in target:
            raise ValueError("memory_make 的 target 必须是新建完整路径，不能是根")
        return target.rsplit(".", 1)[0]

    # ------------------------------------------------------------------
    # 所有者解析
    # ------------------------------------------------------------------

    @staticmethod
    def _has_device_prefix(core: 'Core', path: str) -> bool:
        """路径的任意前缀是否命中已挂载设备。

        权限检查不能触发设备 resolve_path（如 $MEM.inputs.next 会阻塞等待输入），
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
    def _owner_of(cls, core: 'Core', path: str, space=None) -> Optional[int]:
        """按 2.6 查找 memory 路径所有者。

        只接受 memory kind；device kind 直接返回 None（设备授权走 authorize_listener）。
        """
        if not isinstance(path, str):
            return None
        try:
            kind, segments = core.mem.split_path(path, space=space)[:2]
        except Exception:
            return None
        if kind != "memory":
            return None

        # 目标路径必须真实存在；不存在时不做祖先兜底。
        ref = "$MEM" if not segments else "$MEM." + ".".join(segments)
        try:
            core.mem.resolve_path(["MEM"] + list(segments), for_llm=False)
        except Exception:
            return None

        # 从完整路径向上逐级找第一个带 ctrl.owner 的 MetaDict/MetaList。
        for i in range(len(segments), -1, -1):
            prefix = segments[:i]
            try:
                node = core.mem.resolve_path(["MEM"] + list(prefix), for_llm=False)
            except Exception:
                continue
            if hasattr(node, "get_ctrl"):
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
    def resolve_handler(core: 'Core', handler_ref: str, space=None) -> str:
        if not isinstance(handler_ref, str) or not handler_ref:
            raise ValueError("handler_ref 必须是字符串引用")
        try:
            node = core.unwrap(handler_ref, for_llm=False, space=space)
        except (VMMemoryError, KeyError, IndexError, TypeError, ValueError) as e:
            raise ValueError(f"handler_ref 不存在：{handler_ref}（{e}）")
        if not isinstance(node, MetaDict):
            raise ValueError("handler_ref 必须指向 ctrl.type='handler' 的 dict 节点")
        if (node.get_ctrl() or {}).get("type") != "handler":
            raise ValueError("handler_ref 必须指向 ctrl.type='handler' 的 dict 节点")
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
    def _merge_output(event: dict, out: dict, listener_id: Optional[int] = None,
                      listener: Optional[dict] = None, owner_kernel: bool = True) -> None:
        """把处理器允许修改的字段合并进事件。

        身份字段不可变；未知键忽略；校验失败抛 EventHandlerError。
        before 的决策动作：execute 继续、replace 拒绝、allow 放行短路；
        allow 只有 kernel 级监听器可以返回（normal 级返回按处理器错误 fail closed）。
        after 阶段遇到 notify 自我投递时丢弃该条并记入 handler_errors；
        before 阶段遇到 notify 自我投递视为处理器错误。
        """
        if "action" in out:
            action = out["action"]
            if event["phase"] == "before":
                if action not in ("execute", "replace", "allow"):
                    raise EventHandlerError(
                        0,  # 调用方随后用真实 listener_id 包装
                        f"before 阶段 action 必须是 execute/replace/allow，得到 {action!r}",
                    )
                if action == "allow" and (listener or {}).get("tier") != KERNEL_TIER:
                    raise EventHandlerError(0, "只有内核级监听器可以返回 allow")
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
            # 调用者要能分辨"被拦截"与"成功"：统一加前缀；处理器自定义文本也加。
            event["response"] = f"Interrupted: {response or DEFAULT_REPLACE_RESPONSE}"

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
                owner_cid = (listener or {}).get("owner_cid")
                if not owner_kernel and item["cid"] != owner_cid:
                    error = (
                        f"用户态监听器只能把 notify 投递给注册者自己"
                        f"（owner cid={owner_cid}，请求 cid={item['cid']}）"
                    )
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

    def _listener_owner_kernel(self, listener: dict) -> bool:
        conv = self._core._conv_by_cid.get(listener.get("owner_cid"))
        return conv is not None and conv.caps.mode == "kernel"


    # ------------------------------------------------------------------
    # 声明式条件解析与匹配
    # ------------------------------------------------------------------

    def _resolve_condition(self, node, owner_cid: int, owner_kernel: bool,
                           space=None, owner_conv=None) -> dict:
        """把 handler 节点的 condition 数据解析成内部条件（canonical 目标）。"""
        raw = _condition_leaf(node)
        if raw is None:
            raise IllegalOperationError(
                "handler 节点没有 condition 数据；新式监听器必须在 handler 节点声明条件"
            )
        if not isinstance(raw, dict):
            raise IllegalOperationError("handler 节点的 condition 必须是 JSON 对象")
        unknown = sorted(set(raw) - CONDITION_KEYS)
        if unknown:
            raise IllegalOperationError(f"handler 节点 condition 含未知字段：{unknown}")
        if "preset" in raw:
            preset = raw["preset"]
            if preset not in USER_CONDITION_PRESETS:
                raise IllegalOperationError(
                    f"未知的用户态条件子集 {preset!r}（可用：{sorted(USER_CONDITION_PRESETS)}）"
                )
            raw = dict(USER_CONDITION_PRESETS[preset])
            raw["target"] = dict(raw.get("target") or {})
            raw["preset"] = preset
            if raw["target"].get("workspace_root"):
                root_name = getattr(owner_conv, "workspace_root_name", None) or "work"
                raw["target"] = {"kind": "subtree", "ref": "$" + root_name}
        elif not owner_kernel:
            raise IllegalOperationError(
                "用户态对话只能使用命名条件子集（preset），不能自定义 condition"
            )
        events = raw.get("events")
        if events is None:
            events = list(EVENT_TYPES)
        if not isinstance(events, list) or not all(isinstance(x, str) for x in events):
            raise IllegalOperationError("condition.events 必须是事件类型字符串列表")
        unknown_events = [x for x in events if x not in EVENT_TYPES]
        if unknown_events:
            raise IllegalOperationError(f"condition.events 含未知事件类型：{unknown_events}")
        phases = raw.get("phase", ["before", "after"])
        if phases == "all":
            phases = ["before", "after"]
        if not isinstance(phases, list) or not all(x in ("before", "after") for x in phases):
            raise IllegalOperationError("condition.phase 必须是 before/after 列表")
        target = raw.get("target")
        resolved_target = None
        if target is not None:
            if not isinstance(target, dict):
                raise IllegalOperationError("condition.target 必须是对象")
            kind = target.get("kind")
            if kind not in ("exact", "subtree", "owned_by_owner", "cid_owner",
                            "self_lifecycle"):
                raise IllegalOperationError(f"condition.target.kind 不支持：{kind!r}")
            resolved = {"kind": kind}
            if kind in ("exact", "subtree"):
                ref = target.get("ref")
                if not isinstance(ref, str) or not ref.startswith("$"):
                    raise IllegalOperationError("condition.target.ref 必须是 $ 引用")
                normal = self._core.canonical_ref(ref, space=space)
                delete = self._core.canonical_ref(ref, space=space, follow_final=False)
                if normal is None:
                    raise IllegalOperationError(f"condition.target.ref 无法解析：{ref}")
                resolved["refs"] = {"normal": normal, "delete": delete}
            elif kind == "owned_by_owner":
                resolved["cid"] = owner_cid
            elif kind == "cid_owner":
                resolved["cid"] = owner_cid
            elif kind == "self_lifecycle":
                resolved["cid"] = owner_cid
            resolved_target = resolved
        source = raw.get("source")
        resolved_source = None
        if source is not None:
            if not isinstance(source, dict):
                raise IllegalOperationError("condition.source 必须是对象")
            resolved_source = {}
            for key in ("mode", "groups", "cid"):
                if key in source:
                    value = source[key]
                    if not isinstance(value, list):
                        raise IllegalOperationError(f"condition.source.{key} 必须是列表")
                    resolved_source[key] = value
        return {
            "preset": raw.get("preset"),
            "events": list(events),
            "phases": list(phases),
            "target": resolved_target,
            "source": resolved_source,
        }

    def register_condition(self, owner_cid: int, handler_ref: str, space=None,
                           tier: str = NORMAL_TIER, priority: int = 0,
                           kernel_mode: bool = False) -> int:
        """按 handler 节点自带的 condition 数据注册监听器（新式入口）。"""
        if tier not in LISTENER_TIERS:
            raise IllegalOperationError(f"tier 必须是 normal/kernel，得到 {tier!r}")
        if tier == KERNEL_TIER and not kernel_mode:
            raise IllegalOperationError("内核级监听器只有内核态对话可以注册")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise IllegalOperationError(f"priority 必须是非负整数，得到 {priority!r}")
        if priority > 0 and not kernel_mode:
            raise IllegalOperationError("priority>0 只有内核态对话可以设置")
        node = self._core.unwrap(handler_ref, for_llm=False, space=space)
        owner_conv = self._core._conv_by_cid.get(owner_cid)
        condition = self._resolve_condition(node, owner_cid, kernel_mode, space=space,
                                            owner_conv=owner_conv)
        handler_code = self.resolve_handler(self._core, handler_ref, space=space)
        listener_id = self._next_listener_id
        self._next_listener_id += 1
        display_events = condition["events"]
        display_event = display_events[0] if len(display_events) == 1 else "*"
        target = condition.get("target") or {}
        display_target = "*"
        if target.get("kind") in ("exact", "subtree") and target.get("refs"):
            ref0 = target["refs"]["normal"]
            display_target = "$" + ref0[0] + ("." + ".".join(ref0[1]) if ref0[0] == "memory" and ref0[1] else "")
        self.listeners.append({
            "listener_id": listener_id,
            "owner_cid": owner_cid,
            "event_type": display_event,
            "target": display_target,
            "target_ref": target.get("refs", {}).get("normal") if target else None,
            "handler_ref": handler_ref,
            "handler_code": handler_code,
            "phase": "all" if set(condition["phases"]) == {"before", "after"} else condition["phases"][0],
            "enabled": True,
            "tier": tier,
            "priority": priority,
            "space": space,
            "condition": condition,
        })
        if self._journaling():
            self._note("listener_register", listener=listener_to_snapshot(self.listeners[-1]))
        return listener_id

    def _actor_conv(self, event: dict):
        source = event.get("source") or {}
        cid = source.get("initiator_cid") or (event.get("payload") or {}).get("actor_cid")
        return self._core._conv_by_cid.get(cid) if isinstance(cid, int) else None

    def _condition_target_matches(self, target: dict, event: dict) -> bool:
        kind = target.get("kind")
        if kind in ("exact", "subtree"):
            refs = target.get("refs") or {}
            ref = refs.get("delete" if event.get("event_type") == "memory_delete" else "normal")
            if ref is None:
                return False
            ev = event.get("target_ref")
            if ev is None:
                return len(ref) >= 2 and event.get("target", "") == "$" + ref[0] + (
                    "." + ".".join(ref[1]) if ref[1] else "")
            if kind == "exact":
                return tuple(ev) == tuple(ref)
            if ref[0] != ev[0]:
                if not (ref[0] == "memory" and ev[0] == "memory"):
                    return False
            if ref[0] == "memory":
                return tuple(ev[1][:len(ref[1])]) == tuple(ref[1]) if len(ref[1]) <= len(ev[1]) else False
            return ref[1] == ev[1] and len(ref[2]) <= len(ev[2]) and tuple(ev[2][:len(ref[2])]) == tuple(ref[2])
        if kind == "owned_by_owner":
            ev = event.get("target_ref")
            if not ev or ev[0] != "memory":
                return False
            owner = self._owner_of_segments(ev[1])
            return owner == target.get("cid")
        if kind == "cid_owner":
            ev = event.get("target_ref")
            return bool(ev) and ev[0] == "cid" and ev[1] == target.get("cid")
        if kind == "self_lifecycle":
            cid = target.get("cid")
            ev = event.get("target_ref")
            if ev and ev[0] == "cid" and ev[1] == cid:
                return True
            conv = self._actor_conv(event)
            return conv is not None and conv.cid == cid
        return False

    def _condition_source_matches(self, source_filter, event: dict) -> bool:
        if not source_filter:
            return True
        conv = self._actor_conv(event)
        if conv is None:
            return False
        if "mode" in source_filter and conv.caps.mode not in source_filter["mode"]:
            return False
        if "groups" in source_filter and not (set(source_filter["groups"]) & set(conv.caps.groups)):
            return False
        if "cid" in source_filter and conv.cid not in source_filter["cid"]:
            return False
        return True

    def _owner_of_segments(self, segments):
        try:
            return self._owner_of(self._core, "$MEM." + ".".join(segments), space=None)
        except Exception:
            return None

    def _condition_matches(self, listener: dict, event: dict) -> bool:
        condition = listener.get("condition")
        if condition is None:
            return False
        if event["event_type"] not in condition["events"]:
            return False
        if event["phase"] not in condition["phases"]:
            return False
        target = condition.get("target")
        if target is not None and not self._condition_target_matches(target, event):
            return False
        if not self._condition_source_matches(condition.get("source"), event):
            return False
        return True

    def _match(self, event: dict) -> List[dict]:
        """匹配监听器并排序：canonical target 优先，priority 降序、同优先级按注册顺序。"""
        target_ref = event.get("target_ref")
        matched = []
        for listener in self.listeners:
            if listener.get("enabled", True) is False:
                continue  # 调试控制台的暂时屏蔽：记录留着，只是不参与匹配
            if listener.get("condition") is not None:
                if self._condition_matches(listener, event):
                    matched.append(listener)
                continue
            if listener["event_type"] != event["event_type"]:
                continue
            listener_phase = listener.get("phase", "all")
            if listener_phase not in ("all", event["phase"]):
                continue
            if listener["target"] == "*":
                matched.append(listener)
                continue
            listener_ref = listener.get("target_ref")
            if listener_ref is not None and target_ref is not None:
                if listener_ref == target_ref:
                    matched.append(listener)
            elif listener["target"] == event["target"]:
                # 兼容：两侧都没有 canonical 目标时按别名字符串匹配
                matched.append(listener)
        matched.sort(key=lambda l: (-int(l.get("priority", 0)), l["listener_id"]))
        return matched

    def dispatch(self, event: dict) -> dict:
        matched = self._match(event)
        if matched:
            self._enrich_event(event, matched)
        if event["phase"] == "before":
            for listener in matched:
                before_errors = len(event.get("handler_errors") or [])
                try:
                    out = self._run_listener(listener, self._event_view(event, listener))
                    # _merge_output 中 action 为 replace/allow 时也会校验 response/notify
                    self._merge_output(event, out, listener["listener_id"], listener,
                                       owner_kernel=self._listener_owner_kernel(listener))
                except EventHandlerError as e:
                    # fail closed 改为事件结果：不抛异常、不在这里打印 stderr；
                    # 指令执行方负责向 stderr 报告一次并返回 Error 工具响应。
                    if e.listener_id == 0:
                        e = EventHandlerError(listener["listener_id"], e.message)
                    self._note_listener_error(listener, str(e))
                    event["action"] = "handler_error"
                    event["response"] = f"Error: {e}"
                    event["error"] = str(e)
                    break
                else:
                    for entry in (event.get("handler_errors") or [])[before_errors:]:
                        if entry.get("listener_id") == listener["listener_id"]:
                            self._note_listener_error(listener, entry.get("error", ""))
                if event["action"] in ("replace", "allow"):
                    break  # 首个 allow/replace 决定结果
        else:  # after：只审计与通知，不改变结果
            for listener in matched:
                before_errors = len(event.get("handler_errors") or [])
                try:
                    out = self._run_listener(listener, self._event_view(event, listener))
                    self._merge_output(event, out, listener["listener_id"], listener,
                                       owner_kernel=self._listener_owner_kernel(listener))
                except EventHandlerError as e:
                    # fail open：记录 stderr，跳过该处理器的 notify，继续后续监听器
                    if e.listener_id == 0:
                        e = EventHandlerError(listener["listener_id"], e.message)
                    self._note_listener_error(listener, str(e))
                    event.setdefault("handler_errors", []).append({
                        "listener_id": listener["listener_id"],
                        "error": str(e),
                    })
                    print(f"[AVM] {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    self._note_listener_error(listener, err)
                    event.setdefault("handler_errors", []).append({
                        "listener_id": listener["listener_id"],
                        "error": err,
                    })
                    print(f"[AVM] 事件监听器 {listener['listener_id']} 处理失败: {err}", file=sys.stderr)
                    continue
                for entry in (event.get("handler_errors") or [])[before_errors:]:
                    if entry.get("listener_id") == listener["listener_id"]:
                        self._note_listener_error(listener, entry.get("error", ""))

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
        for observer in list(self._observers):
            try:
                observer(event)
            except Exception as e:
                # 观察者只是旁听，失败不能影响派发
                print(
                    f"[AVM] 事件观察者处理失败: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

    def deliver_notifies(self, event: dict) -> None:
        """按 2.7 内部投递 notify：不经事件管道、不产生 send_instruction 事件。

        TODO(issue): 不校验目标授权——监听器所有者可唤醒任意存活对话。作者倾向
        “处理器只允许返回到自己”，需要单独讨论（question-notify-target-authorization-2026-10-24）。
        """
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
            if core._is_waiting_for_return(target):
                # 等亚对话返回的对话：notify 只入批次，不唤醒；亚对话结束后父对话先跑
                continue
            if cid in core._dormant_cids:
                core._dormant_cids.remove(cid)
            if cid in core._ready_cids:
                core._ready_cids.remove(cid)
            core._ready_cids.append(cid)

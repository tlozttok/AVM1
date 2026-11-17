"""AVM 事件模型 v1：事件构造、监听器注册表、权限校验、处理器执行与管道分发。

批次二（权限体系）：10 条指令都产生 before 请求事件与 after 事实事件；
事件载荷分 normal / kernel 两级，监听器有 priority 与决策动作 allow/replace/execute；
匹配与权限判定用 canonical 目标（target_ref），处理器看到的目标仍是请求别名。
监听器是 ctrl.type="handler" 节点中的代码，每次事件在全新受限命名空间执行，
不保留实例状态。

监听注册的 memory 目标授权不在 Core：Core 只提供机制，策略由软件层的 Guard
监听器（监听 listen_event before）按 caps/roots 实现。设备目标仍由设备授权。

"这个目标指的是什么"这类只读读法（别名 → 节点/对话/监听器状态、normal 与
kernel 两级的载荷形状）在 event_targets.TargetProbe 里：判权和拼载荷问的是
同一组问题，写在两处就会漂移。
"""

# TODO(issue-2026-11-01): 用户态事件子集（named presets）的名单与粒度是临时方案，待作者
# 后续细化：preset 是否读写分离、是否允许用户自定义条件子集、是否把
# send_instruction/listen_event 纳入子集。设计见
# docs/private/Agents/implemented/design-unified-event-bus-2026-10-31.md §2.3、§8。
import copy
import sys
from typing import Any, Dict, List, Optional

from .capabilities import AddressSpace
from .event_conditions import EventConditions, condition_leaf
from .event_contracts import (  # noqa: F401  (对外转出，见下方说明)
    EVENT_TYPES,
    KERNEL_TIER,
    LISTENER_TIERS,
    MAX_ANCESTORS_DEPTH,
    NORMAL_TIER,
)
from .event_targets import TargetProbe
from .exceptions import IllegalOperationError, VMMemoryError
from .modules import BASE_MODULES, catalog_exports, import_function, shared_view
from .sandbox import check_code, safe_globals
from .types import MetaDict

# 生命周期事件：通配监听与创建/关闭/监听注册的权限规则与内存事件不同
LIFECYCLE_TYPES = {
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
}

# 地址类事件：target 是内存路径，权限判定落在"这段地址归谁"上（设备问设备，内存交给 Guard）
MEMORY_EVENT_TYPES = {
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
}

# 层级名与祖先深度上限住在 event_contracts（事件系统与目标探针共用）；这里重新
# 导出，`from .events import NORMAL_TIER` 这类外部写法不用改。

# kernel 级 source_context 的截断上限（消息条数）；超出的旧消息不放进载荷
SOURCE_CONTEXT_MAX_MESSAGES = 20


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


def _cid_from_ref(target: str, type_error: str, number_error: str) -> int:
    """从 `cid:<n>` 形态的 target 里取出数字。

    两条错误消息由调用方给：不同事件类型对同一个错误有各自的措辞，
    放进本函数就会变成一句谁都不像的话。
    """
    if not target.startswith("cid:"):
        raise IllegalOperationError(type_error)
    try:
        return int(target[4:])
    except ValueError:
        raise IllegalOperationError(number_error)


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
    imports = listener.get("imports")
    if imports is not None:
        if not isinstance(imports, list) or not all(isinstance(name, str) for name in imports):
            raise ValueError("监听器的 imports 必须是字符串列表")
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



# 声明式监听条件的解析与匹配在 event_conditions（EventConditions）。


def _merge_action(event: dict, out: dict, listener: Optional[dict]) -> None:
    """action 只在 before 阶段生效；allow 是内核级特权（normal 级返回视为错误）。"""
    if "action" not in out:
        return
    action = out["action"]
    if event["phase"] != "before":
        return  # after 阶段忽略 action
    if action not in ("execute", "replace", "allow"):
        raise EventHandlerError(
            0,  # 调用方随后用真实 listener_id 包装
            f"before 阶段 action 必须是 execute/replace/allow，得到 {action!r}",
        )
    if action == "allow" and (listener or {}).get("tier") != KERNEL_TIER:
        raise EventHandlerError(0, "只有内核级监听器可以返回 allow")
    event["action"] = action


def _merge_replace_response(event: dict, out: dict) -> None:
    """replace 的拒答文本：统一加 Interrupted 前缀，让调用者能分辨"被拦截"与"成功"。"""
    if event.get("action") != "replace":
        return
    response = out.get("response", "")
    if response is None:
        response = ""
    if not isinstance(response, str):
        raise EventHandlerError(
            0,
            f"response 必须是字符串，得到 {type(response).__name__}",
        )
    # 处理器自定义文本也加前缀：调用者只看这一句，不能有一类漏掉标记。
    event["response"] = f"Interrupted: {response or DEFAULT_REPLACE_RESPONSE}"


def _merge_payload(event: dict, out: dict) -> None:
    """载荷里只有该事件类型声明为可变的键能改；类型按事件类型校验。"""
    mutable = MUTABLE_PAYLOAD_KEYS.get(event["event_type"], set())
    payload_out = out.get("payload")
    if not isinstance(payload_out, dict):
        return
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


def _reject_notify(event: dict, listener_id: Optional[int], error: str) -> None:
    """notify 违规：before 阶段当处理器错误抛出；after 阶段记账、丢弃这一条、继续。"""
    if event["phase"] == "before":
        raise EventHandlerError(0, error)
    event.setdefault("handler_errors", []).append({
        "listener_id": listener_id,
        "error": error,
    })
    print(f"[AVM] 事件监听器 {listener_id} 处理失败: {error}", file=sys.stderr)


def _merge_notify(event: dict, out: dict, listener_id: Optional[int],
                  listener: Optional[dict], owner_kernel: bool) -> None:
    """notify 投递列表：形状校验 + 两条"不许投给谁"，通过后并进事件。"""
    if "notify" not in out:
        return
    notifies = out["notify"]
    if not isinstance(notifies, list):
        raise EventHandlerError(0, "notify 必须是 list")
    # 谁在发起这次投递：before 看事件来源，after 看载荷里的执行者
    source_cid = None
    if event["phase"] == "before":
        source = event.get("source") or {}
        if source.get("kind") == "core":
            source_cid = source.get("initiator_cid")
    else:
        source_cid = (event.get("payload") or {}).get("actor_cid")
    owner_cid = (listener or {}).get("owner_cid")
    for item in notifies:
        if not isinstance(item, dict) or not isinstance(item.get("cid"), int) or not isinstance(item.get("content"), str):
            raise EventHandlerError(
                0,
                f"notify 每项必须是 {{\"cid\": int, \"content\": str}}，得到 {item!r}",
            )
        if source_cid is not None and item["cid"] == source_cid:
            _reject_notify(event, listener_id, f"notify 不允许自我投递（cid={source_cid} 是当前对话）")
            continue
        if not owner_kernel and item["cid"] != owner_cid:
            _reject_notify(
                event, listener_id,
                f"用户态监听器只能把 notify 投递给注册者自己"
                f"（owner cid={owner_cid}，请求 cid={item['cid']}）",
            )
            continue
        event["notify"].append({"cid": item["cid"], "content": item["content"]})


class EventSystem:
    def __init__(self, core: 'Core'):
        self._core = core
        # 声明式条件的解析与匹配（handler 节点上的 condition 数据）。
        self.conditions = EventConditions(core)
        # 目标读数（别名 → 节点/对话/监听器的真实状态）交给 TargetProbe。传的是
        # "取法"而不是列表本身：cancel_all_for_owner 会整个重建 self.listeners。
        self.targets = TargetProbe(core, lambda: self.listeners)
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

    def is_journaling(self) -> bool:
        """当前是否正在为回合日志记录（无日志/重放/非回合执行期为假）。"""
        check = getattr(self._core, "is_journaling", None)
        return bool(check()) if callable(check) else False

    def _note(self, kind: str, **payload) -> None:
        """把监听器变化记进回合日志（无日志/重放/非回合执行期时是空操作）。"""
        note = getattr(self._core, "note_effect", None)
        if callable(note):
            note(kind, **payload)

    def reserve_listener_id(self, listener_id) -> None:
        """重放/恢复用：保证下一个分配号大于已用过的 id。

        日志里记录的是当时的 id，号段可能有跳号（注册后又撤销、或来自旧代际），
        所以不能按"条数"推，只能按"见过的最大号"抬分配器。
        """
        if isinstance(listener_id, int) and listener_id >= self._next_listener_id:
            self._next_listener_id = listener_id + 1

    def reset_listener_ids(self) -> None:
        """监听器 id 分配器归零（快照重建：监听器表整体替换后从 1 重发）。"""
        self._next_listener_id = 1

    def reset_event_ids(self) -> None:
        """事件 id 分配器归零（快照重建：事件流水不跨重启，id 从 1 重发）。"""
        self._next_event_id = 1

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
            "target_ref": self.targets.canonical_ref(
                target, space=space, follow_final=(event_type != "memory_delete")
            ),
            "handler_ref": handler_ref,
            "handler_code": handler_code,
            "imports": self._owner_imports(owner_cid),
            "phase": phase,
            "enabled": True,
            "tier": tier,
            "priority": priority,
            "space": space,
        })
        if self.is_journaling():
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
            if condition_leaf(node) is not None:
                owner_kernel = self._listener_owner_kernel(listener)
                condition = self.conditions.parse(
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
        if self.is_journaling():
            self._note("listener_modify", listener_id=listener_id,
                       listener=listener_to_snapshot(listener))

    def cancel(self, listener_id: int, owner_cid: int) -> None:
        listener = self._get_listener(listener_id)
        self._check_owner(listener, owner_cid)
        self.listeners.remove(listener)
        if self.is_journaling():
            self._note("listener_cancel", listener_id=listener_id)

    @staticmethod
    def _note_listener_error(listener: dict, message: str) -> None:
        """记录监听器最近一次失败：$MEM.listeners 里可自查（B1）。"""
        listener["error_count"] = int(listener.get("error_count", 0)) + 1
        listener["last_error"] = message

    def cancel_all_for_owner(self, owner_cid: int) -> None:
        self.listeners = [l for l in self.listeners if l.get("owner_cid") != owner_cid]
        if self.is_journaling():
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
        listener["target_ref"] = self.targets.canonical_ref(
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

        memory：Core 不做所有者授权——监听策略由软件层的 Guard 监听器负责；
        device：由设备 authorize_listener 决定（内核态不自动越过设备策略）；
        send/close：普通对话只能监听发往自己/关闭自己的事件；
        lifecycle：通配只有内核态可注册，其余按“自己相关”判定。
        """
        if event_type in MEMORY_EVENT_TYPES:
            self._check_memory_target(event_type, target, owner_cid, space, kernel_mode)
            return
        if event_type == "send_instruction":
            self._check_send_target(target, owner_cid, kernel_mode)
            return
        if event_type in LIFECYCLE_TYPES:
            self._check_lifecycle_target(event_type, target, owner_cid, space, kernel_mode)
            return
        raise ValueError(f"不支持的事件注册类型：{event_type}")

    def _check_memory_target(self, event_type: str, target: str, owner_cid: int,
                             space, kernel_mode: bool) -> None:
        """地址类事件：内核态直接放行；普通对话由设备裁决，内存节点交给软件层 Guard。"""
        if kernel_mode:
            return
        self._check_device_target(event_type, target, owner_cid, space)

    def _check_device_target(self, event_type: str, target: str, owner_cid: int, space) -> None:
        """目标落在设备上就问设备的 authorize_listener；内存节点不做所有者判定。"""
        split = self._core.mem.split_path(target, space=space)
        if split[0] != "device":
            return
        _, mount_key, remainder = split
        device = self._core.mem.devices[mount_key]
        if not device.authorize_listener(event_type, remainder, owner_cid):
            raise ValueError(f"设备 {mount_key} 未授权对话 {owner_cid} 监听 {target}")

    @staticmethod
    def _check_send_target(target: str, owner_cid: int, kernel_mode: bool) -> None:
        """send_instruction：target 必须是 cid:<cid>，普通对话只能监听发往自己的。"""
        cid = _cid_from_ref(
            target,
            "send_instruction 事件的 target 必须是 cid:<cid>",
            f"send_instruction 事件的 target cid 必须是数字：{target}",
        )
        if kernel_mode or cid == owner_cid:
            return
        raise ValueError(f"对话只能监听发往自己的消息，不能监听 {target}")

    def _check_lifecycle_target(self, event_type: str, target: str, owner_cid: int,
                                space, kernel_mode: bool) -> None:
        """生命周期事件：通配先判，其余按事件类型分流。"""
        if target == "*":
            if not kernel_mode:
                raise ValueError("通配监听只有内核态对话可以注册")
            return
        if event_type == "close_conversation":
            self._check_close_target(target, owner_cid, kernel_mode)
            return
        if event_type == "listen_event":
            self._check_listen_target(target, owner_cid, space, kernel_mode)
            return
        self._check_program_target(event_type, target, space, kernel_mode)

    @staticmethod
    def _check_close_target(target: str, owner_cid: int, kernel_mode: bool) -> None:
        """close_conversation：普通对话只能监听关闭自己的事件。"""
        cid = _cid_from_ref(
            target,
            "close_conversation 事件的 target 必须是 cid:<cid>",
            f"cid 必须是数字：{target}",
        )
        if kernel_mode or cid == owner_cid:
            return
        raise ValueError("普通对话只能监听关闭自己的事件")

    def _check_listen_target(self, target: str, owner_cid: int, space, kernel_mode: bool) -> None:
        """listen_event 的三种 target 形态：listener:<id>、cid:<id>、$<内存路径>。"""
        if target.startswith("listener:"):
            self._check_listener_target(target, owner_cid, kernel_mode)
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
            # action=listen 的 before target 是准备注册的目标（condition 的 canonical
            # ref）：memory 目标不做所有者判定（交给 Guard），设备照旧询问设备。
            self._check_device_target("listen_event", target, owner_cid, space)
            return
        raise IllegalOperationError(
            f"listen_event 事件的 target 不是可识别的形式：{target}"
        )

    def _check_listener_target(self, target: str, owner_cid: int, kernel_mode: bool) -> None:
        """listener:<id>：监听器必须存在，普通对话只能监听自己的监听器。"""
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

    def _check_program_target(self, event_type: str, target: str, space, kernel_mode: bool) -> None:
        """create_cmd / create_sub：target 是程序节点引用（内存节点；不做所有者判定）。"""
        if kernel_mode:
            return
        if not isinstance(target, str) or not target.startswith("$"):
            raise IllegalOperationError(f"{event_type} 事件的 target 必须是程序节点引用")
        split = self._core.mem.split_path(target, space=space)
        if split[0] != "memory":
            raise ValueError(f"不能监听设备上的程序节点：{target}")

    def _source_context(self, cid) -> Optional[dict]:
        """kernel 级来源对话上下文（消息列表，截断旧消息并标注）。"""
        if not isinstance(cid, int):
            return None
        conv = self._core.get_conversation(cid)
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
        actor = self.targets.actor_cid(event)
        if actor is not None and "conversation" not in source:
            try:
                source["conversation"] = self.targets.conversation_snapshot(actor)
            except Exception:
                source["conversation"] = None
        follow_final = event.get("event_type") != "memory_delete"
        try:
            event["target_info"] = self.targets.target_info(
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
                    kernel["target_full"] = self.targets.target_full(ref, follow_final=follow_final)
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

    def _owner_imports(self, owner_cid: int) -> list:
        """注册者对话可 import 的模块名：基线 ∪ owner 对话 caps.imports。

        注册监听器时抄进记录（与 handler_code 同一种「注册时快照」语义）；owner 对话
        关闭/恢复后处理器仍按注册时的白名单运行。
        """
        conv = self._core.get_conversation(owner_cid)
        granted = conv.caps.imports if conv is not None else frozenset()
        return sorted(set(BASE_MODULES) | set(granted))

    @staticmethod
    def _handler_namespace(allowed) -> dict:
        """处理器命名空间：基线视图 + 该监听器白名单的实例 `__import__`。"""
        namespace = safe_globals({
            "math": shared_view("math"),
            "json": shared_view("json"),
            "re": shared_view("re"),
        })
        namespace["__builtins__"]["__import__"] = import_function(allowed, {})
        return namespace

    def _run_listener(self, listener: dict, event: dict) -> dict:
        """在全新受限命名空间执行处理器，返回规范化输出。

        处理器错误统一抛 EventHandlerError（before fail closed / after fail open 由 dispatch 决定）。
        导入白名单取注册时抄下的 `listener["imports"]`（旧快照没有该字段时只有基线）。
        """
        allowed = listener.get("imports") or list(BASE_MODULES)
        namespace = self._handler_namespace(allowed)
        try:
            check_code(listener["handler_code"], allow_import=True,
                       imports=allowed, exports=catalog_exports())
            exec(compile(listener["handler_code"], "<handler>", "exec"), namespace)
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

        四类字段各一个 _merge_xxx：动作、拒答文本、载荷、通知。它们只在"哪个阶段
        抛错、哪个阶段记账"上不同，混在一起会看不出这个差别。
        """
        _merge_action(event, out, listener)
        _merge_replace_response(event, out)
        _merge_payload(event, out)
        _merge_notify(event, out, listener_id, listener, owner_kernel)

    # ------------------------------------------------------------------
    # 分发
    # ------------------------------------------------------------------

    def _listener_owner_kernel(self, listener: dict) -> bool:
        conv = self._core.get_conversation(listener.get("owner_cid"))
        return conv is not None and conv.caps.mode == "kernel"


    # ------------------------------------------------------------------
    # 声明式条件
    #
    # 解析与匹配都在 EventConditions 里；这里只保留对外方法名（指令与 Guard 用）。
    # ------------------------------------------------------------------

    def parse_condition(self, owner_cid: int, handler_ref: str, space=None,
                        kernel_mode: bool = False) -> dict:
        """解析 handler 节点自带的 condition 数据；只解析，不注册监听器。

        listen_event before 事件需要在派发前拿到解析后的 condition（Guard 授权依据），
        解析结果随后原样交给 register_condition，避免注册时重复解析。
        """
        node = self._core.unwrap(handler_ref, for_llm=False, space=space)
        return self.conditions.parse(node, owner_cid, kernel_mode, space=space)

    def register_condition(self, owner_cid: int, handler_ref: str, space=None,
                           tier: str = NORMAL_TIER, priority: int = 0,
                           kernel_mode: bool = False, condition: Optional[dict] = None) -> int:
        """按 handler 节点自带的 condition 数据注册监听器（新式入口）。

        condition 是已由 parse_condition 解析好的结果（listen_event 指令先解析再派发
        before 事件）；缺省时在本函数里解析。
        """
        if tier not in LISTENER_TIERS:
            raise IllegalOperationError(f"tier 必须是 normal/kernel，得到 {tier!r}")
        if tier == KERNEL_TIER and not kernel_mode:
            raise IllegalOperationError("内核级监听器只有内核态对话可以注册")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise IllegalOperationError(f"priority 必须是非负整数，得到 {priority!r}")
        if priority > 0 and not kernel_mode:
            raise IllegalOperationError("priority>0 只有内核态对话可以设置")
        if condition is None:
            condition = self.parse_condition(
                owner_cid, handler_ref, space=space, kernel_mode=kernel_mode,
            )
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
            "imports": self._owner_imports(owner_cid),
            "phase": "all" if set(condition["phases"]) == {"before", "after"} else condition["phases"][0],
            "enabled": True,
            "tier": tier,
            "priority": priority,
            "space": space,
            "condition": condition,
        })
        if self.is_journaling():
            self._note("listener_register", listener=listener_to_snapshot(self.listeners[-1]))
        return listener_id

    def _match(self, event: dict) -> List[dict]:
        """匹配监听器并排序：canonical target 优先，priority 降序、同优先级按注册顺序。"""
        target_ref = event.get("target_ref")
        matched = []
        for listener in self.listeners:
            if listener.get("enabled", True) is False:
                continue  # 调试控制台的暂时屏蔽：记录留着，只是不参与匹配
            if listener.get("condition") is not None:
                if self.conditions.matches(listener, event):
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
        self._core.note_turn_event(summary)
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
            target = core.get_conversation(cid)
            if target is None or core.is_finished(cid):
                print(
                    f"[AVM] 事件通知投递失败：对话 {cid} 不存在或已关闭",
                    file=sys.stderr,
                )
                continue
            target.user_batch.add_user_content(content)
            if core.is_waiting_for_return(target):
                # 等亚对话返回的对话：notify 只入批次，不唤醒；亚对话结束后父对话先跑
                continue
            core.wake(cid)

"""回合级效果日志（journal）：持久运行的第一块地基。

一轮 = 一次推理 + 该轮产生的全部指令。指令执行完回到
`Core.advance_conversation` 的安全点（`monitor.record` 之后）记一条记录；异常路径
（API 失败重试、未捕获异常）不追加，该回合视为未提交。

记录是"状态增量"，不是操作流水：

```json
{"seq": 3, "cid": 0, "at": "2026-11-07T21:00:00+0800",
 "history": [{"role": "tool", "content": "...", "tool_call_id": "c1"}],
 "call": {"content": "...", "reasoning_content": "", "tool_calls": [...]},
 "batch": {"tool_responses": [...], "user_contents": [...]},
 "scheduler": {"active": null, "ready": [1], "dormant": [0], "finished": [],
               "activation_instructions": 0, "next_cid": 2},
 "effects": [{"kind": "memory_set", "ref": "$MEM.x", "value": "v", ...}]}
```

- `history` 是本轮真正消费进历史的非 assistant 消息（上一轮指令产生的 tool 响应、
  投递进来的 user 内容）。**按实际追加的消息记**，不按生成它的效果推导——before
  事件处理器可以改写 payload，指令最终写入的值才是状态。
- `call` 是本轮 assistant 消息（content / reasoning_content / tool_calls）。
- `batch` 是回合结束后本对话的待发批次（指令产生的工具响应、被投递的输入）。
- `scheduler` 是回合结束时的调度状态。调度迁移散在指令、亚对话完成、预算中断等多处，
  逐条补效果容易漏，所以整块记下；它是重建调度用的输入，不是逐字段一致的契约。
- `effects` 是语义效果（内存写、对话生命周期、投递、监听器、设备装卸）。重放期间
  不调模型、不派发事件、不访问设备：设备写只记审计条目，重放时跳过。

计划里的 `call` 原样保留；计划的 `tool_responses` 落在 `history` 里（真正进历史，
顺序才与真实请求前缀一致）；`batch` / `scheduler` 是恢复后接着调度所必需的增量。
设备写审计（计划"待定夺"第 3 项的默认选择）以 `device_write` 效果出现，重放时跳过。

Journal 是纯追加的 JSONL：临时想丢弃日志尾部用 `truncate(seq)`（v1 不自动压缩）。
"""

import json
import os
import sys
from typing import List, Optional

from .capabilities import AddressSpace
from .exceptions import VMMemoryError, IllegalOperationError
from .memory import Memory
from .types import (
    Conversation, MetaDict, MetaList,
    batch_from_dict, batch_to_dict, message_from_dict, message_to_dict,
)


class JournalError(ValueError):
    """日志格式错误或重放与当前状态不一致：面向运行者，应停下而不是继续。"""


# ----------------------------------------------------------------------
# 日志文件
# ----------------------------------------------------------------------

class Journal:
    """一条 JSONL 日志。seq 是单调递增的提交序号，由本类分配。

    append 每次打开、写一行、flush + fsync、关闭：崩溃最多丢最后半行，
    读回时遇到解析失败的行就停在那里（半截日志 = 上一条提交后的状态）。
    """

    def __init__(self, path: str):
        if not isinstance(path, str) or not path:
            raise ValueError("日志路径必须是非空字符串")
        self.path = path
        self._last_seq = 0
        self._rescan()

    def _rescan(self) -> None:
        """扫描已有文件确定 last_seq；坏行/半行之后的内容忽略。"""
        last = 0
        if not os.path.isfile(self.path):
            self._last_seq = 0
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    break
                seq = record.get("seq") if isinstance(record, dict) else None
                if not isinstance(seq, int) or seq != last + 1:
                    break
                last = seq
        self._last_seq = last

    def append(self, record: dict) -> int:
        """追加一条记录并分配 seq，返回 seq。"""
        if not isinstance(record, dict):
            raise JournalError("日志记录必须是对象")
        record = dict(record)
        seq = self._last_seq + 1
        record["seq"] = seq
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_seq = seq
        return seq

    def read(self, after_seq: int = 0) -> List[dict]:
        """读回 seq > after_seq 的记录；坏行处停止（半截日志的恢复语义）。"""
        if not os.path.isfile(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    print(
                        f"[AVM] 日志 {self.path}: 遇到无法解析的行，之后的内容按未提交处理",
                        file=sys.stderr,
                    )
                    break
                if not isinstance(record, dict) or not isinstance(record.get("seq"), int):
                    print(
                        f"[AVM] 日志 {self.path}: 记录形状不合法，之后的内容按未提交处理",
                        file=sys.stderr,
                    )
                    break
                if record["seq"] > after_seq:
                    records.append(record)
        return records

    def last_seq(self) -> int:
        return self._last_seq

    def truncate(self, seq: int) -> None:
        """物理丢弃 seq 之后的记录（只保留 seq <= seq）。v1 由宿主手动调用。"""
        if not isinstance(seq, int) or seq < 0:
            raise ValueError("truncate 的 seq 必须是非负整数")
        records = [r for r in self.read(0) if r["seq"] <= seq]
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._last_seq = records[-1]["seq"] if records else 0


# ----------------------------------------------------------------------
# 快照形状的辅助（回合记录里的 conversation / scheduler / batch 用）
# ----------------------------------------------------------------------

def scheduler_to_snapshot(core) -> dict:
    """回合结束时的调度状态（日志记录与重启快照共用同一形状）。"""
    return {
        "active": core._active_cid,
        "ready": list(core._ready_cids),
        "dormant": list(core._dormant_cids),
        "finished": sorted(core._finished),
        "activation_instructions": core._activation_instructions,
        "next_cid": core._next_cid,
    }


def conversation_create_payload(core, conv) -> dict:
    """新建对话的效果载荷：对话全部字段 + 工作空间分配（若本次创建分配了）。

    工作空间是 `_build_conversation` 在对话注册前写进内存树的节点；它不是对话字段，
    快照/日志都必须单独带上，否则恢复后用户态对话的根目标不存在。
    """
    doc = conv.to_snapshot()
    doc.pop("batch", None)  # 新建对话没有待发批次
    workspace = None
    root_name = getattr(conv, "workspace_root_name", None)
    if isinstance(root_name, str) and root_name:
        target = conv.caps.roots.get(root_name)
        if target is None:
            raise JournalError(f"对话 {conv.cid} 声明了工作空间根 {root_name!r} 但能力根表里没有")
        node = core.mem.resolve_path(["MEM"] + list(target), for_llm=False)
        workspace = {
            "root_name": root_name,
            "target": list(target),
            "node": Memory.serialize_node(node),
        }
    return {"conversation": doc, "workspace": workspace}


def _space_roots(space) -> Optional[dict]:
    """AddressSpace → 根表 dict（JSON 安全）；None 保持 None。"""
    if space is None:
        return None
    return {name: list(segments) for name, segments in space.as_roots().items()}


def _space_from_roots(roots: Optional[dict]):
    if roots is None:
        return None
    return AddressSpace({name: tuple(segments) for name, segments in roots.items()})


# ----------------------------------------------------------------------
# 重放
# ----------------------------------------------------------------------

def apply(core, record: dict) -> None:
    """把一条回合记录应用到 Core 上（重放）。

    调用方保证记录顺序；本函数不调模型、不派发事件、不访问设备。应用失败抛
    JournalError——日志与镜像/快照不一致时继续跑只会把状态带得更歪。
    幂等由 `Core.replay_journal` 的 seq 游标保证（已应用过的记录不再进入这里）。
    """
    if not isinstance(record, dict):
        raise JournalError("日志记录必须是对象")
    seq = record.get("seq")
    cid = record.get("cid")
    conv = core._conv_by_cid.get(cid)
    if conv is None:
        raise JournalError(f"日志 seq={seq}: 目标对话 cid={cid} 不存在，日志与镜像/快照不匹配")
    was_replaying = core._replaying
    core._replaying = True
    try:
        for item in record.get("history") or []:
            conv.messages.append(message_from_dict(item))
        call = record.get("call")
        if call is not None:
            payload = dict(call)
            payload["role"] = "assistant"
            conv.messages.append(message_from_dict(payload))
        for effect in record.get("effects") or []:
            _apply_effect(core, effect, seq)
        # 对话可能在本轮效果里被删除/关闭：批次与调度按"结果"重建（找不到就跳过）。
        target = core._conv_by_cid.get(cid)
        if target is not None and record.get("batch") is not None:
            target.user_batch = batch_from_dict(record["batch"])
        apply_scheduler(core, record.get("scheduler"), f"日志 seq={seq}")
        # 这段历史来自日志（不是本进程跑出来的）：Python 对话的执行器需要预热
        core._warmup_needed.add(cid)
    finally:
        core._replaying = was_replaying


def _apply_effect(core, effect, seq) -> None:
    if not isinstance(effect, dict) or not isinstance(effect.get("kind"), str):
        raise JournalError(f"日志 seq={seq}: 效果条目形状不合法：{effect!r}")
    kind = effect["kind"]
    try:
        handler = _EFFECTS.get(kind)
        if handler is None:
            raise JournalError(f"日志 seq={seq}: 未知效果种类 {kind!r}")
        handler(core, effect)
    except JournalError:
        raise
    except (VMMemoryError, IllegalOperationError, KeyError, IndexError, TypeError, ValueError) as e:
        raise JournalError(f"日志 seq={seq}: 应用效果 {kind} 失败：{e}") from e


def apply_scheduler(core, scheduler, source: str = "日志") -> None:
    if not isinstance(scheduler, dict):
        return

    def _known(cid):
        if cid in core._conv_by_cid:
            return True
        print(
            f"[AVM] {source}: 调度状态引用不存在的对话 cid={cid}（宿主侧对话不随恢复）",
            file=sys.stderr,
        )
        return False

    active = scheduler.get("active")
    core._active_cid = active if active is not None and _known(active) else None
    core._ready_cids = [cid for cid in scheduler.get("ready") or [] if _known(cid)]
    core._dormant_cids = [cid for cid in scheduler.get("dormant") or [] if _known(cid)]
    core._finished = {cid for cid in scheduler.get("finished") or [] if _known(cid)}
    if isinstance(scheduler.get("activation_instructions"), int):
        core._activation_instructions = scheduler["activation_instructions"]
    if isinstance(scheduler.get("next_cid"), int):
        core._next_cid = scheduler["next_cid"]


# ----------------------------------------------------------------------
# 单条效果的恢复
# ----------------------------------------------------------------------

def _effect_memory_set(core, effect) -> None:
    core.mem.set(
        effect["ref"], effect["value"],
        owner_cid=effect.get("owner_cid"), space=_space_from_roots(effect.get("space")),
    )


def _effect_memory_make(core, effect) -> None:
    core.mem.make(
        effect["ref"], effect["key"], effect["mem_type"],
        owner_cid=effect.get("owner_cid"), space=_space_from_roots(effect.get("space")),
    )


def _effect_memory_delete(core, effect) -> None:
    core.mem.delete(effect["ref"], space=_space_from_roots(effect.get("space")))


def _effect_memory_attach(core, effect) -> None:
    """把一段子树挂到真实段路径（程序包安装用；内核态整体替换，不建中间节点元数据）。"""
    path = effect.get("path")
    if not isinstance(path, list) or not path:
        raise VMMemoryError("memory_attach 的 path 必须是非空段列表")
    node = Memory.deserialize_node(effect.get("node"))
    parent = core.mem._data
    for key in path[:-1]:
        if isinstance(parent, (dict, MetaDict)):
            child = parent.get(key)
            if child is None:
                child = Memory.deserialize_node({"kind": "dict", "value": {}})
                parent[key] = child
            parent = child
        else:
            raise VMMemoryError(f"memory_attach 的父段不是容器：{'/'.join(str(x) for x in path)}")
    last = path[-1]
    if isinstance(parent, (dict, MetaDict)):
        parent[last] = node
    elif isinstance(parent, (list, MetaList)):
        idx = int(last)
        if idx >= len(parent):
            parent.append(node)
        else:
            parent[idx] = node
    else:
        raise VMMemoryError(f"memory_attach 的父段不是容器：{'/'.join(str(x) for x in path)}")


def _effect_memory_detach(core, effect) -> None:
    """摘除真实段路径上的子树（程序包卸载用；整棵子树回收留给内存管理计划）。"""
    path = effect.get("path")
    if not isinstance(path, list) or not path:
        raise VMMemoryError("memory_detach 的 path 必须是非空段列表")
    parent = core.mem._data
    for key in path[:-1]:
        if not isinstance(parent, (dict, MetaDict)) or key not in parent:
            raise VMMemoryError(f"memory_detach 的路径不存在：{'/'.join(str(x) for x in path)}")
        parent = parent[key]
    last = path[-1]
    if isinstance(parent, (dict, MetaDict)) and last in parent:
        del parent[last]
    elif isinstance(parent, (list, MetaList)) and int(last) < len(parent):
        del parent[int(last)]
    else:
        raise VMMemoryError(f"memory_detach 的路径不存在：{'/'.join(str(x) for x in path)}")


def _effect_edit_metadata(core, effect) -> None:
    op = effect.get("type")
    key = effect.get("key")
    node = core.unwrap(effect["ref"], for_llm=False, space=_space_from_roots(effect.get("space")))
    if op not in ("set", "del"):
        raise VMMemoryError(f"edit_metadata 日志只记录 set/del，得到 {op!r}")
    ctrl = dict(node.get_ctrl() or {})
    if op == "set":
        ctrl[key] = effect.get("value")
    else:
        ctrl.pop(key, None)
    node.set_ctrl(ctrl)


def _effect_device_write(core, effect) -> None:
    """设备写审计：真实执行发生在设备自己身上，重放不访问设备。"""
    return


def _effect_conversation_create(core, effect) -> None:
    payload = effect.get("conversation")
    if not isinstance(payload, dict):
        raise VMMemoryError("conversation_create 缺少 conversation 载荷")
    conv = Conversation.from_snapshot(payload)
    workspace = effect.get("workspace")
    if isinstance(workspace, dict) and workspace.get("target"):
        _attach_workspace_node(core, workspace["target"], workspace.get("node"))
    parent_cid = payload.get("parent_cid")
    if parent_cid is not None:
        parent = core._conv_by_cid.get(parent_cid)
        if parent is None:
            raise VMMemoryError(f"conversation_create 的父对话 cid={parent_cid} 不存在")
        conv.parent = parent
    core._register(conv, cid=payload["cid"])
    # 日志里创建出来的对话在续跑时同样需要 Python 执行器预热
    core._warmup_needed.add(conv.cid)


def _attach_workspace_node(core, target, node_image) -> None:
    """把工作空间节点挂到真实段路径（父段不存在时补空 dict，与 _ensure_region_node 一致）。"""
    parent = core.mem._data
    for key in list(target)[:-1]:
        if not isinstance(parent, (dict, MetaDict)):
            raise VMMemoryError(f"工作空间区域父段不是容器：{'/'.join(str(x) for x in target)}")
        child = parent.get(key)
        if child is None:
            child = Memory.deserialize_node({"kind": "dict", "value": {}})
            parent[key] = child
        parent = child
    node = Memory.deserialize_node(node_image) if node_image is not None else None
    if isinstance(parent, (dict, MetaDict)) and node is not None:
        parent[target[-1]] = node


def _effect_conversation_close(core, effect) -> None:
    cid = effect["cid"]
    conv = core._conv_by_cid.get(cid)
    if conv is None:
        return
    core._finished.add(cid)
    for queue in (core._ready_cids, core._dormant_cids):
        while cid in queue:
            queue.remove(cid)
    core._executors.pop(cid, None)


def _effect_conversation_delete(core, effect) -> None:
    cid = effect["cid"]

    def _delete(target_cid):
        for child in list(core._conv_by_cid.values()):
            if child.is_sub and child.parent is not None and child.parent.cid == target_cid:
                _delete(child.cid)
        conv = core._conv_by_cid.pop(target_cid, None)
        if conv is None:
            return
        for queue in (core._ready_cids, core._dormant_cids):
            while target_cid in queue:
                queue.remove(target_cid)
        core._finished.discard(target_cid)
        core._executors.pop(target_cid, None)

    _delete(cid)


def _effect_delivery(core, effect) -> None:
    cid = effect.get("to_cid")
    target = core._conv_by_cid.get(cid)
    if target is None or cid in core._finished:
        print(f"[AVM] 日志投递效果：对话 {cid} 不存在或已关闭，跳过", file=sys.stderr)
        return
    for item in effect.get("tool_responses") or []:
        target.user_batch.add_tool_response(item.get("content", ""), item.get("tool_call_id", ""))
    for content in effect.get("user_contents") or []:
        target.user_batch.add_user_content(content)


def _effect_listener_register(core, effect) -> None:
    from .events import listener_from_snapshot

    data = effect.get("listener")
    if not isinstance(data, dict):
        raise VMMemoryError("listener_register 缺少 listener 载荷")
    listener = listener_from_snapshot(data)
    core.events.listeners.append(listener)
    listener_id = listener.get("listener_id")
    if isinstance(listener_id, int) and listener_id >= core.events._next_listener_id:
        core.events._next_listener_id = listener_id + 1


def _effect_listener_modify(core, effect) -> None:
    from .events import listener_from_snapshot

    listener_id = effect.get("listener_id")
    for index, listener in enumerate(core.events.listeners):
        if listener.get("listener_id") == listener_id:
            replacement = listener_from_snapshot(effect["listener"])
            core.events.listeners[index] = replacement
            return
    raise VMMemoryError(f"listener_modify 的目标监听器 {listener_id} 不存在")


def _effect_listener_cancel(core, effect) -> None:
    listener_id = effect.get("listener_id")
    core.events.listeners = [
        listener for listener in core.events.listeners
        if listener.get("listener_id") != listener_id
    ]


def _effect_listener_cancel_all(core, effect) -> None:
    owner_cid = effect.get("owner_cid")
    core.events.listeners = [
        listener for listener in core.events.listeners
        if listener.get("owner_cid") != owner_cid
    ]


def _effect_device_mount(core, effect) -> None:
    from .image import mount_device_entry

    mount = effect.get("mount")
    spec = effect.get("spec") or {}
    mount_device_entry(core, {
        "path": mount,
        "file": spec.get("path"),
        "class": spec.get("class"),
        "args": spec.get("args") or {},
        "id": spec.get("id"),
    }, base=None)
    # 来源表以日志记录为准（重放时重新解析的路径可能与记录形态不同）
    core.device_specs[mount] = dict(spec)


def _effect_device_unmount(core, effect) -> None:
    core.unmount_device(effect.get("mount"))


_EFFECTS = {
    "memory_set": _effect_memory_set,
    "memory_make": _effect_memory_make,
    "memory_delete": _effect_memory_delete,
    "memory_attach": _effect_memory_attach,
    "memory_detach": _effect_memory_detach,
    "edit_metadata": _effect_edit_metadata,
    "device_write": _effect_device_write,
    "conversation_create": _effect_conversation_create,
    "conversation_close": _effect_conversation_close,
    "conversation_delete": _effect_conversation_delete,
    "delivery": _effect_delivery,
    "listener_register": _effect_listener_register,
    "listener_modify": _effect_listener_modify,
    "listener_cancel": _effect_listener_cancel,
    "listener_cancel_all": _effect_listener_cancel_all,
    "device_mount": _effect_device_mount,
    "device_unmount": _effect_device_unmount,
}

"""回合级效果日志（journal）：持久运行的第一块地基。

一轮 = 一次推理 + 该轮产生的全部指令。指令执行完回到
`Core.advance_conversation` 的安全点（`monitor.record` 之后）记一条记录；异常路径
（API 失败重试、未捕获异常）不追加，该回合视为未提交。

记录是**有序变化序列**（v2），不是按字段分组的快照：

```json
{"v": 2, "seq": 3, "cid": 0, "at": "2026-11-09T21:00:00+0800",
 "changes": [
   {"kind": "message", "message": {"role": "tool", "content": "...", "tool_call_id": "c1"}},
   {"kind": "message", "message": {"role": "assistant", "content": "", "tool_calls": [...]}},
   {"kind": "effect", "effect": {"kind": "memory_set", "ref": "$MEM.x", "value": "v"}},
   {"kind": "batch", "batch": {"tool_responses": [], "user_contents": []}},
   {"kind": "scheduler", "active": null, "ready_add": [1], "dormant_remove": [0],
    "activation_instructions": 0, "next_cid": 2}
 ]}
```

- `message`：本轮真正追加进历史的实际消息（tool 响应、投递进来的 user 内容、
  assistant 消息）。按实际追加记，不按生成它的效果推导——before 事件处理器可以
  改写 payload。
- `effect`：语义效果（内存写、对话生命周期、投递、监听器、设备装卸）。重放期间
  不调模型、不派发事件、不访问设备；设备写只记审计条目，重放时跳过。
- `batch`：回合结束时的待发批次；只在相对上次提交有变化时写。
- `scheduler`：调度增量（active、就绪/休眠的增删、finished、预算计数、next_cid）；
  只在有变化时写。调度迁移散在指令、亚对话完成、预算中断等多处，增量由回合前后
  的状态差算出，保证不漏。

旧格式（带 `history` / `call` / `batch` / `scheduler` / `effects` 字段、没有 `v`）
不再读取；遇到时 `apply` 直接报错并提示归档。

设备的写审计（计划"待定夺"第 3 项的默认选择）以 `device_write` 效果出现，重放时跳过。

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

    def iter_records(self, after_seq: int = 0):
        """逐条读回 seq > after_seq 的记录；坏行处停止（半截日志的恢复语义）。

        重放用这个接口：内存里只留一条记录，不把整段尾部读成列表。
        """
        if not os.path.isfile(self.path):
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
                if not isinstance(seq, int) or seq <= after_seq:
                    if isinstance(seq, int) and seq <= after_seq:
                        continue
                    break
                yield record

    def read(self, after_seq: int = 0) -> List[dict]:
        """读回 seq > after_seq 的记录（小规模/测试用；重放走 iter_records）。"""
        return list(self.iter_records(after_seq))

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
    """新建对话的效果载荷：对话全部字段。

    工作目录不再由内核分配：软件用 memory_make 建目录、再写
    `$MEM.conversations.<cid>.roots`（`conversation_roots` 效果）动态加根。
    旧日志的 `conversation_create` 载荷可能仍带 `workspace` 字段，重放侧保留兼容。
    """
    doc = conv.to_snapshot()
    doc.pop("batch", None)  # 新建对话没有待发批次
    return {"conversation": doc}


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

JOURNAL_RECORD_VERSION = 2

_OLD_FORMAT_HINT = (
    "旧格式回合日志（v!=2）：日志已改为增量变化序列，不再读取旧记录。"
    "请归档旧日志后重开世界（或换一个镜像目录）。"
)


def apply(core, record: dict) -> None:
    """把一条回合记录应用到 Core 上（重放）。

    调用方保证记录顺序；本函数不调模型、不派发事件、不访问设备。应用失败抛
    JournalError——日志与镜像/快照不一致时继续跑只会把状态带得更歪。
    幂等由 `Core.replay_journal` 的 seq 游标保证（已应用过的记录不再进入这里）。

    记录形状（v2）：`{"v": 2, "seq", "cid", "at", "changes": [...]}`，逐个应用
    changes 里的变化：消息追加、效果、待发批次、调度增量。顺序就是执行顺序。
    """
    if not isinstance(record, dict):
        raise JournalError("日志记录必须是对象")
    seq = record.get("seq")
    if record.get("v") != JOURNAL_RECORD_VERSION:
        raise JournalError(f"日志 seq={seq}: {_OLD_FORMAT_HINT}")
    cid = record.get("cid")
    conv = core._conv_by_cid.get(cid)
    if conv is None:
        from .core import PANEL_CID
        if cid == PANEL_CID:
            # 保留对话是世界的固定成员，身份固定且可恢复：日志里先出现、快照里
            # 还没有它时（只靠日志恢复），在这里补建。
            conv = core.ensure_operator_conversation()
        else:
            raise JournalError(f"日志 seq={seq}: 目标对话 cid={cid} 不存在，日志与镜像/快照不匹配")
    changes = record.get("changes")
    if not isinstance(changes, list):
        raise JournalError(f"日志 seq={seq}: changes 必须是数组")
    was_replaying = core._replaying
    core._replaying = True
    try:
        for change in changes:
            _apply_change(core, change, cid, seq)
        # 这段历史来自日志（不是本进程跑出来的）：Python 对话的执行器需要预热
        core._warmup_needed.add(cid)
    finally:
        core._replaying = was_replaying


def _apply_change(core, change, cid: int, seq) -> None:
    if not isinstance(change, dict) or not isinstance(change.get("kind"), str):
        raise JournalError(f"日志 seq={seq}: 变化条目形状不合法：{change!r}")
    kind = change["kind"]
    if kind == "message":
        conv = core._conv_by_cid.get(cid)
        if conv is None:
            raise JournalError(f"日志 seq={seq}: 消息的目标对话 cid={cid} 不存在")
        message = change.get("message")
        if not isinstance(message, dict):
            raise JournalError(f"日志 seq={seq}: message 载荷必须是对象")
        conv.messages.append(message_from_dict(message))
        return
    if kind == "effect":
        _apply_effect(core, change.get("effect"), seq)
        return
    if kind == "batch":
        target_cid = change.get("cid", cid)
        conv = core._conv_by_cid.get(target_cid)
        if conv is not None:
            conv.user_batch = batch_from_dict(change.get("batch"))
        return
    if kind == "scheduler":
        _apply_scheduler_delta(core, change, seq)
        return
    raise JournalError(f"日志 seq={seq}: 未知变化种类 {kind!r}")


def _apply_scheduler_delta(core, change, seq) -> None:
    """应用一条调度增量；引用不存在对话时警告并跳过（与快照恢复同一策略）。"""

    def _known(cid):
        if cid in core._conv_by_cid:
            return True
        print(
            f"[AVM] 日志 seq={seq}: 调度增量引用不存在的对话 cid={cid}，跳过",
            file=sys.stderr,
        )
        return False

    if "active" in change:
        active = change.get("active")
        core._active_cid = active if active is not None and _known(active) else None
    for key, attr in (("ready", "_ready_cids"), ("dormant", "_dormant_cids")):
        target = getattr(core, attr)
        for cid in change.get(key + "_remove") or []:
            if cid in target:
                target.remove(cid)
        for cid in change.get(key + "_add") or []:
            if _known(cid) and cid not in target:
                target.append(cid)
    for cid in change.get("finished_remove") or []:
        core._finished.discard(cid)
    for cid in change.get("finished_add") or []:
        if _known(cid):
            core._finished.add(cid)
    if isinstance(change.get("activation_instructions"), int):
        core._activation_instructions = change["activation_instructions"]
    if isinstance(change.get("next_cid"), int):
        core._next_cid = change["next_cid"]


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
    # 旧日志效果可能带 owner_cid 字段：按未知字段忽略（节点所有权已移出 MEM）。
    core.mem.set(
        effect["ref"], effect["value"],
        space=_space_from_roots(effect.get("space")),
    )


def _effect_memory_make(core, effect) -> None:
    # 旧日志效果可能带 owner_cid 字段：按未知字段忽略。
    core.mem.make(
        effect["ref"], effect["key"], effect["mem_type"],
        space=_space_from_roots(effect.get("space")),
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
    """旧日志兼容：把工作空间节点挂到真实段路径（父段不存在时补空 dict）。

    内核不再分配工作空间；这段只用于重放本机制移除前写下的日志。
    """
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


def _effect_conversation_roots(core, effect) -> None:
    """还原一次动态授予根：整体替换对话的非 core 根表（core 自述根保留）。"""
    cid = effect.get("cid")
    conv = core._conv_by_cid.get(cid)
    if conv is None:
        print(
            f"[AVM] 日志 conversation_roots 效果：对话 {cid} 不存在（可能已关闭），跳过",
            file=sys.stderr,
        )
        return
    roots = effect.get("roots")
    if not isinstance(roots, dict):
        raise VMMemoryError("conversation_roots 缺少 roots 映射")
    new_roots = core._core_roots(conv)
    for name, segments in roots.items():
        new_roots[name] = tuple(segments)
    conv.caps = conv.caps.copy(roots=new_roots)


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
    from .events import drop_removed_condition_listener, listener_from_snapshot

    data = effect.get("listener")
    if not isinstance(data, dict):
        raise VMMemoryError("listener_register 缺少 listener 载荷")
    listener = listener_from_snapshot(data)
    # 旧日志里依赖节点所有权 cid 的条件形态已删除：重放时丢弃该监听器，不中断启动。
    if drop_removed_condition_listener(listener, "日志重放"):
        return
    core.events.listeners.append(listener)
    listener_id = listener.get("listener_id")
    if isinstance(listener_id, int) and listener_id >= core.events._next_listener_id:
        core.events._next_listener_id = listener_id + 1


def _effect_listener_modify(core, effect) -> None:
    from .events import drop_removed_condition_listener, listener_from_snapshot

    listener_id = effect.get("listener_id")
    replacement = listener_from_snapshot(effect["listener"])
    if drop_removed_condition_listener(replacement, "日志重放"):
        # 旧条件形态的 modify：丢弃被改写的监听器本身；记录缺失也不报错（可能已随
        # listener_register 一起被丢弃），重放继续。
        core.events.listeners = [
            listener for listener in core.events.listeners
            if listener.get("listener_id") != listener_id
        ]
        return
    for index, listener in enumerate(core.events.listeners):
        if listener.get("listener_id") == listener_id:
            core.events.listeners[index] = replacement
            return
    print(
        f"[AVM] 日志重放：listener_modify 的目标监听器 {listener_id} 不存在，跳过",
        file=sys.stderr,
    )


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
    dev = {
        "path": mount,
        "file": spec.get("path"),
        "class": spec.get("class"),
        "args": spec.get("args") or {},
        "id": spec.get("id"),
    }
    if spec.get("sha256"):
        # 重放按日志记录的内容哈希重建；旧日志没有该字段时由挂载入口计算并补记。
        dev["sha256"] = spec["sha256"]
    new_spec = mount_device_entry(core, dev, base=None)
    # 来源表以日志记录为准（重放时重新解析的路径可能与记录形态不同）；
    # 旧日志的 spec 没有 sha256，用本次重算值补上，避免下次快照变成"旧快照"。
    recorded = dict(spec)
    if not recorded.get("builtin") and not recorded.get("sha256"):
        recorded["sha256"] = new_spec.get("sha256")
    core.device_specs[mount] = recorded


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
    "conversation_roots": _effect_conversation_roots,
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

"""重启快照：持久运行的加速缓存。

日志是持久化依据，快照是它的缓存。快照记录"某一安全点上的状态"，日志记录之后
提交的回合；恢复顺序是：装载镜像（跳过 init）→ 恢复快照 → 按 `journal_to` 重放
seq 之后的日志。快照缺失或损坏时退回"镜像 + 日志"重放，不阻塞启动。

快照形状与内核转储（诊断用）同源但目标不同：

- 转储会截断、只作诊断；快照不截断，用于加速重启。
- 转储只有 mount/type/id 的设备表；快照还带 `device_specs`（设备来源表），
  否则运行期热装的设备重启后无法复原。
- 快照带 `seq`（已提交的日志位置），恢复时用来定位日志尾部。

承诺的边界（见 docs/runtime.md 的「持久运行」）：`$MEM` 与对话历史/身份/待发批次
是硬承诺；监听器、调度、设备实例是重建结果；设备内部状态、未提交回合的副作用、
宿主特权操作不在承诺内；Python 对话的跨轮状态必须落在 `$MEM`。
"""

import json
import os
import sys
import time
from typing import Optional

from .events import listener_from_snapshot, listener_to_snapshot
from .journal import apply_scheduler, scheduler_to_snapshot
from .memory import Memory
from .types import MetaDict

SNAPSHOT_FORMAT = "avm.snapshot"
SNAPSHOT_VERSION = 1


class SnapshotError(ValueError):
    """快照格式错误或与当前镜像/设备文件不匹配：面向运行者，应修复而不是吞掉。"""


# ----------------------------------------------------------------------
# 生成 / 读写
# ----------------------------------------------------------------------

def snapshot(core) -> dict:
    """把当前 Core 拍成重启快照（JSON 安全）。

    安全点约束：只在没有半截回合的位置调用（宿主退出路径、测试）。快照的 seq 取
    `_journal_seq`——已提交的日志位置，不是日志文件长度：文件里可能存在尚未重放
    的尾部记录，它们不属于当前状态。
    """
    scheduler = scheduler_to_snapshot(core)
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seq": int(getattr(core, "_journal_seq", 0) or 0),
        "image": os.path.join(core.image_dir or "", core.image_name or "") or None,
        "host_flags": dict(core.host_flags or {}),
        "para_ref": core.para_ref,
        "note": "重启快照（加速缓存）；memory 段与 persist_to 同格式",
        "scheduler": scheduler,
        "next_cid": core._next_cid,
        "memory": core.mem.to_image_dict(),
        "security": {
            "roots": {name: list(target) for name, target in core.root_registry.items()},
        },
        "conversations": [
            core._conv_by_cid[cid].to_snapshot()
            for cid in sorted(core._conv_by_cid) if cid >= 0
        ],
        "listeners": [listener_to_snapshot(listener) for listener in core.events.listeners],
        "devices": [
            {"mount": mount, "type": type(device).__name__,
             "id": getattr(device, "device_id", None)}
            for mount, device in list(core.mem._devices.items())
        ],
        "device_specs": {
            mount: dict(spec) for mount, spec in core.device_specs.items()
        },
        "interrupts": {"last_dispatch": list(core.interrupts.last_dispatch)},
    }


def write_atomic(path: str, doc: dict) -> str:
    """原子写快照：同目录临时文件 + os.replace（半截快照不会覆盖掉旧快照）。"""
    if not isinstance(path, str) or not path:
        raise SnapshotError("快照路径必须是非空字符串")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_snapshot(path: str) -> dict:
    """读快照并校验格式与版本；格式不对抛 SnapshotError。"""
    if not isinstance(path, str) or not path:
        raise SnapshotError("快照路径必须是非空字符串")
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        raise SnapshotError(f"快照不存在或不可读：{path}: {e}") from e
    except ValueError as e:
        raise SnapshotError(f"快照不是合法 JSON：{path}: {e}") from e
    if not isinstance(doc, dict):
        raise SnapshotError(f"快照根节点必须是对象：{path}")
    if doc.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError(
            f"不是 AVM 重启快照（format={doc.get('format')!r}）：{path}"
            "——单对话检查点请用 --checkpoint-in / restore_conversation"
        )
    version = doc.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(f"不支持的快照版本 {version}（当前支持 {SNAPSHOT_VERSION}）：{path}")
    for key in ("scheduler", "memory", "conversations", "device_specs"):
        if key not in doc:
            raise SnapshotError(f"快照缺少 {key} 段：{path}")
    return doc


# ----------------------------------------------------------------------
# 恢复
# ----------------------------------------------------------------------

def restore(core, doc: dict) -> None:
    """把快照恢复到 Core（重启恢复：调度、对话、内存、监听器、设备表、安全配置）。

    顺序有依赖：先换内存树（其中含设备挂载点的链接节点），再按 `device_specs`
    重建设备并重新物化挂载点；最后恢复对话/监听器/调度。
    """
    doc = _validate_doc(doc)
    _rebuild_memory_and_devices(core, doc)
    _rebuild_conversations(core, doc)
    _rebuild_listeners(core, doc)
    _restore_security_and_pointer(core, doc)
    apply_scheduler(core, doc.get("scheduler"), source="重启快照")
    core._journal_seq = int(doc.get("seq", 0) or 0)
    core._last_snapshot_seq = core._journal_seq
    core._refresh_journal_batch()
    core._check_invariants()


def _validate_doc(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise SnapshotError("快照必须是对象")
    if doc.get("format") != SNAPSHOT_FORMAT or doc.get("version") != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"不是可恢复的重启快照（format/version 不对）："
            f"{doc.get('format')!r}/{doc.get('version')!r}"
        )
    return doc


def _rebuild_memory_and_devices(core, doc: dict) -> None:
    from .image import mount_device_entry

    old_devices = dict(core.mem._devices)
    specs = doc.get("device_specs") or {}
    # 设备表与来源表按快照重建：镜像已挂载的镜像设备也重新实例化（设备状态不序列化）
    core.mem._devices = {}
    core.mem._device_ids = {}
    core.device_specs = {}
    memory_doc = doc.get("memory")
    if not isinstance(memory_doc, dict):
        raise SnapshotError("快照 memory 段必须是对象")
    core.mem._data = Memory.deserialize_node(memory_doc)
    if not isinstance(core.mem._data, MetaDict):
        raise SnapshotError("快照 memory 段的根必须是 dict 节点")
    for mount, spec in specs.items():
        if not isinstance(spec, dict):
            raise SnapshotError(f"device_specs.{mount} 必须是对象")
        if spec.get("builtin"):
            device = old_devices.get(mount)
            if device is None:
                raise SnapshotError(f"内建设备 {mount} 不在当前内核里，无法恢复")
            device.mount_path = mount
            core.mem.mount(mount, device)
            device.attach_core(core)
            core.device_specs[mount] = dict(spec)
            continue
        file_path = spec.get("path")
        if not isinstance(file_path, str) or not os.path.isfile(file_path):
            raise SnapshotError(
                f"设备 {mount} 的文件不存在：{file_path!r}——快照依赖设备文件留在原路径，"
                "恢复前请把设备文件放回"
            )
        mount_device_entry(core, {
            "path": mount,
            "file": file_path,
            "class": spec.get("class"),
            "args": spec.get("args") or {},
            "id": spec.get("id"),
        }, base=None)
        # 来源表以快照记录为准（挂载路径可能被规范化，比较/重启要求逐字段一致）
        core.device_specs[mount] = dict(spec)
    # 设备挂载点重新物化（mount 已建链接；这里兜底快照树里没带的挂载点）
    for mount in list(core.mem._devices):
        node = core.mem._data
        for segment in mount.split(".")[:-1]:
            if not isinstance(node, (dict, MetaDict)):
                break
            node = node.get(segment)
        if isinstance(node, (dict, MetaDict)) and mount.split(".")[-1] not in node:
            core.mem._plant_link(
                mount, getattr(core.mem._devices[mount], "device_id", None) or mount,
                type(core.mem._devices[mount]).__name__,
            )


def _rebuild_conversations(core, doc: dict) -> None:
    from .types import Conversation

    core._conv_by_cid = {}
    core._ready_cids = []
    core._dormant_cids = []
    core._active_cid = None
    core._finished = set()
    core._executors = {}
    briefs = doc.get("conversations") or []
    if not isinstance(briefs, list):
        raise SnapshotError("快照 conversations 段必须是数组")
    for brief in briefs:
        if not isinstance(brief, dict) or not isinstance(brief.get("cid"), int):
            raise SnapshotError(f"快照里的对话条目形状不合法：{brief!r}")
        if brief["cid"] in core._conv_by_cid:
            raise SnapshotError(f"快照里对话 cid 重复：{brief['cid']}")
        conv = Conversation.from_snapshot(brief)
        core._register(conv, cid=brief["cid"])
        core._warmup_needed.add(conv.cid)  # 恢复出来的对话：Python 执行器首次创建时预热
    for brief in briefs:
        parent_cid = brief.get("parent_cid")
        if parent_cid is None:
            continue
        conv = core._conv_by_cid[brief["cid"]]
        parent = core._conv_by_cid.get(parent_cid)
        if parent is None:
            raise SnapshotError(
                f"对话 {brief['cid']} 的父对话 cid={parent_cid} 不在快照里"
            )
        conv.parent = parent


def _rebuild_listeners(core, doc: dict) -> None:
    from .events import drop_removed_condition_listener

    listeners = doc.get("listeners") or []
    if not isinstance(listeners, list):
        raise SnapshotError("快照 listeners 段必须是数组")
    core.events.listeners = []
    core.events._next_listener_id = 1
    core.events._next_event_id = 1
    for data in listeners:
        try:
            listener = listener_from_snapshot(data)
        except (TypeError, ValueError) as e:
            raise SnapshotError(f"快照里的监听器记录不合法：{e}") from e
        # 旧快照里依赖节点所有权 cid 的条件形态已删除：丢弃该监听器并告警，不中断启动。
        if drop_removed_condition_listener(listener, "重启快照"):
            continue
        core.events.listeners.append(listener)
        listener_id = listener.get("listener_id")
        if isinstance(listener_id, int) and listener_id >= core.events._next_listener_id:
            core.events._next_listener_id = listener_id + 1


def _restore_security_and_pointer(core, doc: dict) -> None:
    security = doc.get("security") or {}
    roots = security.get("roots") or {}
    core.root_registry = {
        name: tuple(segments) for name, segments in roots.items()
    }
    # 旧快照可能带 security.workspace_region：内核已不分配工作空间，按未知字段忽略。
    para_ref = doc.get("para_ref")
    if isinstance(para_ref, str) and para_ref:
        core.para_ref = para_ref
    next_cid = doc.get("next_cid")
    if not isinstance(next_cid, int):
        next_cid = (max(core._conv_by_cid) + 1) if core._conv_by_cid else 0
    core._next_cid = next_cid

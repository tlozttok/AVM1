"""信息查询设备：只读，向对话暴露 VM 内部状态（类似 /proc）。"""

from .exceptions import VMMemoryError
from .memory_device import MemoryDevice


def _state_of(core, cid):
    if core._active_cid == cid:
        return "active"
    if cid in core._dormant_cids:
        return "dormant"
    if cid in core._ready_cids:
        return "ready"
    return "finished"


class ConversationsDevice(MemoryDevice):
    """$MEM.conversations：对话列表；子路径 <cid> 查单个对话。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        lines = []
        for cid, conv in self._core._conv_by_cid.items():
            lines.append(
                f"  cid={cid} {conv.identity} state={_state_of(self._core, cid)} "
                f"para={conv.para_ref} python={conv.is_python}"
            )
        return "对话列表:\n" + ("\n".join(lines) if lines else "（无）")

    def resolve_path(self, path):
        if len(path) != 1:
            raise VMMemoryError("$MEM.conversations 只支持单个 cid 子路径")
        try:
            cid = int(path[0])
        except ValueError:
            raise VMMemoryError(f"cid 必须是数字: {path[0]}")
        conv = self._core._conv_by_cid.get(cid)
        if conv is None:
            raise VMMemoryError(f"对话 {cid} 不存在")
        return (
            f"cid={cid} name={conv.name} identity={conv.identity} "
            f"state={_state_of(self._core, cid)} is_sub={conv.is_sub} "
            f"is_python={conv.is_python} para={conv.para_ref} messages={len(conv.messages)}"
        )


class SchedulerDevice(MemoryDevice):
    """$MEM.scheduler：调度状态（活跃/就绪/休眠 + 指令预算）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return (
            f"active={self._core._active_cid}\n"
            f"ready={self._core._ready_cids}\n"
            f"dormant={self._core._dormant_cids}\n"
            f"instruction_budget={self._core._instruction_budget} "
            f"(当前链计数={self._core._activation_instructions})"
        )

    def resolve_path(self, path):
        if len(path) != 1 or path[0] not in ("active", "ready", "dormant", "budget"):
            raise VMMemoryError("$MEM.scheduler 支持 active/ready/dormant/budget")
        return {
            "active": self._core._active_cid,
            "ready": list(self._core._ready_cids),
            "dormant": list(self._core._dormant_cids),
            "budget": f"{self._core._activation_instructions}/{self._core._instruction_budget}",
        }[path[0]]


class IccDevice(MemoryDevice):
    """$MEM.icc：待处理 ICC 记录摘要。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        if not self._core._icc:
            return "暂无待处理 ICC 记录"
        lines = [
            f"  {icc_id} → caller={rec['caller_cid']} callee={rec.get('callee_cid')} mode={rec.get('mode', 'tool')}"
            for icc_id, rec in self._core._icc.items()
        ]
        return "ICC 记录:\n" + "\n".join(lines)


class MemorySummaryDevice(MemoryDevice):
    """$MEM.memory：内存树摘要（顶层键 + 节点统计）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        from .types import MetaDict, MetaList

        root = self._core.mem._data
        counts = {"dict": 0, "list": 0, "str": 0}

        def walk(v):
            if isinstance(v, MetaDict):
                counts["dict"] += 1
                for x in v.values():
                    walk(x)
            elif isinstance(v, MetaList):
                counts["list"] += 1
                for x in v:
                    walk(x)
            else:
                counts["str"] += 1

        walk(root)
        return f"内存树: 顶层键={list(root.keys())} 节点统计={counts}"

    def resolve_path(self, path):
        raise VMMemoryError("$MEM.memory 只读摘要；读取具体路径请用 memory_read")


class MonitorDevice(MemoryDevice):
    """$MEM.monitor：监测帧摘要；子路径 <seq> 查单帧。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        frames = self._core.monitor.frames()
        if not frames:
            return "监测器无帧"
        last = frames[-1]
        return (
            f"监测帧: {len(frames)} 帧（基线+{len(frames) - 1}轮）\n"
            f"最近 seq={last.seq} active={last.sched['active_cid']} "
            f"tools={[t.get('cmd_type') for t in (last.lmu.get('tool_calls') or [])]}"
        )

    def resolve_path(self, path):
        if len(path) != 1:
            raise VMMemoryError("$MEM.monitor 只支持单帧号子路径")
        try:
            seq = int(path[0])
        except ValueError:
            raise VMMemoryError(f"帧号必须是数字: {path[0]}")
        f = self._core.monitor.frame(seq)
        if f is None:
            raise VMMemoryError(f"帧 {seq} 不存在")
        return (
            f"seq={f.seq} active={f.sched['active_cid']} "
            f"ready={f.sched['ready_cids']} dormant={f.sched['dormant_cids']} "
            f"result={f.lmu.get('result')!r} error={f.lmu.get('error')}"
        )

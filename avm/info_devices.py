"""信息查询设备：只读，向对话暴露 VM 内部状态（类似 /proc）。"""

import json

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
        """按状态分组列出存活对话；finished 与运行态分开，cid 空洞不做伪造补全。

        宿主/面板保留 cid（PANEL_CID）不列出：它对普通对话不可见。
        """
        groups = {}
        for cid, conv in self._core._conv_by_cid.items():
            if self._core.is_hidden_cid(cid):
                continue
            groups.setdefault(_state_of(self._core, cid), []).append((cid, conv))
        lines = ["对话列表（按状态分组；cid 空洞表示已关闭或已回收的个体，不保留记录）："]
        for state in ("active", "ready", "dormant", "finished"):
            entries = groups.get(state)
            if not entries:
                continue
            lines.append(f"  {state}:")
            for cid, conv in entries:
                lines.append(
                    f"    cid={cid} {conv.identity} para={conv.para_ref} python={conv.is_python}"
                )
        return "\n".join(lines)

    def resolve_path(self, path):
        if len(path) != 1:
            raise VMMemoryError("$MEM.conversations 只支持单个 cid 子路径")
        try:
            cid = int(path[0])
        except ValueError:
            raise VMMemoryError(f"cid 必须是数字: {path[0]}")
        conv = self._core._conv_by_cid.get(cid)
        if conv is None or self._core.is_hidden_cid(cid):
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

    def _visible(self, cid):
        return None if self._core.is_hidden_cid(cid) else cid

    def _visible_list(self, cids):
        return [cid for cid in cids if not self._core.is_hidden_cid(cid)]

    def to_llm_string(self):
        return (
            f"active={self._visible(self._core._active_cid)}\n"
            f"ready={self._visible_list(self._core._ready_cids)}\n"
            f"dormant={self._visible_list(self._core._dormant_cids)}\n"
            f"instruction_budget={self._core._instruction_budget} "
            f"(当前链计数={self._core._activation_instructions})"
        )

    def resolve_path(self, path):
        if len(path) != 1 or path[0] not in ("active", "ready", "dormant", "budget"):
            raise VMMemoryError("$MEM.scheduler 支持 active/ready/dormant/budget")
        return {
            "active": self._visible(self._core._active_cid),
            "ready": self._visible_list(self._core._ready_cids),
            "dormant": self._visible_list(self._core._dormant_cids),
            "budget": f"{self._core._activation_instructions}/{self._core._instruction_budget}",
        }[path[0]]


class MemorySummaryDevice(MemoryDevice):
    """$MEM.memory：内存树摘要（顶层键 + 节点统计）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        from .types import MetaDict, MetaList, LinkNode

        root = self._core.mem._data
        counts = {"dict": 0, "list": 0, "str": 0, "link": 0}

        def walk(v):
            if isinstance(v, MetaDict):
                counts["dict"] += 1
                for x in v.values():
                    walk(x)
            elif isinstance(v, MetaList):
                counts["list"] += 1
                for x in v:
                    walk(x)
            elif isinstance(v, LinkNode):
                # 链接只计数，不跟随：摘要是只读展示，跟随会带来寻址副作用
                counts["link"] += 1
            else:
                counts["str"] += 1

        walk(root)
        return f"内存树: 顶层键={list(root.keys())} 节点统计={counts}"

    def resolve_path(self, path):
        raise VMMemoryError("$MEM.memory 只读摘要；读取具体路径请用 memory_read")


class ListenersDevice(MemoryDevice):
    """$MEM.listeners：事件监听器列表；子路径 <listener_id> 查单条详情。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    @staticmethod
    def _format(l):
        text = (
            f"listener_id={l['listener_id']} owner_cid={l['owner_cid']} "
            f"event_type={l['event_type']} phase={l.get('phase', 'all')} "
            f"target={l['target']} handler_ref={l['handler_ref']} "
            f"enabled={l.get('enabled', True)}"
        )
        if l.get("error_count"):
            text += f" error_count={l['error_count']} last_error={l.get('last_error', '')}"
        return text

    def to_llm_string(self):
        lines = ["  " + self._format(l) for l in self._core.events.listeners]
        return "监听器列表:\n" + ("\n".join(lines) if lines else "（无）")

    def resolve_path(self, path):
        if len(path) != 1:
            raise VMMemoryError("$MEM.listeners 只支持单个 listener_id 子路径")
        try:
            listener_id = int(path[0])
        except ValueError:
            raise VMMemoryError(f"listener_id 必须是数字: {path[0]}")
        for l in self._core.events.listeners:
            if l["listener_id"] == listener_id:
                return self._format(l)
        raise VMMemoryError(f"监听器 {listener_id} 不存在")


class MonitorDevice(MemoryDevice):
    """$MEM.monitor：监测帧摘要。单帧细节在 frames.<seq>（不写进设备自述）。"""

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
            f"tools={[t.get('cmd_type') for t in (last.lmu.get('tool_calls') or [])]} "
            f"events={len(last.lmu.get('events') or [])}"
        )

    def resolve_path(self, path):
        """单帧细节：frames.<seq>。

        不在设备自述/to_llm_string 里宣传这个路径：先保证功能全面，读取方式的
        宣传面留给后续安全批次决定。
        """
        if len(path) != 2 or path[0] != "frames":
            raise VMMemoryError("$MEM.monitor 只提供汇总读取")
        try:
            seq = int(path[1])
        except ValueError:
            raise VMMemoryError(f"帧号必须是数字: {path[1]}")
        f = self._core.monitor.frame(seq)
        if f is None:
            raise VMMemoryError(f"帧 {seq} 不存在")
        return (
            f"seq={f.seq} active={self._core.is_hidden_cid(f.sched['active_cid']) and None or f.sched['active_cid']} "
            f"ready={[c for c in f.sched['ready_cids'] if not self._core.is_hidden_cid(c)]} "
            f"dormant={[c for c in f.sched['dormant_cids'] if not self._core.is_hidden_cid(c)]} "
            f"result={f.lmu.get('result')!r} error={f.lmu.get('error')} "
            f"events={json.dumps(f.lmu.get('events') or [], ensure_ascii=False)}"
        )



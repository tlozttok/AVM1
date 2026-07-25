"""AVM 调试工具集"""

import json
from typing import List, Dict, Any, Optional
from .core import Core


class DebugTracer:

    def __init__(self, core: Core):
        self.core: Core = core
        self.history: List[Dict[str, Any]] = []
        self._step_count: int = 0

    def step(self) -> bool:
        if self.core._active_cid is None:
            return False
        conv = self.core._conv_by_cid[self.core._active_cid]
        self._record(f"STEP-{self._step_count:03d} BEFORE | cid={conv.cid}")
        self.core.advance_conversation()
        self._record(f"STEP-{self._step_count:03d} AFTER  | cid={conv.cid}")
        self._step_count += 1
        return True

    def run(self) -> None:
        while self.step():
            pass

    def last_diff(self, max_width: int = 80) -> str:
        if len(self.history) < 2:
            return "(历史不足 2 条，无法 diff)"
        before = self.history[-2]
        after = self.history[-1]
        lines = ["=" * max_width, f"DIFF: {before['label']} -> {after['label']}", "-" * max_width]

        b_conv = before["conversations"]
        a_conv = after["conversations"]
        if b_conv != a_conv:
            lines.append(f"[conversations] {len(b_conv)} -> {len(a_conv)}")

        b_mem = before["mem_keys"]
        a_mem = after["mem_keys"]
        if b_mem != a_mem:
            added = a_mem - b_mem
            removed = b_mem - a_mem
            if added:
                lines.append(f"[mem] +keys: {added}")
            if removed:
                lines.append(f"[mem] -keys: {removed}")

        return "\n".join(lines)

    def dump_history(self, path: Optional[str] = None) -> str:
        data = {"total_steps": self._step_count, "snapshots": self.history}
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        lines = ["=" * 60, "AVM 状态摘要", "=" * 60]
        lines.append(f"active: {self.core._active_cid}")
        lines.append(f"ready: {self.core._ready_cids}")
        lines.append(f"dormant: {self.core._dormant_cids}")
        lines.append(f"conversations ({len(self.core._conv_by_cid)}):")
        for cid, conv in self.core._conv_by_cid.items():
            b = conv.user_batch
            lines.append(f"  [{cid}] {len(conv.messages)} msgs, {len(b.tool_responses)} tool_resp, {len(b.user_contents)} user_content")
        lines.append(f"mem top-level keys: {list(self.core.mem._data.keys())}")
        lines.append(f"mounted devices: {list(self.core.mem._devices.keys())}")
        return "\n".join(lines)

    def _record(self, label: str):
        self.history.append({
            "label": label,
            "active_cid": self.core._active_cid,
            "ready_cids": list(self.core._ready_cids),
            "dormant_cids": list(self.core._dormant_cids),
            "conversations": {cid: self._conv_summary(c) for cid, c in self.core._conv_by_cid.items()},
            "mem_keys": set(self.core.mem._data.keys()),
            "mem_devices": list(self.core.mem._devices.keys()),
        })

    @staticmethod
    def _conv_summary(conv) -> str:
        roles = [m.role for m in conv.messages]
        return f"Conversation(cid={conv.cid}, roles={roles})"


def inspect_core(core: Core, title: str = "CORE INSPECT") -> str:
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    lines.append(f"active: {core._active_cid}")
    lines.append(f"ready: {core._ready_cids}")
    lines.append(f"dormant: {core._dormant_cids}")
    lines.append(f"conversations: {len(core._conv_by_cid)}")
    for cid, c in core._conv_by_cid.items():
        msgs = [(m.role, m.content[:40]) for m in c.messages]
        b = c.user_batch
        lines.append(f"  [{cid}] msgs={msgs}, tools={b.tool_responses}, users={b.user_contents}")
    lines.append(f"mem top keys: {list(core.mem._data.keys())}")
    lines.append(f"devices: {list(core.mem._devices.keys())}")
    lines.append("=" * 60)
    return "\n".join(lines)

"""AVM 调试工具集

提供步进执行、状态快照、执行历史记录等能力。
用法示例:
    from debug_utils import DebugTracer
    tracer = DebugTracer(core)
    while tracer.step():
        print(tracer.last_diff())
"""

import json
from typing import List, Dict, Any, Optional
from AVM import Core, parse_instruction, CRT
from memory_device import MemoryDevice


class DebugTracer:
    """指令级调试追踪器
    
    记录每一次指令执行前后的完整状态快照，支持 diff 对比和导出。
    """

    def __init__(self, core: Core):
        self.core = core
        self.history: List[Dict[str, Any]] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """执行一条指令并记录快照。返回 False 表示命令栈已空。"""
        if not self.core.command_stack:
            return False
        raw = self.core.command_stack[-1]
        self._record(f"STEP-{self._step_count:03d} BEFORE | {raw}")
        instruction = parse_instruction(raw)
        return_type = instruction.execute(self.core)
        self._record(f"STEP-{self._step_count:03d} AFTER  | {raw} -> {return_type.name}")
        if return_type == CRT.EXIT:
            self.core.command_stack.pop()
        self._step_count += 1
        return True

    def run(self) -> None:
        """持续执行直到命令栈为空。"""
        while self.step():
            pass

    # ------------------------------------------------------------------
    # 查询 / 格式化
    # ------------------------------------------------------------------

    def last_diff(self, max_width: int = 80) -> str:
        """返回最近两次快照之间的差异摘要。"""
        if len(self.history) < 2:
            return "(历史不足 2 条，无法 diff)"
        before = self.history[-2]
        after = self.history[-1]
        lines = ["=" * max_width, f"DIFF: {before['label']} -> {after['label']}", "-" * max_width]

        # command_stack 变化
        b_stack = before["command_stack"]
        a_stack = after["command_stack"]
        if b_stack != a_stack:
            lines.append(f"[command_stack] {len(b_stack)} -> {len(a_stack)} items")
            if len(b_stack) > 0 and len(a_stack) > 0:
                lines.append(f"  OLD top: {b_stack[-1]}")
                lines.append(f"  NEW top: {a_stack[-1] if a_stack else '(empty)'}")

        # last_msg_reg 变化
        b_lm = before["last_msg_reg"]
        a_lm = after["last_msg_reg"]
        if b_lm != a_lm:
            lines.append(f"[last_msg_reg] {len(b_lm)} -> {len(a_lm)} conversations")

        # usr_tool_reg 变化
        b_ut = before["usr_tool_reg"]
        a_ut = after["usr_tool_reg"]
        if b_ut != a_ut:
            lines.append(f"[usr_tool_reg] {len(b_ut)} -> {len(a_ut)} batches")
            for i, (old, new) in enumerate(zip(b_ut, a_ut)):
                if old != new:
                    lines.append(f"  batch[{i}] changed")
            if len(a_ut) > len(b_ut):
                lines.append(f"  batch[{len(b_ut)}..{len(a_ut)-1}] added")
            if len(b_ut) > len(a_ut):
                lines.append(f"  batch[{len(a_ut)}..{len(b_ut)-1}] removed")

        # mem 变化（仅顶层 key 数量）
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
        """导出完整历史为 JSON 字符串或写入文件。"""
        data = {
            "total_steps": self._step_count,
            "snapshots": self.history,
        }
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def summary(self) -> str:
        """返回当前状态的文本摘要。"""
        lines = ["=" * 60, "AVM 状态摘要", "=" * 60]
        lines.append(f"command_stack ({len(self.core.command_stack)}):")
        for i, cmd in enumerate(reversed(self.core.command_stack)):
            marker = "<<< TOP" if i == 0 else ""
            lines.append(f"  {cmd} {marker}")
        lines.append(f"last_msg_reg ({len(self.core.last_msg_reg)}):")
        for i, conv in enumerate(self.core.last_msg_reg):
            lines.append(f"  [{i}] {len(conv.messages)} messages")
        lines.append(f"usr_tool_reg ({len(self.core.usr_tool_reg)}):")
        for i, batch in enumerate(self.core.usr_tool_reg):
            lines.append(f"  [{i}] {len(batch.tool_responses)} tool_responses, {len(batch.user_contents)} user_contents")
        lines.append(f"mem top-level keys: {list(self.core.mem._data.keys())}")
        lines.append(f"mounted devices: {list(self.core.mem._devices.keys())}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _record(self, label: str):
        self.history.append({
            "label": label,
            "command_stack": list(self.core.command_stack),
            "last_msg_reg": [self._conv_summary(c) for c in self.core.last_msg_reg],
            "usr_tool_reg": [self._batch_summary(b) for b in self.core.usr_tool_reg],
            "mem_keys": set(self.core.mem._data.keys()),
            "mem_devices": list(self.core.mem._devices.keys()),
        })

    @staticmethod
    def _conv_summary(conv) -> str:
        roles = [m.role for m in conv.messages]
        return f"Conversation(roles={roles})"

    @staticmethod
    def _batch_summary(batch) -> str:
        return f"Batch(tr={len(batch.tool_responses)}, uc={len(batch.user_contents)})"


# ---------------------------------------------------------------------------
# 便捷的 inspect 函数
# ---------------------------------------------------------------------------

def inspect_core(core: Core, title: str = "CORE INSPECT") -> str:
    """一次性打印 Core 的当前完整状态。"""
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    lines.append(f"command_stack  : {core.command_stack}")
    lines.append(f"last_msg_reg   : {len(core.last_msg_reg)} conversation(s)")
    for i, c in enumerate(core.last_msg_reg):
        msgs = [(m.role, m.content[:40]) for m in c.messages]
        lines.append(f"  [{i}] {msgs}")
    lines.append(f"usr_tool_reg   : {len(core.usr_tool_reg)} batch(es)")
    for i, b in enumerate(core.usr_tool_reg):
        lines.append(f"  [{i}] tools={b.tool_responses}, users={b.user_contents}")
    lines.append(f"mem top keys   : {list(core.mem._data.keys())}")
    lines.append(f"devices        : {list(core.mem._devices.keys())}")
    lines.append("=" * 60)
    return "\n".join(lines)

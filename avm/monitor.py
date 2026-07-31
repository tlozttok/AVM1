"""AVM 变量监测器：按内核行为循环记录步进帧

一周期 = 一次对话推进（advance_conversation，含一次 LMU 调用）。
每轮推进结束后生成一帧（Frame），帧内是全部被监测变量在该轮结束后的值。
第 0 帧为基线（run 启动后、第一轮推进前）。

双存储：
- 内存环形缓冲：帧序列，消息/输出截断
- 全文文件（可选）：每次 API 调用的完整输入/输出，按帧号分段
"""

import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional


def _truncate(text: Any, limit: int) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "..."
    return text


@dataclass
class Frame:
    """一轮推进结束后的状态向量"""
    seq: int
    ts: float
    lmu: Dict[str, Any]                 # result / tool_calls / elapsed_ms / error
    sched: Dict[str, Any]               # active_cid / ready_cids / dormant_cids / next_cid
    conversations: Dict[int, Dict[str, Any]]  # cid -> state / msgs / batch


class Monitor:
    """按内核循环周期记录步进帧，并提供帧查询、轨迹、diff、查找接口"""

    def __init__(
        self,
        max_frames: int = 2000,
        transcript_path: Optional[str] = None,
        max_item_len: int = 200,
    ):
        self._frames: Deque[Frame] = deque(maxlen=max_frames)
        self._next_seq: int = 0
        self._transcript_path = transcript_path
        self._max_item_len = max_item_len

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record_baseline(self, core) -> None:
        """run 启动后、第一轮推进前记录基线帧"""
        self._frames.append(self._build_frame(core))
        self._next_seq += 1

    def record(self, core, error: Optional[BaseException] = None) -> None:
        """一轮推进结束时调用：写全文段（可选）+ 追加一帧"""
        last_call = getattr(core.lmu, "last_call", None)
        if self._transcript_path and last_call is not None:
            self._write_transcript(self._next_seq, last_call)
        self._frames.append(self._build_frame(core, last_call=last_call, error=error))
        self._next_seq += 1

    # ------------------------------------------------------------------
    # 帧构建
    # ------------------------------------------------------------------

    def _build_frame(
        self,
        core,
        last_call: Optional[dict] = None,
        error: Optional[BaseException] = None,
    ) -> Frame:
        if last_call:
            lmu = {
                "result": _truncate(last_call.get("result"), self._max_item_len),
                "tool_calls": [self._summarize_tc(tc) for tc in (last_call.get("tool_calls") or [])],
                "elapsed_ms": last_call.get("elapsed_ms"),
                "error": last_call.get("error") or (str(error) if error else None),
            }
        else:
            lmu = {
                "result": None,
                "tool_calls": [],
                "elapsed_ms": None,
                "error": str(error) if error else None,
            }

        sched = {
            "active_cid": core._active_cid,
            "ready_cids": list(core._ready_cids),
            "dormant_cids": list(core._dormant_cids),
            "next_cid": core._next_cid,
        }

        conversations = {}
        for cid, conv in core._conv_by_cid.items():
            if core._active_cid == cid:
                state = "active"
            elif cid in core._dormant_cids:
                state = "dormant"
            elif cid in core._ready_cids:
                state = "ready"
            else:
                state = "finished"
            batch = conv.user_batch
            conversations[cid] = {
                "state": state,
                "msgs": len(conv.messages),
                "batch": {
                    "tool_responses": [_truncate(r.content, self._max_item_len) for r in batch.tool_responses],
                    "user_contents": [_truncate(c, self._max_item_len) for c in batch.user_contents],
                },
            }

        return Frame(seq=self._next_seq, ts=time.time(), lmu=lmu, sched=sched, conversations=conversations)

    @staticmethod
    def _summarize_tc(tc: dict) -> dict:
        args = {}
        for k, v in (tc.get("args") or {}).items():
            args[k] = _truncate(str(v), 80)
        return {"cmd_type": tc.get("cmd_type"), "call_id": tc.get("call_id"), "args": args}

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def frame(self, seq: int) -> Optional[Frame]:
        """按帧号取帧"""
        for f in self._frames:
            if f.seq == seq:
                return f
        return None

    def frames(self) -> List[Frame]:
        """全部帧（按时间顺序）"""
        return list(self._frames)

    def trail(self, path: str) -> List[tuple]:
        """单变量跨帧轨迹，如 "sched.active_cid"、"conversations.2.batch.tool_responses"。
        返回 [(seq, value), ...]
        """
        return [(f.seq, self._get_path(f, path)) for f in self._frames]

    def diff(self, seq_a: int, seq_b: int) -> Dict[str, tuple]:
        """两帧间全部变量变化，返回 {path: (a 值, b 值)}"""
        a, b = self.frame(seq_a), self.frame(seq_b)
        if a is None or b is None:
            raise KeyError(f"帧不存在：{seq_a} / {seq_b}")
        fa, fb = self._flatten(a), self._flatten(b)
        out = {}
        for key in set(fa) | set(fb):
            if fa.get(key) != fb.get(key):
                out[key] = (fa.get(key), fb.get(key))
        return out

    def find(self, pred: Callable[[Frame], bool]) -> List[Frame]:
        """按谓词找帧，如 find(lambda f: f.lmu["error"])"""
        return [f for f in self._frames if pred(f)]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(frame: Frame) -> Dict[str, Any]:
        out = {}
        for k, v in frame.lmu.items():
            out[f"lmu.{k}"] = v
        for k, v in frame.sched.items():
            out[f"sched.{k}"] = v
        for cid, c in frame.conversations.items():
            out[f"conversations.{cid}.state"] = c["state"]
            out[f"conversations.{cid}.msgs"] = c["msgs"]
            out[f"conversations.{cid}.batch.tool_responses"] = list(c["batch"]["tool_responses"])
            out[f"conversations.{cid}.batch.user_contents"] = list(c["batch"]["user_contents"])
        return out

    @classmethod
    def _get_path(cls, frame: Frame, path: str) -> Any:
        return cls._flatten(frame).get(path)

    def _write_transcript(self, seq: int, call: dict) -> None:
        # TODO(issue): 全文文件写入失败（OSError）当前被静默吞掉；需要决定错误如何暴露
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(f"===== FRAME {seq} =====\n")
                f.write(f"model={call.get('model')} elapsed_ms={call.get('elapsed_ms')}\n")
                if call.get("error"):
                    f.write(f"--- error ---\n{call['error']}\n")
                f.write("--- request ---\n")
                f.write(json.dumps(call.get("messages"), ensure_ascii=False, indent=2) + "\n")
                f.write("--- response ---\n")
                if call.get("result") is not None:
                    f.write(str(call["result"]) + "\n")
                if call.get("tool_calls"):
                    f.write(json.dumps(call["tool_calls"], ensure_ascii=False, indent=2) + "\n")
                f.write("\n")
        except OSError:
            pass

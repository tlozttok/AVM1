"""中断向量表：把外部发起的控制权请求转成调度动作。

中断改变"下一个执行谁"，与事件是两回事：事件通知发生了什么，中断夺取执行权。
本模块做两件事——注册向量、在安全点派发；派发时把执行权交给目标，并把 payload
作为正文投进目标的输入批次（中断是新信息，处理对话要知道自己为什么在跑）。
优先级、屏蔽位、以及它与其他中断源（预算中断、事件 notify）的关系都不在这里。

安全点：Core 在一轮推理（含该轮产生的全部指令）执行完之后派发。这与预算中断
用的是同一个边界，已被证明是状态完整、可恢复的。
"""

import queue
import sys
import threading
from typing import Dict, List, Optional, Tuple


class InterruptTable:
    """中断向量表：中断类型 → 目标对话 cid。

    - 任意线程都可以调 `raise_` 挂起中断（控制台的后端线程就是这么做的）；
    - Core 在安全点调 `dispatch` 派发：把执行权交给向量指向的对话，
      被打断的活跃对话插回就绪队列**队首**，让出后它先跑；
    - 派发时把 payload 投进目标对话的 `user_batch`（`raise_` 不给 payload 就不投）。
      目标已是活跃对话时也投：中断是新信息，对话有必要知道。
    """

    def __init__(self, core):
        self._core = core
        self._vectors: Dict[str, int] = {}
        self._pending: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        self._lock = threading.Lock()
        self._pending_types: set = set()
        self.last_dispatch: List[dict] = []  # 最近一次派发的记录，供控制台与监测查看

    # ------------------------------------------------------------------
    # 向量表
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> Dict[str, int]:
        """向量表的只读副本。"""
        return dict(self._vectors)

    def register(self, itype: str, cid: int) -> None:
        if not isinstance(itype, str) or not itype:
            raise ValueError("中断类型必须是非空字符串")
        if not isinstance(cid, int):
            raise ValueError("中断向量的目标必须是对话 cid")
        self._vectors[itype] = cid

    def unregister(self, itype: str) -> None:
        self._vectors.pop(itype, None)

    # ------------------------------------------------------------------
    # 挂起与派发
    # ------------------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    def pending_types(self) -> List[str]:
        with self._lock:
            return sorted(self._pending_types)

    def raise_(self, itype: str, payload: Optional[str] = None) -> None:
        """挂起一个中断。线程安全。

        同类型已有未派发的挂起中断时不重复入队：操作员连点两次暂停与点一次等价。
        payload 是要投给目标对话的正文；不给就不投。payload 怎么产生由发起方决定，
        中断表只管把它送进目标的 user_batch。
        """
        if not isinstance(itype, str) or not itype:
            raise ValueError("中断类型必须是非空字符串")
        with self._lock:
            if itype in self._pending_types:
                return
            self._pending_types.add(itype)
        self._pending.put((itype, payload))

    def dispatch(self) -> List[dict]:
        """派发全部挂起中断，返回派发记录。只应由 Core 在安全点调用。"""
        records = []
        while True:
            try:
                itype, payload = self._pending.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending_types.discard(itype)
            records.append(self._deliver(itype, payload))
        if records:
            self.last_dispatch = records
        return records

    # ------------------------------------------------------------------
    # 单条中断
    # ------------------------------------------------------------------

    def _deliver(self, itype: str, payload: Optional[str]) -> dict:
        core = self._core
        cid = self._vectors.get(itype)
        if cid is None:
            return self._drop(itype, f"没有为 {itype!r} 注册向量")

        target = core._conv_by_cid.get(cid)
        if target is None or cid in core._finished:
            return self._drop(itype, f"目标对话 {cid} 不存在或已关闭")

        # 投递放在 noop 判断之前：switch 与 noop 两条路径都投，dropped 没有目标才不投。
        # 目标已是活跃对话时，这条消息留给它下一轮消费（user_batch 每轮清空）。
        # 中断不走事件系统：它夺取执行权，若事件处理器能拦下投递，就会出现
        # "执行权已经切过去、消息却没送到"的中间态。
        # TODO(issue): payload 的类型与产生方式待作者在完善中断系统时定义；当前按正文文本处理，
        # 非字符串会被 UserMessageBatch.add_user_content 的 str() 转成 Python 字面量。
        if payload is not None:
            target.user_batch.add_user_content(payload)

        current = core._active_cid
        if current == cid:
            return {"type": itype, "target": cid, "action": "noop",
                    "reason": "目标已是活跃对话"}

        preempted = None
        if current is not None and current not in core._finished:
            preempted = current
            if current in core._dormant_cids:
                core._dormant_cids.remove(current)
            if current not in core._ready_cids:
                core._ready_cids.insert(0, current)

        # 与 _pick_next_active 取活跃对话的两步保持一致：把目标从等待队列里摘掉，
        # 并在调用链断开时重新给满指令预算（否则 _activation_instructions 会带着
        # 上一条链的计数进入被中断的对话）。中断夺取执行权改变的就是"下一个执行
        # 谁"，不摘队列会让目标在本轮结束时被当成"自己让出"而立刻交出执行权。
        if cid in core._ready_cids:
            core._ready_cids.remove(cid)
        if cid in core._dormant_cids:
            core._dormant_cids.remove(cid)
        if not core._same_chain(current, cid):
            core._activation_instructions = 0

        core._active_cid = cid
        return {"type": itype, "target": cid, "action": "switch",
                "preempted": preempted, "payload": payload}

    @staticmethod
    def _drop(itype: str, reason: str) -> dict:
        print(f"[AVM] 中断 {itype!r} 丢弃：{reason}", file=sys.stderr)
        return {"type": itype, "action": "dropped", "reason": reason}

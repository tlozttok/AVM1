"""宿主侧 IO 总线：输入与输出各一条有序流，按 seq 增量读。

总线属于宿主，不属于 $MEM：操作面板宿主的 `IOInputDevice` / `IOOutputDevice`
（见 avm/io_devices.py）是它的设备前端，面板断开、终端模式都不影响运行时本身。
标准镜像设备（avm/memory_device.py）不读这条总线，终端宿主行为不变。

形状与 OperatorChannel 一致：有界缓冲、seq 单调递增、丢弃计数进 status()。
"""

import threading
import time
from typing import Any, Dict, List, Optional


class IOBus:
    """两条流：output 只追加供订阅；input 追加后由目标对话按身份取走。"""

    def __init__(self, capacity: int = 4000):
        self._lock = threading.RLock()
        self._capacity = capacity
        self._streams: Dict[str, List[dict]] = {"input": [], "output": []}
        self._next_seq: Dict[str, int] = {"input": 1, "output": 1}
        self._dropped: Dict[str, int] = {"input": 0, "output": 0}
        # 输入是消费型的：一条记录被谁取走就对所有人消失。用全局消费标记，
        # 不按消费者身份分桶——否则对话改名、或用不同身份组合来读，会重复取到同一条。
        self._consumed_input: set = set()

    def _publish(self, stream: str, record: dict) -> dict:
        with self._lock:
            buf = self._streams[stream]
            if len(buf) >= self._capacity:
                removed = buf[: len(buf) - self._capacity + 1]
                del buf[: len(removed)]
                self._dropped[stream] += len(removed)
                if stream == "input":
                    self._consumed_input.difference_update(e["seq"] for e in removed)
            seq = self._next_seq[stream]
            self._next_seq[stream] += 1
            entry = {"seq": seq, "time": time.time(), **record}
            buf.append(entry)
            return entry

    # -- 发布 ----------------------------------------------------------

    def publish_output(self, source: Any, text: str, kind: str = "text",
                       meta: Optional[dict] = None) -> dict:
        """一条输出。source 是产出它的对话标识（cid 或 None）。"""
        return self._publish("output", {
            "kind": kind, "source": source, "target": None,
            "text": str(text), "meta": dict(meta or {}),
        })

    def publish_input(self, target: Any, text: str,
                      meta: Optional[dict] = None) -> dict:
        """一条输入。target 为 None/空/'*' 表示入口对话可读的全局输入。"""
        return self._publish("input", {
            "kind": "text", "source": None,
            "target": None if target in (None, "", "*") else str(target),
            "text": str(text), "meta": dict(meta or {}),
        })

    # -- 增量读 --------------------------------------------------------

    def drain_io(self, since: int = 0) -> List[dict]:
        """输出流增量读：面板重连用 after=<seq> 续读，不消费。"""
        with self._lock:
            return [e for e in self._streams["output"] if e["seq"] > since]

    def drain_input(self, target: Any = None, since: int = 0) -> List[dict]:
        """输入流增量读（观测用，不消费）；target 给 None 表示全部。"""
        with self._lock:
            return [
                e for e in self._streams["input"]
                if e["seq"] > since and (target is None or e["target"] == target)
            ]

    @staticmethod
    def _identity_set(identities: List[Any]) -> set:
        return {str(i) for i in identities if i is not None and str(i) != ""}

    def _find_pending(self, ids: set) -> Optional[dict]:
        for entry in self._streams["input"]:
            if entry["seq"] in self._consumed_input:
                continue
            target = entry["target"]
            if target is not None and target not in ids:
                continue
            return entry
        return None

    def take_input(self, identities: List[Any]) -> Optional[dict]:
        """取走一条发给这些身份的输入；取不到返回 None。

        身份是 [cid, name, identity] 之类的字符串集合。目标为 None（全局）或
        命中任一身份的记录都算匹配；按流顺序取第一条未被消费的，取走即全局标记。
        """
        ids = self._identity_set(identities)
        with self._lock:
            entry = self._find_pending(ids)
            if entry is not None:
                self._consumed_input.add(entry["seq"])
            return entry

    def peek_input(self, identities: List[Any]) -> Optional[dict]:
        """看一条发给这些身份、还没取走的输入（不消费）。"""
        with self._lock:
            return self._find_pending(self._identity_set(identities))

    def pending_input(self, identities: List[Any]) -> int:
        """还有几条发给这些身份的输入没被取走（不消费）。"""
        ids = self._identity_set(identities)
        with self._lock:
            return sum(
                1 for e in self._streams["input"]
                if e["seq"] not in self._consumed_input
                and (e["target"] is None or e["target"] in ids)
            )

    # -- 状态 ----------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "input": {
                    "next_seq": self._next_seq["input"],
                    "buffered": len(self._streams["input"]),
                    "dropped": self._dropped["input"],
                },
                "output": {
                    "next_seq": self._next_seq["output"],
                    "buffered": len(self._streams["output"]),
                    "dropped": self._dropped["output"],
                },
            }

"""操作面板宿主专用的 IO 设备：宿主侧 `IOBus` 的 $MEM 前端。

与 `avm.memory_device.InputsListDevice` / `OutputsListDevice` 完全独立。那两台是镜像
层的标准设备（终端打印、终端 input()、调试队列），本文件的两台只在操作面板宿主里由
`Console` 挂载，只认 `core.io_bus`，不碰终端、不做终端退化。

挂载点沿用 `inputs` / `outputs`，读法不变（`$MEM.inputs.next`、
`$MEM.outputs.out`），因此镜像里的指令与镜像声明都不用改：镜像装载时仍按
`devices` 段挂标准设备，操作面板宿主在 `Console` 里把它们替换成这两台。
非面板宿主（含标准 CLI）不构造 `Console`，标准设备原样工作。

`debug_actions` 保留与标准输入设备同名的两条动作（"投喂一行输入" / "查询队列"），
调试抽屉与既有 e2e 的投喂路径因此不需要改动。
"""

import threading
from typing import Any

from .exceptions import MemoryIndexOutOfRangeError, MemoryTypeError, VMMemoryError
from .memory_device import MetaListDevice


class IOInputDevice(MetaListDevice):
    """`$MEM.inputs`：从宿主 IO 总线取一条发给当前对话的输入。

    - `next`  阻塞等到一条输入并追加进列表（调试队列优先，其次总线定向输入）。
    - `peek`  非阻塞看一眼，取不到返回空串。
    - `pending` 还没被取走的输入条数。

    等待发生在 `Condition` 上，由 `/api/io/input` 发布输入后调 `notify_input()` 唤醒。
    """

    def __init__(self, data=None, metadata=None):
        super().__init__(data=data, metadata=metadata)
        self._core = None
        self._feed_queue: list = []
        self._cv = threading.Condition()
        self._waiting = False

    def attach_core(self, core) -> None:
        self._core = core

    # -- 总线与身份 ----------------------------------------------------

    def _bus(self):
        return getattr(self._core, "io_bus", None) if self._core is not None else None

    def _identities(self) -> list:
        """当前对话的可寻址身份：cid、名字、identity。"""
        if self._core is None or getattr(self._core, "_active_cid", None) is None:
            return []
        conv = self._core._conv_by_cid.get(self._core._active_cid)
        if conv is None:
            return []
        return [str(self._core._active_cid), getattr(conv, "name", None),
                getattr(conv, "identity", None)]

    # -- 调试动作（与标准输入设备同名同形）----------------------------

    def debug_actions(self) -> list:
        def feed(args):
            content = args.get("content")
            if not isinstance(content, str):
                raise ValueError("content 必须是字符串")
            with self._cv:
                self._feed_queue.append(content)
                self._cv.notify_all()
            return {"queued": len(self._feed_queue)}

        def pending(args):
            bus = self._bus()
            with self._cv:
                queued = len(self._feed_queue)
            bus_pending = (bus.pending_input(self._identities())
                           if bus is not None else 0)
            return {"queued": queued + bus_pending, "waiting": self._waiting}

        return [
            {
                "name": "投喂一行输入",
                "description": "把一行文本排进输入队列；下一次 $MEM.inputs.next 直接取走它。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "要交给对话的一行文本"},
                    },
                    "required": ["content"],
                },
                "run": feed,
            },
            {
                "name": "查询队列",
                "description": "看队列里还有几行、设备当前是否正等着输入。",
                "parameters": {"type": "object", "properties": {}},
                "run": pending,
            },
        ]

    def notify_input(self) -> None:
        """宿主总线来了输入：唤醒可能正等在 `next` 上的读。"""
        with self._cv:
            self._cv.notify_all()

    # -- 读 ------------------------------------------------------------

    def __getitem__(self, index):
        if index == "next":
            return self._request_input()
        if index == "peek":
            bus = self._bus()
            entry = bus.peek_input(self._identities()) if bus is not None else None
            return entry["text"] if entry is not None else ""
        if index == "pending":
            bus = self._bus()
            with self._cv:
                queued = len(self._feed_queue)
            return queued + (bus.pending_input(self._identities())
                             if bus is not None else 0)
        index = self._parse_index(index)
        if isinstance(index, int) and index < 0:
            raise MemoryIndexOutOfRangeError(
                "IOInputDevice 不支持负下标；请求下一行输入请读 $MEM.inputs.next"
            )
        return super().__getitem__(index)

    def _request_input(self):
        self._waiting = True
        try:
            value = self._next_value()
        finally:
            self._waiting = False
        # 读取成功才追加：读失败时列表不应多出一个元素
        self._data.append(value)
        return value

    def _next_value(self) -> str:
        with self._cv:
            while True:
                if self._feed_queue:
                    return self._feed_queue.pop(0)
                bus = self._bus()
                entry = bus.take_input(self._identities()) if bus is not None else None
                if entry is not None:
                    return entry["text"]
                self._cv.wait()

    # -- 与标准列表设备一致的其余接口 --------------------------------

    def __setitem__(self, index, value):
        raise MemoryTypeError("IOInputDevice 是只读的，不允许写入")

    def set_value(self, value):
        raise MemoryTypeError("IOInputDevice 是只读的，不允许写入")

    def resolve_path(self, path: list) -> Any:
        if len(path) == 1:
            return self.__getitem__(path[0])
        raise VMMemoryError("IOInputDevice 不支持多级子路径")

    def to_llm_string(self) -> str:
        if self._waiting:
            return "[等待用户输入...]"
        if not self._data:
            return f"inputs[list_len=0,metadata={self._metadata!r}]"
        return f"inputs[list_len={len(self._data)}, recent={self._data[-3:]!r},metadata={self._metadata!r}]"


class IOOutputDevice(MetaListDevice):
    """`$MEM.outputs`：写入即发布到宿主 IO 总线，绝不打印到终端。

    - `out`    追加一条普通输出。
    - `notify` 追加一条通知输出（kind=notify，面板据此高亮/响铃）。
    """

    def __init__(self, data=None, metadata=None):
        super().__init__(data=data, metadata=metadata)
        self._core = None

    def attach_core(self, core) -> None:
        self._core = core

    def _publish(self, value: str, notify: bool) -> None:
        bus = getattr(self._core, "io_bus", None) if self._core is not None else None
        if bus is None:
            return
        bus.publish_output(
            source=getattr(self._core, "_active_cid", None),
            text=value,
            kind="notify" if notify else "text",
            meta={"notify": True} if notify else None,
        )

    def __getitem__(self, index):
        index = self._parse_index(index)
        if isinstance(index, int) and index < 0:
            raise MemoryIndexOutOfRangeError(
                "IOOutputDevice 不支持负下标；输出文本请写 $MEM.outputs.out"
            )
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if index == "out":
            self.append(value)
            return
        if index == "notify":
            self.append(value, notify=True)
            return
        raise MemoryTypeError("IOOutputDevice 只支持命名写入 $MEM.outputs.out / notify")

    def append(self, value, notify: bool = False):
        if not isinstance(value, str):
            raise MemoryTypeError(f"IOOutputDevice 只接受 str，got {type(value).__name__}")
        self._data.append(value)
        self._publish(value, notify)
        # 不打印：操作面板的输出由 /api/io/stream 订阅，终端不是本设备的出口。

    def set_value(self, value):
        if isinstance(value, list):
            self._data = [str(v) for v in value]
        elif isinstance(value, str):
            self.append(value)
        else:
            raise MemoryTypeError(
                f"IOOutputDevice 只接受 str 或 list，got {type(value).__name__}"
            )

    def to_llm_string(self) -> str:
        return f"outputs[list_len={len(self._data)}, items={self._data!r},metadata={self._metadata!r}]"

    def resolve_path(self, path: list) -> Any:
        if len(path) == 1:
            return self.__getitem__(path[0])
        raise VMMemoryError("IOOutputDevice 不支持多级子路径")

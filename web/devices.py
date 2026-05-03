"""Web 版 IO 设备 — 通过 Event/Queue 与 FastAPI 通信，不阻塞 stdin/stdout"""
import queue
import threading

from avm.memory_device import InputsListDevice, OutputsListDevice


class WebInputDevice(InputsListDevice):
    """Web 版输入设备：通过 Event 同步获取用户输入，不阻塞 stdin"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_input = ""
        self._input_ready = threading.Event()
        self._on_input_wait = None   # 回调: () -> None
        self._on_input_received = None  # 回调: (text: str) -> None

    def __getitem__(self, index):
        index = self._parse_index(index)
        if index == -1:
            self._pending_input = True
            if self._on_input_wait:
                try:
                    self._on_input_wait()
                except Exception:
                    pass
            self._input_ready.clear()
            self._input_ready.wait()
            user_input = self._current_input
            self._pending_input = False
            if self._on_input_received:
                try:
                    self._on_input_received(user_input)
                except Exception:
                    pass
            self._data.append(user_input)
            return user_input
        return self._data[index]

    def provide_input(self, text: str):
        self._current_input = text
        self._input_ready.set()


class WebOutputDevice(OutputsListDevice):
    """Web 版输出设备：消息推入 SSE 广播队列，不写 stdout"""

    def __init__(self, output_queues: list, **kwargs):
        super().__init__(**kwargs)
        self._output_queues = output_queues

    def append(self, value):
        if not isinstance(value, str):
            from avm.exceptions import MemoryTypeError
            raise MemoryTypeError(f"OutputsListDevice 只接受 str，got {type(value).__name__}")
        self._data.append(value)
        for q in self._output_queues:
            try:
                q.put_nowait(value)
            except queue.Full:
                pass

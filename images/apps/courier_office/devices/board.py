"""courier_office 的完工记录板设备。

挂载到 $MEM.board。路径语义：
- 读 $MEM.board.latest ：最新一条记录；
- 写 $MEM.board.append ：追加一条 str 记录；
- 数字索引（含 -1）保持普通列表语义：-1 是最后一条，读写都按索引操作，不追加。

设备 after 事件贡献 board_len / latest，供事件监听者使用。
"""
from avm.exceptions import MemoryIndexOutOfRangeError, MemoryTypeError, VMMemoryError
from avm.memory_device import MetaListDevice


class BoardDevice(MetaListDevice):
    """记录板：append 路径追加，数字索引（含 -1）是普通列表读写，latest 读最新。"""

    def _parse_index(self, index):
        if isinstance(index, str):
            try:
                return int(index)
            except ValueError:
                raise MemoryTypeError(f"BoardDevice 索引必须是整数、latest 或 append，got {index!r}")
        return index

    def __getitem__(self, index):
        if index == "latest":
            if not self._data:
                raise MemoryIndexOutOfRangeError("记录板为空，无法读取 latest")
            return self._data[-1]
        return super().__getitem__(self._parse_index(index))

    def __setitem__(self, index, value):
        if index == "append":
            if not isinstance(value, str):
                raise MemoryTypeError(f"BoardDevice.append 只接受 str 记录，got {type(value).__name__}")
            self.append(value)
            return
        index = self._parse_index(index)
        if not isinstance(value, str):
            raise MemoryTypeError(f"BoardDevice 索引写入只接受 str 记录，got {type(value).__name__}")
        self._data[index] = value

    def resolve_path(self, path):
        if len(path) == 1:
            return self.__getitem__(path[0])
        raise VMMemoryError("BoardDevice 只支持单级子路径访问")

    def set_value(self, value):
        """设备整体写入：str 追加一条，list 替换全部记录。"""
        if isinstance(value, str):
            self.append(value)
            return
        if isinstance(value, list):
            self._data = [str(v) for v in value]
            return
        raise MemoryTypeError(f"BoardDevice 只接受 str 或 list，got {type(value).__name__}")

    def after_event(self, event_type, remainder, result):
        if event_type == "memory_write":
            return {
                "board_len": len(self._data),
                "latest": self._data[-1] if self._data else "",
            }
        return None

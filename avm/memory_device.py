"""AVM 内存设备：可挂载到内存路径的虚拟设备"""
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryDevice:
    """设备基类"""

    def pretend_as_type(self) -> str:
        """返回设备假装类型：'str' | 'MetaList' | 'MetaDict'"""
        raise NotImplementedError

    def to_llm_string(self) -> str:
        """返回给 LLM 的字符串表示"""
        raise NotImplementedError


class StringDevice(MemoryDevice):
    """假装是字符串的设备"""

    def __init__(self, value: str = ""):
        self._value = value

    def pretend_as_type(self) -> str:
        return "str"

    def get_value(self) -> str:
        logger.debug("[StringDevice.get_value] -> %r", self._value)
        return self._value

    def set_value(self, value: str) -> None:
        logger.debug("[StringDevice.set_value] %r", value)
        if not isinstance(value, str):
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"StringDevice 只接受 str，got {type(value).__name__}")
        self._value = value

    def to_llm_string(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"StringDevice(value={self._value!r})"


class MetaListDevice(MemoryDevice):
    """假装是 MetaList 的设备"""

    def __init__(self, data=None, metadata=None):
        self._data = data if data is not None else []
        self._metadata = metadata

    def pretend_as_type(self) -> str:
        return "MetaList"

    def __getitem__(self, index: int) -> Any:
        try:
            return self._data[index]
        except IndexError:
            from .exceptions import MemoryIndexOutOfRangeError
            raise MemoryIndexOutOfRangeError(f"索引越界：{index}，列表长度：{len(self._data)}") from None

    def __setitem__(self, index: int, value: Any) -> None:
        self._data[index] = value

    def append(self, value: Any) -> None:
        self._data.append(value)

    def get_metadata(self) -> Optional[str]:
        return self._metadata

    def set_metadata(self, metadata: str) -> None:
        self._metadata = metadata

    def set_value(self, value: list) -> None:
        """整体替换内部数据（用于 Memory.set_by_path 的设备写入）"""
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        if not isinstance(value, list):
            raise MemoryTypeError(f"MetaListDevice 只接受 list，got {type(value).__name__}")
        self._data = value

    def __contains__(self, value) -> bool:
        return value in self._data

    def to_llm_string(self) -> str:
        return f"list[len={len(self._data)},metadata={self._metadata!r}]"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f"MetaListDevice(data={self._data}, metadata={self._metadata})"


class InputsListDevice(MetaListDevice):
    """伪列表用户输入设备
    挂载到 $MEM.inputs，模拟一个不断增长的输入列表。
    - 读取 $MEM.inputs 时返回列表元数据
    - 读取 $MEM.inputs.-1 时阻塞请求用户输入，追加到列表后返回
    - 读取其他索引时，像正常列表一样返回已有元素
    """

    def __init__(self, data=None, metadata=None):
        super().__init__(data=data, metadata=metadata)
        self._pending_input = False

    def _parse_index(self, index):
        if isinstance(index, str):
            try:
                return int(index)
            except ValueError:
                from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
                raise MemoryTypeError(f"InputsListDevice 索引必须是整数，got {index!r}")
        return index

    def __getitem__(self, index):
        index = self._parse_index(index)
        if index == -1:
            # 伪列表语义：读最后一个元素 = 请求新用户输入
            self._pending_input = True
            try:
                print("\n[用户输入请求] 请输入：", end="", flush=True)
                user_input = input()
            except EOFError:
                user_input = ""
            finally:
                self._pending_input = False
            self._data.append(user_input)
            logger.info("[InputsListDevice] received input: %r", user_input)
            return user_input
        # 正常列表访问
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        raise MemoryTypeError("InputsListDevice 是只读的，不允许写入")

    def set_value(self, value):
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        raise MemoryTypeError("InputsListDevice 是只读的，不允许写入")

    def to_llm_string(self) -> str:
        if self._pending_input:
            return "[等待用户输入...]"
        if not self._data:
            return f"inputs[list_len=0,metadata={self._metadata!r}]"
        recent = self._data[-3:]
        return f"inputs[list_len={len(self._data)}, recent={recent!r},metadata={self._metadata!r}]"


class OutputsListDevice(MetaListDevice):
    """伪列表用户输出设备
    挂载到 $MEM.outputs，模拟一个输出列表。
    - 读取 $MEM.outputs 时返回列表元数据
    - 写入 $MEM.outputs.-1 时以追加方式写入，并打印到屏幕
    - 读取其他索引时，像正常列表一样返回已有元素
    - 不可通过索引修改已有元素，只支持追加（通过 -1 写入）
    """

    def _parse_index(self, index):
        if isinstance(index, str):
            try:
                return int(index)
            except ValueError:
                from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
                raise MemoryTypeError(f"OutputsListDevice 索引必须是整数，got {index!r}")
        return index

    def __getitem__(self, index):
        index = self._parse_index(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        index = self._parse_index(index)
        if index == -1:
            self.append(value)
        else:
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError("OutputsListDevice 只支持追加写入（索引 -1），不允许修改已有元素")

    def append(self, value):
        if not isinstance(value, str):
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"OutputsListDevice 只接受 str，got {type(value).__name__}")
        self._data.append(value)
        print(f"\n[Agent 输出] {value}")

    def set_value(self, value):
        """支持直接写入单个字符串（追加）或列表（替换）"""
        if isinstance(value, list):
            self._data = [str(v) for v in value]
        elif isinstance(value, str):
            self.append(value)
        else:
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"OutputsListDevice 只接受 str 或 list，got {type(value).__name__}")

    def to_llm_string(self) -> str:
        return f"outputs[list_len={len(self._data)}, items={self._data!r},metadata={self._metadata!r}]"


class MetaDictDevice(MemoryDevice):
    """假装是 MetaDict 的设备"""

    def __init__(self, data=None, metadata=None):
        self._data = data if data is not None else {}
        self._metadata = metadata

    def pretend_as_type(self) -> str:
        return "MetaDict"

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_metadata(self) -> Optional[str]:
        return self._metadata

    def set_metadata(self, metadata: str) -> None:
        self._metadata = metadata

    def set_value(self, value: dict) -> None:
        """整体替换内部数据（用于 Memory.set_by_path 的设备写入）"""
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        if not isinstance(value, dict):
            raise MemoryTypeError(f"MetaDictDevice 只接受 dict，got {type(value).__name__}")
        self._data = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_llm_string(self) -> str:
        return f"dict[keys={list(self._data.keys())},metadata={self._metadata!r}]"

    def __repr__(self) -> str:
        return f"MetaDictDevice(data={self._data}, metadata={self._metadata})"

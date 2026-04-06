"""AVM 内存设备：可挂载到内存路径的虚拟设备"""
from typing import Any, Optional


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
        return self._value

    def set_value(self, value: str) -> None:
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
        return self._data[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._data[index] = value

    def append(self, value: Any) -> None:
        self._data.append(value)

    def get_metadata(self) -> Optional[str]:
        return self._metadata

    def set_metadata(self, metadata: str) -> None:
        self._metadata = metadata

    def to_llm_string(self) -> str:
        if self._metadata is not None:
            return self._metadata
        return f"list[len={len(self._data)}]"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f"MetaListDevice(data={self._data}, metadata={self._metadata})"


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

    def to_llm_string(self) -> str:
        if self._metadata is not None:
            return self._metadata
        return f"dict[keys={list(self._data.keys())}]"

    def __repr__(self) -> str:
        return f"MetaDictDevice(data={self._data}, metadata={self._metadata})"

"""AVM 类型系统：带元数据的集合类"""


class MetaList:
    """带元数据的列表，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None):
        self._data = data if data is not None else []
        self._metadata = metadata

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def append(self, value):
        self._data.append(value)

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        if self._metadata is not None:
            return self._metadata
        return f"list[len={len(self._data)}]"

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"MetaList(data={self._data}, metadata={self._metadata})"


class MetaDict:
    """带元数据的字典，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None):
        self._data = data if data is not None else {}
        self._metadata = metadata

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def keys(self):
        return self._data.keys()
    
    def get(self, key, default=None):
        return self._data.get(key, default)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        if self._metadata is not None:
            return self._metadata
        return f"dict[keys={list(self._data.keys())}]"

    def  to_dict(self):
        return self._data.copy()

    def __repr__(self):
        return f"MetaDict(data={self._data}, metadata={self._metadata})"

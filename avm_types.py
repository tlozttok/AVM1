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

    def __contains__(self, key):
        return key in self._data

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        return f"list[len={len(self._data)},metadata={self._metadata!r}]"

    def to_list(self):
        """递归转换为普通 list（用于 API 参数等需要纯 JSON 结构的场景）"""
        result = []
        for v in self._data:
            if isinstance(v, MetaDict):
                result.append(v.to_dict())
            elif isinstance(v, MetaList):
                result.append(v.to_list())
            else:
                result.append(v)
        return result

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        if isinstance(other, MetaList):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented

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

    def __delitem__(self, key):
        del self._data[key]

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)

    def copy(self):
        return MetaDict(data=self._data.copy(), metadata=self._metadata)

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

    def __contains__(self, key):
        return key in self._data

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        return f"dict[keys={list(self._data.keys())},metadata={self._metadata!r}]"

    def to_dict(self):
        """递归转换为普通 dict（用于 API 参数等需要纯 JSON 结构的场景）"""
        result = {}
        for k, v in self._data.items():
            if isinstance(v, MetaDict):
                result[k] = v.to_dict()
            elif isinstance(v, MetaList):
                result[k] = v.to_list()
            else:
                result[k] = v
        return result

    def __eq__(self, other):
        if isinstance(other, MetaDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def __repr__(self):
        return f"MetaDict(data={self._data}, metadata={self._metadata})"

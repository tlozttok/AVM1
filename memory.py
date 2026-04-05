"""AVM 内存系统定义"""
from typing import Any, Optional
from avm_types import MetaList, MetaDict


class Memory:
    """
    内存类
    封装内存访问逻辑，支持 $（递归解引用）和 &（一层解引用）
    """

    def __init__(self):
        self._data: dict = {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return self._data.setdefault(key, default)

    def unwrap(self, value: list, for_llm: bool = False) -> Any:
        """
        解引用值
        :param value: 引用列表，如 ['$', 'key'] 或 ['&', 'key', 'subkey']
        :param for_llm: 如果为 True，对 MetaList/MetaDict 返回 to_llm_string()
        :return: 解引用后的值
        """
        if value[0] == "$":
            return self._unwrap_dollar(value[1:], for_llm=for_llm)
        elif value[0] == "&":
            return self._unwrap_ampersand(value[1:])
        else:
            # 无前缀，直接返回路径对应的值
            return self._get_by_path(value)

    def _unwrap_dollar(self, path: list, for_llm: bool = False, seen: Optional[set] = None) -> Any:
        """
        $ 引用：递归解引用
        """
        if seen is None:
            seen = set()

        # 获取路径对应的值
        temp = self._get_by_path(path)

        # 检查是否是字符串且以 $ 开头（需要递归解引用）
        if isinstance(temp, str) and temp.startswith("$"):
            # 防止循环引用
            if temp in seen:
                return temp
            seen.add(temp)
            temp_value = [temp[0], *temp[1:].split(".")]
            return self._unwrap_dollar(temp_value[1:], for_llm=for_llm, seen=seen)

        # 处理类型转换
        return self._convert_for_llm(temp, for_llm)

    def _unwrap_ampersand(self, path: list) -> Any:
        """
        & 引用：一层解引用（不递归）
        """
        temp = self._get_by_path(path)

        # 如果是字符串且以 $ 开头，只解一层
        if isinstance(temp, str) and temp.startswith("$"):
            temp_value = [temp[0], *temp[1:].split(".")]
            temp = self._get_by_path(temp_value[1:])

        return temp

    def _get_by_path(self, path: list) -> Any:
        """
        根据路径获取值
        :param path: 路径列表，如 ['key', 'subkey']
        """
        temp = self._data
        for key in path:
            temp = temp[key]
        return temp

    def _convert_for_llm(self, value: Any, for_llm: bool) -> Any:
        """
        根据类型转换为 LLM 可读的字符串
        """
        if not for_llm:
            return value

        if isinstance(value, MetaList):
            return value.to_llm_string()
        elif isinstance(value, MetaDict):
            return value.to_llm_string()
        elif isinstance(value, list):
            # 约定：列表中第一个元素是元数据
            return value[0] if value else ""
        elif isinstance(value, dict):
            return f"dict.keys:{value.keys()}"
        else:
            return value

    def set_by_path(self, path: list, value: Any) -> None:
        """
        根据路径设置值
        :param path: 路径列表，如 ['key', 'subkey']
        """
        if len(path) == 1:
            self._data[path[0]] = value
            return

        temp = self._data
        for key in path[:-1]:
            if key not in temp:
                temp[key] = {}
            temp = temp[key]
        temp[path[-1]] = value

    def to_dict(self) -> dict:
        """返回底层字典的副本"""
        return self._data.copy()

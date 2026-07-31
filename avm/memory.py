"""AVM 内存系统定义"""
from typing import Any, Optional
from .types import MetaList, MetaDict
from .exceptions import VMMemoryError, MemoryKeyNotFoundError, MemoryIndexOutOfRangeError, MemoryTypeError
from .memory_device import MemoryDevice, StringDevice


def _wrap_value(value: Any) -> Any:
    """将普通 dict/list 递归包装为 MetaDict/MetaList。已包装的也递归检查内部数据。"""
    if isinstance(value, MetaDict):
        for k, v in list(value.items()):
            value[k] = _wrap_value(v)
        return value
    if isinstance(value, MetaList):
        for i in range(len(value)):
            value[i] = _wrap_value(value[i])
        return value
    if isinstance(value, dict):
        return MetaDict(data={k: _wrap_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return MetaList(data=[_wrap_value(v) for v in value])
    return value


class Memory:
    """
    内存类
    封装内存访问逻辑，支持 $（递归解引用）和 &（一层解引用）
    注意：MEM 中只存储 MetaDict/MetaList/str，不会出现普通 dict/list
    """

    def __init__(self):
        self._data: MetaDict = MetaDict(data={})
        self._devices: dict = {}  # 存储已挂载的设备，key 为路径字符串

    def __getitem__(self, key: str) -> Any:
        if key in self._devices:
            return self._devices[key]
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._devices:
            device = self._devices[key]
            if hasattr(device, 'set_value'):
                device.set_value(value)
            else:
                raise VMMemoryError(f"设备 {type(device).__name__} 不支持直接写入")
            return
        self._data[key] = _wrap_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._devices or key in self._data

    def __delitem__(self, key: str) -> None:
        if key in self._devices:
            raise VMMemoryError(f"不能删除已挂载的设备：{key}，请先卸载")
        del self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._devices:
            return self._devices[key]
        return self._data.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self._devices:
            return self._devices[key]
        return self._data.setdefault(key, default)

    def unwrap(self, value: list, for_llm: bool = True) -> Any:
        """
        解引用值
        :param value: 路径列表，如 ['$', 'MEM', 'key'] 或 ['$', 'device', 'subpath']
        """
        if value[0] in ("$", "&"):
            path = value[1:]
        else:
            path = value
        return self.resolve_path(path, for_llm=for_llm)

    def resolve_path(self, path: list, for_llm: bool = False) -> Any:
        if not path:
            return self._convert_for_llm(self._data, for_llm)

        # 不以 MEM 开头 — 必须先命中设备
        if path[0] != "MEM":
            for i in range(len(path)):
                prefix_str = ".".join(path[:i + 1])
                if prefix_str in self._devices:
                    device = self._devices[prefix_str]
                    remaining = path[i + 1:]
                    result = device.resolve_path(remaining) if remaining else device
                    return self._convert_for_llm(result, for_llm)
            raise VMMemoryError(f"未知路径前缀：{path[0]}，只允许 MEM 或已挂载设备")

        # MEM 路径：先检查设备命中，再走数据
        path = path[1:]  # strip MEM
        if not path:
            return self._convert_for_llm(self._data, for_llm)

        return self._resolve_mem_path(path, for_llm=for_llm)

    def _resolve_mem_path(self, path: list, for_llm: bool) -> Any:
        # 检查路径任意前缀是否命中设备
        for i in range(len(path)):
            prefix_str = ".".join(path[:i + 1])
            if prefix_str in self._devices:
                device = self._devices[prefix_str]
                remaining = path[i + 1:]
                result = device.resolve_path(remaining) if remaining else device
                return self._convert_for_llm(result, for_llm)

        # 普通数据访问
        current = self._data
        for key in path:
            self._check_access(current, key, path)
            current = current[key]
        return self._convert_for_llm(current, for_llm)

    @staticmethod
    def _check_access(current, key: str, path: list):
        """检查 key 是否可访问，不可则抛出对应的内存子类异常"""
        from .memory_device import MemoryDevice

        if isinstance(current, MemoryDevice):
            return
        partial = ".".join(str(k) for k in path)
        if isinstance(current, str):
            raise MemoryTypeError(f"字符串没有子节点：{partial}")
        if isinstance(current, (dict, MetaDict)):
            if key not in current:
                raise MemoryKeyNotFoundError(f"键 {key!r} 不存在：{partial}（可用 memory_make 创建）")
        elif isinstance(current, (list, MetaList)):
            try:
                idx = int(key)
            except ValueError:
                raise MemoryTypeError(f"列表索引必须是数字，got {key!r}：{partial}")
            if idx < -len(current) or idx >= len(current):
                raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(current)}：{partial}")
        else:
            raise MemoryTypeError(f"无法访问子路径：父节点是 {type(current).__name__} 类型：{partial}")

    def _convert_for_llm(self, value: Any, for_llm: bool) -> Any:
        """
        根据类型转换为 LLM 可读的字符串
        """
        if not for_llm:
            return value

        # 检查是否是设备
        if isinstance(value, MemoryDevice):
            return value.to_llm_string()

        if isinstance(value, MetaList):
            return value.to_llm_string()
        elif isinstance(value, MetaDict):
            return value.to_llm_string()
        elif isinstance(value, dict):
            raise VMMemoryError(f"MEM 中不应存在普通 dict，请检查写入路径。keys={list(value.keys())[:5]}")
        elif isinstance(value, list):
            raise VMMemoryError(f"MEM 中不应存在普通 list，请检查写入路径。len={len(value)}")
        else:
            return value

    def set_by_path(self, path: list, value: Any) -> None:
        """
        根据路径设置值
        :param path: 路径列表，如 ['key', 'subkey']
        特殊规则：如果 path 终点的现有值是 MetaList/MetaDict，
        则写入被解释为修改其元数据，而不是替换整个对象。
        """
        if not path:
            if isinstance(self._data, MetaDict):
                self._data.set_metadata(str(value))
                return
            self._data = _wrap_value(value)
            return

        # 检查是否是设备路径
        if self.is_device_path(path):
            device = self.get_device(path)
            if hasattr(device, 'set_value'):
                device.set_value(value)
            else:
                raise VMMemoryError(f"设备 {type(device).__name__} 不支持直接写入")
            return

        # 检查路径前缀是否命中设备：如 outputs.-1 中 outputs 是设备
        for i in range(len(path) - 1, 0, -1):
            prefix = path[:i]
            if self.is_device_path(prefix):
                device = self.get_device(prefix)
                current = device
                for key in path[i:-1]:
                    current = current[key]
                last_key = path[-1]
                current[last_key] = value
                return

        # 普通路径处理
        temp = self._data
        for key in path[:-1]:
            try:
                exists = key in temp
            except (TypeError, KeyError):
                exists = False
            if not exists:
                try:
                    temp[key] = MetaDict(data={})
                except TypeError:
                    raise VMMemoryError(f"无法在 {type(temp).__name__} 类型下创建子路径")
            temp = temp[key]

        last_key = path[-1]
        # 如果前缀路径落到了叶子节点（字符串等），无法继续索引
        if isinstance(temp, str):
            prefix_path = ".".join(str(k) for k in path[:-1])
            raise MemoryTypeError(
                f"无法写入：{prefix_path} 是字符串，没有子节点 (.{last_key})"
            )
        if not isinstance(temp, (dict, MetaDict, list, MetaList)):
            prefix_path = ".".join(str(k) for k in path[:-1])
            raise MemoryTypeError(
                f"无法写入：{prefix_path} 是 {type(temp).__name__} 类型，不支持子路径"
            )

        # 列表目标：将 key 转为整数索引
        if isinstance(temp, (list, MetaList)):
            try:
                idx = int(last_key)
            except ValueError:
                raise MemoryTypeError(f"列表索引必须是数字：{last_key}")
            if idx < -len(temp) or idx >= len(temp):
                raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(temp)}")
            temp[idx] = _wrap_value(value)
            return

        # 如果终点已存在且是 MetaList/MetaDict，写入即修改元数据
        if last_key in temp:
            existing = temp[last_key]
            if isinstance(existing, (MetaList, MetaDict)):
                existing.set_metadata(value)
                return

        temp[last_key] = _wrap_value(value)

    def set(self, ref: str, value: Any) -> None:
        """
        根据引用设置值
        :param ref: 引用字符串，如 '$MEM.key' 或 '$MEM.key.subkey'
        :param value: 要设置的值
        """
        if not ref.startswith("$"):
            raise ValueError(f"引用必须以 $ 开头：{ref}")

        # 解析引用路径 - 移除 $MEM 前缀
        parts = ref[1:].split(".")  # 例如：["$MEM", "user_output"] -> ["MEM", "user_output"]
        if parts[0] == "MEM":
            parts = parts[1:]  # 移除 MEM 前缀，得到 ["user_output"]

        self.set_by_path(parts, value)

    def mount(self, path: str, device: MemoryDevice) -> None:
        """
        挂载设备到指定路径
        :param path: 路径字符串，如 'sys.log'（不包含 $MEM 前缀）
        :param device: 要挂载的设备
        """
        if not isinstance(device, MemoryDevice):
            raise VMMemoryError(f"必须挂载 MemoryDevice 类型的设备， got {type(device).__name__}")

        self._devices[path] = device

    def unmount(self, path: str) -> None:
        """
        从指定路径卸载设备
        :param path: 路径字符串
        """
        if path not in self._devices:
            raise VMMemoryError(f"路径 {path} 没有挂载设备")
        del self._devices[path]

    def is_device_path(self, path: list) -> bool:
        """检查路径是否对应设备"""
        path_str = ".".join(path)
        return path_str in self._devices

    def get_device(self, path: list) -> Optional[MemoryDevice]:
        """获取路径对应的设备"""
        path_str = ".".join(path)
        return self._devices.get(path_str)

    def make(self, ref: str, key: str, mem_type: str) -> None:
        """
        在指定路径创建新的内存地址
        :param ref: 引用字符串，如 '$MEM.key' 或 '$MEM.key.subkey'
        :param key: 新地址的名字
        :param mem_type: 类型，'str' | 'dict' | 'list'
        """
        if not ref.startswith("$"):
            raise VMMemoryError(f"引用必须以 $ 开头：{ref}")

        if mem_type not in ('str', 'dict', 'list'):
            raise VMMemoryError(f"不支持的类型：{mem_type}，必须是 'str', 'dict', 'list' 之一")

        parts = ref[1:].split(".")

        # 获取 ref 指向的值
        try:
            current = self.resolve_path(parts, for_llm=False)
        except (VMMemoryError, MemoryKeyNotFoundError, KeyError, TypeError):
            raise VMMemoryError(f"路径不存在：{ref}")

        # 检查类型
        if isinstance(current, str):
            raise VMMemoryError(f"不能在字符串类型的路径下创建新键：{ref}")
        elif isinstance(current, (list, MetaList)):
            if not key.isdigit():
                raise VMMemoryError(f"列表类型的路径下，key 必须是数字索引：{key}")
            index = int(key)
            if index < 0 or index >= len(current):
                raise VMMemoryError(f"索引越界：{index}，列表长度：{len(current)}")
            if mem_type == 'str':
                current[index] = ""
            elif mem_type == 'dict':
                current[index] = MetaDict(data={})
            elif mem_type == 'list':
                current[index] = MetaList(data=[])
        elif isinstance(current, (dict, MetaDict)):
            if mem_type == 'str':
                current[key] = ""
            elif mem_type == 'dict':
                current[key] = MetaDict(data={})
            elif mem_type == 'list':
                current[key] = MetaList(data=[])
        else:
            raise VMMemoryError(f"不支持在类型 {type(current).__name__} 下创建新键")

    def to_dict(self) -> dict:
        """返回底层字典的副本"""
        return self._data.copy()

    def save(self, filepath: str) -> None:
        """将内存数据持久化到 JSON 文件（仅 _data，不含设备）"""
        import json

        def _serialize(value):
            if isinstance(value, MetaDict):
                return {
                    "__type": "MetaDict",
                    "meta": value.get_metadata(),
                    "data": {k: _serialize(v) for k, v in value.items()},
                }
            if isinstance(value, MetaList):
                return {
                    "__type": "MetaList",
                    "meta": value.get_metadata(),
                    "data": [_serialize(v) for v in value],
                }
            if isinstance(value, str):
                return value
            return str(value)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(_serialize(self._data), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Memory":
        """从 JSON 文件加载内存数据并返回 Memory 实例"""
        import json, os
        from .types import MetaList, MetaDict

        mem = cls()
        if not os.path.isfile(filepath):
            return mem

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _deserialize(value):
            if isinstance(value, dict) and value.get("__type") == "MetaDict":
                return MetaDict(
                    data={k: _deserialize(v) for k, v in value["data"].items()},
                    metadata=value.get("meta"),
                )
            if isinstance(value, dict) and value.get("__type") == "MetaList":
                return MetaList(
                    data=[_deserialize(v) for v in value["data"]],
                    metadata=value.get("meta"),
                )
            return value

        mem._data = _deserialize(raw)
        return mem

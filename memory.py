"""AVM 内存系统定义"""
from typing import Any, Optional
from avm_types import MetaList, MetaDict
from exceptions import VMMemoryError
from memory_device import MemoryDevice, StringDevice
import logging

logger = logging.getLogger(__name__)


class Memory:
    """
    内存类
    封装内存访问逻辑，支持 $（递归解引用）和 &（一层解引用）
    """

    def __init__(self):
        self._data: dict = {}
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
        self._data[key] = value

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
        :param value: 引用列表，如 ['$', 'key'] 或 ['&', 'key', 'subkey']
        :param for_llm: 如果为 True，对 MetaList/MetaDict 返回 to_llm_string()
        :return: 解引用后的值
        """
        logger.debug("[unwrap] value=%s for_llm=%s", value, for_llm)
        if value[0] == "$":
            path = value[1:]
            # 移除 MEM 前缀（$MEM.key -> $key）
            if path and path[0] == "MEM":
                path = path[1:]
            return self._unwrap_dollar(path, for_llm=for_llm)
        elif value[0] == "&":
            path = value[1:]
            if path and path[0] == "MEM":
                path = path[1:]
            return self._unwrap_ampersand(path, for_llm=for_llm)
        else:
            # 无前缀，直接返回路径对应的值
            return self._get_by_path(value)

    def _unwrap_dollar(self, path: list, for_llm: bool, seen: Optional[set] = None) -> Any:
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
            next_path = temp_value[1:]
            # 移除 MEM 前缀
            if next_path and next_path[0] == "MEM":
                next_path = next_path[1:]
            return self._unwrap_dollar(next_path, for_llm=for_llm, seen=seen)

        # 处理类型转换
        return self._convert_for_llm(temp, for_llm)

    def _unwrap_ampersand(self, path: list, for_llm: bool) -> Any:
        """
        & 引用：直接返回路径对应的原始值，不做额外解引用
        """
        temp = self._get_by_path(path)
        return self._convert_for_llm(temp, for_llm)

    def _get_by_path(self, path: list) -> Any:
        """
        根据路径获取值
        :param path: 路径列表，如 ['key', 'subkey']
        """
        # 首先检查是否是设备路径
        if self.is_device_path(path):
            return self.get_device(path)

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

        # 检查是否是设备
        if isinstance(value, MemoryDevice):
            return value.to_llm_string()

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
        logger.debug("[set_by_path] path=%s value=%r", path, value)
        # 检查是否是设备路径
        if self.is_device_path(path):
            device = self.get_device(path)
            if hasattr(device, 'set_value'):
                device.set_value(value)
            else:
                raise VMMemoryError(f"设备 {type(device).__name__} 不支持直接写入")
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
                    temp[key] = {}
                except TypeError:
                    raise VMMemoryError(f"无法在 {type(temp).__name__} 类型下创建子路径")
            temp = temp[key]
        temp[path[-1]] = value

    def set(self, ref: str, value: Any) -> None:
        """
        根据引用设置值
        :param ref: 引用字符串，如 '$MEM.key' 或 '$MEM.key.subkey'
        :param value: 要设置的值
        """
        logger.info("[mem.set] ref=%s", ref)
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
        logger.info("[mem.mount] path=%s device=%s", path, type(device).__name__)
        if not isinstance(device, MemoryDevice):
            raise VMMemoryError(f"必须挂载 MemoryDevice 类型的设备， got {type(device).__name__}")

        self._devices[path] = device

    def unmount(self, path: str) -> None:
        """
        从指定路径卸载设备
        :param path: 路径字符串
        """
        logger.info("[mem.unmount] path=%s", path)
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
        logger.info("[mem.make] ref=%s key=%s type=%s", ref, key, mem_type)
        if not ref.startswith("$"):
            raise VMMemoryError(f"引用必须以 $ 开头：{ref}")

        if mem_type not in ('str', 'dict', 'list'):
            raise VMMemoryError(f"不支持的类型：{mem_type}，必须是 'str', 'dict', 'list' 之一")

        # 解析引用路径
        parts = ref[1:].split(".")
        # 移除 MEM 前缀
        if parts and parts[0] == "MEM":
            parts = parts[1:]

        # 获取父路径和当前值
        parent_path = parts[:-1] if len(parts) > 1 else []
        current_path = parts

        # 获取 ref 对应的值
        if parent_path:
            parent = self._get_by_path(parent_path)
        else:
            parent = self._data

        # 检查 ref 是否存在
        if current_path:
            try:
                current = self._get_by_path(current_path)
            except (KeyError, TypeError):
                raise VMMemoryError(f"路径不存在：{ref}")
        else:
            current = parent

        # 如果 ref 是空路径（即 $MEM），则直接在 _data 上操作
        if not current_path:
            current = self._data

        # 检查类型
        if isinstance(current, str):
            raise VMMemoryError(f"不能在字符串类型的路径下创建新键：{ref}")
        elif isinstance(current, list):
            # 期望第二个值是索引数字
            if not key.isdigit():
                raise VMMemoryError(f"列表类型的路径下，key 必须是数字索引：{key}")
            index = int(key)
            if index < 0 or index >= len(current):
                raise VMMemoryError(f"索引越界：{index}，列表长度：{len(current)}")
            # 在列表指定索引位置创建新值
            if mem_type == 'str':
                current[index] = ""
            elif mem_type == 'dict':
                current[index] = {}
            elif mem_type == 'list':
                current[index] = []
        elif isinstance(current, dict):
            # 在字典中创建新键
            if mem_type == 'str':
                current[key] = ""
            elif mem_type == 'dict':
                current[key] = {}
            elif mem_type == 'list':
                current[key] = []
        else:
            raise VMMemoryError(f"不支持在类型 {type(current).__name__} 下创建新键")

    def to_dict(self) -> dict:
        """返回底层字典的副本"""
        return self._data.copy()

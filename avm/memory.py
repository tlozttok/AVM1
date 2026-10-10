"""AVM 内存系统定义"""
from typing import Any, Optional
from .types import MetaList, MetaDict
from .exceptions import VMMemoryError, MemoryKeyNotFoundError, MemoryIndexOutOfRangeError, MemoryTypeError, MemoryCircularReferenceError
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


def _with_owner(node: Any, owner_cid: Optional[int]) -> Any:
    """若节点是 MetaDict/MetaList 且提供了 owner_cid，则复制现有 ctrl 并写入 owner。

    返回传入节点，方便链式调用。str 叶子没有 ctrl，不写。
    """
    if owner_cid is not None and isinstance(node, (MetaDict, MetaList)):
        ctrl = dict(node.get_ctrl() or {})
        ctrl["owner"] = str(owner_cid)
        node.set_ctrl(ctrl)
    return node



class Memory:
    """
    内存类
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
        """公开入口：带空链接栈解析路径。"""
        return self._resolve_path(path, for_llm=for_llm, seen_links=None)

    def _resolve_path(self, path: list, for_llm: bool = False, seen_links: Optional[list] = None) -> Any:
        if seen_links is None:
            seen_links = []
        if not path:
            return self._convert_for_llm(self._data, for_llm)

        # 不以 MEM 开头 — 必须先命中设备
        if path[0] != "MEM":
            for i in range(len(path)):
                prefix_str = ".".join(path[:i + 1])
                if prefix_str in self._devices:
                    device = self._devices[prefix_str]
                    remaining = path[i + 1:]
                    result = self._resolve_device(device, remaining, seen_links, prefix_str)
                    return self._convert_for_llm(result, for_llm)
            raise VMMemoryError(f"未知路径前缀：{path[0]}，只允许 MEM 或已挂载设备")

        # MEM 路径：先检查设备命中，再走数据
        path = path[1:]  # strip MEM
        if not path:
            return self._convert_for_llm(self._data, for_llm)

        return self._resolve_mem_path(path, for_llm=for_llm, seen_links=seen_links)

    def _resolve_device(self, device, remaining: list, seen_links: list, prefix: Optional[str] = None):
        """解析设备命中。LinkDevice 即使 remaining 为空也透明跟随目标。"""
        if getattr(device, "is_link", False):
            if prefix is not None and prefix in seen_links:
                chain = " -> ".join(seen_links + [prefix])
                raise MemoryCircularReferenceError(f"链接成环: {chain}")
            if prefix is not None:
                seen_links.append(prefix)
            return device.resolve_path(remaining or [], seen_links or ([prefix] if prefix else []))
        if remaining:
            return device.resolve_path(remaining)
        return device

    def _resolve_mem_path(self, path: list, for_llm: bool, seen_links: Optional[list] = None) -> Any:
        if seen_links is None:
            seen_links = []
        # 检查路径任意前缀是否命中设备
        for i in range(len(path)):
            prefix_str = ".".join(path[:i + 1])
            if prefix_str in self._devices:
                device = self._devices[prefix_str]
                remaining = path[i + 1:]
                result = self._resolve_device(device, remaining, seen_links, prefix_str)
                return self._convert_for_llm(result, for_llm)

        # 普通数据访问
        current = self._data
        for key in path:
            self._check_access(current, key, path)
            current = current[key]
        return self._convert_for_llm(current, for_llm)

    def _follow_real(self, segments: list, seen_links: Optional[list] = None) -> tuple:
        """把 MEM 段（不含根）跟随链接，返回 ("memory", segments) 或 ("device", mount, remainder)。

        不做数据访问副作用；只按挂载注册表决定最终目标。
        """
        if seen_links is None:
            seen_links = []
        for i in range(len(segments), 0, -1):
            prefix = ".".join(segments[:i])
            if prefix in self._devices:
                device = self._devices[prefix]
                if getattr(device, "is_link", False):
                    if prefix in seen_links:
                        chain = " -> ".join(seen_links + [prefix])
                        raise MemoryCircularReferenceError(f"链接成环: {chain}")
                    seen_links.append(prefix)
                    return device._follow_real(segments[i:], seen_links)
                return ("device", prefix, list(segments[i:]))
        return ("memory", list(segments))

    def follow_link_target(self, ref: str) -> tuple:
        """返回链接后的真实目标：(kind, real_segments) 或 ("device", mount, remainder)。"""
        split = self.split_path(ref)
        kind = split[0]
        if kind == "memory":
            return ("memory", ref[1:].split(".")[1:])
        _, mount, remainder = split
        device = self._devices[mount]
        if getattr(device, "is_link", False):
            return device._follow_real(remainder, seen_links=[mount])
        return ("device", mount, remainder)

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

    def set_by_path(self, path: list, value: Any, owner_cid: Optional[int] = None) -> None:
        """
        根据路径设置值
        :param path: 路径列表，如 ['key', 'subkey']
        :param owner_cid: 新创建的 MetaDict/MetaList 节点的所有者 cid（可选）。
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
            if getattr(device, "is_link", False):
                device.set_value(value, owner_cid=owner_cid)
            elif hasattr(device, 'set_value'):
                device.set_value(value)
            else:
                raise VMMemoryError(f"设备 {type(device).__name__} 不支持直接写入")
            return

        # 检查路径前缀是否命中设备：如 outputs.-1 中 outputs 是设备
        for i in range(len(path) - 1, 0, -1):
            prefix = path[:i]
            if self.is_device_path(prefix):
                device = self.get_device(prefix)
                if getattr(device, "is_link", False):
                    # 链接子路径写：委托到真实目标路径（含环检测）
                    real_segments = device._real_mem_segments(path[i:])
                    self.set_by_path(real_segments, value, owner_cid=owner_cid)
                    return
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
                    temp[key] = _with_owner(MetaDict(data={}), owner_cid)
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
            temp[idx] = _with_owner(_wrap_value(value), owner_cid)
            return

        # 如果终点已存在且是 MetaList/MetaDict，写入即修改元数据
        if last_key in temp:
            existing = temp[last_key]
            if isinstance(existing, (MetaList, MetaDict)):
                existing.set_metadata(value)
                return

        temp[last_key] = _with_owner(_wrap_value(value), owner_cid)

    def set(self, ref: str, value: Any, owner_cid: Optional[int] = None) -> None:
        """
        根据引用设置值
        :param ref: 引用字符串，如 '$MEM.key' 或 '$MEM.key.subkey'
        :param value: 要设置的值
        :param owner_cid: 新创建容器节点的所有者 cid（可选）
        """
        if not ref.startswith("$"):
            raise ValueError(f"引用必须以 $ 开头：{ref}")

        # 解析引用路径 - 移除 $MEM 前缀
        parts = ref[1:].split(".")  # 例如：["$MEM", "user_output"] -> ["MEM", "user_output"]
        if parts[0] == "MEM":
            parts = parts[1:]  # 移除 MEM 前缀，得到 ["user_output"]

        self.set_by_path(parts, value, owner_cid=owner_cid)

    # ------------------------------------------------------------------
    # 调试原语：屏蔽与还原（宿主侧，不经指令、不产生事件）
    # ------------------------------------------------------------------

    def mounted_devices(self) -> dict:
        """已挂载设备的只读视图（调试控制台扫描设备用）。"""
        return dict(self._devices)

    def _locate(self, ref: str):
        """把 $MEM 引用解析成 (容器, 键)，跟随链接，但拒绝落在设备上。"""
        split = self.split_path(ref)
        if split[0] == "device":
            raise VMMemoryError(
                f"{ref} 落在设备上：调试原语只作用于数据节点，设备请用 unmount/mount"
            )
        segments = ref[1:].split(".")[1:]  # 去掉 $MEM
        if not segments:
            raise VMMemoryError("不能摘除 $MEM 根节点")
        target = self._follow_real(segments)
        if target[0] != "memory":
            raise VMMemoryError(
                f"{ref} 落在设备上：调试原语只作用于数据节点，设备请用 unmount/mount"
            )
        path = target[1]
        container = self._data
        for key in path[:-1]:
            self._check_access(container, key, path)
            container = container[key]
        return container, path[-1]

    @staticmethod
    def _index_of(container, key):
        """列表容器把键转成整数下标；字典容器原样返回键。"""
        if isinstance(container, (list, MetaList)):
            try:
                return int(key)
            except (TypeError, ValueError):
                raise MemoryTypeError(f"列表索引必须是数字：{key}") from None
        return key

    def detach(self, ref: str) -> Any:
        """调试用：按路径摘除节点并原样返回（meta / ctrl / owner 一并保留）。

        与指令无关，不产生事件：屏蔽期间对话只会看到键不存在，注册在该路径上的
        监听器不会被触发。还原用 attach。设备路径拒绝——先 unmount 设备。
        """
        container, key = self._locate(ref)
        if isinstance(container, (dict, MetaDict)):
            if key not in container:
                raise MemoryKeyNotFoundError(f"键 {key!r} 不存在：{ref}")
            node = container[key]
            del container[key]
            return node
        if isinstance(container, (list, MetaList)):
            idx = self._index_of(container, key)
            if idx < 0 or idx >= len(container):
                raise MemoryIndexOutOfRangeError(
                    f"索引越界：{idx}，列表长度：{len(container)}"
                )
            node = container[idx]
            del container[idx]
            return node
        raise MemoryTypeError(
            f"无法摘除 {ref}：父节点是 {type(container).__name__}，没有子节点"
        )

    def attach(self, ref: str, node: Any) -> None:
        """调试用：把 detach 摘下来的节点原样放回。

        路径已存在时报还原冲突，不静默覆盖——屏蔽期间该路径可能已被对话重建，
        由操作员决定丢弃哪一边。
        """
        container, key = self._locate(ref)
        if isinstance(container, (dict, MetaDict)):
            if key in container:
                raise VMMemoryError(f"还原冲突：{ref} 已存在，不覆盖")
            container[key] = node
            return
        if isinstance(container, (list, MetaList)):
            idx = self._index_of(container, key)
            if idx < 0 or idx > len(container):
                raise VMMemoryError(
                    f"还原冲突：{ref} 的下标 {idx} 越界（当前长度 {len(container)}）"
                )
            container.insert(idx, node)
            return
        raise MemoryTypeError(
            f"无法还原 {ref}：父节点是 {type(container).__name__}，没有子节点"
        )

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

    def split_path(self, ref: str):
        """把 $MEM 引用拆分为 memory 或 device 路径。

        返回：
        - ("memory", segments)
        - ("device", mount_key, remainder)
        段序列按最长设备挂载键匹配；`box` 不会错误匹配 `$MEM.box2`。
        """
        if not isinstance(ref, str) or not ref.startswith("$MEM"):
            raise ValueError(f"split_path 只接受 $MEM 引用：{ref!r}")
        parts = ref[1:].split(".")
        if not parts or parts[0] != "MEM":
            raise ValueError(f"split_path 只接受 $MEM 引用：{ref!r}")
        rest = parts[1:]
        # 最长挂载键优先：同时挂载 box 与 box.dev 时，$MEM.box.dev 命中 box.dev。
        for i in range(len(rest), 0, -1):
            prefix = ".".join(rest[:i])
            if prefix in self._devices:
                return ("device", prefix, rest[i:])
        return ("memory", rest)

    def make(self, ref: str, key: str, mem_type: str, owner_cid: Optional[int] = None) -> None:
        """
        在指定路径创建新的内存地址
        :param ref: 引用字符串，如 '$MEM.key' 或 '$MEM.key.subkey'
        :param key: 新地址的名字
        :param mem_type: 类型，'str' | 'dict' | 'list'
        :param owner_cid: 新容器节点的所有者 cid（可选；str 节点不写 owner）
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
                current[index] = _with_owner(MetaDict(data={}), owner_cid)
            elif mem_type == 'list':
                current[index] = _with_owner(MetaList(data=[]), owner_cid)
        elif isinstance(current, (dict, MetaDict)):
            if mem_type == 'str':
                current[key] = ""
            elif mem_type == 'dict':
                current[key] = _with_owner(MetaDict(data={}), owner_cid)
            elif mem_type == 'list':
                current[key] = _with_owner(MetaList(data=[]), owner_cid)
        else:
            raise VMMemoryError(f"不支持在类型 {type(current).__name__} 下创建新键")

    def to_dict(self) -> dict:
        """返回底层字典的副本"""
        return self._data.copy()

    def save(self, filepath: str) -> None:
        """将内存数据持久化到 JSON 文件（仅 _data，不含设备）。
        格式与系统镜像 mem 段一致：显式节点 {kind, meta, ctrl, value}。"""
        import json

        def _serialize(value):
            if isinstance(value, MetaDict):
                node = {"kind": "dict", "value": {k: _serialize(v) for k, v in value.items()}}
            elif isinstance(value, MetaList):
                node = {"kind": "list", "value": [_serialize(v) for v in value]}
            else:
                node = {"kind": "str", "value": str(value)}
            meta = value.get_metadata() if hasattr(value, "get_metadata") else None
            ctrl = value.get_ctrl() if hasattr(value, "get_ctrl") else None
            if meta is not None:
                node["meta"] = meta
            if ctrl is not None:
                node["ctrl"] = ctrl
            return node

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(_serialize(self._data), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Memory":
        """从 JSON 文件加载内存数据并返回 Memory 实例。
        支持显式节点格式（kind/meta/ctrl/value）与旧 __type 格式。"""
        import json, os

        mem = cls()
        if not os.path.isfile(filepath):
            return mem

        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        mem._data = cls._deserialize_node(raw)
        return mem

    @classmethod
    def _deserialize_node(cls, value):
        """解析单个显式节点；兼容旧 __type 格式。"""
        from .types import MetaList, MetaDict

        if isinstance(value, dict):
            if value.get("kind") in ("str", "dict", "list"):
                kind = value["kind"]
                v = value.get("value")
                meta = value.get("meta")
                ctrl = value.get("ctrl")
                if kind == "str":
                    return v
                if kind == "dict":
                    return MetaDict(
                        data={k: cls._deserialize_node(x) for k, x in (v or {}).items()},
                        metadata=meta, ctrl=ctrl,
                    )
                if kind == "list":
                    return MetaList(
                        data=[cls._deserialize_node(x) for x in (v or [])],
                        metadata=meta, ctrl=ctrl,
                    )
            if value.get("__type") == "MetaDict":  # 旧格式兼容
                return MetaDict(
                    data={k: cls._deserialize_node(x) for k, x in value.get("data", {}).items()},
                    metadata=value.get("meta"),
                )
            if value.get("__type") == "MetaList":  # 旧格式兼容
                return MetaList(
                    data=[cls._deserialize_node(x) for x in value.get("data", [])],
                    metadata=value.get("meta"),
                )
        return value

    @classmethod
    def from_tree(cls, tree: dict) -> "Memory":
        """从镜像 mem 段（顶层键 → 显式节点）构建内存。
        注意：kind=device 的标记节点已在镜像加载器中处理，此处不接收。"""
        from .types import MetaDict

        mem = cls()
        mem._data = MetaDict(data={k: cls._deserialize_node(v) for k, v in tree.items()})
        return mem

"""AVM 内存系统定义"""
from typing import Any, Optional
from .types import MetaList, MetaDict, LinkNode, StrNode
from .exceptions import (VMMemoryError, IllegalOperationError, MemoryKeyNotFoundError, MemoryIndexOutOfRangeError,
                          MemoryTypeError, MemoryCircularReferenceError, MemoryLinkBrokenError)
from .capabilities import AddressSpace, MEM_ROOT_NAME
from .memory_device import MemoryDevice, StringDevice


_MAX_PATH_SEGMENTS = 64  # 单条引用的段数上限（戊4）
_MAX_VALUE_DEPTH = 64  # 写入值的嵌套深度上限（戊4）


def _show(parts) -> str:
    """把显示段序列拼成 `$根.段...` 形态（错误与地址回显用）。"""
    return "$" + ".".join(str(s) for s in parts)


def _root_unavailable(name: str) -> VMMemoryError:
    """未知/未授予的根：统一错误，不区分拼错与未授权。"""
    return VMMemoryError(f"引用根不可用：{name!r}（本对话根表不含该根）")


def check_new_key(key) -> None:
    """新建地址的键语法（指令边界与 Memory.make 共用一套）。

    键是路径的一段，点号是段分隔符——带点的键建得出来但永远读不回来（模板：`$MEM.a.b`
    会被切成 a、b 两段）。空键同理，还会让 `$MEM.` 与读根混淆。
    """
    if not isinstance(key, str):
        raise IllegalOperationError(f"新地址的键必须是字符串，得到 {type(key).__name__}")
    if key == "":
        raise IllegalOperationError("新地址的键不能为空字符串")
    if "." in key:
        raise IllegalOperationError(f"新地址的键不能包含点号（点号是路径分隔符，该地址将无法寻址）：{key!r}")


def _device_child(node: Any, key: Any) -> Any:
    """设备空间的只读下钻：把设备自己的异常类型统一翻成 VMMemoryError。

    设备是宿主代码，抛什么类型不由内核决定；不统一翻译就会以裸 TypeError/KeyError
    穿过 memory_write 的边界（子路径写入不该成为内核退出路径）。
    """
    try:
        return node[key]
    except VMMemoryError:
        raise
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        raise VMMemoryError(
            f"设备 {type(node).__name__} 不支持子路径 {key!r}：{type(e).__name__}: {e}"
        )


def _device_put(node: Any, key: Any, value: Any) -> None:
    """设备空间的写入：同上，统一翻译异常类型。"""
    try:
        node[key] = value
    except VMMemoryError:
        raise
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        raise VMMemoryError(
            f"设备 {type(node).__name__} 不支持写入 {key!r}：{type(e).__name__}: {e}"
        )


def _wrap_value(value: Any, depth: int = 0) -> Any:
    """将普通 dict/list 递归包装为 MetaDict/MetaList。已包装的也递归检查内部数据。

    嵌套深度设上限：超限按非法操作拒绝（VMMemoryError），不让 RecursionError 变成内核退出。
    """
    if depth > _MAX_VALUE_DEPTH:
        raise VMMemoryError(f"写入值嵌套超过上限 {_MAX_VALUE_DEPTH} 层")
    if isinstance(value, MetaDict):
        for k, v in list(value.items()):
            value[k] = _wrap_value(v, depth + 1)
        return value
    if isinstance(value, MetaList):
        for i in range(len(value)):
            value[i] = _wrap_value(value[i], depth + 1)
        return value
    if isinstance(value, dict):
        return MetaDict(data={k: _wrap_value(v, depth + 1) for k, v in value.items()})
    if isinstance(value, list):
        return MetaList(data=[_wrap_value(v, depth + 1) for v in value])
    if isinstance(value, StrNode):
        return value
    if isinstance(value, str):
        return StrNode(value)
    return value


def _with_owner(node: Any, owner_cid: Optional[int]) -> Any:
    """若节点是 MetaDict/MetaList 且提供了 owner_cid，则复制现有 ctrl 并写入 owner。

    返回传入节点，方便链式调用。str 叶子没有 ctrl，不写。
    """
    if owner_cid is not None and isinstance(node, (MetaDict, MetaList, StrNode)):
        ctrl = dict(node.get_ctrl() or {})
        ctrl["owner"] = str(owner_cid)
        node.set_ctrl(ctrl)
    return node



class Memory:
    """
    内存类
    注意：MEM 中只存储 MetaDict/MetaList/str/LinkNode，不会出现普通 dict/list
    """

    def __init__(self):
        self._data: MetaDict = MetaDict(data={})
        self._devices: dict = {}  # 存储已挂载的设备，key 为路径字符串（设备表保留地址）
        self._device_ids: dict = {}  # 设备唯一标识符 → 设备；链接 target 不以 $ 开头时按此解析

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

    def _expand_ref_parts(self, parts: list, space: Optional[AddressSpace] = None):
        """根展开：返回 ("memory", 真实段, 显示段) 或 ("device", 请求段, 请求段)。

        `space=None` 时按旧规则（`$MEM` 或设备路径简写）；提供 space 时第一段按根表
        精确匹配，不含 `MEM` 的根表只认根名（设备经根目标里的挂载链接进入）。
        语法与前缀错误抛 `IllegalOperationError`。
        """
        first = parts[0]
        if space is not None:
            target = space.root(first)
            if target is not None:
                return ("memory", list(target) + list(parts[1:]), list(parts))
            if not space.has_mem():
                raise _root_unavailable(first)
        if first == MEM_ROOT_NAME:
            return ("memory", list(parts[1:]), list(parts))
        # 设备路径简写：space=None 或根表含 MEM 时保留
        for i in range(len(parts)):
            if ".".join(str(s) for s in parts[:i + 1]) in self._devices:
                return ("device", list(parts), list(parts))
        shown = _show(parts)
        raise IllegalOperationError(
            f"无法识别的引用前缀：{first!r}（引用 {shown}）；只允许已声明的根或已挂载设备路径"
        )

    def _parse_ref(self, ref: str) -> list:
        """校验引用字符串并切段（读、写、删除共用一套语法）。"""
        if not isinstance(ref, str):
            raise IllegalOperationError(f"引用必须是字符串，得到 {type(ref).__name__}")
        if not ref.startswith("$"):
            raise IllegalOperationError(f"引用必须以 $ 开头：{ref}")
        parts = ref[1:].split(".")
        if any(part == "" for part in parts):
            if all(part == "" for part in parts):
                raise IllegalOperationError(f"引用不能为空：{ref!r}（至少要写 $MEM）")
            raise IllegalOperationError(f"引用不能有空段：{ref!r}")
        if len(parts) > _MAX_PATH_SEGMENTS:
            raise IllegalOperationError(f"引用段数超过上限 {_MAX_PATH_SEGMENTS}：{ref!r}")
        return parts

    def unwrap(self, value: list, for_llm: bool = True, space: Optional[AddressSpace] = None) -> Any:
        """
        解引用值
        :param value: 路径列表，如 ['$', 'MEM', 'key']、['$', 'work', 'key'] 或 ['$', 'device', 'subpath']
        :param space: 地址上下文（根表）；None = 只允许 $MEM 根
        """
        if value[0] in ("$", "&"):
            path = list(value[1:])
        else:
            path = list(value)
        return self.resolve_path(path, for_llm=for_llm, space=space)

    def resolve_path(self, path: list, for_llm: bool = False, space: Optional[AddressSpace] = None) -> Any:
        """公开入口：带空链接栈解析路径。"""
        return self._resolve_path(path, for_llm=for_llm, seen_links=None, space=space)

    def _resolve_path(self, path: list, for_llm: bool = False, seen_links: Optional[list] = None,
                      space: Optional[AddressSpace] = None) -> Any:
        if seen_links is None:
            seen_links = []
        if any(seg == "" for seg in path):
            # 读侧与写侧用同一套语法：空段是语法错误，不是"键 '' 不存在"
            shown = _show(path)
            if all(seg == "" for seg in path):
                raise IllegalOperationError(f"引用不能为空：{shown!r}（至少要写 $MEM）")
            raise IllegalOperationError(f"引用不能有空段：{shown}")
        if not path:
            return self._convert_for_llm(self._data, for_llm)

        kind, real, display = self._expand_ref_parts(list(path), space)
        # 根别名：目标本身缺失时报"根无效"（不通过别名路径的错误表达目标位置）
        if space is not None and kind == "memory" and path[0] != MEM_ROOT_NAME:
            target = space.root(path[0])
            if target is not None and not self._root_target_complete(list(target)):
                raise MemoryLinkBrokenError(f"引用根无效：{path[0]!r}（目标不存在）")
        if kind == "device":
            for i in range(len(real)):
                prefix_str = ".".join(str(s) for s in real[:i + 1])
                if prefix_str in self._devices:
                    device = self._devices[prefix_str]
                    remaining = list(real[i + 1:])
                    result = self._resolve_device(device, remaining, seen_links, prefix_str)
                    return self._convert_for_llm(result, for_llm)
            shown = _show(display)
            raise IllegalOperationError(
                f"无法识别的引用前缀：{real[0]!r}（引用 {shown}）；只允许已声明的根或已挂载设备路径"
            )

        if not real:
            return self._convert_for_llm(self._data, for_llm)
        return self._resolve_mem_path(real, for_llm=for_llm, seen_links=seen_links, display=display)

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

    def _resolve_mem_path(self, path: list, for_llm: bool, seen_links: Optional[list] = None,
                          display: Optional[list] = None) -> Any:
        """按内存树逐段解析；遇到 LinkNode 就进入 target 指向的地址。

        链接 target 以 `$` 开头 → 内存树地址；否则 → 设备唯一标识符。
        经过的链接节点路径记入 seen_links（本次解析的访问集合），再次命中即报成环。
        `display` 是报错用的请求段序列（含根名）；链接跳转不改它。
        """
        if seen_links is None:
            seen_links = []
        if display is None:
            display = [MEM_ROOT_NAME] + list(path)
        node: Any = self._data
        walked: list = []
        index = 0
        while True:
            if isinstance(node, LinkNode):
                address = ".".join(walked)
                if address in seen_links:
                    chain = " -> ".join(seen_links + [address])
                    raise MemoryCircularReferenceError(f"链接成环: {chain}")
                seen_links.append(address)
                kind, payload = self._link_target(node, address)
                if kind == "device":
                    device = self._require_device(payload, address)
                    remainder = list(path[index:])
                    # 当前这一跳已记入 seen_links，交给设备命中逻辑时不让它重复计数
                    device_seen = [x for x in seen_links if x != address]
                    resolved = self._resolve_device(device, remainder, device_seen, address)
                    return self._convert_for_llm(resolved, for_llm)
                # 内存树地址：拼上剩余段，从根重新解析（访问集合继续累积）
                # 内存目标必须完整存在：目标消失时链接保留但访问报"链接无效"
                self._check_link_target(address, payload, seen_links)
                path = list(payload) + list(path[index:])
                node = self._data
                walked = []
                index = 0
                continue
            if index >= len(path):
                return self._convert_for_llm(node, for_llm)
            key = path[index]
            self._check_access(node, key, display)
            node = node[self._index_key(node, key)]
            walked.append(key)
            index += 1

    @staticmethod
    def _link_target(node: LinkNode, address: str = "") -> tuple:
        """解析链接 target：返回 ("memory", segments) 或 ("device", 标识符)。"""
        target = node.target
        if not isinstance(target, str) or not target:
            raise VMMemoryError(f"链接 {address or '$MEM'} 的 target 必须是非空字符串")
        if target.startswith("$"):
            parts = target[1:].split(".")
            if not parts or parts[0] != "MEM":
                raise VMMemoryError(
                    f"链接 {address or '$MEM'} 的 target 只支持 $MEM 地址或设备标识符：{target!r}"
                )
            return ("memory", parts[1:])
        return ("device", target)

    def _require_device(self, identifier: str, address: str = "") -> MemoryDevice:
        device = self._device_ids.get(identifier)
        if device is None:
            where = f"（来自链接 {address}）" if address else ""
            raise MemoryLinkBrokenError(
                f"链接无效：{address or '$MEM'} 指向的设备标识符未知：{identifier!r}{where}"
            )
        return device

    def _check_link_target(self, address: str, payload: list, seen_links) -> None:
        """内存链接的目标路径必须完整存在；目标消失时抛"链接无效"。

        像符号链接：目标被删除不删除链接节点本身，访问到它时才报错。
        """
        result = self._walk_links(list(payload), list(seen_links or []))
        if result[0] == "memory" and result[2] is False:
            # 报错只写链接所在路径，不写真实目标（能力边界）
            raise MemoryLinkBrokenError(f"链接无效：{address or '$MEM'}（目标不存在）")

    def _address_of(self, device: MemoryDevice) -> Optional[str]:
        """一个设备当前登记的第一个挂载地址（事件边界用）。

        取快照迭代：控制台线程可以在内核线程推进时卸载设备（戊3）。
        """
        for address, mounted in list(self._devices.items()):
            if mounted is device:
                return address
        return None

    def _walk_links(self, segments: list, seen_links: Optional[list] = None) -> tuple:
        """沿内存树走段序列，展开链接与设备。

        返回：
        - ("memory", segments, complete)：落在数据节点（段序列已按链接重写；complete 表示整条段序列都走到了）
        - ("device", device, address, remainder)：落在设备空间

        只走树上已存在的节点；遇到不存在的键即停止，把剩余段交回调用者按普通路径处理。
        """
        if seen_links is None:
            seen_links = []
        node: Any = self._data
        walked: list = []
        current = list(segments)
        index = 0
        while True:
            if isinstance(node, LinkNode):
                address = ".".join(walked)
                if address in seen_links:
                    chain = " -> ".join(seen_links + [address])
                    raise MemoryCircularReferenceError(f"链接成环: {chain}")
                seen_links.append(address)
                kind, payload = self._link_target(node, address)
                if kind == "device":
                    device = self._require_device(payload, address)
                    remainder = list(current[index:])
                    if getattr(device, "is_link", False):
                        # 设备层别名：交给它自己跟随，seen_links 继续累积（跨设备成环在此断掉）
                        sub = device._follow_real(remainder, seen_links)
                        if sub[0] == "device":
                            real = self._devices.get(sub[1])
                            if real is None:
                                raise MemoryLinkBrokenError(
                                    f"链接无效：{address} 的设备别名目标未登记挂载地址：{sub[1]}"
                                )
                            return ("device", real, sub[1], list(sub[2]))
                        current = list(sub[1])
                        node = self._data
                        walked = []
                        index = 0
                        continue
                    return ("device", device, self._address_of(device), remainder)
                # 内存目标必须完整存在：目标消失时链接保留但访问报"链接无效"
                self._check_link_target(address, payload, seen_links)
                current = list(payload) + list(current[index:])
                node = self._data
                walked = []
                index = 0
                continue
            if index >= len(current):
                return ("memory", current, True)
            key = current[index]
            if isinstance(node, (dict, MetaDict)):
                if key not in node:
                    return ("memory", current, False)
                node = node[key]
            elif isinstance(node, (list, MetaList)):
                try:
                    idx = int(key)
                except (ValueError, TypeError):
                    return ("memory", current, False)
                if idx < 0:
                    raise IllegalOperationError(
                        f"负下标不支持：{'.'.join(walked + [key])}（-1 只表示追加位置）"
                    )
                try:
                    node = node[idx]
                except IndexError:
                    return ("memory", current, False)
            else:
                return ("memory", current, False)
            walked.append(key)
            index += 1

    def _follow_real(self, segments: list, seen_links: Optional[list] = None) -> tuple:
        """把 MEM 段（不含根）跟随链接，返回 ("memory", segments) 或 ("device", address, remainder)。

        不做数据访问副作用；只按内存树与挂载注册表决定最终目标。
        """
        target = self._walk_links(segments, [] if seen_links is None else seen_links)
        if target[0] == "device":
            _, _device, address, remainder = target
            if address is None:
                raise VMMemoryError("设备未登记挂载地址")
            return ("device", address, remainder)
        return ("memory", target[1])

    def follow_link_target(self, ref: str, space: Optional[AddressSpace] = None) -> tuple:
        """返回链接后的真实目标：(kind, real_segments) 或 ("device", mount, remainder)。"""
        split = self.split_path(ref, space=space)
        kind = split[0]
        if kind == "memory":
            target = self._walk_links(list(split[1]))
            if target[0] == "device":
                _, _device, address, remainder = target
                return ("device", address, remainder)
            return ("memory", target[1])
        _, mount, remainder = split
        # 并发卸载的窗口：查表失败按未登记处理，不让 KeyError 穿到写入路径
        device = self._devices.get(mount)
        if device is None:
            raise VMMemoryError(f"设备未登记挂载地址：{mount}")
        if getattr(device, "is_link", False):
            return device._follow_real(remainder, seen_links=[mount])
        return ("device", mount, remainder)

    @staticmethod
    def _index_key(node: Any, key: Any) -> Any:
        """列表节点按引用寻址时用整数下标取值。

        引用是一段字符串（`$MEM.a.0`），`_check_access` 已经确认这段是合法下标，
        但取值前必须把字符串转成整数——否则 `MetaList.__getitem__("0")` 会抛
        `TypeError: list indices must be integers`。设备列表自己会转，树里的列表靠这里。
        """
        return int(str(key)) if isinstance(node, (list, MetaList)) else key

    @staticmethod
    def _check_access(current, key: str, path: list, shown: Optional[str] = None):
        """检查 key 是否可访问，不可则抛出对应的内存子类异常。

        `path` 是显示段序列（含根名）；`shown` 指定时直接用它做错误里的地址。
        """
        from .memory_device import MemoryDevice

        if isinstance(current, MemoryDevice):
            return
        partial = shown if shown is not None else _show(path)
        if isinstance(current, str):
            raise MemoryTypeError(f"字符串没有子节点：{partial}")
        if isinstance(current, (dict, MetaDict)):
            if key == "-1":
                raise IllegalOperationError(f"字典里不能用 -1（-1 只表示列表追加位置）：{partial}")
            if key not in current:
                raise MemoryKeyNotFoundError(f"键 {key!r} 不存在：{partial}（可用 memory_make 创建）")
        elif isinstance(current, (list, MetaList)):
            try:
                idx = int(key)
            except ValueError:
                raise IllegalOperationError(f"列表索引必须是数字，got {key!r}：{partial}")
            if idx < 0:
                raise IllegalOperationError(f"负下标不支持：{partial}（-1 只表示追加位置）")
            if idx >= len(current):
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

    def set_by_path(self, path: list, value: Any, owner_cid: Optional[int] = None,
                    display: Optional[list] = None) -> Optional[str]:
        """
        根据路径设置值
        :param path: 真实路径列表（`MEM` 之后），如 ['key', 'subkey']
        :param owner_cid: 新创建的 MetaDict/MetaList 节点的所有者 cid（可选）。
        :param display: 报错与追加地址用的请求段序列（含根名）；缺省按 `$MEM` 拼。
        特殊规则：如果 path 终点的现有值是 MetaList/MetaDict，
        则写入被解释为修改其元数据，而不是替换整个对象。
        """
        if display is None:
            display = [MEM_ROOT_NAME] + list(path)
        if not path:
            if isinstance(self._data, MetaDict):
                self._data.set_metadata(str(value))
                return
            self._data = _wrap_value(value)
            return

        # 列表追加：-1 只表示"末尾的下一个位置"，不是下标。
        # 追加是显式位置语法，不是隐式建路径：父容器必须已存在。
        if path and path[-1] == "-1" and not self.is_device_path(path[:-1]):
            parent = self.resolve_path(["MEM"] + list(path[:-1]), for_llm=False)
            if isinstance(parent, MemoryDevice):
                _device_put(parent, "-1", value)  # 设备层本轮保持旧语义
                return None
            if isinstance(parent, (list, MetaList)):
                self._check_append_same_type(parent, "str")
                parent.append(_with_owner(_wrap_value(value), owner_cid))
                return _show(list(display[:-1]) + [str(len(parent) - 1)])
            if isinstance(parent, (dict, MetaDict)):
                raise IllegalOperationError("字典里不能用 -1（-1 只表示列表追加位置）")
            if isinstance(parent, str):
                raise VMMemoryError("字符串没有子节点：不能用 -1 追加")
            raise VMMemoryError(f"不支持在类型 {type(parent).__name__} 下追加")

        # 链接透明：先沿树展开链接（含成环检测），得到真实落点
        followed = self._walk_links(path)
        if followed[0] == "device":
            _, device, _address, remainder = followed
            if not remainder:
                if hasattr(device, 'set_value'):
                    device.set_value(value)
                    return
                raise VMMemoryError(f"设备 {type(device).__name__} 不支持直接写入")
            current = device
            for key in remainder[:-1]:
                current = _device_child(current, key)
            _device_put(current, remainder[-1], value)
            return
        path = followed[1]

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

        # 检查路径前缀是否命中设备：如 outputs.out 中 outputs 是设备
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
                    current = _device_child(current, key)
                _device_put(current, path[-1], value)
                return

        # 普通路径处理：只改已存在的地址（创建归 memory_make）。
        node = self._data
        for i, key in enumerate(path[:-1]):
            child, exists = self._child(node, key, display, i)
            if not exists:
                raise MemoryKeyNotFoundError(
                    f"路径不存在：{_show(display)}"
                    "——memory_write 不建节点，先用 memory_make 创建"
                )
            node = child

        _, exists = self._child(node, path[-1], display, len(path) - 1)
        if not exists:
            raise MemoryKeyNotFoundError(
                f"键 {path[-1]!r} 不存在：{_show(display)}"
                "——memory_write 不建节点，先用 memory_make 创建"
            )
        self._assign(node, path[-1], _with_owner(_wrap_value(value), owner_cid), display, len(path) - 1)

    @staticmethod
    def _child(node: Any, key: Any, display: list, index: int):
        """(子节点, 是否存在)。node 不是容器时抛 MemoryTypeError（含请求路径，不含链接目标）。"""
        shown = _show(display)
        if isinstance(node, (dict, MetaDict)):
            return node.get(key), (key in node)
        if isinstance(node, (list, MetaList)):
            idx = Memory._index_of(node, key)
            if idx < 0:
                raise MemoryIndexOutOfRangeError(
                    f"负下标不支持：{shown}（-1 只表示追加位置）"
                )
            if idx >= len(node):
                raise MemoryIndexOutOfRangeError(
                    f"索引越界：{idx}，列表长度：{len(node)}：{shown}"
                )
            return node[idx], True
        raise MemoryTypeError(
            f"无法写入：{shown} 是 {type(node).__name__} 类型，没有子节点 (.{key})"
        )

    @staticmethod
    def _assign(node: Any, key: Any, value: Any, display: list, index: int) -> None:
        """把 value 落到 node 的 key 上。已存在的容器节点按既有规则解释为改元数据。"""
        if isinstance(node, (dict, MetaDict)):
            existing = node.get(key)
            if isinstance(existing, (MetaList, MetaDict)):
                existing.set_metadata(value)
                return
            if isinstance(existing, str):
                # 写字符串叶子：只改内容。owner 属于节点创建（make/append），写入不夺权：
                # 已有 StrNode 保留其 ctrl，普通 str 保持无 ctrl（由祖先容器提供 owner）。
                keep = existing.get_ctrl() if isinstance(existing, StrNode) else None
                node[key] = StrNode(str(value), ctrl=keep)
                return
            node[key] = value
            return
        if isinstance(node, (list, MetaList)):
            idx = Memory._index_of(node, key)
            if idx < 0:
                raise IllegalOperationError("负下标不支持（-1 只表示追加位置）")
            if idx >= len(node):
                raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(node)}")
            existing = node[idx]
            if isinstance(existing, str):
                keep = existing.get_ctrl() if isinstance(existing, StrNode) else None
                node[idx] = StrNode(str(value), ctrl=keep)
                return
            node[idx] = value
            return
        raise MemoryTypeError(
            f"无法写入：{_show(display)} 是 {type(node).__name__} 类型，不支持子路径"
        )

    def _expand_write_ref(self, ref: str, space: Optional[AddressSpace] = None):
        """写入引用：校验语法后展开，返回 (真实段, 显示段)。

        与 resolve_path 同一套语法与根规则；设备路径简写返回请求段本身。
        """
        parts = self._parse_ref(ref)
        kind, real, display = self._expand_ref_parts(parts, space)
        return real, display

    def delete(self, ref: str, space: Optional[AddressSpace] = None) -> tuple:
        """删除一个内存节点，返回 (类型名, 地址)。

        可删：str 叶子、空容器、指向 `$MEM` 的链接节点（删链接本身）。
        拒绝：根名本身、设备挂载点/指向设备的链接、设备空间路径、非空容器
        （整棵子树删除保留给内核态对话，尚未实现）。中间段跟随链接，最终段不跟随。
        """
        parts = self._parse_ref(ref)
        kind, real, display = self._expand_ref_parts(parts, space)
        if kind == "device":
            raise IllegalOperationError(
                f"{ref} 落在设备上：不能在设备空间删除节点（设备内容由设备自己管理）"
            )
        if len(display) == 1:
            if display[0] == MEM_ROOT_NAME:
                raise IllegalOperationError("不能删除 $MEM 根节点")
            raise IllegalOperationError(f"不能删除根：{ref}（根是内核声明，不是内存节点）")
        # 请求段（根名之后）的负下标检查
        if any(seg == "-1" or (seg.startswith("-") and seg[1:].isdigit()) for seg in display[1:]):
            raise IllegalOperationError(f"负下标不支持：{ref}（-1 只表示追加位置）")

        segments = list(real)
        split = self.split_path(ref, space=space)
        if split[0] == "device":
            if not split[2]:
                raise IllegalOperationError(
                    f"{ref} 是设备挂载点，不能删除（挂载点请用 unmount/mount）"
                )
            raise IllegalOperationError(f"{ref} 落在设备上：不能在设备空间删除节点（设备内容由设备自己管理）")

        # 中间段跟随链接；最终段不跟随（删链接本身，不删目标）
        followed = self._follow_real(list(segments[:-1]))
        if followed[0] == "device":
            raise IllegalOperationError(f"{ref} 落在设备上：不能在设备空间删除节点（设备内容由设备自己管理）")
        resolved = list(followed[1])

        container = self._data
        for key in resolved:
            if isinstance(container, LinkNode):
                raise VMMemoryError(f"{ref} 的中间段落在链接上，无法删除")
            self._check_access(container, key, [], shown=ref)
            container = container[self._index_key(container, key)]

        last = segments[-1]
        if isinstance(container, (dict, MetaDict)):
            if last not in container:
                raise MemoryKeyNotFoundError(f"键 {last!r} 不存在：{ref}")
            node = container[last]
            real_address = ".".join(resolved + [last])
        elif isinstance(container, (list, MetaList)):
            idx = self._index_of(container, last)
            if idx < 0:
                raise IllegalOperationError("负下标不支持（-1 只表示追加位置）")
            if idx >= len(container):
                raise MemoryIndexOutOfRangeError(f"索引越界：{idx}，列表长度：{len(container)}")
            node = container[idx]
            real_address = ".".join(resolved + [str(idx)])
        elif isinstance(container, str):
            raise MemoryTypeError(f"字符串没有子节点：{ref}")
        else:
            raise VMMemoryError(f"不支持在类型 {type(container).__name__} 下删除节点：{ref}")

        if isinstance(node, LinkNode):
            if real_address in self._devices or not node.target.startswith("$MEM"):
                raise IllegalOperationError(f"{ref} 是设备挂载点或指向设备的链接，不能删除（挂载点请用 unmount/mount）")
            deleted_type = "link"
        elif isinstance(node, (dict, MetaDict)):
            size = len(node.to_dict()) if isinstance(node, MetaDict) else len(node)
            if size > 0:
                raise VMMemoryError(
                    f"非空容器不能删除：{ref}（整棵子树删除保留给内核态对话，尚未实现）"
                )
            deleted_type = "dict"
        elif isinstance(node, (list, MetaList)):
            size = len(node.to_list()) if isinstance(node, MetaList) else len(node)
            if size > 0:
                raise VMMemoryError(
                    f"非空容器不能删除：{ref}（整棵子树删除保留给内核态对话，尚未实现）"
                )
            deleted_type = "list"
        else:
            deleted_type = "str"

        if isinstance(container, (dict, MetaDict)):
            del container[last]
        else:
            del container[idx]  # 列表元素删除后，后续元素下标前移
        return deleted_type, ref

    def set(self, ref: str, value: Any, owner_cid: Optional[int] = None,
            space: Optional[AddressSpace] = None) -> Optional[str]:
        """
        根据引用设置值
        :param ref: 引用字符串，如 '$MEM.key'、'$work.key' 或 '$data.key.subkey'
        :param value: 要设置的值
        :param owner_cid: 新创建容器节点的所有者 cid（可选）
        :param space: 地址上下文（根表）；None = 只允许 $MEM 根
        :return: 追加到列表时的地址（请求别名形态，如 '$data.list.3'）；普通写入返回 None
        """
        real, display = self._expand_write_ref(ref, space)
        return self.set_by_path(real, value, owner_cid=owner_cid, display=display)

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
            if isinstance(container, LinkNode):
                raise VMMemoryError(
                    f"{ref} 落在链接上：调试原语只作用于数据节点，链接请用 unmount/mount"
                )
            self._check_access(container, key, path)
            container = container[self._index_key(container, key)]
        if isinstance(container, (dict, MetaDict)) and isinstance(container.get(path[-1]), LinkNode):
            raise VMMemoryError(
                f"{ref} 落在链接上：调试原语只作用于数据节点，链接请用 unmount/mount"
            )
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

        三件事一起做：登记地址（设备表保留地址）、登记设备标识符、在挂载点物化一个
        链接节点（这样挂载点出现在内存树里，可枚举、可持久化）。
        标识符缺省取挂载地址；父段不存在时自动物化空 MetaDict。

        :param path: 路径字符串，如 'sys.log'（不包含 $MEM 前缀）
        :param device: 要挂载的设备
        """
        if not isinstance(device, MemoryDevice):
            raise VMMemoryError(f"必须挂载 MemoryDevice 类型的设备， got {type(device).__name__}")
        if not isinstance(path, str) or not path:
            raise VMMemoryError("挂载路径必须是非空字符串")

        device_id = getattr(device, "device_id", None) or path
        if not isinstance(device_id, str) or not device_id:
            raise VMMemoryError(f"设备标识符必须是非空字符串：{device_id!r}")
        if device_id.startswith("$"):
            raise VMMemoryError(f"设备标识符不能以 $ 开头：{device_id!r}")
        holder = self._device_ids.get(device_id)
        if holder is not None and holder is not device:
            raise VMMemoryError(f"设备标识符重复：{device_id!r}")
        # 挂载点唯一：同一路径上已有另一台设备时拒绝。容忍覆盖会让旧设备的标识符
        # 失去全部挂载地址，设备表从此不一致（丙1 的不变量会在安全点报出来）。
        mounted = self._devices.get(path)
        if mounted is not None and mounted is not device:
            raise VMMemoryError(f"挂载路径已被另一台设备占用：{path}")

        self._plant_link(path, device_id, type(device).__name__)
        self._devices[path] = device
        self._device_ids[device_id] = device

    def unmount(self, path: str) -> None:
        """
        从指定路径卸载设备（同时摘除挂载点的链接节点与标识符登记）
        :param path: 路径字符串
        """
        if path not in self._devices:
            raise VMMemoryError(f"路径 {path} 没有挂载设备")
        device = self._devices.pop(path)
        device_id = getattr(device, "device_id", None) or path
        if self._device_ids.get(device_id) is device:
            del self._device_ids[device_id]
        self._remove_link(path)

    def _plant_link(self, path: str, device_id: str, metadata: str) -> None:
        """在挂载点物化链接节点；父段不存在则自动建空 MetaDict。

        挂载点已有普通数据节点、或父段是链接节点时拒绝——挂载点与数据节点不能同名，
        链接节点里也放不下第二个挂载点（同时挂 box 与 box.dev 属于这种冲突）。
        """
        segments = path.split(".")
        node: Any = self._data
        for i, seg in enumerate(segments[:-1]):
            if isinstance(node, LinkNode):
                raise VMMemoryError(f"挂载点 {path} 落在链接 {'.'.join(segments[:i])} 里")
            if not isinstance(node, (dict, MetaDict)):
                raise VMMemoryError(f"挂载路径 {path} 的父段不是容器：{'.'.join(segments[:-1])}")
            if seg not in node:
                node[seg] = MetaDict(data={})
            node = node[seg]
        if isinstance(node, LinkNode):
            raise VMMemoryError(f"挂载点 {path} 落在链接里，链接节点里放不下第二个挂载点")
        if not isinstance(node, (dict, MetaDict)):
            raise VMMemoryError(f"挂载路径 {path} 的父段不是容器：{'.'.join(segments[:-1])}")
        last = segments[-1]
        existing = node.get(last)
        if existing is not None and not isinstance(existing, LinkNode):
            raise VMMemoryError(f"挂载点 {path} 已有数据节点，拒绝覆盖")
        node[last] = LinkNode(target=device_id, metadata=metadata)

    def _remove_link(self, path: str) -> None:
        """摘除挂载点的链接节点（找不到就跳过：设备表是权威）。"""
        segments = path.split(".")
        node: Any = self._data
        for seg in segments[:-1]:
            if not isinstance(node, (dict, MetaDict)) or seg not in node:
                return
            node = node[seg]
        if not isinstance(node, (dict, MetaDict)):
            return
        if isinstance(node.get(segments[-1]), LinkNode):
            del node[segments[-1]]

    def materialize_mount_links(self) -> None:
        """按设备表重建全部挂载点的链接节点（幂等）。

        用途：整树替换之后把已挂载设备重新物化进树。镜像装载会在
        `core.mem._data = root` 之后调用——Core 的内建设备在 `Core.__init__`
        里挂载，早于整树替换，不重建就只剩设备表、树里看不到挂载点。
        """
        for path, device in list(self._devices.items()):
            device_id = getattr(device, "device_id", None) or path
            self._plant_link(path, device_id, type(device).__name__)

    def is_device_path(self, path: list) -> bool:
        """检查路径是否对应设备"""
        path_str = ".".join(path)
        return path_str in self._devices

    def get_device(self, path: list) -> Optional[MemoryDevice]:
        """获取路径对应的设备"""
        path_str = ".".join(path)
        return self._devices.get(path_str)

    def split_path(self, ref: str, space: Optional[AddressSpace] = None):
        """把根引用拆分为 memory 或 device 路径。

        返回：
        - ("memory", segments)
        - ("device", mount_key, remainder)
        space=None 时只接受 `$MEM` 引用（旧行为）；提供 space 时接受根名。
        段序列按最长设备挂载键匹配；`box` 不会错误匹配 `$MEM.box2`。
        设备路径简写（`$inputs...`）不是根引用，仍按旧行为拒绝。
        """
        parts = self._parse_ref(ref)
        kind, real, display = self._expand_ref_parts(parts, space)
        if kind != "memory":
            raise ValueError(f"split_path 只接受根名引用：{ref!r}")
        # 最长挂载键优先：同时挂载 box 与 box.dev 时，$MEM.box.dev 命中 box.dev。
        for i in range(len(real), 0, -1):
            prefix = ".".join(real[:i])
            if prefix in self._devices:
                return ("device", prefix, real[i:])
        return ("memory", real)

    def make(self, ref: str, key: str, mem_type: str, owner_cid: Optional[int] = None,
             space: Optional[AddressSpace] = None) -> str:
        """在指定路径创建新地址，返回地址字符串（请求别名形态，追加时带真实下标）。

        `ref` 是路径：缺失的名字段按下一段推断建节点（下一段是 `-1` → list，否则 dict）；
        `-1` 段表示在列表末尾追加一个容器节点（中间段只能是 dict，非空列表要同类）并继续。
        `key` 是单段：名字 → 在 ref 落点创建该名字的节点（已存在报错，只创建不覆盖）；
        `-1` → 在 ref 落点（必须是 list）追加一个 `mem_type` 类型的节点。

        `-1` 只表示追加位置，不是下标：`-2` 及以下一律报错。
        """
        if mem_type not in ('str', 'dict', 'list'):
            raise IllegalOperationError(f"不支持的类型：{mem_type}，必须是 'str', 'dict', 'list' 之一")

        check_new_key(key)
        parts = self._parse_ref(ref)
        kind, real_ref, display = self._expand_ref_parts(parts, space)

        # 根别名：目标本身必须存在（根无效），不通过别名补建根目标
        if space is not None and kind == "memory" and parts[0] != MEM_ROOT_NAME:
            target = space.root(parts[0])
            if target is not None and not self._root_target_complete(list(target)):
                raise MemoryLinkBrokenError(f"引用根无效：{parts[0]!r}（目标不存在）")

        # 先按引用解析（链接与设备别名透明）：能解析就直接当落点用
        node = None
        if kind == "memory":
            try:
                node = self._resolve_mem_path(list(real_ref), for_llm=False, display=list(display))
            except MemoryCircularReferenceError:
                raise
            except VMMemoryError:
                node = None
        else:
            try:
                node = self._resolve_path(list(real_ref), for_llm=False, space=space)
            except MemoryCircularReferenceError:
                raise
            except VMMemoryError:
                node = None

        address_parts = list(display)
        if node is None:
            if kind == "device":
                raise VMMemoryError(f"不支持在设备路径下创建新键：{ref}")
            split = self.split_path(ref, space=space)
            if split[0] != "memory":
                raise VMMemoryError(f"不支持在设备路径下创建新键：{ref}")
            real_segments = list(split[1])
            request_rest = list(parts[1:])
            offset = len(real_segments) - len(request_rest)
            node, address_parts = self._walk_make_path(
                real_segments, list(display), offset, ref, owner_cid
            )

        if isinstance(node, (dict, MetaDict)):
            if key == "-1":
                raise IllegalOperationError(f"字典里不能用 -1（-1 只表示列表追加位置）：{ref}")
            if key in node:
                raise VMMemoryError(
                    f"地址已存在：{_show(address_parts + [key])}——memory_make 只创建新地址，不覆盖"
                )
            node[key] = self._new_node(mem_type, owner_cid)
            return _show(address_parts + [key])
        if isinstance(node, (list, MetaList)):
            if key != "-1":
                raise IllegalOperationError(
                    f"列表位置的 key 只能是 -1（追加），得到 {key!r}；列表长度 {len(node)}"
                )
            # 仅新列表可随意指定类型；列表已有值时必须与已有元素同类型
            self._check_append_same_type(node, mem_type)
            node.append(self._new_node(mem_type, owner_cid))
            return _show(address_parts + [str(len(node) - 1)])
        if isinstance(node, str):
            raise VMMemoryError(f"不能在字符串类型的路径下创建新键：{ref}")
        raise VMMemoryError(f"不支持在类型 {type(node).__name__} 下创建新键：{ref}")

    def _root_target_complete(self, target_segments: list) -> bool:
        """根目标是否完整存在（不触发设备 resolve_path 副作用）。"""
        result = self._walk_links(list(target_segments))
        return not (result[0] == "memory" and result[2] is False)

    @staticmethod
    def _list_elem_type(node: list) -> str:
        """列表的元素类型（同类规则的依据）。混合列表报错：类型不唯一时不做猜测。"""
        kinds = set()
        for item in node:
            if isinstance(item, MetaDict) or isinstance(item, dict):
                kinds.add("dict")
            elif isinstance(item, MetaList) or isinstance(item, list):
                kinds.add("list")
            elif isinstance(item, LinkNode):
                kinds.add("link")
            else:
                kinds.add("str")
        if not kinds:
            return "str"  # 空列表默认 str（write 追加的默认类型）
        if len(kinds) > 1:
            raise VMMemoryError(
                f"列表元素类型不一致：{sorted(kinds)}，同类规则的依据不唯一（该列表来自旧数据，需重建）"
            )
        return kinds.pop()

    @staticmethod
    def _check_append_same_type(node: list, append_kind: str) -> None:
        """非空列表追加必须与已有元素同类型；空列表不检查（新列表可任意指定）。"""
        if not node:
            return
        elem = Memory._list_elem_type(node)
        if elem != append_kind:
            raise VMMemoryError(
                f"同类节点必须与已有元素同类型：列表元素是 {elem}，不能追加 {append_kind}"
            )

    @staticmethod
    def _new_node(mem_type: str, owner_cid: Optional[int] = None):
        """按 mem_type 造一个新节点（str 叶子不带 ctrl，容器写 owner）。"""
        if mem_type == 'str':
            return _with_owner(StrNode(""), owner_cid)
        if mem_type == 'dict':
            return _with_owner(MetaDict(data={}), owner_cid)
        return _with_owner(MetaList(data=[]), owner_cid)

    def _walk_make_path(self, segments: list, display: list, offset: int, ref: str,
                        owner_cid: Optional[int] = None) -> tuple:
        """沿 make 的真实路径下钻并建出缺失段，返回 (终端容器, 地址显示段)。

        先找最深可解析前缀（走既有 resolve_path，链接透明），再在普通 dict/list 上补建：
        缺失名字段按下一段推断类型（`-1` → list，否则 dict）；`-1` 段追加 dict 容器。
        `display` 是请求显示段（含根名）；`offset` 是真实段里属于根目标的前缀长度，
        用于把 `-1` 追加段换回真实下标写进地址。错误信息用请求引用（不披露目标路径）。
        """
        node = self._data
        existing = 0
        for i in range(len(segments), 0, -1):
            try:
                node = self.resolve_path(["MEM"] + list(segments[:i]), for_llm=False)
                existing = i
                break
            except MemoryCircularReferenceError:
                raise
            except VMMemoryError:
                continue
        address = list(display)

        def display_index(real_index: int):
            return 1 + (real_index - offset) if real_index >= offset else None

        for j in range(existing, len(segments)):
            seg = segments[j]
            next_seg = segments[j + 1] if j + 1 < len(segments) else None
            if isinstance(node, (dict, MetaDict)):
                if seg == "-1":
                    raise IllegalOperationError(f"字典里不能用 -1（-1 只表示列表追加位置）：{ref}")
                if seg in node:
                    node = node[seg]
                    continue
                created = self._new_node('list' if next_seg == "-1" else 'dict', owner_cid)
                node[seg] = created
                node = created
                continue
            if isinstance(node, (list, MetaList)):
                if seg != "-1":
                    raise VMMemoryError(
                        f"列表位置的路径段只能是 -1（追加），得到 {seg!r}：{ref}"
                    )
                self._check_append_same_type(node, "dict")
                created = self._new_node('dict', owner_cid)
                node.append(created)
                di = display_index(j)
                if di is not None:
                    address[di] = str(len(node) - 1)
                node = created
                continue
            if isinstance(node, LinkNode):
                raise VMMemoryError(f"不能通过链接创建节点（链接目标不存在）：{ref}")
            if isinstance(node, str):
                raise VMMemoryError(f"不能在字符串类型的路径下创建新键：{ref}")
            raise VMMemoryError(f"不支持在类型 {type(node).__name__} 下创建新键：{ref}")
        return node, address

    def to_dict(self) -> dict:
        """返回底层字典的副本"""
        return self._data.copy()

    def to_image_dict(self) -> dict:
        """内存树的显式节点表示（与镜像 mem 段、persist_to、内核转储同格式）。

        不可 JSON 化的值降级为 `str(value)`（与既有行为一致）。
        """
        def _serialize(value):
            if isinstance(value, MetaDict):
                node = {"kind": "dict", "value": {k: _serialize(v) for k, v in value.items()}}
            elif isinstance(value, MetaList):
                node = {"kind": "list", "value": [_serialize(v) for v in value]}
            elif isinstance(value, LinkNode):
                node = {"kind": "link", "target": value.target, "value": None}
            else:
                node = {"kind": "str", "value": str(value)}
                ctrl = value.get_ctrl() if isinstance(value, StrNode) else None
                if ctrl is not None:
                    node["ctrl"] = ctrl
                return node
            meta = value.get_metadata() if hasattr(value, "get_metadata") else None
            ctrl = value.get_ctrl() if hasattr(value, "get_ctrl") else None
            if meta is not None:
                node["meta"] = meta
            if ctrl is not None:
                node["ctrl"] = ctrl
            return node

        return _serialize(self._data)

    def save(self, filepath: str) -> None:
        """将内存数据持久化到 JSON 文件（仅 _data，不含设备）。
        格式与系统镜像 mem 段一致：显式节点 {kind, meta, ctrl, value}。"""
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_image_dict(), f, ensure_ascii=False, indent=2)

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
            if value.get("kind") in ("str", "dict", "list", "link"):
                kind = value["kind"]
                v = value.get("value")
                meta = value.get("meta")
                ctrl = value.get("ctrl")
                if kind == "link":
                    target = value.get("target")
                    if not isinstance(target, str) or not target:
                        raise VMMemoryError(f"link 节点的 target 必须是非空字符串：{target!r}")
                    return LinkNode(target=target, metadata=meta, ctrl=ctrl)
                if kind == "str":
                    return StrNode(v if isinstance(v, str) else str(v), ctrl=ctrl)
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

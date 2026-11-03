"""内核信息与控制设备：观测面只读，控制面可写（批次三）。

观测面：$MEM.scheduler、$MEM.memory、$MEM.monitor（只读）。
控制面：$MEM.conversations、$MEM.listeners（读实时状态；写固定字段直接改
Core 运行时对象，不写内存树）。设备只做结构校验，授权由镜像层事件监听器承担。
"""

import json

from .capabilities import (
    CONTROL_DEVICE_ROOTS, check_root_name, grant_target_segments, path_is_prefix,
)
from .exceptions import IllegalOperationError, VMMemoryError
from .memory_device import MemoryDevice

# 监听器 priority 的结构上限（非负整数）
MAX_LISTENER_PRIORITY = 10000


def _caller_is_kernel(core, owner_cid) -> bool:
    conv = core._conv_by_cid.get(owner_cid)
    return conv is not None and conv.caps.mode == "kernel"


def _parse_json_field(value, field: str):
    """控制面写入值都是 memory_write 的字符串内容：解析 JSON 文本。"""
    if not isinstance(value, str):
        raise IllegalOperationError(f"{field} 必须是 JSON 文本（memory_write 的 content 是字符串）")
    try:
        return json.loads(value)
    except ValueError as e:
        raise IllegalOperationError(f"{field} 不是合法 JSON：{e}")


def _state_of(core, cid):
    if core._active_cid == cid:
        return "active"
    if cid in core._dormant_cids:
        return "dormant"
    if cid in core._ready_cids:
        return "ready"
    return "finished"


class _MetaProxy:
    """$MEM.conversations.<cid>.meta 的单键写代理。"""

    def __init__(self, device, cid):
        self._device = device
        self._cid = cid

    def __setitem__(self, key, value):
        self._device._write_meta_key(self._cid, key, value)

    def __getitem__(self, key):
        return self._device._read_meta_key(self._cid, key)


class _ConversationProxy:
    """$MEM.conversations.<cid> 的字段写代理（set_by_path 的设备入口）。"""

    def __init__(self, device, cid):
        self._device = device
        self._cid = cid

    def __getitem__(self, field):
        if field == "meta":
            return _MetaProxy(self._device, self._cid)
        return self._device._read_field(self._cid, field)

    def __setitem__(self, field, value):
        self._device._write_field(self._cid, field, value)


class ConversationsDevice(MemoryDevice):
    """$MEM.conversations：读对话列表/详情；写 groups/roots/meta（批次三控制面）。"""

    WRITABLE_FIELDS = ("groups", "roots", "meta")
    READABLE_FIELDS = ("groups", "roots", "meta", "mode", "tools")

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        """按状态分组列出存活对话；finished 与运行态分开，cid 空洞不做伪造补全。"""
        groups = {}
        for cid, conv in self._core._conv_by_cid.items():
            if self._core.is_hidden_cid(cid):
                continue
            groups.setdefault(_state_of(self._core, cid), []).append((cid, conv))
        lines = ["对话列表（按状态分组；cid 空洞表示已关闭或已回收的个体，不保留记录）："]
        for state in ("active", "ready", "dormant", "finished"):
            entries = groups.get(state)
            if not entries:
                continue
            lines.append(f"  {state}:")
            for cid, conv in entries:
                lines.append(
                    f"    cid={cid} {conv.identity} para={conv.para_ref} python={conv.is_python} "
                    f"mode={conv.caps.mode} groups={list(conv.caps.groups)}"
                )
        return "\n".join(lines)

    def _conv_read(self, cid: int):
        conv = self._core._conv_by_cid.get(cid)
        if conv is None or self._core.is_hidden_cid(cid):
            raise VMMemoryError(f"对话 {cid} 不存在")
        return conv

    def _conv_write(self, cid: int):
        conv = self._conv_read(cid)
        if cid in self._core._finished:
            raise VMMemoryError(f"对话 {cid} 已关闭")
        return conv

    @staticmethod
    def _parse_cid(raw):
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise VMMemoryError(f"cid 必须是数字: {raw}")

    def _in_workspace(self, target) -> bool:
        region = tuple(self._core.workspace_region)
        return path_is_prefix(region, tuple(target))

    def _workspace_roots(self, conv):
        """Core 拥有的根：工作空间根（按 workspace_root_name/区域识别）与能力自述根。"""
        result = {}
        name = getattr(conv, "workspace_root_name", None)
        for root_name, target in conv.caps.roots.items():
            if root_name == "MEM":
                continue
            if root_name in ("caps", name) or self._in_workspace(target):
                result[root_name] = tuple(target)
        return result

    def _read_roots_json(self, conv) -> str:
        core_roots = self._workspace_roots(conv)
        data = {}
        for name, target in conv.caps.roots.items():
            if name == "MEM":
                continue
            data[name] = {
                "target": "$MEM." + ".".join(target),
                "core": name in core_roots,
            }
        return json.dumps(data, ensure_ascii=False, sort_keys=True)

    def _read_field(self, cid: int, field: str):
        conv = self._conv_read(cid)
        if field == "groups":
            return json.dumps(list(conv.caps.groups), ensure_ascii=False)
        if field == "roots":
            return self._read_roots_json(conv)
        if field == "meta":
            return json.dumps(dict(conv.attrs or {}), ensure_ascii=False, sort_keys=True)
        if field == "mode":
            return conv.caps.mode
        if field == "tools":
            return json.dumps(sorted(conv.caps.tools), ensure_ascii=False)
        raise VMMemoryError(
            f"$MEM.conversations.<cid> 字段只支持 {list(self.READABLE_FIELDS)}，得到 {field!r}"
        )

    def _read_meta_key(self, cid: int, key: str) -> str:
        conv = self._conv_read(cid)
        attrs = dict(conv.attrs or {})
        if key not in attrs:
            raise VMMemoryError(f"meta 中没有键 {key!r}")
        return str(attrs[key])

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def _write_field(self, cid: int, field: str, value) -> None:
        conv = self._conv_write(cid)
        if field == "groups":
            data = _parse_json_field(value, "groups")
            if not isinstance(data, list):
                raise IllegalOperationError("groups 必须是 JSON 字符串数组")
            groups = []
            for item in data:
                if not isinstance(item, str) or not item.strip():
                    raise IllegalOperationError(f"组名必须是非空字符串，得到 {item!r}")
                if item not in groups:
                    groups.append(item.strip())
            conv.caps = conv.caps.copy(groups=tuple(groups))
            return
        if field == "roots":
            self._write_roots(conv, value)
            return
        if field == "meta":
            data = _parse_json_field(value, "meta")
            if not isinstance(data, dict):
                raise IllegalOperationError("meta 必须是 JSON 对象（字符串 -> 字符串）")
            attrs = {}
            for key, val in data.items():
                if not isinstance(key, str) or not isinstance(val, str):
                    raise IllegalOperationError("meta 的键和值都必须是字符串")
                attrs[key] = val
            conv.attrs = attrs
            return
        if field in ("mode", "tools"):
            raise IllegalOperationError(f"{field} 是只读字段：运行期改写留到后续批次")
        raise IllegalOperationError(
            f"$MEM.conversations.<cid> 可写字段只有 {list(self.WRITABLE_FIELDS)}，得到 {field!r}"
        )

    def _write_meta_key(self, cid: int, key: str, value) -> None:
        conv = self._conv_write(cid)
        if not isinstance(key, str) or not key:
            raise IllegalOperationError("meta 键必须是非空字符串")
        if not isinstance(value, str):
            raise IllegalOperationError("meta 值必须是字符串")
        attrs = dict(conv.attrs or {})
        attrs[key] = value
        conv.attrs = attrs

    def _write_roots(self, conv, value) -> None:
        if conv.caps.mode == "kernel":
            raise IllegalOperationError(
                "内核态对话的根表固定含 MEM，不能通过控制面改写（用户态对话才可授予根）"
            )
        data = _parse_json_field(value, "roots")
        if not isinstance(data, dict):
            raise IllegalOperationError("roots 必须是 JSON 对象（根名 -> $MEM 路径）")
        core_roots = self._workspace_roots(conv)
        new_roots = dict(core_roots)
        for name, ref in data.items():
            if not isinstance(name, str) or not name:
                raise IllegalOperationError(f"根名必须是非空字符串，得到 {name!r}")
            check_root_name(name)
            if name in core_roots:
                raise IllegalOperationError(f"根名 {name!r} 由 Core 分配，不能通过控制面覆盖")
            if not isinstance(ref, str):
                raise IllegalOperationError(f"根 {name!r} 的目标必须是 $MEM 引用字符串")
            new_roots[name] = grant_target_segments(ref)
        self._validate_grants(conv, new_roots, core_roots)
        conv.caps = conv.caps.copy(roots=new_roots)

    def _validate_grants(self, conv, new_roots, core_roots) -> None:
        """先全量校验后提交：保留区、前缀无关、工作空间绑定、设备路径。"""
        core = self._core
        region = tuple(core.workspace_region)
        grants = [(name, tuple(target)) for name, target in new_roots.items()
                  if name not in core_roots]
        for name, target in grants:
            if not target:
                raise IllegalOperationError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if path_is_prefix(region, target) or path_is_prefix(target, region):
                raise IllegalOperationError(f"根 {name!r} 的目标落在保留的工作空间区域")
            if target[0] in CONTROL_DEVICE_ROOTS:
                raise IllegalOperationError(
                    f"根 {name!r} 的目标落在内核控制面设备路径 $MEM.{target[0]}"
                )
            for mount in core.mem._devices:
                mount_segments = tuple(mount.split("."))
                if target == mount_segments:
                    continue  # 显式授予设备本身是允许的
                if path_is_prefix(target, mount_segments) or path_is_prefix(mount_segments, target):
                    raise IllegalOperationError(
                        f"根 {name!r} 的目标落在保留的内核设备路径 $MEM.{mount}"
                    )
        # 新根之间前缀无关（相等目标允许共享）
        for i, (n1, t1) in enumerate(grants):
            for n2, t2 in grants[i + 1:]:
                if t1 == t2:
                    continue
                if path_is_prefix(t1, t2) or path_is_prefix(t2, t1):
                    raise IllegalOperationError(f"根 {n1!r} 与 {n2!r} 的目标前缀冲突")
        # 与其他存活对话的根表、根注册表比较
        others = []
        for cid, other in core._conv_by_cid.items():
            if cid == conv.cid:
                continue
            for name, target in other.caps.roots.items():
                if name == "MEM":
                    continue
                others.append((cid, name, tuple(target), self._in_workspace(target)))
        for name, target in core.root_registry.items():
            others.append((None, name, tuple(target), False))
        for name, target in grants:
            for cid, other_name, other_target, other_is_workspace in others:
                if target == other_target:
                    if other_is_workspace:
                        raise IllegalOperationError(
                            f"根 {name!r} 不能绑定对话 {cid} 的工作空间目标"
                        )
                    continue  # 相同目标允许共享
                if path_is_prefix(target, other_target) or path_is_prefix(other_target, target):
                    if other_is_workspace:
                        raise IllegalOperationError(
                            f"根 {name!r} 与其他对话 {cid} 的工作空间目标重叠"
                        )
                    raise IllegalOperationError(
                        f"根 {name!r} 与已有根 {other_name!r} 的目标前缀冲突"
                    )

    # ------------------------------------------------------------------
    # 设备接口
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        return _ConversationProxy(self, self._parse_cid(key))

    def resolve_path(self, path):
        if not path:
            raise VMMemoryError("$MEM.conversations 根路径请直接读设备自述")
        cid = self._parse_cid(path[0])
        if len(path) == 1:
            conv = self._conv_read(cid)
            return (
                f"cid={cid} name={conv.name} identity={conv.identity} "
                f"state={_state_of(self._core, cid)} is_sub={conv.is_sub} "
                f"is_python={conv.is_python} para={conv.para_ref} messages={len(conv.messages)} "
                f"mode={conv.caps.mode} groups={list(conv.caps.groups)} "
                f"roots={sorted(conv.caps.roots)} tools={len(conv.caps.tools)} "
                f"meta_keys={sorted((conv.attrs or {}))}"
            )
        if len(path) == 2:
            return self._read_field(cid, path[1])
        if len(path) == 3 and path[1] == "meta":
            return self._read_meta_key(cid, path[2])
        raise VMMemoryError(
            "$MEM.conversations 支持 <cid>、<cid>.groups/roots/meta/mode/tools、"
            "<cid>.meta.<key>"
        )

    def authorize_listener(self, event_type, remainder, owner_cid):
        return _caller_is_kernel(self._core, owner_cid)

    def after_event(self, event_type, remainder, result):
        if event_type != "memory_write" or len(remainder) < 2:
            return None
        try:
            cid = int(remainder[0])
        except (TypeError, ValueError):
            return None
        field = ".".join(str(x) for x in remainder[1:])
        return {"cid": cid, "field": field, "value": (result or {}).get("content")}


class SchedulerDevice(MemoryDevice):
    """$MEM.scheduler：调度状态（活跃/就绪/休眠 + 指令预算）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def _visible(self, cid):
        return None if self._core.is_hidden_cid(cid) else cid

    def _visible_list(self, cids):
        return [cid for cid in cids if not self._core.is_hidden_cid(cid)]

    def to_llm_string(self):
        return (
            f"active={self._visible(self._core._active_cid)}\n"
            f"ready={self._visible_list(self._core._ready_cids)}\n"
            f"dormant={self._visible_list(self._core._dormant_cids)}\n"
            f"instruction_budget={self._core._instruction_budget} "
            f"(当前链计数={self._core._activation_instructions})"
        )

    def resolve_path(self, path):
        if len(path) != 1 or path[0] not in ("active", "ready", "dormant", "budget"):
            raise VMMemoryError("$MEM.scheduler 支持 active/ready/dormant/budget")
        return {
            "active": self._visible(self._core._active_cid),
            "ready": self._visible_list(self._core._ready_cids),
            "dormant": self._visible_list(self._core._dormant_cids),
            "budget": f"{self._core._activation_instructions}/{self._core._instruction_budget}",
        }[path[0]]


class MemorySummaryDevice(MemoryDevice):
    """$MEM.memory：内存树摘要（顶层键 + 节点统计）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        from .types import MetaDict, MetaList, LinkNode

        root = self._core.mem._data
        counts = {"dict": 0, "list": 0, "str": 0, "link": 0}

        def walk(v):
            if isinstance(v, MetaDict):
                counts["dict"] += 1
                for x in v.values():
                    walk(x)
            elif isinstance(v, MetaList):
                counts["list"] += 1
                for x in v:
                    walk(x)
            elif isinstance(v, LinkNode):
                # 链接只计数，不跟随：摘要是只读展示，跟随会带来寻址副作用
                counts["link"] += 1
            else:
                counts["str"] += 1

        walk(root)
        return f"内存树: 顶层键={list(root.keys())} 节点统计={counts}"

    def resolve_path(self, path):
        raise VMMemoryError("$MEM.memory 只读摘要；读取具体路径请用 memory_read")


class _ListenerProxy:
    """$MEM.listeners.<id> 的字段写代理。"""

    def __init__(self, device, listener_id):
        self._device = device
        self._listener_id = listener_id

    def __getitem__(self, field):
        return self._device._read_field(self._listener_id, field)

    def __setitem__(self, field, value):
        self._device._write_field(self._listener_id, field, value)


class ListenersDevice(MemoryDevice):
    """$MEM.listeners：读监听器列表/详情；写 priority/enabled（批次三控制面）。"""

    WRITABLE_FIELDS = ("priority", "enabled")
    READABLE_FIELDS = ("priority", "enabled", "event_type", "target", "phase", "tier", "handler_ref")

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    @staticmethod
    def _format(l):
        text = (
            f"listener_id={l['listener_id']} owner_cid={l['owner_cid']} "
            f"event_type={l['event_type']} phase={l.get('phase', 'all')} "
            f"target={l['target']} handler_ref={l['handler_ref']} "
            f"enabled={l.get('enabled', True)} "
            f"priority={l.get('priority', 0)} tier={l.get('tier', 'normal')}"
        )
        if l.get("error_count"):
            text += f" error_count={l['error_count']} last_error={l.get('last_error', '')}"
        return text

    def to_llm_string(self):
        lines = ["  " + self._format(l) for l in self._core.events.listeners]
        return "监听器列表:\n" + ("\n".join(lines) if lines else "（无）")

    def _record(self, listener_id: int) -> dict:
        for l in self._core.events.listeners:
            if l["listener_id"] == listener_id:
                return l
        raise VMMemoryError(f"监听器 {listener_id} 不存在")

    @staticmethod
    def _parse_id(raw):
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise VMMemoryError(f"listener_id 必须是数字: {raw}")

    def _read_field(self, listener_id: int, field: str):
        record = self._record(listener_id)
        if field == "priority":
            return str(record.get("priority", 0))
        if field == "enabled":
            return "true" if record.get("enabled", True) else "false"
        if field in ("event_type", "target", "phase", "tier", "handler_ref"):
            return str(record.get(field, ""))
        raise VMMemoryError(
            f"$MEM.listeners.<id> 字段只支持 {list(self.READABLE_FIELDS)}，得到 {field!r}"
        )

    def _write_field(self, listener_id: int, field: str, value) -> None:
        record = self._record(listener_id)
        if field == "priority":
            if not isinstance(value, str) or not value.isdigit():
                raise IllegalOperationError(f"priority 必须是非负整数，得到 {value!r}")
            priority = int(value)
            if priority > MAX_LISTENER_PRIORITY:
                raise IllegalOperationError(
                    f"priority 不能超过 {MAX_LISTENER_PRIORITY}，得到 {priority}"
                )
            record["priority"] = priority
            return
        if field == "enabled":
            if value not in ("true", "false"):
                raise IllegalOperationError(f"enabled 必须是 true/false，得到 {value!r}")
            record["enabled"] = (value == "true")
            return
        raise IllegalOperationError(
            f"$MEM.listeners.<id> 可写字段只有 {list(self.WRITABLE_FIELDS)}，得到 {field!r}"
        )

    # ------------------------------------------------------------------
    # 设备接口
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        return _ListenerProxy(self, self._parse_id(key))

    def resolve_path(self, path):
        if not path:
            raise VMMemoryError("$MEM.listeners 根路径请直接读设备自述")
        listener_id = self._parse_id(path[0])
        if len(path) == 1:
            return self._format(self._record(listener_id))
        if len(path) == 2:
            return self._read_field(listener_id, path[1])
        raise VMMemoryError("$MEM.listeners 支持 <id> 与 <id>.<字段>")

    def authorize_listener(self, event_type, remainder, owner_cid):
        return _caller_is_kernel(self._core, owner_cid)

    def after_event(self, event_type, remainder, result):
        if event_type != "memory_write" or len(remainder) < 2:
            return None
        try:
            listener_id = int(remainder[0])
        except (TypeError, ValueError):
            return None
        return {"listener_id": listener_id, "field": remainder[1],
                "value": (result or {}).get("content")}


class RootsDevice(MemoryDevice):
    """$MEM.roots：运行期根注册表控制面（读 JSON；写整体替换或按根名写入）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    @staticmethod
    def _target_ref(target) -> str:
        return "$MEM." + ".".join(target)

    def _table(self) -> dict:
        return {name: self._target_ref(target)
                for name, target in self._core.root_registry.items()}

    def to_llm_string(self):
        return json.dumps(self._table(), ensure_ascii=False, sort_keys=True)

    def resolve_path(self, path):
        if not path:
            return self.to_llm_string()
        if len(path) != 1:
            raise VMMemoryError("$MEM.roots.<根名> 只支持单段子路径")
        name = path[0]
        target = self._core.root_registry.get(name)
        if target is None:
            raise VMMemoryError(f"根注册表中没有 {name!r}")
        return self._target_ref(target)

    def __getitem__(self, name):
        return self.resolve_path([name])

    def __setitem__(self, name, value):
        if not isinstance(name, str) or not name:
            raise IllegalOperationError("根名必须是非空字符串")
        if not isinstance(value, str):
            raise IllegalOperationError("根目标必须是 $MEM 引用字符串")
        new = dict(self._core.root_registry)
        new[name] = grant_target_segments(value)
        try:
            self._core.replace_root_registry(new)
        except VMMemoryError as e:
            raise IllegalOperationError(str(e))

    def set_value(self, value):
        data = _parse_json_field(value, "roots")
        if not isinstance(data, dict):
            raise IllegalOperationError("roots 必须是 JSON 对象（根名 -> $MEM 路径）")
        spec = {}
        for name, ref in data.items():
            if not isinstance(name, str) or not isinstance(ref, str):
                raise IllegalOperationError("roots 的键和值都必须是字符串")
            spec[name] = grant_target_segments(ref)
        try:
            self._core.replace_root_registry(spec)
        except VMMemoryError as e:
            raise IllegalOperationError(str(e))

    def authorize_listener(self, event_type, remainder, owner_cid):
        return _caller_is_kernel(self._core, owner_cid)

    def after_event(self, event_type, remainder, result):
        if event_type != "memory_write":
            return None
        if remainder:
            return {"name": remainder[0], "value": (result or {}).get("content")}
        return {"action": "replace"}


class CapsDevice(MemoryDevice):
    """$MEM.caps：只读能力自述。用户态对话经保留根 `caps` 绑定到 $MEM.caps.<cid>。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def _conv(self, cid: int):
        conv = self._core._conv_by_cid.get(cid)
        if conv is None or self._core.is_hidden_cid(cid):
            raise VMMemoryError(f"对话 {cid} 不存在")
        return conv

    def _summary(self, conv) -> dict:
        core_roots = {"caps"}
        for name, target in conv.caps.roots.items():
            if name in ("MEM", "caps"):
                continue
            if path_is_prefix(tuple(self._core.workspace_region), tuple(target)):
                core_roots.add(name)
        roots = {}
        for name, target in conv.caps.roots.items():
            if name == "MEM":
                continue
            roots[name] = {"core": name in core_roots}
        return {
            "cid": conv.cid,
            "identity": conv.identity,
            "name": getattr(conv, "name", None),
            "mode": conv.caps.mode,
            "groups": list(conv.caps.groups),
            "roots": roots,
            "tools": sorted(conv.caps.tools),
            "workspace": getattr(conv, "workspace_root_name", None),
        }

    def to_llm_string(self):
        cids = [cid for cid in sorted(self._core._conv_by_cid)
                if not self._core.is_hidden_cid(cid)]
        return "能力自述：读 $MEM.caps.<cid> 查询（可用 cid：" + str(cids) + "）"

    def resolve_path(self, path):
        if not path:
            return self.to_llm_string()
        try:
            cid = int(path[0])
        except (TypeError, ValueError):
            raise VMMemoryError(f"cid 必须是数字: {path[0]}")
        conv = self._conv(cid)
        if len(path) == 1:
            return json.dumps(self._summary(conv), ensure_ascii=False, sort_keys=True)
        if len(path) == 2:
            data = self._summary(conv)
            if path[1] in data:
                value = data[path[1]]
                return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            if path[1] in conv.caps.roots and path[1] != "MEM":
                return "$" + path[1]
            raise VMMemoryError(
                "$MEM.caps.<cid> 支持 mode/groups/roots/tools/workspace/identity/name"
            )
        if len(path) == 3 and path[1] == "roots":
            name = path[2]
            if name in conv.caps.roots and name != "MEM":
                return "$" + name
            raise VMMemoryError(f"对话没有根 {name!r}")
        raise VMMemoryError("$MEM.caps 支持 <cid>、<cid>.<字段>、<cid>.roots.<根名>")

    def authorize_listener(self, event_type, remainder, owner_cid):
        return _caller_is_kernel(self._core, owner_cid)


class MonitorDevice(MemoryDevice):
    """$MEM.monitor：监测帧摘要。单帧细节在 frames.<seq>（不写进设备自述）。"""

    def __init__(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        frames = self._core.monitor.frames()
        if not frames:
            return "监测器无帧"
        last = frames[-1]
        return (
            f"监测帧: {len(frames)} 帧（基线+{len(frames) - 1}轮）\n"
            f"最近 seq={last.seq} active={last.sched['active_cid']} "
            f"tools={[t.get('cmd_type') for t in (last.lmu.get('tool_calls') or [])]} "
            f"events={len(last.lmu.get('events') or [])}"
        )

    def resolve_path(self, path):
        """单帧细节：frames.<seq>。

        不在设备自述/to_llm_string 里宣传这个路径：先保证功能全面，读取方式的
        宣传面留给后续安全批次决定。
        """
        if len(path) != 2 or path[0] != "frames":
            raise VMMemoryError("$MEM.monitor 只提供汇总读取")
        try:
            seq = int(path[1])
        except ValueError:
            raise VMMemoryError(f"帧号必须是数字: {path[1]}")
        f = self._core.monitor.frame(seq)
        if f is None:
            raise VMMemoryError(f"帧 {seq} 不存在")
        return (
            f"seq={f.seq} active={self._core.is_hidden_cid(f.sched['active_cid']) and None or f.sched['active_cid']} "
            f"ready={[c for c in f.sched['ready_cids'] if not self._core.is_hidden_cid(c)]} "
            f"dormant={[c for c in f.sched['dormant_cids'] if not self._core.is_hidden_cid(c)]} "
            f"result={f.lmu.get('result')!r} error={f.lmu.get('error')} "
            f"events={json.dumps(f.lmu.get('events') or [], ensure_ascii=False)}"
        )



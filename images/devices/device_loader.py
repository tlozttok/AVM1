"""设备装载入口：申请 / 面板批准 / 挂载三段式。

这是**宿主代码挂载**的入口，不是 VM 的软件包机制：VM 只有内存树、指令与设备；软件包
怎么分发、装到哪里、装什么，是 OS 的事情（OS 层约定见 docs/porting-contract.md）。

挂到 `$MEM.device_loader` 之类的路径上后，地址空间是：

```
读 $MEM.device_loader                  说明书 + 申请摘要（无副作用）
读 $MEM.device_loader.request          申请列表：<名字> <状态> <短哈希>
读 $MEM.device_loader.request.<名字>   申请详情（请求方原样字段、状态、时间、原因）
写 $MEM.device_loader.request.<名字>   提交申请：{"file","class","args"[,"id"]}
写 $MEM.device_loader.resume.<名字>    面板批准后完成挂载（值忽略）
```

审批值（`approved_sha256` / `approved_at` / `approved_via`）由**面板设备动作**在批准时
生成，申请方不提供。把根地址、`.request`、`.resume` 当写入点都会得到 Illegal；旧协议的
`action` / `path` / `approved_*` 字段一律拒绝。

面板通过 `debug_actions()` 看到每一条待批准申请的两个动作：`批准装载 <名字>` 与
`拒绝装载 <名字>`。批准动作只核对文件哈希并把批准记在设备状态里（控制台 HTTP 线程、
不经指令集、不产生事件）；真正的挂载只在写 `resume.<名字>` 的指令里发生，因此 journal
的 `device_mount` 效果、快照恢复与日志重放语义不变。

四条约束沿用：

1. **不实现 `emit_device_event` / `after_event`**：设备不发事件。
2. **错误信息不回显写入原文**：只给原因类别。
3. **成功返回 None**：结果靠观察地址空间（申请详情 / 挂载后的设备）。
4. **申请与批准状态不持久化**：重启后重新申请、重新批准；已挂载设备从 `device_mount`
   记录重放，不重新审批。
"""

import ast
import hashlib
import json
import os
import threading
import time

from avm.exceptions import IllegalOperationError, VMMemoryError
from avm.memory_device import MemoryDevice

_APPROVED_VIA = "panel-device-action"
_LEGACY_FIELDS = ("action", "path", "approved_sha256", "approved_at", "approved_via")
_REQUEST_FIELDS = ("file", "class", "args", "id")


def _sha256_file(path: str) -> str:
    """流式计算文件内容哈希（设备文件可能不小，不整块读进内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_classes_in_file(path: str):
    """从源码里找继承 MemoryDevice 的类名；只做 AST 解析，不 import（不执行设备代码）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        raise IllegalOperationError(f"设备文件无法作为 Python 源码解析：{e}") from e
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else (
                base.attr if isinstance(base, ast.Attribute) else None
            )
            if isinstance(name, str) and name.endswith("MemoryDevice"):
                classes.append(node.name)
                break
    return classes


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class _RequestProxy:
    """`$MEM.device_loader.request.<名字>` 的写代理（设备子地址写的入口）。"""

    def __init__(self, device: "DeviceLoader"):
        self._device = device

    def __setitem__(self, name, value):
        self._device._submit(name, value)

    def __getitem__(self, name):
        raise VMMemoryError(
            "request 子地址只写；读申请进度用 memory_read "
            "$MEM.device_loader.request.<名字>")


class _ResumeProxy:
    """`$MEM.device_loader.resume.<名字>` 的写代理。"""

    def __init__(self, device: "DeviceLoader"):
        self._device = device

    def __setitem__(self, name, value):
        self._device._resume(name)

    def __getitem__(self, name):
        raise VMMemoryError(
            "resume 子地址只写；读申请进度用 memory_read "
            "$MEM.device_loader.request.<名字>")


class DeviceLoader(MemoryDevice):
    """把设备代码挂载拆成申请、人工批准、挂载三步的宿主设备。"""

    def __init__(self, base_dir: str = ""):
        self._base_dir = str(base_dir or "")
        self._core = None
        self._records = {}          # 设备名（挂载名） → 申请记录
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # MemoryDevice 接口
    # ------------------------------------------------------------------

    def attach_core(self, core) -> None:
        self._core = core

    def pretend_as_type(self) -> str:
        return "str"

    def set_value(self, value) -> None:
        raise IllegalOperationError(
            "device_loader 的根地址不接收写入：提交申请写 "
            "$MEM.device_loader.request.<名字>，批准后写 "
            "$MEM.device_loader.resume.<名字>")

    def __getitem__(self, key):
        if key == "request":
            return _RequestProxy(self)
        if key == "resume":
            return _ResumeProxy(self)
        raise VMMemoryError(
            f"device_loader 不支持子地址 {key!r}：可用 request.<名字> / resume.<名字>")

    def __setitem__(self, key, value) -> None:
        raise IllegalOperationError(
            f"device_loader 的 {key!r} 需要一个设备名：提交申请写 "
            "$MEM.device_loader.request.<名字>，完成挂载写 "
            "$MEM.device_loader.resume.<名字>")

    def resolve_path(self, path):
        if not path:
            raise VMMemoryError("读根地址请直接读 $MEM.device_loader")
        head = path[0]
        if head == "request":
            if len(path) == 1:
                return self._requests_text()
            if len(path) == 2:
                return self._record_detail(path[1])
            raise VMMemoryError(
                "$MEM.device_loader.request 只支持 .request 与 .request.<名字>")
        if head == "resume":
            raise VMMemoryError(
                "$MEM.device_loader.resume.<名字> 只写；进度读 "
                "$MEM.device_loader.request.<名字>")
        raise VMMemoryError(
            f"device_loader 不支持子地址 {head!r}：可用 request.<名字> / resume.<名字>")

    def to_llm_string(self) -> str:
        with self._lock:
            statuses = {name: record["status"] for name, record in self._records.items()}
        pending = sorted(name for name, status in statuses.items() if status == "pending")
        approved = sorted(name for name, status in statuses.items() if status == "approved")
        rejected = sorted(name for name, status in statuses.items() if status == "rejected")
        lines = [
            "device_loader：设备装载入口（申请 / 面板批准 / 挂载）",
            '  写 $MEM.device_loader.request.<名字>   提交申请：{"file","class","args"[,"id"]}；不挂载',
            "  读 $MEM.device_loader.request.<名字>   查状态与详情",
            "  写 $MEM.device_loader.resume.<名字>    面板批准后完成挂载（值忽略）",
            "  根地址、.request、.resume 不接受写入；审批值由面板设备动作生成，申请方不提供",
        ]
        if pending:
            lines.append(f"  待批准 {len(pending)} 台：{', '.join(pending)}")
        if approved:
            lines.append(f"  已批准待挂载 {len(approved)} 台：{', '.join(approved)}")
        if rejected:
            lines.append(f"  已拒绝 {len(rejected)} 台：{', '.join(rejected)}")
        if not statuses:
            lines.append("  当前没有装载申请")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 面板动作：批准 / 拒绝（控制台 HTTP 线程；只改设备状态）
    # ------------------------------------------------------------------

    def debug_actions(self) -> list:
        with self._lock:
            pending = [(name, dict(record)) for name, record in sorted(self._records.items())
                       if record["status"] == "pending"]
        actions = []
        for name, record in pending:
            detail = "\n".join(self._record_lines(record))
            actions.append({
                "name": f"批准装载 {name}",
                "description": (
                    "校验设备文件自申请以来没有变化，记下人工批准；不执行挂载。\n"
                    + detail),
                "parameters": {"type": "object", "properties": {}},
                "run": (lambda args, _name=name: self._approve(_name)),
            })
            actions.append({
                "name": f"拒绝装载 {name}",
                "description": "拒绝这次装载申请并记录原因；不执行挂载。\n" + detail,
                "parameters": {"type": "object", "properties": {
                    "reason": {"type": "string", "description": "可选的拒绝原因"}}},
                "run": (lambda args, _name=name: self._reject(
                    _name, (args or {}).get("reason"))),
            })
        return actions

    # ------------------------------------------------------------------
    # 地址内容
    # ------------------------------------------------------------------

    @staticmethod
    def _record_lines(record: dict) -> list:
        args_text = json.dumps(record.get("args") or {}, ensure_ascii=False)
        return [
            f"  设备名（挂载名）：{record['name']}",
            f"  宿主文件：{record['file_abs']}",
            f"  设备类：{record['class']}",
            f"  args：{args_text}",
            f"  sha256：{record['sha256']}",
            f"  申请时间：{record['requested_at']}  申请人 cid：{record['requested_by']}",
            f"  请求方写的 file：{record['file']}",
        ]

    def _requests_text(self) -> str:
        with self._lock:
            records = [(name, dict(record)) for name, record in sorted(self._records.items())]
        if not records:
            return ("没有装载申请。提交申请写 $MEM.device_loader.request.<名字>，"
                    '值为 {"file","class","args"[,"id"]}。')
        lines = ["装载申请："]
        for name, record in records:
            lines.append(
                f"  {name}  status={record['status']}  "
                f"sha256={record['sha256'][:12]}  "
                f"requested_at={record['requested_at']}  by={record['requested_by']}")
        return "\n".join(lines)

    def _record_detail(self, name: str) -> str:
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise VMMemoryError(f"没有设备名 {name!r} 的装载申请记录")
            record = dict(record)
        lines = self._record_lines(record)
        lines.append(f"  状态：{record['status']}")
        if record["status"] == "pending":
            lines.append("  下一步：等操作员在面板 $MEM.device_loader 设备上批准或拒绝")
        elif record["status"] == "approved":
            lines.append(f"  批准 sha256：{record['approved_sha256']}")
            lines.append(f"  批准时间：{record['approved_at']}  批准来源：{record['approved_via']}")
            lines.append(
                f'  下一步：写 $MEM.device_loader.resume.{name} ""（值忽略）完成挂载')
        elif record["status"] == "rejected":
            if record.get("reason"):
                lines.append(f"  拒绝原因：{record['reason']}")
            lines.append("  下一步：重新写 $MEM.device_loader.request.<名字> 提交新申请")
        if record.get("last_error"):
            lines.append(f"  上次挂载失败：{record['last_error']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 申请 / 批准 / 拒绝 / 挂载
    # ------------------------------------------------------------------

    @staticmethod
    def _check_name(name) -> str:
        if not isinstance(name, str) or not name or name != name.strip():
            raise IllegalOperationError("设备名必须是非空字符串（不含首尾空白）")
        if "." in name:
            raise IllegalOperationError("设备名不能含点号：设备名就是单段挂载名")
        if name.startswith("$"):
            raise IllegalOperationError("设备名不能以 $ 开头")
        return name

    @staticmethod
    def _parse_request(value) -> dict:
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise IllegalOperationError("装载申请不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise IllegalOperationError("装载申请必须是 JSON 对象")
        legacy = [key for key in _LEGACY_FIELDS if key in payload]
        if legacy:
            raise IllegalOperationError(
                "装载申请不再接受字段 " + ", ".join(legacy)
                + "：审批值由面板设备动作生成，申请方只提供 file/class/args/id")
        unknown = [key for key in payload if key not in _REQUEST_FIELDS]
        if unknown:
            raise IllegalOperationError(
                "装载申请有未知字段：" + ", ".join(sorted(unknown)))
        file_value = payload.get("file")
        class_name = payload.get("class")
        args = payload.get("args")
        device_id = payload.get("id")
        if not isinstance(file_value, str) or not file_value:
            raise IllegalOperationError("装载申请需要非空字符串 file")
        if not isinstance(class_name, str) or not class_name:
            raise IllegalOperationError("装载申请需要非空字符串 class")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise IllegalOperationError("装载申请的 args 必须是对象")
        if device_id is not None and (not isinstance(device_id, str) or not device_id):
            raise IllegalOperationError("装载申请的 id 必须是非空字符串或省略")
        return {
            "file": file_value,
            "class": class_name,
            "args": args,
            "id": device_id,
        }

    def _check_mount_name_free(self, name: str) -> None:
        core = self._core
        if name in core.mem.mounted_devices():
            raise IllegalOperationError(f"挂载名 {name!r} 已被设备占用")
        try:
            core.mem.resolve_path(["MEM", name], for_llm=False)
        except VMMemoryError:
            return
        raise IllegalOperationError(f"挂载名 {name!r} 已被内存节点占用")

    def _submit(self, name, value) -> None:
        core = self._core
        if core is None:
            raise VMMemoryError("设备装载入口未接入 Core")
        name = self._check_name(name)
        payload = self._parse_request(value)
        file_abs = self._resolve(payload["file"])
        try:
            data_sha = _sha256_file(file_abs)
        except OSError as e:
            raise IllegalOperationError(f"设备文件不可读：{e}") from None
        classes = _device_classes_in_file(file_abs)
        if payload["class"] not in classes:
            raise IllegalOperationError(
                f"文件里没有继承 MemoryDevice 的类 {payload['class']!r}"
                f"（发现：{', '.join(classes) or '无'}）")
        self._check_mount_name_free(name)
        with self._lock:
            existing = self._records.get(name)
            if existing is not None:
                if existing["status"] == "approved":
                    raise IllegalOperationError(
                        f"设备 {name!r} 的申请已批准，先写 "
                        f"$MEM.device_loader.resume.{name} 完成挂载")
                if existing["status"] == "pending":
                    same = (
                        existing["file"] == payload["file"]
                        and existing["class"] == payload["class"]
                        and existing["args"] == payload["args"]
                        and existing["id"] == payload["id"]
                        and existing["file_abs"] == file_abs
                        and existing["sha256"] == data_sha
                    )
                    if same:
                        return None
                    raise IllegalOperationError(
                        f"设备 {name!r} 已有内容不同的待批准申请；"
                        "先由操作员拒绝或等待处理")
            requested_by = getattr(core, "_active_cid", None)
            self._records[name] = {
                "name": name,
                "file": payload["file"],
                "file_abs": file_abs,
                "class": payload["class"],
                "args": payload["args"],
                "id": payload["id"],
                "sha256": data_sha,
                "requested_at": _now(),
                "requested_by": requested_by,
                "status": "pending",
                "approved_sha256": None,
                "approved_at": None,
                "approved_via": None,
                "reason": "",
                "last_error": None,
            }
        return None

    def _approve(self, name):
        core = self._core
        if core is None:
            raise VMMemoryError("设备装载入口未接入 Core")
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise KeyError(f"设备 {name!r} 没有装载申请")
            if record["status"] != "pending":
                raise ValueError(
                    f"设备 {name!r} 的申请状态是 {record['status']}，不是待批准")
            record = dict(record)
        try:
            current = _sha256_file(record["file_abs"])
        except OSError as e:
            raise ValueError(f"设备文件不可读，拒绝批准：{e}") from None
        if current != record["sha256"]:
            raise ValueError(
                f"设备文件在申请后已变化（申请时 {record['sha256']}，现在 {current}）；"
                "这次批准作废，请拒绝后让申请方重新提交")
        with self._lock:
            live = self._records.get(name)
            if live is None or live["status"] != "pending":
                raise ValueError(f"设备 {name!r} 的申请状态已变化，未批准")
            live["status"] = "approved"
            live["approved_sha256"] = live["sha256"]
            live["approved_at"] = _now()
            live["approved_via"] = _APPROVED_VIA
            return {
                "approved": name,
                "sha256": live["sha256"],
                "next": f"写 $MEM.device_loader.resume.{name} 完成挂载",
            }

    def _reject(self, name, reason=None):
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise KeyError(f"设备 {name!r} 没有装载申请")
            if record["status"] != "pending":
                raise ValueError(
                    f"设备 {name!r} 的申请状态是 {record['status']}，不是待批准")
            record["status"] = "rejected"
            record["reason"] = str(reason) if reason is not None else ""
            record["rejected_at"] = _now()
            return {"rejected": name, "reason": record["reason"]}

    def _reject_after_approval(self, name, reason) -> None:
        record = self._records.get(name)
        if record is None:
            return
        record["status"] = "rejected"
        record["reason"] = reason
        record["rejected_at"] = _now()

    def _resume(self, name) -> None:
        from avm.image import ImageError, mount_device_entry

        core = self._core
        if core is None:
            raise VMMemoryError("设备装载入口未接入 Core")
        name = self._check_name(name)
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise IllegalOperationError(f"设备 {name!r} 没有装载申请记录")
            if record["status"] != "approved":
                raise IllegalOperationError(
                    f"设备 {name!r} 的申请状态是 {record['status']}，未获批准；不能挂载")
            record = dict(record)
        try:
            current = _sha256_file(record["file_abs"])
        except OSError as e:
            with self._lock:
                self._reject_after_approval(name, f"设备文件不可读：{e}")
            raise VMMemoryError(f"设备装载失败：设备文件不可读：{e}") from None
        if current != record["approved_sha256"]:
            with self._lock:
                self._reject_after_approval(
                    name,
                    "批准后设备文件已变化（批准 "
                    f"{record['approved_sha256']}，现在 {current}）")
            raise VMMemoryError(
                f"设备 {name} 装载失败：批准后设备文件已变化，请重新提交申请")
        dev = {
            "path": name,
            "file": record["file_abs"],
            "class": record["class"],
            "args": record["args"],
            "id": record["id"],
            "approved_sha256": record["approved_sha256"],
            "approved_at": record["approved_at"],
            "approved_via": record["approved_via"],
        }
        try:
            spec = mount_device_entry(core, dev, base=None)
        except (ImageError, VMMemoryError) as e:
            message = f"设备装载失败：{e}"
            with self._lock:
                live = self._records.get(name)
                if live is not None and live["status"] == "approved":
                    live["last_error"] = message
            raise VMMemoryError(message) from None
        # 运行期热装的设备进回合日志：重启后按 spec 重新挂载
        core.note_effect("device_mount", mount=name, spec=spec)
        with self._lock:
            self._records.pop(name, None)
        return None

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------

    def _base(self) -> str:
        if self._base_dir:
            return self._base_dir
        image_dir = getattr(self._core, "image_dir", None)
        return image_dir or os.getcwd()

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(
            os.path.join(self._base(), path))

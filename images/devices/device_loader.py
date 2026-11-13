"""设备装载入口：宿主/操作员审批后把一段宿主 Python 挂成设备。

这是**宿主代码挂载**的入口，不是 VM 的软件包机制：VM 只有内存树、指令与设备；软件包
怎么分发、装到哪里、装什么，是 OS 的事情（OS 层约定见 docs/porting-contract.md）。
本设备只做一件事——把一条经过审批的设备条目交给内核的 `mount_device_entry`。

把它挂到 `$MEM.device_loader` 之类的路径上，操作员对话用 `memory_write` 写入一段 JSON：

```json
{"action": "load_device", "path": "extra", "file": "echo.py",
 "class": "EchoDevice", "args": {},
 "approved_sha256": "<设备文件 sha256>"}
```

装载必须带 `approved_sha256`（操作员在面板审批弹窗里看到的文件哈希）：设备装载入口把
它透传给挂载入口，由内核的 `mount_device_entry` 在 import 之前重算文件哈希并核对。
密码不再是凭证：审批是人的仪式，批准的是"哪段代码"。

三条约束：

1. **不实现 `emit_device_event` / `after_event`**：设备不发事件。装载是纯内存操作
   （Memory.mount 只改设备表与挂载点链接节点），不触发事件管道。
2. **错误信息不回显写入原文**：只给原因类别（批准哈希缺失/不匹配、命令不是合法
   JSON……）。
3. **成功返回 None**：结果靠观察设备表得到，不从返回通道塞回执。

旧命令里的 `password` 字段一律拒绝。
"""

import json
import os
import time

from avm.exceptions import IllegalOperationError, VMMemoryError
from avm.memory_device import MemoryDevice

_ACTIONS = ("load_device",)


def _approval(payload: dict) -> tuple:
    """取审批三元组；approved_sha256 是装载的唯一必需字段。

    approved_at / approved_via 是宿主/面板写下的审批事实，缺省时由本设备补
    （直接写 JSON 的调用没有面板会话；时间取写入时刻）。
    """
    approved = payload.get("approved_sha256")
    if not isinstance(approved, str) or not approved.strip():
        raise VMMemoryError("装载命令必须带 approved_sha256（设备审批哈希）")
    approved_at = payload.get("approved_at")
    if not isinstance(approved_at, str) or not approved_at.strip():
        approved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    approved_via = payload.get("approved_via")
    if not isinstance(approved_via, str) or not approved_via.strip():
        approved_via = "device_loader"
    return approved.strip().lower(), approved_at, approved_via


class DeviceLoader(MemoryDevice):
    """把"宿主审批后挂载设备代码"变成对挂载路径的一次 memory_write。"""

    def __init__(self, base_dir: str = ""):
        self._base_dir = str(base_dir or "")
        self._core = None

    # ------------------------------------------------------------------
    # MemoryDevice 接口
    # ------------------------------------------------------------------

    def attach_core(self, core) -> None:
        self._core = core

    def pretend_as_type(self) -> str:
        return "str"

    def to_llm_string(self) -> str:
        return "device_loader：写入 JSON 命令（load_device）"

    # ------------------------------------------------------------------
    # 写入 = 命令
    # ------------------------------------------------------------------

    def set_value(self, value) -> None:
        if self._core is None:
            raise VMMemoryError("设备装载入口未接入 Core")
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise IllegalOperationError("装载命令不是合法 JSON") from None
        if not isinstance(payload, dict):
            raise IllegalOperationError("装载命令必须是 JSON 对象")
        if "password" in payload:
            raise IllegalOperationError(
                "装载命令不再接受 password：请改用 approved_sha256 走设备审批"
            )
        action = payload.get("action")
        if action not in _ACTIONS:
            raise IllegalOperationError(f"装载命令 action 不支持：{action!r}")
        approved, approved_at, approved_via = _approval(payload)
        self._load_device(payload, approved, approved_at, approved_via)

    def _base(self) -> str:
        if self._base_dir:
            return self._base_dir
        image_dir = getattr(self._core, "image_dir", None)
        return image_dir or os.getcwd()

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(
            os.path.join(self._base(), path))

    def _load_device(self, payload: dict, approved: str, approved_at: str,
                     approved_via: str) -> None:
        from avm.image import ImageError, mount_device_entry

        mount = payload.get("path")
        file_path = payload.get("file")
        class_name = payload.get("class")
        if not isinstance(mount, str) or not mount:
            raise IllegalOperationError("load_device 需要非空挂载 path")
        if not isinstance(file_path, str) or not file_path:
            raise IllegalOperationError("load_device 需要非空设备 file")
        if not isinstance(class_name, str) or not class_name:
            raise IllegalOperationError("load_device 需要非空 class")
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            raise IllegalOperationError("load_device 的 args 必须是对象")
        dev = {
            "path": mount,
            "file": self._resolve(file_path),
            "class": class_name,
            "args": args,
            "id": payload.get("id"),
            "approved_sha256": approved,
            "approved_at": approved_at,
            "approved_via": approved_via,
        }
        try:
            spec = mount_device_entry(self._core, dev, base=None)
        except ImageError as e:
            raise VMMemoryError(f"设备装载失败：{e}") from None
        # 运行期热装的设备进回合日志：重启后按 spec 重新挂载
        self._core.note_effect("device_mount", mount=mount, spec=spec)

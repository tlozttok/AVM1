"""AVM 调试控制台后端。

它不是 AVM 的一部分，是宿主侧的调试工具：看正在跑的 Core，并在需要时介入。

三条边界：
- **观测端点直接读状态**：调度、对话、内存树、监听器、设备、事件；
- **有副作用的端点经面板对话**：先中断、再提交一次工具调用，因此走正常指令路径、
  产生事件、发起者是面板对话。控制台不另开写内存的路径；
- **设备调试动作是例外**：直接调用设备自己声明的 `run`，不入核心、不产生事件。

面板对话是 AVM 里的一条宿主对话：保留魔数 cid（-1），对普通对话不可见、不可寻址
（conversations/scheduler 列表过滤，send/close 报不存在），由中断向量表激活、
结束交互即让出执行权。面板本身不主动抢占。

用法：
    python -m avm.console <image.json> [--port 8700] [--checkpoint-in cp.json]

只绑定回环地址。前端可以跨端口访问：端点带 CORS 头，并应答 OPTIONS 预检。
"""

import argparse
import ast
import json
import os
import queue
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from .core import _build_tools, _instruction_registry
from .exceptions import VMMemoryError, RunnerStop
from .io_bus import IOBus
from .io_devices import IOInputDevice, IOOutputDevice
from .memory_device import MemoryDevice, MetaListDevice
from .types import MetaDict, MetaList, LinkNode


# ----------------------------------------------------------------------
# 前端产物
# ----------------------------------------------------------------------

# web/ 是 Vite 工程，构建产物 web/dist 不入库。没构建过时 / 给一段构建提示，
# 而不是一个 404——面板是调试工具，缺产物要说清怎么补。
_WEB_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "dist"
)

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json; charset=utf-8",
}


# ----------------------------------------------------------------------
# JSON 化
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 设备文件检查（审批弹窗的只读数据源）
# ----------------------------------------------------------------------

def _device_classes_in_file(path: str) -> List[str]:
    """从源码里找继承 MemoryDevice 的类名；只做 AST 解析，不 import（不执行设备代码）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, UnicodeDecodeError, SyntaxError) as e:
        raise ValueError(f"设备文件无法作为 Python 源码解析：{path}: {e}") from e
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


def _path_within(path: str, roots: List[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([root, path]) == root:
                return True
        except ValueError:
            continue  # 不同盘符：不是同一棵树
    return False


def _describe_device(device: MemoryDevice) -> dict:
    """设备的运行者视图：类型、挂载键、LLM 表示。取不到就报错，不抛。"""
    out = {"kind": "device", "type": type(device).__name__, "mount": device.mount_path}
    try:
        out["pretend_as_type"] = device.pretend_as_type()
    except Exception as e:
        out["pretend_as_type"] = None
        out["error"] = f"pretend_as_type: {type(e).__name__}: {e}"
    try:
        out["preview"] = device.to_llm_string()
    except Exception as e:
        out["preview"] = None
        out["error"] = f"{out.get('error', '')} to_llm_string: {type(e).__name__}: {e}".strip()
    return out


def _jsonable(value: Any) -> Any:
    """把内存节点与运行时对象转成可 JSON 化的结构。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, MetaDict):
        return {
            "kind": "dict",
            "metadata": value.get_metadata(),
            "ctrl": value.get_ctrl(),
            "children": {str(k): _jsonable(v) for k, v in value.items()},
        }
    if isinstance(value, MetaList):
        return {
            "kind": "list",
            "metadata": value.get_metadata(),
            "ctrl": value.get_ctrl(),
            "children": [_jsonable(v) for v in value],
        }
    if isinstance(value, LinkNode):
        # 链接节点：只报 target 与元数据，不跟随（面板展示的是树本身）
        return {
            "kind": "link",
            "target": value.target,
            "metadata": value.get_metadata(),
            "ctrl": value.get_ctrl(),
        }
    if isinstance(value, MemoryDevice):
        return _describe_device(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _walk(node: Any, path: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把内存树摊平成 路径 → 摘要，供快照与 diff 使用。"""
    if out is None:
        out = {}
    if isinstance(node, MetaDict):
        out[path or "$MEM"] = {"kind": "dict", "ctrl": node.get_ctrl(), "count": len(node.keys())}
        for k, v in node.items():
            _walk(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(node, MetaList):
        out[path or "$MEM"] = {"kind": "list", "ctrl": node.get_ctrl(), "count": len(node)}
        for i in range(len(node)):
            _walk(node[i], f"{path}.{i}", out)
    elif isinstance(node, LinkNode):
        out[path] = {"kind": "link", "target": node.target}
    elif isinstance(node, MemoryDevice):
        out[path] = {"kind": "device", "type": type(node).__name__}
    else:
        out[path] = {"kind": "str", "preview": str(node)[:120]}
    return out


# ----------------------------------------------------------------------
# 通道
# ----------------------------------------------------------------------

class OperatorChannel:
    """面板对话与控制台之间的双向通道。

    - 控制台 → 面板：`submit_action` 投一个动作，面板执行器的 `wait_action` 取走；
    - 面板 → 控制台：`publish` 记一条输出，控制台按 seq 增量读取。

    输出是有界环形缓冲：面板长期开着也不会把内存吃光，丢了多少条记在 status 里。
    """

    def __init__(self, capacity: int = 4000):
        self._actions: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()
        self._output: List[dict] = []
        self._capacity = capacity
        self._next_seq = 1
        self._dropped = 0

    # -- 面板执行器侧 --------------------------------------------------

    def wait_action(self) -> dict:
        return self._actions.get()

    def publish(self, kind: str, data: Any) -> None:
        with self._lock:
            if len(self._output) >= self._capacity:
                del self._output[: len(self._output) - self._capacity + 1]
                self._dropped += 1
            self._output.append({
                "seq": self._next_seq,
                "kind": kind,
                "data": data,
                "time": time.time(),
            })
            self._next_seq += 1

    # -- 控制台侧 ------------------------------------------------------

    def submit_action(self, action: dict) -> None:
        self._actions.put(action)

    def drain(self, since: int = 0) -> List[dict]:
        with self._lock:
            return [e for e in self._output if e["seq"] > since]

    def status(self) -> dict:
        with self._lock:
            return {
                "next_seq": self._next_seq,
                "buffered": len(self._output),
                "dropped": self._dropped,
                "pending_actions": self._actions.qsize(),
            }


# ----------------------------------------------------------------------
# 控制台
# ----------------------------------------------------------------------

class ConsoleConflict(RuntimeError):
    """宿主策略冲突（生命周期或客户端席位）：HTTP 层映射为 409。"""


class Console:
    """调试控制台：面板对话 + 观测 + 介入 + 屏蔽账本。"""

    def __init__(self, core, web_root: Optional[str] = None,
                 auth_token: Optional[str] = None, force_stop_hook=None,
                 single_client: bool = False, client_lease_seconds: float = 15.0,
                 force_stop_delay: float = 10.0, mount_io_devices: bool = True):
        self.core = core
        # 前端构建产物目录（web/dist）。传 None 用仓库默认位置；测试传临时目录。
        self.web_root = web_root if web_root is not None else _WEB_DIST

        # 操作面板：宿主侧 IO 总线与生命周期。auth_token 为 None 时不启用认证
        # （单测与 e2e mock 走这条路），真实宿主由 main() 生成 token 后启用。
        if getattr(core, "io_bus", None) is None:
            core.io_bus = IOBus()
        self.io_bus = core.io_bus
        # 把镜像声明的 inputs / outputs 换成只认总线的两台 IO 设备。标准设备留在
        # avm/memory_device.py 供终端宿主使用，本宿主不碰终端、不打印。
        self.io_input_devices: List[IOInputDevice] = []
        self.io_output_devices: List[IOOutputDevice] = []
        if mount_io_devices:
            self._install_io_devices()

        self.channel = OperatorChannel()
        # 保留对话（cid -1）在装载时/重放时已经存在；这里只把操作员通道绑上去。
        # 没有（例如测试直接构造 Core）时 bind 会创建一条。
        self.panel = core.bind_console_conversation(self.channel)
        core.interrupts.register("operator", self.panel.cid)

        self.auth_token = auth_token
        self.force_stop_hook = force_stop_hook
        self.lifecycle = "running"                 # running / stopping / stopped
        self.stopping_since: Optional[float] = None
        # 关机流程：请求关机后要等满 force_stop_delay 秒才允许强制退出。
        self.force_stop_delay = float(force_stop_delay)
        # 单客户端准入是宿主策略：真实面板宿主开启，单测与 e2e mock 关闭。
        self.single_client = bool(single_client)
        self.client_lease_seconds = float(client_lease_seconds)
        self._client_lock = threading.Lock()
        self._client_lease: Optional[dict] = None  # {client_id, since, last_seen}

        self.running = False                       # Core.run 是否在跑，由 runner 设置
        self._memory_mask: Dict[str, Any] = {}     # ref → 摘下来的节点
        self._device_mask: Dict[str, Any] = {}     # 挂载键 → 设备实例
        self._action_log: List[dict] = []          # 设备调试动作的调用记录
        self._snapshots: List[Dict[str, Any]] = [_walk(core.unwrap("$MEM", for_llm=False))]
        core.events.add_observer(self._on_event)

    # -- 操作面板 IO 设备挂载 ------------------------------------------

    def _install_io_devices(self) -> None:
        """在 `inputs` / `outputs` 上换成总线前端；镜像没声明的路径不新增。

        数据和元数据从镜像装出来的标准设备原样搬过去（快照恢复后的历史不能丢），
        设备标识符与挂载键也照抄。已经挂了 IO 设备的宿主（e2e mock）只登记引用。
        """
        core = self.core
        for mount, cls, bucket in (
            ("inputs", IOInputDevice, self.io_input_devices),
            ("outputs", IOOutputDevice, self.io_output_devices),
        ):
            old = core.mem.mounted_devices().get(mount)
            if old is None:
                continue
            if isinstance(old, cls):
                dev = old
            elif isinstance(old, MetaListDevice):
                dev = cls(data=list(old._data),
                          metadata=old.get_metadata())
                dev.device_id = getattr(old, "device_id", None)
                core.mem.unmount(mount)
                core.mem.mount(mount, dev)
            else:
                # 镜像在 inputs / outputs 上挂了别的设备：不猜它的语义，保持原样
                continue
            dev.mount_path = mount
            dev.attach_core(core)
            bucket.append(dev)

    # -- 单客户端准入 --------------------------------------------------

    def _lease_expired(self, lease: dict, now: float) -> bool:
        return now - lease["last_seen"] > self.client_lease_seconds

    def client_status(self) -> dict:
        with self._client_lock:
            lease = self._client_lease
            if lease is not None and self._lease_expired(lease, time.time()):
                self._client_lease = None
                lease = None
            return {
                "single_client": self.single_client,
                "held": lease is not None,
                "client_id": lease["client_id"] if lease else None,
                "since": lease["since"] if lease else None,
                "lease_seconds": self.client_lease_seconds,
            }

    def acquire_client(self, client_id: Optional[str]) -> dict:
        """占住唯一席位；已声明身份的其它客户端在租约有效期内一律被挡。"""
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id 必须是非空字符串")
        if not self.single_client:
            return {"client_id": client_id, "single_client": False}
        now = time.time()
        with self._client_lock:
            lease = self._client_lease
            if (lease is not None and lease["client_id"] != client_id
                    and not self._lease_expired(lease, now)):
                raise ConsoleConflict(
                    f"已有客户端连接（{lease['client_id']}，"
                    f"自 {time.strftime('%H:%M:%S', time.localtime(lease['since']))} 起）；"
                    f"关闭它或等待租约 {self.client_lease_seconds:g} 秒过期"
                )
            since = lease["since"] if (lease and lease["client_id"] == client_id
                                       and not self._lease_expired(lease, now)) else now
            self._client_lease = {"client_id": client_id, "since": since,
                                  "last_seen": now}
        return {"client_id": client_id, "single_client": True, "since": since}

    def release_client(self, client_id: Optional[str]) -> dict:
        with self._client_lock:
            lease = self._client_lease
            released = bool(lease and client_id and lease["client_id"] == client_id)
            if released:
                self._client_lease = None
        return {"released": released}

    def check_client(self, client_id: Optional[str]) -> bool:
        """普通 API 请求：带身份的必须就是席位持有者；不带身份的按宿主本地调用放行。

        面板前端所有请求都带 `X-AVM-Client`（SSE 靠 cookie），所以第二个浏览器
        标签页即使知道 token 也会被挡在 `/api/` 之外。
        """
        if not self.single_client or client_id is None:
            return True
        now = time.time()
        with self._client_lock:
            lease = self._client_lease
            if lease is None or self._lease_expired(lease, now):
                self._client_lease = {"client_id": client_id, "since": now,
                                      "last_seen": now}
                return True
            if lease["client_id"] != client_id:
                return False
            lease["last_seen"] = now
            return True

    # -- 事件观察 ------------------------------------------------------

    def _on_event(self, event: dict) -> None:
        self.channel.publish("event", _jsonable(event))

    # -- 观测 ----------------------------------------------------------

    def schedule(self) -> dict:
        core = self.core
        return {
            "active_cid": core._active_cid,
            "ready": list(core._ready_cids),
            "dormant": list(core._dormant_cids),
            "finished": sorted(core._finished),
            "instruction_budget": core._instruction_budget,
            "activation_instructions": core._activation_instructions,
            "panel_cid": self.panel.cid,
        }

    def state(self) -> dict:
        core = self.core
        return {
            "running": self.running,
            "schedule": self.schedule(),
            "conversations": [
                self._conversation_brief(cid, conv)
                for cid, conv in sorted(core._conv_by_cid.items())
            ],
            "panel": {
                "cid": self.panel.cid,
                "active": core._active_cid == self.panel.cid,
                "messages": len(self.panel.messages),
                "channel": self.channel.status(),
            },
            "interrupts": {
                "vectors": core.interrupts.vectors,
                "pending": core.interrupts.pending_types(),
                "last_dispatch": core.interrupts.last_dispatch,
            },
            "mask": {
                "memory": sorted(self._memory_mask),
                "devices": sorted(self._device_mask),
            },
            "snapshots": len(self._snapshots),
            # 操作面板首屏：生命周期、本次恢复摘要、IO 两条流的 seq 与丢弃计数
            "lifecycle": self.lifecycle,
            "recovery": getattr(core, "recovery", {}),
            "io": self.io_bus.status(),
            "force_stop_after": self.force_stop_delay,
            "force_stop_remaining": self.force_stop_remaining(),
            "client": self.client_status(),
        }

    def force_stop_remaining(self) -> float:
        """还要等几秒才允许强制退出；尚未请求关机时给满额，已停止时给 0。"""
        with self._client_lock:
            since = self.stopping_since
            lifecycle = self.lifecycle
        if lifecycle == "stopped":
            return 0.0
        if since is None:
            return self.force_stop_delay
        return max(0.0, self.force_stop_delay - (time.time() - since))

    def _conversation_brief(self, cid: int, conv) -> dict:
        core = self.core
        if cid in core._finished:
            status = "finished"
        elif core._active_cid == cid:
            status = "active"
        elif cid in core._ready_cids:
            status = "ready"
        elif cid in core._dormant_cids:
            status = "dormant"
        else:
            status = "unknown"
        return {
            "cid": cid,
            "identity": conv.identity,
            "name": conv.name,
            "status": status,
            "is_python": conv.is_python,
            "is_sub": conv.is_sub,
            "is_root": conv.is_root,
            "parent": conv.parent.cid if conv.parent is not None else None,
            "messages": len(conv.messages),
            # 宿主观测面：只展示能力摘要（根名/工具名），不展开根目标的真实地址
            "caps": {
                "mode": conv.caps.mode,
                "groups": list(conv.caps.groups),
                "roots": sorted(conv.caps.roots),
                "tools": sorted(conv.caps.tools),
            },
        }

    def conversation(self, cid: int) -> dict:
        conv = self.core._conv_by_cid.get(cid)
        if conv is None:
            raise KeyError(f"对话 {cid} 不存在")
        brief = self._conversation_brief(cid, conv)
        brief["history"] = [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": getattr(m, "tool_calls", None),
                "tool_call_id": getattr(m, "tool_call_id", None),
            }
            for m in conv.messages
        ]
        brief["user_batch"] = {
            "user_contents": list(conv.user_batch.user_contents),
            "tool_responses": [
                {"content": r.content, "tool_call_id": r.tool_call_id}
                for r in conv.user_batch.tool_responses
            ],
        }
        brief["metadata"] = dict(conv.metadata)
        brief["para_ref"] = conv.para_ref
        return brief

    def memory(self, ref: Optional[str] = None) -> dict:
        # 这里走的是普通读取路径，如果 ref 落在设备上，设备的读会被真的执行。
        # 对 InputsListDevice 来说 `$MEM.inputs.next` 就是"请求一次操作员输入"：控制台托管下
        # （host_flags["console"]）这一读等在面板输入队列上，所以这个端点会挂起，
        # 直到有人在面板里喂一行（POST /api/device-action）——不再是挂起在终端上等敲键盘。
        target = ref if ref else "$MEM"
        return {"ref": target, "node": _jsonable(self.core.unwrap(target, for_llm=False))}

    def listeners(self) -> dict:
        from .events import EventSystem
        return {
            "listeners": [dict(l) for l in self.core.events.listeners],
            "editable_fields": list(EventSystem.EDITABLE_FIELDS),
        }

    def devices(self) -> dict:
        """已挂载设备，现取一次各自的调试动作。"""
        out = []
        for mount, device in sorted(self.core.mem.mounted_devices().items()):
            actions = []
            error = None
            try:
                raw = device.debug_actions() or []
            except Exception as e:
                raw, error = [], f"debug_actions: {type(e).__name__}: {e}"
            for index, action in enumerate(raw):
                if not isinstance(action, dict):
                    continue
                actions.append({
                    "index": index,
                    "name": action.get("name"),
                    "description": action.get("description"),
                    "parameters": action.get("parameters"),
                    "callable": callable(action.get("run")),
                })
            out.append({
                "mount": mount,
                "type": type(device).__name__,
                "description": _describe_device(device),
                "actions": actions,
                "error": error,
            })
        return {"devices": out, "action_log": list(self._action_log[-50:])}

    # -- 设备审批：只读检查 -------------------------------------------

    def _inspect_roots(self) -> List[str]:
        """允许检查的目录：镜像目录 + 已安装程序包的源目录。

        审批弹窗只能检查宿主世界里的设备文件；任意路径（/etc/passwd、用户 home）
        一律拒绝。包源目录来自安装记录，已卸载的包不再出现在允许范围里。
        """
        roots = []
        image_dir = getattr(self.core, "image_dir", None)
        if isinstance(image_dir, str) and image_dir:
            roots.append(os.path.realpath(image_dir))
        records = self.core.mem._data.get("package_records")
        if isinstance(records, MetaDict):
            for _name, versions in records.items():
                if not isinstance(versions, MetaDict):
                    continue
                for _version, record in versions.items():
                    if not isinstance(record, MetaDict):
                        continue
                    source = record.get("source")
                    if isinstance(source, str) and source:
                        roots.append(os.path.dirname(os.path.realpath(source)))
        return sorted(set(roots))

    def _existing_device_spec(self, path: str, mount: Optional[str] = None) -> Optional[dict]:
        """同一设备文件已挂在哪些路径；优先返回请求里给出的挂载点。"""
        matches = []
        for spec_mount, spec in sorted(self.core.device_specs.items()):
            if not isinstance(spec, dict) or spec.get("builtin"):
                continue
            spec_path = spec.get("path")
            if isinstance(spec_path, str) and os.path.realpath(spec_path) == path:
                matches.append((spec_mount, spec))
        if not matches:
            return None
        chosen = matches[0]
        if mount is not None:
            for item in matches:
                if item[0] == mount:
                    chosen = item
                    break
        spec_mount, spec = chosen
        return {
            "mount": spec_mount,
            "class": spec.get("class"),
            "args": spec.get("args") or {},
            "sha256": spec.get("sha256"),
            "approved_at": spec.get("approved_at"),
            "approved_via": spec.get("approved_via"),
        }

    def inspect_device(self, path, class_name: Optional[str] = None,
                       mount: Optional[str] = None,
                       args: Optional[dict] = None) -> dict:
        """只读检查一台设备文件：路径、大小、哈希、类名、挂载信息与现有来源表条目。

        不 import 设备代码（类名从 AST 取），不挂载、不写内存。路径必须落在
        镜像目录或程序包源目录内；拒绝只读探测宿主任意文件。
        """
        from .image import _sha256_file

        if not isinstance(path, str) or not path.strip():
            raise ValueError("path 必须是非空字符串")
        if mount is not None and (not isinstance(mount, str) or not mount):
            raise ValueError("mount 必须是非空字符串")
        if args is not None and not isinstance(args, dict):
            raise ValueError("args 必须是对象")
        roots = self._inspect_roots()
        if not roots:
            raise ValueError("当前内核没有镜像目录或程序包目录，拒绝检查任意路径")
        base = roots[0]
        target = os.path.realpath(
            path if os.path.isabs(path) else os.path.join(base, path)
        )
        if not _path_within(target, roots):
            raise ValueError(f"设备文件路径必须落在镜像目录或程序包目录内：{path}")
        if not os.path.isfile(target):
            raise ValueError(f"设备文件不存在：{path}")
        classes = _device_classes_in_file(target)
        if class_name is not None:
            if not isinstance(class_name, str) or not class_name:
                raise ValueError("class 必须是非空字符串")
            if classes and class_name not in classes:
                raise ValueError(
                    f"文件里没有继承 MemoryDevice 的类 {class_name}"
                    f"（发现：{', '.join(classes) or '无'}）"
                )
        elif len(classes) == 1:
            class_name = classes[0]
        existing = self._existing_device_spec(target, mount)
        if mount is None and existing is not None:
            mount = existing["mount"]
        if args is None and existing is not None:
            args = existing["args"]
        try:
            size = os.path.getsize(target)
            digest = _sha256_file(target)
        except OSError as e:
            raise ValueError(f"设备文件不可读：{path}: {e}") from e
        return {
            "path": target,
            "size": size,
            "sha256": digest,
            "class": class_name,
            "mount": mount,
            "args": args,
            "existing": existing,
        }

    # -- 快照与 diff ---------------------------------------------------

    def snapshot(self) -> int:
        self._snapshots.append(_walk(self.core.unwrap("$MEM", for_llm=False)))
        return len(self._snapshots) - 1

    def diff(self, frm: int, to: Optional[int] = None) -> dict:
        if frm < 0 or frm >= len(self._snapshots):
            raise KeyError(f"快照 {frm} 不存在（现有 0..{len(self._snapshots) - 1}）")
        before = self._snapshots[frm]
        after = self._snapshots[to] if to is not None else _walk(
            self.core.unwrap("$MEM", for_llm=False))
        return {
            "from": frm,
            "to": to,
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(
                p for p in set(before) & set(after) if before[p] != after[p]
            ),
        }

    # -- 介入：中断与动作 ----------------------------------------------

    def request_pause(self) -> dict:
        """请求暂停。中断在下一个安全点生效；系统空转结束时不会生效。

        中断会向面板对话投一条 payload 为「已暂停」的消息，操作员在对话流里看得到。
        """
        self.core.interrupts.raise_("operator", payload="已暂停")
        return {
            "requested": True,
            "pending": self.core.interrupts.pending_types(),
            "panel_cid": self.panel.cid,
            "already_active": self.core._active_cid == self.panel.cid,
        }

    def submit_action(self, tool: str, arguments: Optional[dict] = None) -> dict:
        """提交一次操作员动作。未暂停时先自动中断。"""
        if not isinstance(tool, str) or not tool:
            raise ValueError("tool 必须是非空字符串")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments 必须是对象")
        paused = self.core._active_cid == self.panel.cid
        if not paused:
            self.core.interrupts.raise_("operator", payload="已暂停")
        self.channel.submit_action({"tool": tool, "arguments": arguments})
        known = set(_instruction_registry())
        return {
            "submitted": True,
            "tool": tool,
            "auto_paused": not paused,
            "known_tool": tool in known,
            "warning": None if tool in known else (
                f"未知工具 {tool!r}：Core 会在 stderr 报错，且不回工具响应"
            ),
        }

    def resume(self) -> dict:
        """让面板结束本轮交互，把执行权让回就绪队列。"""
        self.channel.submit_action({"end": True})
        return {"submitted": True}

    def invoke_device_action(self, mount: str, index: int,
                             arguments: Optional[dict] = None) -> dict:
        """调用设备自己声明的调试动作。不入核心、不产生事件。"""
        device = self.core.mem.mounted_devices().get(mount)
        if device is None:
            raise KeyError(f"设备 {mount} 未挂载（可能被屏蔽了）")
        actions = device.debug_actions() or []
        if index < 0 or index >= len(actions):
            raise KeyError(f"设备 {mount} 没有编号 {index} 的调试动作")
        action = actions[index]
        if not isinstance(action, dict) or not callable(action.get("run")):
            raise ValueError(f"设备 {mount} 的调试动作 {index} 没有可调用的 run")
        args = arguments if isinstance(arguments, dict) else {}
        started = time.time()
        try:
            result = action["run"](args)
            error = None
        except Exception as e:
            result, error = None, f"{type(e).__name__}: {e}"
        record = {
            "mount": mount,
            "index": index,
            "name": action.get("name"),
            "arguments": args,
            "result": _jsonable(result) if error is None else None,
            "error": error,
            "time": started,
        }
        self._action_log.append(record)
        return record

    # -- 操作面板：IO 与生命周期 ---------------------------------------

    def io_input(self, target, text) -> dict:
        """发布一条定向输入，并唤醒可能正等在读上的 IO 设备。

        target 为 None / 空 / '*' 表示无目标输入（入口对话可取）。
        输入本身只进宿主侧总线，不经面板对话、不产生事件。
        """
        if self.lifecycle != "running":
            raise ConsoleConflict(f"面板处于 {self.lifecycle} 状态，不再接收新输入")
        if not isinstance(text, str) or not text:
            raise ValueError("text 必须是非空字符串")
        entry = self.io_bus.publish_input(target, text)
        for device in self.io_input_devices:
            device.notify_input()
        return {"seq": entry["seq"], "target": entry["target"], "text": entry["text"]}

    def request_shutdown(self) -> dict:
        """请求关机：停在下一个安全点，由宿主写回并退出。

        这里只置标志；真正的写快照与进程退出在 main() 的 core.run() 返回路径上，
        避免半截回合被拍进快照。新输入从这一刻起被拦住。
        """
        with self._client_lock:
            self.lifecycle = "stopping"
            self.stopping_since = time.time()
        request_stop = getattr(self.core, "request_stop", None)
        if callable(request_stop):
            request_stop()
        return {"status": "stopping", "force_stop_after": self.force_stop_delay}

    def force_stop(self) -> dict:
        """强制退出：不写快照。只有注入了回调的真实宿主才真的结束进程。

        只在「已请求关机且等满 force_stop_delay 秒」之后放行，防止顺手把还在
        正常收尾的宿主杀掉；回调由 main() 注入，单测与 mock 不结束进程。
        """
        remaining = self.force_stop_remaining()
        if remaining > 0:
            raise ConsoleConflict(
                f"关机还在收尾：{remaining:.0f} 秒后才可强制退出（不写快照）"
            )
        with self._client_lock:
            self.lifecycle = "stopped"
        hook = self.force_stop_hook
        if callable(hook):
            threading.Thread(target=hook, daemon=True).start()
        return {"status": "force-stopped", "snapshot_written": False}

    # -- 屏蔽与还原 ----------------------------------------------------

    def mask_memory(self, ref: str) -> dict:
        if ref in self._memory_mask:
            raise ValueError(f"{ref} 已经被屏蔽")
        node = self.core.mem.detach(ref)  # 失败时抛 VMMemoryError，不改账本
        self._memory_mask[ref] = node
        return {"masked": ref, "kind": type(node).__name__}

    def unmask_memory(self, ref: str) -> dict:
        if ref not in self._memory_mask:
            raise KeyError(f"{ref} 没有被屏蔽")
        node = self._memory_mask[ref]
        try:
            self.core.mem.attach(ref, node)  # 冲突时抛 VMMemoryError，账本保持不变
        except Exception:
            raise
        del self._memory_mask[ref]
        return {"unmasked": ref}

    def mask_device(self, mount: str) -> dict:
        if mount in self._device_mask:
            raise ValueError(f"设备 {mount} 已经被屏蔽")
        devices = self.core.mem.mounted_devices()
        if mount not in devices:
            raise KeyError(f"路径 {mount} 没有挂载设备")
        device = devices[mount]
        self.core.mem.unmount(mount)
        self._device_mask[mount] = device
        return {"masked": mount, "type": type(device).__name__}

    def unmask_device(self, mount: str) -> dict:
        if mount not in self._device_mask:
            raise KeyError(f"设备 {mount} 没有被屏蔽")
        self.core.mem.mount(mount, self._device_mask[mount])
        del self._device_mask[mount]
        return {"unmasked": mount}

    # -- 监听器管线 ----------------------------------------------------

    def patch_listener(self, listener_id: int, fields: dict) -> dict:
        if not any(l["listener_id"] == listener_id for l in self.core.events.listeners):
            raise KeyError(f"监听器 {listener_id} 不存在")
        return self.core.events.update(listener_id, **fields)


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<meta charset="utf-8"><title>AVM 调试控制台</title>
<h1>AVM 调试控制台后端</h1>
<p>前端还没构建，这里先是后端说明。构建面板：</p>
<pre>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>构建后刷新本页即可看到面板（源码在 <code>web/</code>，产物 <code>web/dist</code> 不入库）。
开发期也可以不走构建：<code>cd web &amp;&amp; npm run dev</code>，由 Vite 代理 <code>/api</code>。</p>
<p>端点：</p>
<ul>
<li><code>GET /api/state</code> · <code>GET /api/tools</code></li>
<li><code>GET /api/conversations/&lt;cid&gt;</code></li>
<li><code>GET /api/memory?ref=$MEM.a.b</code></li>
<li><code>GET /api/memory/diff?from=0&amp;to=1</code></li>
<li><code>POST /api/memory/snapshot</code></li>
<li><code>GET /api/listeners</code> · <code>PATCH /api/listeners/&lt;id&gt;</code></li>
<li><code>GET /api/devices</code> · <code>POST /api/device-action</code> · <code>POST /api/device/inspect</code></li>
<li><code>GET /api/events</code> · <code>GET /api/io/stream?after=&lt;seq&gt;</code>（SSE）</li>
<li><code>POST /api/interrupt</code> · <code>POST /api/resume</code> · <code>POST /api/action</code></li>
<li><code>POST /api/io/input</code> · <code>POST /api/session/acquire</code> · <code>POST /api/session/release</code></li>
<li><code>POST /api/system/shutdown</code> · <code>POST /api/system/force-stop</code></li>
<li><code>POST|DELETE /api/mask/memory</code> · <code>POST|DELETE /api/mask/device</code></li>
</ul>
"""


class _ConsoleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    console: Console = None  # 由 serve() 注入

    # -- 基础设施 ------------------------------------------------------

    def log_message(self, fmt, *args):
        pass  # 静音访问日志

    def _cors(self):
        if self.console.auth_token:
            # 认证宿主不放宽跨源：同源请求本就不需要 CORS 头，只回显同源 Origin。
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and urlparse(origin).netloc == host:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        else:
            # 测试与 e2e mock 不启用认证，保持旧的开发期跨源头
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization, X-AVM-Client")

    def _authorized(self) -> bool:
        token = self.console.auth_token
        if not token:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {token}":
            return True
        return f"avm_token={token}" in self.headers.get("Cookie", "")

    def _same_origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True  # 无 Origin 的请求（curl、同源表单）不算跨源
        return urlparse(origin).netloc == self.headers.get("Host", "")

    def _client_id(self) -> Optional[str]:
        """取客户端身份：普通请求走头，SSE（EventSource 不能带自定义头）走 cookie。"""
        header = self.headers.get("X-AVM-Client")
        if header:
            return header
        for part in self.headers.get("Cookie", "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == "avm_client":
                return value or None
        return None

    def _guard(self, path: str) -> bool:
        """返回 True 表示已自行响应，调用方应直接 return。

        认证与单客户端准入都是宿主策略：前者由 auth_token 决定，后者由
        single_client 决定。/api/session/* 自己处理身份，不走这里的席位校验。
        """
        if not path.startswith("/api/"):
            return False
        if self.console.auth_token:
            if not self._same_origin_ok():
                self._error(403, "跨源请求被拒绝")
                return True
            if not self._authorized():
                self._error(401, "需要面板凭证")
                return True
        if self.console.single_client and not path.startswith("/api/session/"):
            if not self.console.check_client(self._client_id()):
                holder = self.console.client_status()["client_id"]
                self._error(409, f"已有客户端连接（{holder}）；本宿主一次只允许一个面板客户端")
                return True
        return False

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: Any):
        body = json.dumps(obj, ensure_ascii=False, default=repr).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _error(self, code: int, message: str):
        self._send_json(code, {"error": message})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ValueError(f"请求体不是合法 JSON：{e}")
        if not isinstance(parsed, dict):
            raise ValueError("请求体必须是对象")
        return parsed

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    # -- 前端产物 ------------------------------------------------------

    def _static_path(self, url_path: str) -> Optional[str]:
        """把 URL 路径映射到 web_root 下的真实文件；目录穿越与不存在都返回 None。"""
        root = os.path.realpath(self.console.web_root)
        rel = url_path.lstrip("/") or "index.html"
        target = os.path.realpath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            return None
        return target if os.path.isfile(target) else None

    def _send_file(self, fs_path: str):
        try:
            with open(fs_path, "rb") as f:
                body = f.read()
        except OSError as e:
            self._error(500, f"读取静态文件失败：{e}")
            return
        ctype = _MIME.get(os.path.splitext(fs_path)[1].lower(), "application/octet-stream")
        self._send(200, body, ctype)

    def _send_static(self, path: str):
        """非 /api/ 的 GET 都在产物目录里找。构建过就是面板，没构建就在 / 给构建提示。"""
        target = self._static_path(path)
        if target is not None:
            self._send_file(target)
        elif path == "/":
            self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._error(404, f"没有这个文件：{path}")

    # -- 方法 ----------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        query = self._query()
        console = self.console
        if self._guard(path):
            return
        try:
            if not path.startswith("/api/"):
                # 非 API 的 GET 都是前端产物：/ 给面板（或构建提示），/assets/… 给静态文件
                self._send_static(path)
            elif path == "/api/tools":
                self._send_json(200, {"tools": _build_tools()})
            elif path == "/api/state":
                self._send_json(200, console.state())
            elif path == "/api/listeners":
                self._send_json(200, console.listeners())
            elif path == "/api/devices":
                self._send_json(200, console.devices())
            elif path == "/api/memory":
                self._send_json(200, console.memory(query.get("ref")))
            elif path == "/api/memory/diff":
                self._send_json(200, console.diff(
                    int(query.get("from", 0)),
                    int(query["to"]) if "to" in query else None,
                ))
            elif path == "/api/events":
                self._stream_events()
            elif path == "/api/io/stream":
                self._stream_io(int(query.get("after", 0)))
            elif path.startswith("/api/conversations/"):
                self._send_json(200, console.conversation(int(path.rsplit("/", 1)[1])))
            else:
                self._error(404, f"没有这个端点：{path}")
        except KeyError as e:
            self._error(404, str(e.args[0] if e.args else e))
        except VMMemoryError as e:
            # 读不到就是读不到：屏蔽期间键不存在、路径被改掉，都是可预期的结果
            self._error(404, f"{type(e).__name__}: {e}")
        except ValueError as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        path = urlparse(self.path).path
        query = self._query()
        console = self.console
        if self._guard(path):
            return
        try:
            body = self._body()
            if path == "/api/interrupt":
                self._send_json(200, console.request_pause())
            elif path == "/api/resume":
                self._send_json(200, console.resume())
            elif path == "/api/action":
                self._send_json(200, console.submit_action(
                    body.get("tool"), body.get("arguments")))
            elif path == "/api/device-action":
                self._send_json(200, console.invoke_device_action(
                    body.get("mount"), int(body.get("index", 0)), body.get("arguments")))
            elif path == "/api/device/inspect":
                self._send_json(200, console.inspect_device(
                    body.get("path"), class_name=body.get("class"),
                    mount=body.get("mount"), args=body.get("args")))
            elif path == "/api/memory/snapshot":
                self._send_json(200, {"snapshot": console.snapshot()})
            elif path == "/api/mask/memory":
                self._send_json(200, console.mask_memory(body.get("ref")))
            elif path == "/api/mask/device":
                self._send_json(200, console.mask_device(body.get("mount")))
            elif path == "/api/io/input":
                self._send_json(200, console.io_input(
                    body.get("target"), body.get("text")))
            elif path == "/api/session/acquire":
                self._send_json(200, console.acquire_client(body.get("client_id")))
            elif path == "/api/session/release":
                self._send_json(200, console.release_client(
                    body.get("client_id") or self._client_id()))
            elif path == "/api/system/shutdown":
                self._send_json(202, console.request_shutdown())
            elif path == "/api/system/force-stop":
                self._send_json(202, console.force_stop())
            else:
                self._error(404, f"没有这个端点：{path}")
        except KeyError as e:
            self._error(404, str(e.args[0] if e.args else e))
        except VMMemoryError as e:
            self._error(400, f"{type(e).__name__}: {e}")
        except ConsoleConflict as e:
            # 生命周期冲突：关机中拒收输入、强制退出还没到时间、席位被占
            self._error(409, str(e))
        except ValueError as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def do_PATCH(self):
        path = urlparse(self.path).path
        console = self.console
        if self._guard(path):
            return
        try:
            if path.startswith("/api/listeners/"):
                listener_id = int(path.rsplit("/", 1)[1])
                self._send_json(200, console.patch_listener(listener_id, self._body()))
            else:
                self._error(404, f"没有这个端点：{path}")
        except KeyError as e:
            self._error(404, str(e.args[0] if e.args else e))
        except VMMemoryError as e:
            self._error(400, f"{type(e).__name__}: {e}")
        except ValueError as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def do_DELETE(self):
        path = urlparse(self.path).path
        query = self._query()
        console = self.console
        if self._guard(path):
            return
        try:
            if path == "/api/mask/memory":
                self._send_json(200, console.unmask_memory(query.get("ref")))
            elif path == "/api/mask/device":
                self._send_json(200, console.unmask_device(query.get("mount")))
            else:
                self._error(404, f"没有这个端点：{path}")
        except KeyError as e:
            self._error(404, str(e.args[0] if e.args else e))
        except VMMemoryError as e:
            self._error(409, f"{type(e).__name__}: {e}")  # 还原冲突
        except ValueError as e:
            self._error(400, str(e))
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    # -- SSE -----------------------------------------------------------

    def _stream_events(self):
        """一条有序流：面板输出（动作、工具响应、事件）+ 调度变化。"""
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cursor = 0
        last_sched = None
        try:
            while True:
                payloads = []
                for entry in self.console.channel.drain(cursor):
                    cursor = entry["seq"]
                    payloads.append({"type": entry["kind"], "entry": entry})
                sched = self.console.schedule()
                if sched != last_sched:
                    last_sched = sched
                    payloads.append({"type": "schedule", "data": sched})
                for payload in payloads:
                    line = json.dumps(payload, ensure_ascii=False, default=repr)
                    self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                if payloads:
                    self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _stream_io(self, after: int = 0):
        """输出流增量 SSE：面板重连用 after=<seq> 续读，不消费总线记录。

        与 /api/events 是两条独立的流：那条是面板对话与 Core 事件，这条是
        $MEM.outputs 的宿主侧输出。真实宿主一次只允许一个面板客户端（见
        single_client），游标按面板重连续读，不消费总线记录。
        """
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cursor = after
        try:
            while True:
                entries = self.console.io_bus.drain_io(cursor)
                for entry in entries:
                    cursor = entry["seq"]
                    line = json.dumps({"type": "io", "entry": entry},
                                      ensure_ascii=False, default=repr)
                    self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                if entries:
                    self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def serve(console: Console, host: str = "127.0.0.1", port: int = 8700) -> ThreadingHTTPServer:
    """起 HTTP 服务（后台线程），返回服务器实例。只绑定回环地址。"""
    handler = type("_BoundConsoleHandler", (_ConsoleHandler,), {"console": console})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 调试控制台后端")
    parser.add_argument("image", nargs="?", help="系统镜像 JSON 文件路径")
    parser.add_argument("--port", type=int, default=8700, help="监听端口（默认 8700）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认只绑回环）")
    parser.add_argument("--allow-multiple-clients", action="store_true",
                        help="关闭单客户端准入（默认一次只允许一个面板客户端）")
    parser.add_argument("--checkpoint-in", metavar="PATH",
                        help="调试用：启动时加载单对话检查点（跳过 init）")
    from .boot import add_core_dump_args
    add_core_dump_args(parser)
    args = parser.parse_args(argv)

    if not args.image:
        print("[console] 需要一个镜像路径", file=sys.stderr)
        return 1

    from .boot import boot, dump_on_death, write_snapshot
    from .exceptions import VMMemoryError
    from .image import ImageError

    try:
        core = boot(args.image, checkpoint_in=args.checkpoint_in,
                    host_flags={"console": True})
    except (ImageError, VMMemoryError, OSError, ValueError) as e:
        print(f"[console] 镜像错误: {e}", file=sys.stderr)
        return 1

    # 紧耦合登录：一次性 token 只在本次进程有效，登录本身即凭证。
    # 单测与 e2e mock 直接构造 Console（auth_token=None），不启用这一层。
    token = secrets.token_urlsafe(24)

    def force_exit():
        time.sleep(0.3)  # 让 202 响应先写回浏览器
        os._exit(1)

    console = Console(core, auth_token=token, force_stop_hook=force_exit,
                      single_client=not args.allow_multiple_clients)
    server = serve(console, args.host, args.port)
    print(f"[console] 面板对话 cid={console.panel.cid}，"
          f"中断向量 operator → {console.panel.cid}", file=sys.stderr)
    print(f"[console] 单客户端准入：{'开启' if console.single_client else '关闭（--allow-multiple-clients）'}",
          file=sys.stderr)
    print(f"[console] http://{args.host}:{server.server_port}/", file=sys.stderr)
    print(f"[console] 面板凭证：{token}", file=sys.stderr)
    token_path = os.path.join(
        os.path.dirname(os.path.abspath(args.image)), "out", "console.token"
    )
    try:
        os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(token)
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass  # Windows 上 chmod 语义有限，尽力而为
        print(f"[console] 凭证已写入 {token_path}", file=sys.stderr)
    except OSError as e:
        print(f"[console] 凭证文件写入失败（以终端显示为准）：{e}", file=sys.stderr)
    if not os.path.isfile(os.path.join(console.web_root, "index.html")):
        print(f"[console] 前端未构建（{console.web_root} 里没有 index.html）："
              f"cd web && npm install && npm run build；/ 会给出同样的提示", file=sys.stderr)

    console.running = True
    # API 失败停下问运行者：策略由加载区（avm/boot.py）统一安装，面板宿主一样走终端
    if args.no_core_dump:
        core.core_dump_enabled = False
    status = 0
    clean_exit = True
    try:
        core.run()
    except RunnerStop as e:
        print(f"[console] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[console] 收到中断，写回后退出", file=sys.stderr)
        status = 130
    except BaseException as e:
        # 内核死亡：先落现场快照，再保留原始 traceback 往上抛
        clean_exit = False
        dump_on_death(core, e, args.core_dump)
        raise
    finally:
        console.running = False
        console.lifecycle = "stopped"
        # 关机流程缺的那一环：正常退出（含 /api/system/shutdown 走到的安全点）
        # 写重启快照；异常死亡不拍，半截回合不能当恢复输入。
        if clean_exit:
            write_snapshot(core)
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[console] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[console] 写回失败: {e}", file=sys.stderr)
        server.shutdown()
    return status


if __name__ == "__main__":
    sys.exit(main())

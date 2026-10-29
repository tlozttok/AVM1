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
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from .core import _build_tools, _instruction_registry
from .exceptions import VMMemoryError, RunnerStop
from .memory_device import MemoryDevice
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

class Console:
    """调试控制台：面板对话 + 观测 + 介入 + 屏蔽账本。"""

    def __init__(self, core, web_root: Optional[str] = None):
        self.core = core
        # 前端构建产物目录（web/dist）。传 None 用仓库默认位置；测试传临时目录。
        self.web_root = web_root if web_root is not None else _WEB_DIST
        self.channel = OperatorChannel()
        self.panel = core.start_operator_conversation(self.channel)
        core.interrupts.register("operator", self.panel.cid)

        self.running = False                       # Core.run 是否在跑，由 runner 设置
        self._memory_mask: Dict[str, Any] = {}     # ref → 摘下来的节点
        self._device_mask: Dict[str, Any] = {}     # 挂载键 → 设备实例
        self._action_log: List[dict] = []          # 设备调试动作的调用记录
        self._snapshots: List[Dict[str, Any]] = [_walk(core.unwrap("$MEM", for_llm=False))]
        core.events.add_observer(self._on_event)

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
        }

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
<li><code>GET /api/devices</code> · <code>POST /api/device-action</code></li>
<li><code>GET /api/events</code>（SSE）</li>
<li><code>POST /api/interrupt</code> · <code>POST /api/resume</code> · <code>POST /api/action</code></li>
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
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

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
            elif path == "/api/memory/snapshot":
                self._send_json(200, {"snapshot": console.snapshot()})
            elif path == "/api/mask/memory":
                self._send_json(200, console.mask_memory(body.get("ref")))
            elif path == "/api/mask/device":
                self._send_json(200, console.mask_device(body.get("mount")))
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

    def do_PATCH(self):
        path = urlparse(self.path).path
        console = self.console
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
    parser.add_argument("--checkpoint-in", metavar="PATH",
                        help="调试用：启动时加载单对话检查点（跳过 init）")
    from .boot import add_core_dump_args
    add_core_dump_args(parser)
    args = parser.parse_args(argv)

    if not args.image:
        print("[console] 需要一个镜像路径", file=sys.stderr)
        return 1

    from .boot import boot, dump_on_death
    from .exceptions import VMMemoryError
    from .image import ImageError

    try:
        core = boot(args.image, checkpoint_in=args.checkpoint_in,
                    host_flags={"console": True})
    except (ImageError, VMMemoryError, OSError, ValueError) as e:
        print(f"[console] 镜像错误: {e}", file=sys.stderr)
        return 1

    console = Console(core)
    server = serve(console, args.host, args.port)
    print(f"[console] 面板对话 cid={console.panel.cid}，"
          f"中断向量 operator → {console.panel.cid}", file=sys.stderr)
    print(f"[console] http://{args.host}:{server.server_port}/", file=sys.stderr)
    if not os.path.isfile(os.path.join(console.web_root, "index.html")):
        print(f"[console] 前端未构建（{console.web_root} 里没有 index.html）："
              f"cd web && npm install && npm run build；/ 会给出同样的提示", file=sys.stderr)

    console.running = True
    # API 失败停下问运行者：策略由加载区（avm/boot.py）统一安装，面板宿主一样走终端
    if args.no_core_dump:
        core.core_dump_enabled = False
    status = 0
    try:
        core.run()
    except RunnerStop as e:
        print(f"[console] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[console] 收到中断，写回后退出", file=sys.stderr)
        status = 130
    except BaseException as e:
        # 内核死亡：先落现场快照，再保留原始 traceback 往上抛
        dump_on_death(core, e, args.core_dump)
        raise
    finally:
        console.running = False
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[console] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[console] 写回失败: {e}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())

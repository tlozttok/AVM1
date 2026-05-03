"""AVM Web UI — FastAPI + SSE 对话界面"""
import asyncio
import json
import os
import queue
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avm.core import Core
from avm.memory import Memory
from avm.types import MetaDict
from web.devices import WebInputDevice, WebOutputDevice
from web.monitor_state import MonitorState

DEFAULT_CONFIG = {
    "log_level": "INFO",
    "log_file": None,
    "debug": False,
    "prompt": "programs/deepseek_simplified.json",
}


def load_config(path: str = "config.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)
    return config


def _json_to_meta(value):
    if isinstance(value, dict):
        metadata = value.pop("meta", None) if "meta" in value else None
        data = {k: _json_to_meta(v) for k, v in value.items()}
        return MetaDict(data=data, metadata=metadata)
    elif isinstance(value, list):
        return [_json_to_meta(v) for v in value]
    else:
        return value


# --- 全局状态（在 lifespan 中初始化）---

config: dict = {}
core: Core | None = None
output_queues: list = []
queues_lock = threading.Lock()

# 监控面板状态
monitor_state: MonitorState | None = None
monitor_queues: list = []
monitor_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, core, output_queues

    # 加载配置
    config = load_config()

    # 日志配置 (写到 stderr，方便重定向到文件)
    import logging
    logging.basicConfig(
        level=logging.DEBUG if config.get("debug") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 创建 Core
    core = Core()
    core.debug = config.get("debug", False)

    # 持久化
    user_uuid = config.get("uuid", "dev-001")
    persist_level = config.get("persist_level", "off")
    persist_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", f"avm_{user_uuid}.json"
    )
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    core.persist_path = persist_path
    core.persist_level = persist_level

    restart = config.get("restart", False)
    if not restart and os.path.isfile(persist_path):
        core.mem = Memory.load(persist_path)
    else:
        core.mem = Memory()

    # 加载 prompt 到内存
    prompt_path = config["prompt"]
    if restart or not os.path.isfile(persist_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            core.mem[key] = _json_to_meta(value)

    # 模型参数
    if "model_params" not in core.mem or not isinstance(core.mem.get("model_params"), MetaDict):
        core.mem["model_params"] = MetaDict(data={
            "model": "deepseek-v4-flash",
            "extra_body": {"thinking": {"type": "disabled"}},
            "use_tool": "auto",
        })

    # SSE 广播队列
    output_queues = []

    # 挂载 Web 设备
    web_input = WebInputDevice(data=[], metadata="用户输入列表")
    core.mem.mount("inputs", web_input)
    core.mem.mount("outputs", WebOutputDevice(output_queues, data=[], metadata="对用户的输出列表"))

    core.command_stack.append("create 0 -1 $MEM.system $MEM.user $MEM.model_params")

    # 启动监控状态收集器
    global monitor_state
    monitor_state = MonitorState(core)
    monitor_state.attach()

    def _push_monitor_snapshot(event_type, payload):
        """Core 状态变化时，构建快照并推送给所有监控 SSE 客户端"""
        if monitor_state is None:
            return
        try:
            snapshot = monitor_state.build_snapshot()
            with monitor_lock:
                for q in monitor_queues:
                    try:
                        q.put_nowait(snapshot)
                    except queue.Full:
                        pass
        except Exception:
            pass

    core.add_state_observer(_push_monitor_snapshot)

    # 注册输入设备回调，让 monitor 能追踪输入等待事件
    def _on_input_wait():
        if monitor_state is not None:
            monitor_state.on_input_wait()

    def _on_input_received(text):
        if monitor_state is not None:
            monitor_state.on_input_received(text)

    web_input._on_input_wait = _on_input_wait
    web_input._on_input_received = _on_input_received

    # 启动 Core 在后台线程
    core_thread = threading.Thread(target=core.run, daemon=True)
    core_thread.start()

    yield

    # 关闭时可以在这里清理资源
    if monitor_state is not None:
        monitor_state.detach()
        monitor_state = None


app = FastAPI(title="AVM Chat", lifespan=lifespan)


# --- FastAPI 路由 ---

@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/input")
async def send_input(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return {"ok": True}
    if core is None:
        return {"ok": False, "error": "core not initialized"}
    web_input = core.mem.get_device(["inputs"])
    if web_input is not None and hasattr(web_input, "provide_input"):
        web_input.provide_input(text)
    return {"ok": True}


@app.get("/stream")
async def stream():
    q: queue.Queue = queue.Queue()
    with queues_lock:
        output_queues.append(q)

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.get_event_loop().run_in_executor(None, q.get)
                    yield f"data: {json.dumps({'text': msg})}\n\n"
                except Exception:
                    break
        finally:
            with queues_lock:
                if q in output_queues:
                    output_queues.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 监控面板路由 ---

@app.get("/monitor", response_class=HTMLResponse)
async def monitor_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "monitor.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/monitor/stream")
async def monitor_stream():
    q: queue.Queue = queue.Queue(maxsize=8)
    with monitor_lock:
        monitor_queues.append(q)

    # 立即推送一次当前快照
    if monitor_state is not None:
        try:
            init_snapshot = monitor_state.build_snapshot()
            q.put_nowait(init_snapshot)
        except Exception:
            pass

    async def event_generator():
        try:
            while True:
                try:
                    snapshot = await asyncio.get_event_loop().run_in_executor(None, q.get)
                    yield f"data: {json.dumps(snapshot, default=str)}\n\n"
                except Exception:
                    break
        finally:
            with monitor_lock:
                if q in monitor_queues:
                    monitor_queues.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()

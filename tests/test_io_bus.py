"""操作面板底层的离线测试（计划 A/B/D/F、关机收尾、单客户端准入）。

覆盖：宿主侧 IO 总线的 seq 与定向输入、操作面板宿主专用的两台 IO 设备、
Core 的停止标志、Console 的新端点与认证收紧。全程裸 Core / 本地 HTTP，不调真实 API。

打断（ConversationAborted / abort_wait / /api/system/abort）不在这里：那套设计已按
裁决整体删除，另立计划重做。本轮的输入等待只由新输入唤醒。
"""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from avm.console import Console, serve
from avm.core import Core
from avm.io_bus import IOBus
from avm.io_devices import IOInputDevice, IOOutputDevice
from avm.memory_device import InputsListDevice, OutputsListDevice, StringDevice


def _request(base, method, path, body=None, headers=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw, code, resp_headers = resp.read(), resp.status, dict(resp.headers)
    except urllib.error.HTTPError as e:
        raw, code, resp_headers = e.read(), e.code, dict(e.headers)
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = raw.decode("utf-8", "replace")
    return code, parsed, resp_headers


def _start(core=None, **kwargs):
    core = core if core is not None else Core()
    console = Console(core, web_root=kwargs.pop("web_root", "_no_dist_"), **kwargs)
    server = serve(console, port=0)
    return core, console, server, f"http://127.0.0.1:{server.server_port}"


@pytest.fixture
def env(tmp_path):
    core, console, server, base = _start(
        web_root=str(tmp_path / "no-dist"), force_stop_delay=0)
    try:
        yield core, console, base
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def gated_env(tmp_path):
    """真实宿主策略：一个客户端、30 秒后才允许强制退出。"""
    core, console, server, base = _start(
        web_root=str(tmp_path / "no-dist"),
        single_client=True, client_lease_seconds=1.0, force_stop_delay=30.0)
    try:
        yield core, console, base
    finally:
        server.shutdown()
        server.server_close()


def _seeded_core(cid=7, name="助手", identity="assistant-7"):
    core = Core()
    core.io_bus = IOBus()
    core._active_cid = cid
    core._conv_by_cid[cid] = SimpleNamespace(cid=cid, name=name, identity=identity)
    return core


# ----------------------------------------------------------------------
# A. IO 总线
# ----------------------------------------------------------------------

class TestIOBus:
    def test_output_stream_has_monotonic_seq_and_incremental_drain(self):
        bus = IOBus()
        first = bus.publish_output(3, "第一条")
        second = bus.publish_output(None, "第二条", kind="notify", meta={"notify": True})
        assert first["seq"] == 1 and second["seq"] == 2
        assert [e["text"] for e in bus.drain_io(0)] == ["第一条", "第二条"]
        assert [e["text"] for e in bus.drain_io(first["seq"])] == ["第二条"]
        assert bus.drain_io(second["seq"]) == []
        assert bus.status()["output"]["next_seq"] == 3
        assert bus.status()["output"]["dropped"] == 0

    def test_ring_buffer_drops_oldest_and_counts(self):
        bus = IOBus(capacity=3)
        for i in range(5):
            bus.publish_output(None, str(i))
        assert [e["text"] for e in bus.drain_io(0)] == ["2", "3", "4"]
        assert bus.status()["output"]["dropped"] == 2

    def test_targeted_input_only_matching_identity_takes(self):
        bus = IOBus()
        bus.publish_input("7", "给 cid 7")
        bus.publish_input("助手", "给名字")
        bus.publish_input(None, "全局一行")
        assert bus.take_input(["9", "别人"])["text"] == "全局一行"
        assert bus.take_input(["7", "助手"])["text"] == "给 cid 7"
        assert bus.take_input(["助手"])["text"] == "给名字"
        assert bus.take_input(["7", "助手"]) is None

    def test_peek_does_not_consume_and_pending_counts(self):
        bus = IOBus()
        bus.publish_input("7", "a")
        bus.publish_input("8", "b")
        bus.publish_input(None, "c")
        assert bus.pending_input(["7"]) == 2  # a + 全局 c
        assert bus.peek_input(["7"])["text"] == "a"
        assert bus.peek_input(["7"])["text"] == "a"
        assert bus.pending_input(["7"]) == 2
        assert bus.take_input(["7"])["text"] == "a"
        assert bus.pending_input(["7"]) == 1

    def test_wildcard_target_becomes_global(self):
        bus = IOBus()
        assert bus.publish_input("*", "所有人")["target"] is None
        assert bus.take_input([])["text"] == "所有人"


# ----------------------------------------------------------------------
# B. 操作面板宿主专用的两台 IO 设备
# ----------------------------------------------------------------------

class TestIODevices:
    def test_output_publishes_to_bus_with_active_cid(self):
        core = _seeded_core()
        dev = IOOutputDevice()
        dev.attach_core(core)
        dev["out"] = "你好"
        entries = core.io_bus.drain_io(0)
        assert len(entries) == 1
        assert entries[0]["text"] == "你好"
        assert entries[0]["source"] == 7
        assert entries[0]["kind"] == "text"
        assert list(dev) == ["你好"]

    def test_notify_output_is_marked(self):
        core = _seeded_core()
        dev = IOOutputDevice()
        dev.attach_core(core)
        dev["notify"] = "需要操作员注意"
        entry = core.io_bus.drain_io(0)[0]
        assert entry["kind"] == "notify"
        assert entry["meta"]["notify"] is True

    def test_output_device_never_prints_to_terminal(self, capsys):
        dev = IOOutputDevice()
        dev["out"] = "只进设备"
        assert capsys.readouterr().out == ""
        assert list(dev) == ["只进设备"]

    def test_output_without_bus_just_stores(self, capsys):
        dev = IOOutputDevice()
        dev.attach_core(Core())  # io_bus 为 None
        dev["out"] = "没有总线"
        assert capsys.readouterr().out == ""
        assert list(dev) == ["没有总线"]

    def test_input_next_takes_targeted_input(self):
        core = _seeded_core()
        dev = IOInputDevice()
        dev.attach_core(core)
        core.io_bus.publish_input("8", "不是给我的")
        core.io_bus.publish_input("7", "给我的")
        assert dev["next"] == "给我的"
        assert list(dev) == ["给我的"]

    def test_input_peek_and_pending_named_interfaces(self):
        core = _seeded_core()
        dev = IOInputDevice()
        dev.attach_core(core)
        assert dev["pending"] == 0
        assert dev["peek"] == ""
        core.io_bus.publish_input("助手", "一行")
        assert dev["peek"] == "一行"
        assert dev["pending"] == 1
        assert dev["next"] == "一行"  # peek 没消费
        assert dev["pending"] == 0

    def test_debug_feed_queue_still_wins(self):
        core = _seeded_core()
        dev = IOInputDevice()
        dev.attach_core(core)
        dev.debug_actions()[0]["run"]({"content": "调试抽屉喂的"})
        core.io_bus.publish_input("7", "总线来的")
        assert dev["next"] == "调试抽屉喂的"
        assert dev["next"] == "总线来的"

    def test_pending_debug_action_counts_bus_and_queue(self):
        core = _seeded_core()
        dev = IOInputDevice()
        dev.attach_core(core)
        dev.debug_actions()[0]["run"]({"content": "队列一行"})
        core.io_bus.publish_input("7", "总线一行")
        assert dev.debug_actions()[1]["run"]({})["queued"] == 2

    def test_blocking_read_is_woken_by_new_input(self):
        core = _seeded_core()
        dev = IOInputDevice()
        dev.attach_core(core)
        result = []
        thread = threading.Thread(target=lambda: result.append(dev["next"]), daemon=True)
        thread.start()
        deadline = time.monotonic() + 3
        while not dev._waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert dev._waiting, "读没有进入等待"
        core.io_bus.publish_input("7", "唤醒这一读")
        dev.notify_input()
        thread.join(timeout=3)
        assert result == ["唤醒这一读"]
        assert list(dev) == ["唤醒这一读"]


class TestConsoleInstallsIODevices:
    def test_replaces_standard_devices_and_keeps_data(self):
        core = Core()
        core.mem.mount("inputs", InputsListDevice(data=["旧输入"], metadata="输入"))
        core.mem.mount("outputs", OutputsListDevice(data=["旧输出"], metadata="输出"))
        console = Console(core)
        devices = core.mem.mounted_devices()
        assert isinstance(devices["inputs"], IOInputDevice)
        assert isinstance(devices["outputs"], IOOutputDevice)
        assert list(devices["inputs"]) == ["旧输入"]
        assert list(devices["outputs"]) == ["旧输出"]
        assert console.io_input_devices == [devices["inputs"]]
        assert console.io_output_devices == [devices["outputs"]]

    def test_io_devices_already_mounted_are_only_registered(self):
        core = Core()
        dev = IOInputDevice()
        core.mem.mount("inputs", dev)
        console = Console(core)
        assert core.mem.mounted_devices()["inputs"] is dev
        assert console.io_input_devices == [dev]

    def test_non_list_device_at_mount_is_left_alone(self):
        core = Core()
        other = StringDevice("原样")
        core.mem.mount("inputs", other)
        console = Console(core)
        assert core.mem.mounted_devices()["inputs"] is other
        assert console.io_input_devices == []

    def test_undeclared_mounts_are_not_created(self):
        core = Core()
        Console(core)
        assert "inputs" not in core.mem.mounted_devices()
        assert "outputs" not in core.mem.mounted_devices()


# ----------------------------------------------------------------------
# C. Core 生命周期
# ----------------------------------------------------------------------

class TestCoreLifecycle:
    def test_request_stop_returns_at_safe_point_without_advancing(self):
        core = Core()
        core.request_stop()
        advanced = []
        core._pick_next_active = lambda: setattr(core, "_active_cid", 1)
        core.advance_conversation = lambda: advanced.append(1)
        core.interrupts.dispatch = lambda: None
        core._check_invariants = lambda: None
        core.monitor.record_baseline = lambda _core: None
        core.run()
        assert advanced == [], "停止标志命中后不该再推进一轮"
        assert core._stop_requested is True

    def test_recovery_default_is_cold(self):
        assert Core().recovery["source"] == "cold"


# ----------------------------------------------------------------------
# D. Console 端点与认证
# ----------------------------------------------------------------------

class TestConsoleEndpoints:
    def test_state_exposes_lifecycle_recovery_io_client(self, env):
        _, _, base = env
        code, state, _ = _request(base, "GET", "/api/state")
        assert code == 200
        assert state["lifecycle"] == "running"
        assert state["recovery"]["source"] in ("cold", "snapshot", "journal")
        assert set(state["io"]) == {"input", "output"}
        assert state["client"]["single_client"] is False
        assert "force_stop_after" in state

    def test_io_input_endpoint_publishes(self, env):
        core, _, base = env
        code, record, _ = _request(
            base, "POST", "/api/io/input", {"target": "7", "text": "给 7 的输入"})
        assert code == 200
        assert record["debug"] is True, "io/input 是调试通道，响应要标注"
        assert record["seq"] == 1
        assert record["target"] == "7"
        assert core.io_bus.drain_input()[0]["text"] == "给 7 的输入"

    def test_io_input_wakes_waiting_device(self, env):
        core, console, base = env
        dev = IOInputDevice()
        dev.attach_core(core)
        console.io_input_devices.append(dev)
        result = []
        thread = threading.Thread(target=lambda: result.append(dev["next"]), daemon=True)
        thread.start()
        deadline = time.monotonic() + 3
        while not dev._waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert dev._waiting
        code, _, _ = _request(base, "POST", "/api/io/input", {"text": "来自面板"})
        assert code == 200
        thread.join(timeout=3)
        assert result == ["来自面板"]

    def test_io_input_rejects_empty_text(self, env):
        _, _, base = env
        code, _, _ = _request(base, "POST", "/api/io/input", {"text": ""})
        assert code == 400

    def test_shutdown_and_force_stop_endpoints(self, env):
        core, console, base = env
        code, body, _ = _request(base, "POST", "/api/system/shutdown", {})
        assert code == 202 and body["status"] == "stopping"
        # 关机请求走 OS 约定（写 $MEM.sys.power.request），不再直接停内核
        assert core._stop_requested is False
        assert body["entry"] == "$MEM.sys.power.request"
        assert console.lifecycle == "stopping"
        assert console.channel.status()["pending_actions"] == 2
        # 关机后不再收新输入
        code, _, _ = _request(base, "POST", "/api/io/input", {"text": "迟到的输入"})
        assert code == 409
        code, body, _ = _request(base, "POST", "/api/system/force-stop", {})
        assert code == 202
        assert body["snapshot_written"] is False
        assert console.lifecycle == "stopped"

    def test_force_stop_without_hook_is_safe(self, env):
        _, console, base = env
        code, body, _ = _request(base, "POST", "/api/system/force-stop", {})
        assert code == 202
        assert body["snapshot_written"] is False
        assert console.lifecycle == "stopped"

    def test_io_stream_is_sse(self, env):
        _, console, base = env
        console.io_bus.publish_output(None, "先放一条")
        conn = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=3)
        conn.request("GET", "/api/io/stream?after=0")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in resp.getheader("Content-Type")
        conn.close()

    def test_console_installs_bus_on_core(self, env):
        core, console, _ = env
        assert core.io_bus is console.io_bus

    def test_system_abort_endpoint_is_gone(self, env):
        _, _, base = env
        code, _, _ = _request(base, "POST", "/api/system/abort", {})
        assert code == 404


class TestShutdownGrace:
    def test_force_stop_is_blocked_until_grace_expires(self, gated_env):
        _, console, base = gated_env
        code, _, _ = _request(base, "POST", "/api/system/force-stop", {})
        assert code == 409
        code, body, _ = _request(base, "POST", "/api/system/shutdown", {})
        assert code == 202 and body["force_stop_after"] == 30.0
        code, state, _ = _request(base, "GET", "/api/state")
        assert code == 200 and state["force_stop_remaining"] > 0
        code, _, _ = _request(base, "POST", "/api/system/force-stop", {})
        assert code == 409
        console.stopping_since = time.time() - 31
        code, _, _ = _request(base, "POST", "/api/system/force-stop", {})
        assert code == 202


class TestSingleClient:
    def test_second_client_is_rejected_until_lease_expires(self, gated_env):
        _, _, base = gated_env
        code, body, _ = _request(
            base, "POST", "/api/session/acquire", {"client_id": "A"})
        assert code == 200 and body["single_client"] is True
        code, body, _ = _request(
            base, "POST", "/api/session/acquire", {"client_id": "B"})
        assert code == 409 and "A" in body["error"]
        code, _, _ = _request(base, "GET", "/api/state", headers={"X-AVM-Client": "B"})
        assert code == 409
        code, _, _ = _request(base, "GET", "/api/state", headers={"X-AVM-Client": "A"})
        assert code == 200
        # SSE 靠 cookie；宿主本地调用（不带身份）照旧放行
        code, _, _ = _request(base, "GET", "/api/state", headers={"Cookie": "avm_client=A"})
        assert code == 200
        code, _, _ = _request(base, "GET", "/api/state")
        assert code == 200

    def test_release_frees_the_seat(self, gated_env):
        _, _, base = gated_env
        _request(base, "POST", "/api/session/acquire", {"client_id": "A"})
        code, body, _ = _request(
            base, "POST", "/api/session/release", {"client_id": "B"})
        assert code == 200 and body["released"] is False
        code, body, _ = _request(
            base, "POST", "/api/session/release", {"client_id": "A"})
        assert code == 200 and body["released"] is True
        code, _, _ = _request(
            base, "POST", "/api/session/acquire", {"client_id": "B"})
        assert code == 200

    def test_stale_lease_expires(self, gated_env):
        _, _, base = gated_env
        _request(base, "POST", "/api/session/acquire", {"client_id": "A"})
        time.sleep(1.2)
        code, _, _ = _request(
            base, "POST", "/api/session/acquire", {"client_id": "B"})
        assert code == 200
        code, _, _ = _request(base, "GET", "/api/state", headers={"X-AVM-Client": "A"})
        assert code == 409

    def test_acquire_requires_client_id(self, gated_env):
        _, _, base = gated_env
        code, _, _ = _request(base, "POST", "/api/session/acquire", {})
        assert code == 400


class TestConsoleAuth:
    @pytest.fixture
    def auth_env(self, tmp_path):
        core = Core()
        console = Console(core, web_root=str(tmp_path / "no-dist"), auth_token="secret-token")
        server = serve(console, port=0)
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            yield console, base
        finally:
            server.shutdown()
            server.server_close()

    def test_api_requires_token(self, auth_env):
        _, base = auth_env
        code, body, headers = _request(base, "GET", "/api/state")
        assert code == 401
        assert headers.get("Access-Control-Allow-Origin") != "*"

    def test_bearer_token_passes(self, auth_env):
        _, base = auth_env
        code, _, _ = _request(
            base, "GET", "/api/state", headers={"Authorization": "Bearer secret-token"})
        assert code == 200

    def test_static_page_loads_without_token(self, auth_env):
        _, base = auth_env
        code, body, _ = _request(base, "GET", "/")
        assert code == 200
        assert "调试控制台" in body

    def test_cross_origin_request_rejected(self, auth_env):
        _, base = auth_env
        code, _, _ = _request(
            base, "GET", "/api/state",
            headers={"Authorization": "Bearer secret-token",
                     "Origin": "http://evil.example"})
        assert code == 403

"""控制台 HTTP 层冒烟

起真实服务、本地请求每个端点，断言状态码与关键字段。前端的契约变了这里会先红。
"""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from avm.console import Console, serve
from avm.core import Core, _build_tools
from avm.memory_device import InputsListDevice, MemoryDevice
from avm.types import MetaDict


class MockLMU:
    def __init__(self, on_exec=None):
        self.on_exec = on_exec

    def exec(self, conversation, para):
        if self.on_exec is not None:
            self.on_exec(conversation)
        return (f"done-{conversation.cid}", [], None)


class EchoDevice(MemoryDevice):
    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return "echo 设备"

    def debug_actions(self):
        return [{"name": "回显", "description": "原样返回参数",
                 "run": lambda args: {"echo": args}}]


def request(base, method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw, code = resp.read(), resp.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    try:
        return code, json.loads(raw) if raw else None
    except ValueError:
        return code, raw.decode("utf-8", "replace")


def wait_for(predicate, timeout=5.0):
    # 单调钟：墙钟会被系统校时前后跳，用它算 deadline 会提前放弃等待
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def env(tmp_path):
    core = Core()
    core.lmu = MockLMU()
    core.mem["note"] = MetaDict(data={"v": "hi"}, metadata="一条笔记", ctrl={"owner": "0"})
    core.mem["h"] = MetaDict(data={"content": "def handle(event):\n    return {}\n"},
                             ctrl={"type": "python"})
    core.mem["own"] = MetaDict(data={"child": ""}, ctrl={"owner": "0"})
    core.mem.mount("echo", EchoDevice())
    # web_root 指向一个空目录：这里断言「没构建」的分支，测试不依赖开发者是否跑过 npm build
    console = Console(core, web_root=str(tmp_path / "no-dist"))
    listener_id = core.events.register(
        owner_cid=0, event_type="memory_write", target="$MEM.own.child",
        handler_ref="$MEM.h", phase="before",
    )
    server = serve(console, port=0)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield core, console, base, listener_id
    finally:
        server.shutdown()
        server.server_close()


class TestReadEndpoints:
    def test_index_page(self, env):
        """没有构建产物时，/ 给的是构建提示页，不是 404。"""
        code, body = request(env[2], "GET", "/")
        assert code == 200
        assert "调试控制台" in body
        assert "npm run build" in body

    def test_tools_endpoint(self, env):
        code, body = request(env[2], "GET", "/api/tools")
        assert code == 200
        assert body["tools"] == _build_tools()
        assert len(body["tools"]) == 9

    def test_state(self, env):
        core, console, base, _ = env
        code, state = request(base, "GET", "/api/state")
        assert code == 200
        assert state["schedule"]["panel_cid"] == console.panel.cid
        assert state["interrupts"]["vectors"] == {"operator": console.panel.cid}
        assert state["panel"]["channel"]["dropped"] == 0
        assert state["mask"] == {"memory": [], "devices": []}
        assert [c["cid"] for c in state["conversations"]] == [console.panel.cid]
        assert state["conversations"][0]["status"] == "dormant"

    def test_conversation_detail(self, env):
        core, console, base, _ = env
        code, conv = request(base, "GET", f"/api/conversations/{console.panel.cid}")
        assert code == 200
        assert conv["is_python"] is True
        assert conv["history"][0]["role"] == "system"
        assert "user_batch" in conv

    def test_conversation_detail_missing(self, env):
        code, body = request(env[2], "GET", "/api/conversations/99")
        assert code == 404
        assert "error" in body

    def test_memory_node(self, env):
        code, body = request(env[2], "GET", "/api/memory?ref=$MEM.note")
        assert code == 200
        assert body["node"]["kind"] == "dict"
        assert body["node"]["metadata"] == "一条笔记"
        assert body["node"]["ctrl"] == {"owner": "0"}
        assert body["node"]["children"]["v"] == "hi"

    def test_memory_whole_tree(self, env):
        code, body = request(env[2], "GET", "/api/memory")
        assert code == 200
        assert body["ref"] == "$MEM"
        assert "note" in body["node"]["children"]

    def test_listeners(self, env):
        core, console, base, listener_id = env
        code, body = request(base, "GET", "/api/listeners")
        assert code == 200
        assert body["listeners"][0]["listener_id"] == listener_id
        assert body["listeners"][0]["enabled"] is True
        assert "enabled" in body["editable_fields"]

    def test_devices(self, env):
        code, body = request(env[2], "GET", "/api/devices")
        assert code == 200
        mounts = [d["mount"] for d in body["devices"]]
        assert "echo" in mounts
        echo = [d for d in body["devices"] if d["mount"] == "echo"][0]
        assert echo["actions"][0]["name"] == "回显"

    def test_unknown_endpoint(self, env):
        code, body = request(env[2], "GET", "/api/nope")
        assert code == 404
        assert "error" in body

    def test_options_preflight(self, env):
        code, body = request(env[2], "OPTIONS", "/api/state")
        assert code == 204


class TestWriteEndpoints:
    def test_patch_listener_disable_and_enable(self, env):
        core, console, base, listener_id = env
        code, record = request(base, "PATCH", f"/api/listeners/{listener_id}",
                               {"enabled": False})
        assert code == 200
        assert record["enabled"] is False

        code, record = request(base, "PATCH", f"/api/listeners/{listener_id}",
                               {"target": "$MEM.own.child", "enabled": True})
        assert code == 200
        assert record["enabled"] is True

    def test_patch_listener_bad_field(self, env):
        core, console, base, listener_id = env
        code, body = request(base, "PATCH", f"/api/listeners/{listener_id}",
                             {"owner_cid": 3})
        assert code == 400
        assert "不可改写" in body["error"]

    def test_patch_missing_listener(self, env):
        code, body = request(env[2], "PATCH", "/api/listeners/99", {"enabled": False})
        assert code == 404

    def test_snapshot_and_diff(self, env):
        core, console, base, _ = env
        code, body = request(base, "POST", "/api/memory/snapshot")
        assert code == 200
        snapshot_id = body["snapshot"]

        core.mem["added"] = "later"
        code, body = request(base, "GET", f"/api/memory/diff?from=0&to={snapshot_id}")
        assert code == 200
        assert "added" not in body["added"] or body["added"] == []

        code, body = request(base, "GET", "/api/memory/diff?from=0")
        assert code == 200
        assert "added" in body["added"]

    def test_diff_missing_snapshot(self, env):
        code, body = request(env[2], "GET", "/api/memory/diff?from=99")
        assert code == 404

    def test_device_action(self, env):
        code, record = request(env[2], "POST", "/api/device-action",
                               {"mount": "echo", "index": 0, "arguments": {"a": 1}})
        assert code == 200
        assert record["result"] == {"echo": {"a": 1}}
        assert record["error"] is None

    def test_device_action_missing_device(self, env):
        code, body = request(env[2], "POST", "/api/device-action",
                             {"mount": "nope", "index": 0})
        assert code == 404

    def test_panel_input_box_feeds_the_inputs_device(self, env):
        """输入设备的调试台经 HTTP 暴露成面板上的输入框。"""
        core, console, base, _ = env
        core.mem.mount("inputs", InputsListDevice(data=[]))

        code, listing = request(base, "GET", "/api/devices")
        entry = [d for d in listing["devices"] if d["mount"] == "inputs"][0]
        assert entry["actions"][0]["name"] == "投喂一行输入"
        assert entry["actions"][0]["parameters"]["required"] == ["content"]

        code, record = request(base, "POST", "/api/device-action", {
            "mount": "inputs", "index": 0, "arguments": {"content": "面板输入"}})
        assert code == 200
        assert record["result"] == {"queued": 1}

        assert core.mem["inputs"][-1] == "面板输入"

    def test_panel_input_wakes_a_waiting_read(self, env):
        """托管下的端到端：设备等在队列上，HTTP 喂一行把它唤醒。

        host_flags 由 load_image 在挂载设备之前写入，这里手工补上同一步。
        """
        core, console, base, _ = env
        core.host_flags = {"console": True}
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        device.attach_core(core)
        assert device._console_input is True

        got = []
        thread = threading.Thread(target=lambda: got.append(core.mem["inputs"][-1]))
        thread.start()
        for _ in range(300):  # 等到读真的等在队列上
            if device._pending_input:
                break
            time.sleep(0.01)
        assert device._pending_input, "读没有进入等待"

        code, record = request(base, "POST", "/api/device-action", {
            "mount": "inputs", "index": 0, "arguments": {"content": "面板输入"}})
        assert code == 200
        assert record["result"] == {"queued": 1}

        thread.join(timeout=5)
        assert not thread.is_alive(), "喂进来之后读没有返回"
        assert got == ["面板输入"]

    def test_mask_and_unmask_memory(self, env):
        core, console, base, _ = env
        code, body = request(base, "POST", "/api/mask/memory", {"ref": "$MEM.note"})
        assert code == 200
        assert body["masked"] == "$MEM.note"

        code, body = request(base, "GET", "/api/memory?ref=$MEM.note")
        assert code == 404, "屏蔽期间读到键不存在"

        code, body = request(base, "DELETE", "/api/mask/memory?ref=$MEM.note")
        assert code == 200
        code, body = request(base, "GET", "/api/memory?ref=$MEM.note")
        assert code == 200
        assert body["node"]["metadata"] == "一条笔记"

    def test_unmask_conflict_is_409(self, env):
        core, console, base, _ = env
        request(base, "POST", "/api/mask/memory", {"ref": "$MEM.note"})
        core.mem["note"] = "对话重建的"
        code, body = request(base, "DELETE", "/api/mask/memory?ref=$MEM.note")
        assert code == 409
        assert "还原冲突" in body["error"]

    def test_mask_and_unmask_device(self, env):
        core, console, base, _ = env
        code, body = request(base, "POST", "/api/mask/device", {"mount": "echo"})
        assert code == 200
        code, body = request(base, "GET", "/api/devices")
        assert "echo" not in [d["mount"] for d in body["devices"]]

        code, body = request(base, "DELETE", "/api/mask/device?mount=echo")
        assert code == 200
        code, body = request(base, "GET", "/api/devices")
        assert "echo" in [d["mount"] for d in body["devices"]]

    def test_bad_json_body(self, env):
        req = urllib.request.Request(
            env[2] + "/api/mask/memory", data=b"{not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.status
        except urllib.error.HTTPError as e:
            code = e.code
        assert code == 400


class TestInterventionFlow:
    def test_pause_action_resume(self, env):
        core, console, base, _ = env
        first = core.start("sys", "one")
        second = core.start("sys", "two")

        def on_exec(conv):
            if conv.cid == first.cid:
                core.interrupts.raise_("operator")

        core.lmu.on_exec = on_exec
        thread = threading.Thread(target=core.run, daemon=True)
        thread.start()
        try:
            code, body = request(base, "POST", "/api/interrupt")
            assert code == 200
            assert body["requested"] is True

            assert wait_for(lambda: core._active_cid == console.panel.cid), "面板没有取得执行权"
            code, state = request(base, "GET", "/api/state")
            assert state["panel"]["active"] is True
            assert state["schedule"]["ready"][0] == second.cid

            code, body = request(base, "POST", "/api/action", {
                "tool": "memory_write",
                "arguments": {"ref": "$MEM.from_panel", "content": "v"},
            })
            assert code == 200
            assert body["known_tool"] is True
            assert body["auto_paused"] is False

            assert wait_for(lambda: "from_panel" in core.mem)
            assert core.mem["from_panel"] == "v"
            assert core._active_cid == console.panel.cid

            code, body = request(base, "POST", "/api/action",
                                 {"tool": "nope", "arguments": {}})
            assert code == 200
            assert body["known_tool"] is False
            assert "未知工具" in body["warning"]

            code, body = request(base, "POST", "/api/resume")
            assert code == 200
            thread.join(timeout=5)
            assert not thread.is_alive()
        finally:
            console.channel.submit_action({"end": True})
            thread.join(timeout=5)

        assert core._active_cid is None

    def test_action_auto_pauses(self, env):
        """没暂停时提交动作：控制台先自动中断，动作在下一个安全点被面板取走。"""
        core, console, base, _ = env
        first = core.start("sys", "one")
        second = core.start("sys", "two")
        third = core.start("sys", "three")
        release = threading.Event()

        def on_exec(conv):
            if conv.cid == first.cid:
                core.interrupts.raise_("operator")
            elif conv.cid == second.cid:
                release.wait(timeout=5)  # 拖住 Core，让中断留在挂起状态

        core.lmu.on_exec = on_exec
        thread = threading.Thread(target=core.run, daemon=True)
        thread.start()
        try:
            assert wait_for(lambda: core._active_cid == console.panel.cid)
            request(base, "POST", "/api/resume")

            assert wait_for(lambda: core._active_cid == second.cid), "被打断的对话应当先跑"
            code, body = request(base, "POST", "/api/action", {
                "tool": "memory_write",
                "arguments": {"ref": "$MEM.auto", "content": "v"},
            })
            assert code == 200
            assert body["auto_paused"] is True, "没暂停时控制台应当先自动中断"

            release.set()
            assert wait_for(lambda: "auto" in core.mem), "恢复后动作应当被执行"
            assert core.mem["auto"] == "v"
        finally:
            release.set()
            console.channel.submit_action({"end": True})
            thread.join(timeout=5)

    def test_bad_action_body(self, env):
        code, body = request(env[2], "POST", "/api/action", {"tool": ""})
        assert code == 400

    def test_resume_without_pause_is_harmless(self, env):
        code, body = request(env[2], "POST", "/api/resume")
        assert code == 200
        assert body["submitted"] is True


class TestEventStream:
    def test_sse_first_frame_is_the_schedule(self, env):
        core, console, base, _ = env
        response = urllib.request.urlopen(base + "/api/events", timeout=5)
        try:
            assert response.status == 200
            assert "text/event-stream" in response.headers.get("Content-Type")
            payload = None
            # 单调钟：墙钟会被系统校时前后跳，用它算 deadline 会提前放弃等待
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                line = response.readline().decode("utf-8").strip()
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: "):])
                    break
            assert payload is not None, "SSE 没有送出任何帧"
            assert payload["type"] == "schedule"
            assert payload["data"]["panel_cid"] == console.panel.cid
        finally:
            response.close()

    def test_sse_carries_panel_output_and_events(self, env):
        core, console, base, _ = env
        response = urllib.request.urlopen(base + "/api/events", timeout=5)
        try:
            console.channel.submit_action({
                "tool": "memory_write",
                "arguments": {"ref": "$MEM.sse", "content": "v"},
            })
            core.interrupts.register("operator", console.panel.cid)
            core.interrupts.raise_("operator")
            core.interrupts.dispatch()
            core.advance_conversation()  # 面板取走动作、Core 执行、事件被观察者转发

            kinds = []
            # 单调钟：墙钟会被系统校时前后跳，用它算 deadline 会提前放弃等待
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and "event" not in kinds:
                line = response.readline().decode("utf-8").strip()
                if line.startswith("data: "):
                    kinds.append(json.loads(line[len("data: "):])["type"])
            assert "tool_call" in kinds
            assert "event" in kinds
        finally:
            response.close()


@pytest.fixture
def web_env(tmp_path):
    """构建产物齐全的控制台：/ 发面板，/web/<rel> 发静态文件。"""
    core = Core()
    core.lmu = MockLMU()
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><title>AVM 调试面板</title><div id="root"></div>', encoding="utf-8")
    (dist / "assets" / "panel.js").write_text("console.log('panel')", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("不该被读到", encoding="utf-8")
    console = Console(core, web_root=str(dist))
    server = serve(console, port=0)
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield console, base
    finally:
        server.shutdown()
        server.server_close()


class TestFrontendFiles:
    def test_index_is_the_built_panel(self, web_env):
        code, body = request(web_env[1], "GET", "/")
        assert code == 200
        assert "AVM 调试面板" in body
        assert 'id="root"' in body

    def test_asset_is_served_with_js_content_type(self, web_env):
        with urllib.request.urlopen(web_env[1] + "/assets/panel.js", timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/javascript")
            assert resp.read().decode("utf-8") == "console.log('panel')"

    def test_missing_asset_is_404(self, web_env):
        code, body = request(web_env[1], "GET", "/assets/nope.js")
        assert code == 404
        assert "error" in body

    def test_static_never_escapes_web_root(self, web_env):
        """目录穿越要挡住：/../secret.txt 不能读到 dist 之外的文件。

        用 http.client 直接发路径，避免客户端先做一次路径规范化。
        """
        host, port = web_env[1].removeprefix("http://").split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=5)
        try:
            for path in ("/../secret.txt", "/assets/../../secret.txt"):
                conn.request("GET", path)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8", "replace")
                assert resp.status == 404, path
                assert "不该被读到" not in body, path
        finally:
            conn.close()

    def test_api_still_wins_over_static(self, web_env):
        """静态托管没把 /api/* 抢走。"""
        code, body = request(web_env[1], "GET", "/api/state")
        assert code == 200
        assert body["schedule"]["panel_cid"] == web_env[0].panel.cid

"""关机流程的进程级验收：真起一次 `python -m avm.console` 的 main()，再重启。

正常断电 = OS 软件写 `$MEM.sys.power.state = "off"`，宿主在安全点看到后写快照/持久化
并退出（VM 不写 clean、不代替 guest 决定关机）；`/api/system/shutdown` 只提交 OS 关机
请求（写 `$MEM.sys.power.request`），不再直接停内核；优雅关机卡住时进程保持运行；
进程被硬杀后重启按脏关闭（last_exit=forced）报告。

全程离线：驱动脚本把 Core.run 换成"安全点派发中断 → 断电边沿判定 → 推进回合"的循环，
把模型目录刷新换成空操作，不调真实 API。其它环节（boot / token / HTTP / 写快照 /
写回 / 进程退出）都走真路径。
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from persist_helpers import write_image

REPO_ROOT = Path(__file__).resolve().parents[1]

DRIVER = '''\
import sys
import time
from avm.core import Core

# 不触网：模型目录清单只是参考，本用例不校验模型名
Core.refresh_model_catalog = lambda self, timeout=10.0: {
    "source": "unavailable", "reason": "offline shutdown test"}

def fake_run(self):
    """与 Core.run 的安全点同序：派发中断 → 断电边沿 → 推进回合。"""
    self._pick_next_active()
    self._power_state_seen = self.read_power_state()
    while not self._stop_requested:
        self.interrupts.dispatch()
        if self.power_off_requested():
            return
        if self._active_cid is None:
            self._pick_next_active()
            if self._active_cid is None:
                time.sleep(0.02)
                continue
        self.advance_conversation()

Core.run = fake_run

from avm.console import main
sys.exit(main(sys.argv[1:]))
'''


PY_PARA = {
    "kind": "dict", "ctrl": {"type": "para"},
    "value": {"model": {"kind": "str", "value": "plm.python"}},
}


def _power_image_mem():
    """最小镜像 mem：python init（不调 LLM）+ OS 约定的断电节点。"""
    return {
        "para": {"kind": "dict", "value": {"python": PY_PARA}},
        "model_params": {
            "kind": "dict", "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {"kind": "dict", "value": {"init": {
            "kind": "dict", "ctrl": {"type": "python"},
            "value": {
                "content": {"kind": "str", "value": (
                    "def init():\n"
                    "    yield from memory_write('$MEM.boot_ran', '1')\n"
                )},
                "para_ref": {"kind": "str", "value": "$MEM.para.python"},
            },
        }}},
        "boot_ran": {"kind": "str", "value": ""},
        "sys": {
            "kind": "dict",
            "value": {
                "power": {
                    "kind": "dict",
                    "value": {
                        "request": {"kind": "str", "value": ""},
                        "state": {"kind": "str", "value": "on"},
                    },
                },
            },
        },
    }


def _write_power_image(tmp_path, **extra):
    """最小镜像（python init，不触网）+ 断电节点；落点 out/ 与默认一致。"""
    image_path = write_image(
        tmp_path, mem=_power_image_mem(),
        init={"name": "init", "system_ref": "$MEM.programs.init",
              "user": "yield from init()", "para_ref": "$MEM.para.python"},
    )
    doc = json.loads(Path(image_path).read_text(encoding="utf-8"))
    doc["persist_to"] = "out/mem.json"
    doc.update(extra)
    Path(image_path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return image_path


def _request(port, path, method="GET", body=None, token=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw, code = resp.read(), resp.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    return code, json.loads(raw) if raw else None


def _spawn(driver_path, image_path, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(driver_path), image_path, "--port", "0"],
        cwd=str(cwd), env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    lines = queue.Queue()
    seen = []

    def reader():
        for line in proc.stderr:
            seen.append(line.rstrip())
            lines.put(line)

    threading.Thread(target=reader, daemon=True).start()
    deadline = time.time() + 25
    port = None
    while time.time() < deadline and port is None:
        try:
            line = lines.get(timeout=0.2)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        match = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
        if match:
            port = int(match.group(1))
    if port is None:
        proc.kill()
        proc.wait(timeout=5)
        raise AssertionError("操作面板宿主没有在 25 秒内起来；stderr：\n"
                             + "\n".join(seen[-30:]))
    return proc, port


def _token(tmp_path):
    path = tmp_path / "out" / "console.token"
    deadline = time.time() + 10
    while time.time() < deadline:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        time.sleep(0.05)
    raise AssertionError("没有写出 out/console.token")


def _wait_for(predicate, timeout=10, message="条件没有满足"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError(message)


def _power_action(port, token, value, *, wait=True):
    """模拟 OS 监督程序/引导程序写 `$MEM.sys.power.state`（面板动作走真回合）。

    动作可能在面板回合进行中被排队，而那个回合会以 `end` 收尾——不补中断，动作会
    留在通道里没人跑。同类型中断去重，所以这里在等待期间反复请求（幂等）。
    """
    code, body = _request(port, "/api/action", "POST", {
        "tool": "memory_write",
        "arguments": {"ref": "$MEM.sys.power.state", "content": value},
    }, token)
    assert code == 200 and body["submitted"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            _request(port, "/api/interrupt", "POST", {}, token)
            node = _request(port, "/api/memory?ref=$MEM.sys.power.state",
                            token=token)[1]["node"]
        except (urllib.error.URLError, OSError):
            return  # 进程已经断电退出（写 off 成功）
        if not wait or node == value:
            return
        time.sleep(0.1)
    raise AssertionError(f"power.state 没有变成 {value!r}")


def _wait_exit(proc, port, token, timeout=20):
    """等宿主断电退出；期间反复补面板中断（动作可能因去重留在队列里）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            assert proc.returncode == 0, f"宿主退出码 {proc.returncode}"
            return
        try:
            _request(port, "/api/interrupt", "POST", {}, token)
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.1)
    raise AssertionError("正常断电后宿主没有退出")


def test_os_power_off_writes_snapshot_and_restart_recovers(tmp_path):
    image_path = _write_power_image(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")

    proc, port = _spawn(driver, image_path, tmp_path)
    try:
        token = _token(tmp_path)

        def _running_state():
            code, body = _request(port, "/api/state", token=token)
            return body if code == 200 and body.get("lifecycle") == "running" else None

        _wait_for(_running_state, timeout=30,
                  message="宿主没有进入 running（可能还在启动/重放）")
        code, state = _request(port, "/api/state", token=token)
        assert code == 200
        assert state["recovery"]["source"] == "cold"
        assert state["recovery"]["last_exit"] == "unknown", "首次启动没有退出证据"
        assert state["lifecycle"] == "running"
        # 未认证请求必须被挡（进程级验收顺带覆盖登录这一环）
        assert _request(port, "/api/state")[0] == 401

        # /api/system/shutdown 只提交 OS 关机请求，不直接停内核
        code, body = _request(port, "/api/system/shutdown", "POST", {}, token)
        assert code == 202 and body["status"] == "stopping"
        assert body["entry"] == "$MEM.sys.power.request"
        # 请求落在世界节点上（经保留对话的普通写入）
        _wait_for(
            lambda: _request(port, "/api/memory?ref=$MEM.sys.power.request",
                             token=token)[1]["node"] == "shutdown",
            message="关机请求没有写进 $MEM.sys.power.request",
        )
        # OS 没有收尾（没写 off）：优雅关机卡住时进程保持运行，宿主不自行断电
        time.sleep(0.5)
        assert proc.poll() is None, "优雅关机没收尾时进程不应退出"
        assert _request(port, "/api/state", token=token)[1]["lifecycle"] == "stopping"

        # OS 监督程序收尾：写 power.state = off → 宿主在安全点写快照并退出
        _power_action(port, token, "off", wait=False)
        _wait_exit(proc, port, token)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert (tmp_path / "out" / "run.snapshot.json").is_file(), "正常断电没写重启快照"
    assert (tmp_path / "out" / "mem.json").is_file(), "正常退出没写回内存"
    journal = (tmp_path / "out" / "run.journal").read_text(encoding="utf-8")
    assert "shutdown" in journal, "关机请求要进日志（可重放）"

    # 每次启动都是新 token：先删掉上一轮的，避免 _token 读到旧凭证
    (tmp_path / "out" / "console.token").unlink()
    proc2, port2 = _spawn(driver, image_path, tmp_path)
    try:
        token2 = _token(tmp_path)
        code, state2 = _request(port2, "/api/state", token=token2)
        assert code == 200
        assert state2["recovery"]["source"] == "snapshot"
        assert state2["recovery"]["mode"] == "auto"
        assert state2["recovery"]["last_exit"] == "normal", "正常断电后的重启应读到正常退出"
        assert state2["recovery"]["restored"]["conversations"] >= 1

        # 快照把 off 一起恢复了：VM 不把上电前的存量 off 当成新请求。
        # OS 引导程序负责把断电标记复位，然后再次请求关机。
        _power_action(port2, token2, "on")
        code, body = _request(port2, "/api/system/shutdown", "POST", {}, token2)
        assert code == 202
        _wait_for(
            lambda: _request(port2, "/api/memory?ref=$MEM.sys.power.request",
                             token=token2)[1]["node"] == "shutdown",
            message="重启后的关机请求没有写进节点",
        )
        _power_action(port2, token2, "off", wait=False)
        _wait_exit(proc2, port2, token2)
    finally:
        if proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=5)


def test_hard_kill_is_reported_as_forced_shutdown(tmp_path):
    image_path = _write_power_image(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")

    proc, port = _spawn(driver, image_path, tmp_path)
    try:
        token = _token(tmp_path)
        assert _request(port, "/api/state", token=token)[0] == 200
        # 至少提交一个回合（boot 程序写 running），让日志里有记录可重放。
        _wait_for(
            lambda: (tmp_path / "out" / "run.journal").is_file()
            and (tmp_path / "out" / "run.journal").stat().st_size > 0,
            message="启动回合没有进日志",
        )
    finally:
        proc.kill()  # 硬杀：不写快照、不写 clean（模拟 force-stop/断电）
        proc.wait(timeout=5)

    assert not (tmp_path / "out" / "run.snapshot.json").exists()
    (tmp_path / "out" / "console.token").unlink()
    proc2, port2 = _spawn(driver, image_path, tmp_path)
    try:
        token2 = _token(tmp_path)
        code, state = _request(port2, "/api/state", token=token2)
        assert code == 200
        assert state["recovery"]["last_exit"] == "forced", "脏关闭要报告 forced"
        assert state["recovery"]["source"] == "journal"
        assert state["recovery"]["replayed"] >= 1
    finally:
        proc2.kill()
        proc2.wait(timeout=5)

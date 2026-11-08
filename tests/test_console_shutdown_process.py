"""关机流程的进程级验收：真起一次 `python -m avm.console` 的 main()，再重启。

全程离线：驱动脚本把 Core.run 换成「等到停止标志」的循环，把模型目录刷新换成空操作，
不调真实 API。其它环节（boot / token / HTTP / 安全点返回 / 写快照 / 写回 / 进程退出）
都走真路径。
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
    while not self._stop_requested:
        time.sleep(0.02)

Core.run = fake_run

from avm.console import main
sys.exit(main(sys.argv[1:]))
'''


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


def _shutdown(proc, port, token):
    code, body = _request(port, "/api/system/shutdown", "POST", {}, token)
    assert code == 202 and body["status"] == "stopping"
    assert proc.wait(timeout=20) == 0, "关机没有让宿主干净退出"


def test_shutdown_writes_snapshot_and_restart_recovers(tmp_path):
    image_path = write_image(tmp_path)
    doc = json.loads(Path(image_path).read_text(encoding="utf-8"))
    doc["persist_to"] = "out/mem.json"
    Path(image_path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")

    proc, port = _spawn(driver, image_path, tmp_path)
    try:
        token = _token(tmp_path)
        code, state = _request(port, "/api/state", token=token)
        assert code == 200
        assert state["recovery"]["source"] == "cold"
        assert state["lifecycle"] == "running"
        # 未认证请求必须被挡（进程级验收顺带覆盖登录这一环）
        assert _request(port, "/api/state")[0] == 401
        _shutdown(proc, port, token)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert (tmp_path / "out" / "run.snapshot.json").is_file(), "正常退出没写重启快照"
    assert (tmp_path / "out" / "mem.json").is_file(), "正常退出没写回内存"

    # 每次启动都是新 token：先删掉上一轮的，避免 _token 读到旧凭证
    (tmp_path / "out" / "console.token").unlink()
    proc2, port2 = _spawn(driver, image_path, tmp_path)
    try:
        token2 = _token(tmp_path)
        code, state2 = _request(port2, "/api/state", token=token2)
        assert code == 200
        assert state2["recovery"]["source"] == "snapshot"
        assert state2["recovery"]["restored"]["conversations"] >= 1
        _shutdown(proc2, port2, token2)
    finally:
        if proc2.poll() is None:
            proc2.kill()
            proc2.wait(timeout=5)

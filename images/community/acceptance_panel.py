#!/usr/bin/env python3
"""面板验收脚本：起调试面板后端 → 经 `$inputs` 喂社区命令 → 断言回执与快照。

用法：
    python images/community/acceptance_panel.py            # 默认随机端口
    python images/community/acceptance_panel.py --keep     # 验收完不删快照

它走的就是作者在面板里的路径：
1. `run_console.py` 起后端（`host_flags={"console": True}`）；
2. `POST /api/device-action` 等同于面板右栏的「喂输入」框；
3. `GET /api/memory?ref=$MEM.outputs` 读设备的回执列表；
4. 最后喂 `exit`，等进程退出，检查 `persist_to` 写出的内存快照。

本脚本不修改代码、不依赖浏览器；面板 UI 只是同一批端点的前端。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(HERE, "run_console.py")
SNAPSHOT = os.path.join(HERE, "out", "community_mem.json")

# 每条命令 = (输入文本, 回执里必须出现的片段)
COMMANDS = [
    ("HELP", "OK HELP"),
    ("BOARD design", "OK BOARD design"),
    ("POST P000001", "OK POST P000001"),
    ("@D1 COMMENT P000001 | 评论：为什么只允许一层？", "OK COMMENT"),
    ("@D2 REPLY C000004 | 回复这条评论，应外化成分叉帖。", "已外化成分叉帖"),
    ("@D1 REPLY C000004 | 后续回复进同一个分叉帖。", "已有分叉帖"),
    ("THREAD P000005", "OK THREAD P000005"),
    ("NEW code insight 面板验收帖 | 这是一条经面板输入的新帖。", "OK NEW"),
    ("QUOTE P000001 引用帖 | 引用 P000001 的正文。", "OK QUOTE"),
    ("COMMENT P000005 | 评论分叉帖。", "OK COMMENT"),
    ("TAG P000001 面板验收", "OK TAG"),
    ("SOLVE P000002 P000003 方案已记录", "OK SOLVE"),
    ("MERGE P000001 P000002", "OK MERGE"),
    ("DELETE C000004 验收：删除一条评论", "OK DELETE"),
    ("MAINT", "OK MAINT"),
    ("DELETE P000008 验收：删除引用帖（软删除）", "OK DELETE"),
    ("CHECK", "OK CHECK"),
]

# 注释里的评论 id 依赖前面几步的序号：种子 seq=3；第一条评论分到 C000004，
# 第一次回复开分叉帖 P000005、首条评论 C000006。脚本会先用 DEVICES/位置核对，
# 若 id 与预期不符就打印实际回执并以失败退出。

PASS = 0
FAIL = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [ok] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def http_json(url: str, payload: dict | None = None, timeout: float = 5.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(base: str, process: subprocess.Popen, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            http_json(base + "/api/state", timeout=1.0)
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.1)
    return False


def output_items(base: str) -> tuple[list[str], int]:
    body = http_json(base + "/api/memory?ref=$MEM.outputs")
    preview = str((body.get("node") or {}).get("preview") or "")
    match = re.search(r"list_len=(\d+)", preview)
    count = int(match.group(1)) if match else 0
    items: list[str] = []
    item_match = re.search(r"items=(\[.*\]),metadata=", preview, re.S)
    if item_match:
        try:
            parsed = ast.literal_eval(item_match.group(1))
            if isinstance(parsed, list):
                items = [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            items = []
    return items, count


def find_input_action(base: str) -> int:
    devices = http_json(base + "/api/devices")
    if isinstance(devices, dict):
        devices = devices.get("devices") or []
    for device in devices:
        if not isinstance(device, dict) or device.get("mount") != "inputs":
            continue
        actions = device.get("actions") or []
        for action in actions:
            if "投喂" in str(action.get("name") or ""):
                return int(action.get("index", 0))
        if actions:
            return int(actions[0].get("index", 0))
    raise RuntimeError("找不到 $inputs 的喂输入动作；镜像里 inputs 设备没挂上？")


def feed(base: str, action_index: int, content: str) -> None:
    http_json(
        base + "/api/device-action",
        {"mount": "inputs", "index": action_index, "arguments": {"content": content}},
    )


def wait_reply(base: str, previous_count: int, timeout: float = 20.0) -> list[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        items, count = output_items(base)
        if count > previous_count:
            return items
        time.sleep(0.05)
    raise RuntimeError(f"等待回执超时（outputs 仍在 {previous_count} 条）")


def scan_snapshot() -> dict:
    if not os.path.isfile(SNAPSHOT):
        raise RuntimeError(f"快照不存在：{SNAPSHOT}")
    with open(SNAPSHOT, "r", encoding="utf-8") as f:
        return json.load(f)


def node_value(node):
    return node.get("value") if isinstance(node, dict) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="社区地基的面板验收")
    parser.add_argument("--port", type=int, default=0, help="面板端口（默认随机）")
    parser.add_argument("--keep", action="store_true", help="保留快照文件")
    args = parser.parse_args(argv)

    if os.path.isfile(SNAPSHOT) and not args.keep:
        os.remove(SNAPSHOT)
    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"
    print(f"[acceptance] 启动面板：python images/community/run_console.py --port {port}")
    process = subprocess.Popen(
        [sys.executable, RUNNER, "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not wait_ready(base, process):
            out, err = process.communicate(timeout=5)
            print("[acceptance] 面板没有起来" + (f"\nstdout:\n{out}\nstderr:\n{err}" if out or err else ""))
            return 1
        print("[acceptance] 面板已就绪；开始喂命令\n")
        action_index = find_input_action(base)
        print(f"[acceptance] $inputs 喂输入动作 index={action_index}")

        _, previous = output_items(base)
        for command, expected in COMMANDS:
            feed(base, action_index, command)
            items = wait_reply(base, previous)
            previous = len(items)
            reply = items[-1] if items else ""
            compact = reply.splitlines()[0] if reply else ""
            check(expected in reply, command, f"期望含 {expected!r}，实际：{compact!r}")
            if expected == "OK NEW":
                match = re.search(r"OK NEW (P\d{6})", reply)
                check(match is not None, "NEW 回执带 pid", reply[:120])
            if "已外化成分叉帖" in reply:
                match = re.search(r"分叉帖 (P\d{6})", reply)
                check(match is not None, "REPLY 回执带分叉帖 pid", reply[:120])

        # 退出前直接从面板接口读一次内存树：帖子数与评论数
        forum = http_json(base + "/api/memory?ref=$MEM.forum")
        children = (forum.get("node") or {}).get("children") or {}
        check("posts" in children and "comments" in children, "内存树里有 forum.posts / forum.comments")
        posts = (children.get("posts") or {}).get("children") or {}
        check(len(posts) >= 5, f"帖子数 >=5（实际 {len(posts)}）")

        check(http_json(base + "/api/memory?ref=$MEM.forum.admin.log").get("node") is not None, "审计日志节点存在")

        feed(base, action_index, "exit")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            check(False, "process 在 exit 后退出")
            return 1
        out, err = process.communicate()
        check(process.returncode == 0, "面板进程正常退出", f"returncode={process.returncode} err={err[-300:]}")

        snapshot = scan_snapshot()
        root = node_value(snapshot) if isinstance(snapshot.get("value"), dict) else snapshot
        check(isinstance(root, dict) and "forum" in root, "快照里有 $MEM.forum")
        forum_mem = node_value(root.get("forum")) or {}
        posts_mem = node_value(forum_mem.get("posts")) or {}
        comments_mem = node_value(forum_mem.get("comments")) or {}
        admin_mem = node_value(forum_mem.get("admin")) or {}
        log_list = node_value(admin_mem.get("log")) or []
        forms, statuses = [], []
        for node in posts_mem.values():
            meta_text = node_value(node_value(node).get("meta")) if isinstance(node_value(node), dict) else None
            if not isinstance(meta_text, str):
                continue
            meta = json.loads(meta_text)
            forms.append(meta.get("form"))
            statuses.append(meta.get("status"))
        check("branch" in forms, f"快照里有分叉帖（forms={forms}）")
        check("merged" in statuses, f"快照里有已合并墓碑（statuses={statuses}）")
        check("deleted" in statuses, f"快照里有软删除帖子（statuses={statuses}）")
        check(len(log_list) >= 3, f"审计日志 >=3 条（实际 {len(log_list)}）")
        check(len(comments_mem) >= 2, f"评论 >=2 条（实际 {len(comments_mem)}）")
        check(any("面板验收" in str(node_value(node_value(node).get("meta")) or "") for node in posts_mem.values()),
              "新帖内容进入快照")
        print(f"\n[acceptance] 成功 {PASS} 项，失败 {FAIL} 项；快照：{SNAPSHOT}")
        return 0 if FAIL == 0 else 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())

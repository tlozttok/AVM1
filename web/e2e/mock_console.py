#!/usr/bin/env python3
"""前端 e2e 用的测试侧控制台。

起真后端（`avm.console` 的 `Console` + `serve`），执行器是脚本化的 MockLMU：

- `init` 对话（cid 0）第一轮写一次 `$MEM.state.tick`（产生可观测的内存变化与事件），
  第二轮返回纯文本结束交互 → 系统空闲，面板对话取得执行权（顶栏显示「空闲」）；
- `shell` 对话（名字来自节点数据，用于命令台默认目标）平时休眠，收到
  `send_instruction` 后回一句 `shell 收到：<内容>` 然后结束交互；
- `$MEM.inputs` / `$MEM.outputs` 挂操作面板宿主的 IO 设备（`avm/io_devices.py`），
  输出流由后台线程持续写入，供底部系统日志条断言。

它不在 `avm/` 里改任何东西，也不给 `python -m avm.console` 加标志：这是测试基建。
由 `playwright.config.ts` 的 webServer 拉起，端口由 `--port` 指定。

用法：
    python3 web/e2e/mock_console.py --port 8791
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avm.console import Console, serve  # noqa: E402
from avm.core import Core  # noqa: E402
from avm.io_devices import IOInputDevice, IOOutputDevice  # noqa: E402
from avm.memory_device import StringDevice  # noqa: E402
from avm.types import LinkNode, MetaDict, StrNode  # noqa: E402


def _dormant(core: Core, conversation) -> None:
    """把刚创建的对话从就绪队列移到休眠：启动时只有 init 会跑。"""
    if conversation.cid in core._ready_cids:
        core._ready_cids.remove(conversation.cid)
    if conversation.cid not in core._dormant_cids:
        core._dormant_cids.append(conversation.cid)


class ScriptedLMU:
    """按 cid 记账的假执行器：产出真实的指令、事件与内存变化。"""

    def __init__(self, tick_seconds: float = 0.05):
        self.tick_seconds = tick_seconds
        self.rounds: dict = {}

    def exec(self, conversation, para):
        time.sleep(self.tick_seconds)
        n = self.rounds.get(conversation.cid, 0) + 1
        self.rounds[conversation.cid] = n
        name = conversation.name or f"cid-{conversation.cid}"

        if name == "init":
            if n == 1:
                tool, args = "memory_write", {"ref": "$MEM.state.tick", "content": "tick 1"}
                call_id = f"mock-{conversation.cid}-{n}"
                text = f"init 第 {n} 轮"
                conversation.append_assistant_message(
                    text,
                    tool_calls=[{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
                    }],
                )
                return text, [{"call_id": call_id, "cmd_type": tool, "args": args}], None
            text = "init 完成"
            conversation.append_assistant_message(text)
            return text, [], None

        if name == "worker" and n < 30:
            tool, args = "memory_write", {"ref": "$MEM.state.tick", "content": f"worker {n}"}
            call_id = f"mock-worker-{n}"
            text = f"worker 第 {n} 轮"
            conversation.append_assistant_message(
                text,
                tool_calls=[{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
                }],
            )
            return text, [{"call_id": call_id, "cmd_type": tool, "args": args}], None

        # 与真实执行器一致：先把消费掉的输入写进历史，再写助手回复。
        contents = list(conversation.user_batch.user_contents)
        for content in contents:
            conversation.append_user_message(content)
        last = contents[-1] if contents else ""
        reply = f"{name} 收到：{last}"
        conversation.append_assistant_message(reply)
        return reply, [], None


def build(port: int, host: str, tick: float) -> None:
    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "mock"}, ctrl={"type": "para"})
    core.mem["note"] = MetaDict(data={"v": "hi"}, metadata="一条笔记", ctrl={"owner": "0"})
    core.mem["state"] = MetaDict(data={"tick": ""}, metadata="脚本执行器的计数", ctrl={"owner": "0"})
    core.mem._data["alias"] = LinkNode(target="$MEM.note", metadata="指向 note 的链接")

    # 节点工作台三个维度用的节点：程序节点（ctrl.type=python）、para、handler。
    core.mem["programs"] = MetaDict(
        data={
            "shell": MetaDict(
                data={
                    "content": "def handle(cmd):\n    return 'echo ' + cmd\n",
                    "para_ref": "$MEM.mock_para",
                    "name": "shell",
                },
                ctrl={"type": "python"},
                metadata="shell 程序节点（e2e 夹具）",
            ),
        },
        metadata="程序节点",
    )
    core.mem["mock_para"] = MetaDict(data={"model": "mock"}, ctrl={"type": "para"}, metadata="测试参数")
    core.mem["handlers"] = MetaDict(
        data={
            "guard": MetaDict(
                data={
                    "content": "def handle(event):\n    return {}\n",
                    "condition": StrNode(json.dumps({"events": ["close_conversation"], "phase": ["before"]})),
                },
                ctrl={"type": "handler", "priority": "0"},
                metadata="监听处理器（e2e 夹具）",
            ),
        },
        metadata="监听处理器",
    )

    inputs = IOInputDevice(data=[], metadata="输入")
    core.mem.mount("inputs", inputs)
    inputs.mount_path = "inputs"
    inputs.attach_core(core)
    outputs = IOOutputDevice(data=[], metadata="输出")
    core.mem.mount("outputs", outputs)
    outputs.mount_path = "outputs"
    outputs.attach_core(core)
    core.mem.mount("state.log", StringDevice("最后一行"))

    # 监听器编辑界面要有一条可改的记录；用不会触发的 memory_delete 事件。
    core.events.register(
        owner_cid=0, event_type="memory_delete", target="$MEM.state.tick",
        handler_ref="$MEM.handlers.guard", phase="before",
    )

    def emit_output():
        seq = 0
        while True:
            seq += 1
            outputs.append(f"mock 输出 #{seq}", notify=(seq % 2 == 0))
            time.sleep(0.5)

    threading.Thread(target=emit_output, name="mock-output", daemon=True).start()
    core.lmu = ScriptedLMU(tick)

    init = core.start("你是 init 对话，按脚本行动。", "开始", "$MEM.model_params")
    init.name = "init"

    shell = core.start("你是 shell 服务对话，按脚本回声。", "", "$MEM.model_params")
    shell.name = "shell"
    _dormant(core, shell)

    # 长跑用例：收到指令后连续跑若干轮，够 e2e 点「暂停」并观察操作员接管。
    worker = core.start("你是 worker，按脚本长跑。", "等待指令", "$MEM.model_params")
    worker.name = "worker"
    _dormant(core, worker)

    console = Console(core)
    server = serve(console, host, port)

    def run_core():
        console.running = True
        try:
            core.run()
        finally:
            console.running = False

    threading.Thread(target=run_core, name="mock-core", daemon=True).start()
    print(f"[mock-console] http://{host}:{server.server_port}/ 面板 cid={console.panel.cid}",
          file=sys.stderr, flush=True)
    threading.Event().wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="前端 e2e 用的测试侧控制台")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tick", type=float, default=0.05, help="每轮推理的间隔秒数")
    args = parser.parse_args(argv)
    build(args.port, args.host, args.tick)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""前端 e2e 用的测试侧控制台。

起真后端（`avm.console` 的 `Console` + `serve`），执行器是脚本化的 MockLMU：

- init 对话（cid 0）每轮写一次 `$MEM.state.tick`，产出可观测的内存变化与事件；
- 面板输入队列里有货时改读 `$MEM.inputs.next`，把喂进来的那一段取走；
- `$MEM.inputs` / `$MEM.outputs` 直接挂操作面板宿主的 IO 设备（`avm/io_devices.py`），
  走宿主侧总线；标准终端设备（`avm/memory_device.py`）在这个宿主里完全不参与。

它不在 `avm/` 里改任何东西，也不给 `python -m avm.console` 加标志：这是测试基建。
由 `playwright.config.ts` 的 webServer 拉起，端口由 `--port` 指定。

用法：
    python3 web/e2e/mock_console.py --port 8791 [--tick 0.5]
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
from avm.types import LinkNode, MetaDict, MetaList  # noqa: E402


class ScriptedLMU:
    """按 cid 记账的假执行器：产出真实的指令、事件与内存变化。"""

    def __init__(self, inputs: IOInputDevice, tick_seconds: float = 0.5):
        self.inputs = inputs
        self.tick_seconds = tick_seconds
        self.rounds: dict = {}

    def _queued(self) -> int:
        for action in self.inputs.debug_actions() or []:
            if action.get("name") == "查询队列":
                return int(action["run"]({})["queued"])
        return 0

    def exec(self, conversation, para):
        time.sleep(self.tick_seconds)
        n = self.rounds.get(conversation.cid, 0) + 1
        self.rounds[conversation.cid] = n

        if self._queued() > 0:
            # 阻塞读取的命名接口：队列里已有货，这一读立即返回并把内容追加进列表
            tool, args = "memory_read", {"ref": "$MEM.inputs.next"}
        else:
            tool, args = "memory_write", {"ref": "$MEM.state.tick", "content": f"tick {n}"}

        call_id = f"mock-{conversation.cid}-{n}"
        text = f"第 {n} 轮 {tool}"
        conversation.append_assistant_message(
            text,
            tool_calls=[{
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
            }],
        )
        return text, [{"call_id": call_id, "cmd_type": tool, "args": args}], None


def build(port: int, host: str, tick: float) -> None:
    core = Core()
    # 参数节点：Core 取 para 时不再回退默认值、不写 stderr
    core.mem["model_params"] = MetaDict(data={"model": "mock"}, ctrl={"type": "para"})
    core.mem["note"] = MetaDict(data={"v": "hi"}, metadata="一条笔记", ctrl={"owner": "0"})
    # 脚本执行器写这里：$MEM.note 的元数据留给 e2e 断言，别被 tick 覆盖
    core.mem["state"] = MetaDict(data={"tick": ""}, metadata="脚本执行器的计数", ctrl={"owner": "0"})
    # 正式输入落点：面板输入走 /api/input 时保留对话往这里追加条目（可重放）。
    core.mem["sys"] = MetaDict(
        data={"inbox": MetaList(data=[])}, metadata="系统分区（e2e：正式输入落点）")
    # 内存树链接节点（非挂载）：左栏要画成一行 link，读它进入 $MEM.note
    core.mem._data["alias"] = LinkNode(target="$MEM.note", metadata="指向 note 的链接")

    # 操作面板宿主的两台 IO 设备：Console 发现已经是这两台，只登记引用不再替换。
    inputs = IOInputDevice(data=[], metadata="输入")
    core.mem.mount("inputs", inputs)
    inputs.mount_path = "inputs"
    inputs.attach_core(core)
    outputs = IOOutputDevice(data=[], metadata="输出")
    core.mem.mount("outputs", outputs)
    outputs.mount_path = "outputs"
    outputs.attach_core(core)

    # 嵌套挂载：左栏那行的位置由树给出（$MEM.state 展开后是 log 一行），不靠父段算法另注入
    core.mem.mount("state.log", StringDevice("最后一行"))

    # 监听器编辑界面要有一条可改的记录；用不会触发的 memory_delete 事件，
    # 避免拦截 ScriptedLMU 每轮的 memory_write。
    core.mem["handler"] = MetaDict(
        data={"content": "def handle(event):\n    return {}\n"},
        ctrl={"type": "handler"},
    )
    core.events.register(
        owner_cid=0, event_type="memory_delete", target="$MEM.state.tick",
        handler_ref="$MEM.handler", phase="before",
    )

    # 独立线程造输出：不改 ScriptedLMU 的指令序列，既有事件流断言不受影响。
    def emit_output():
        seq = 0
        while True:
            seq += 1
            outputs.append(f"mock 输出 #{seq}", notify=(seq % 2 == 0))
            time.sleep(0.5)

    threading.Thread(target=emit_output, name="mock-output", daemon=True).start()
    core.lmu = ScriptedLMU(inputs, tick)

    init = core.start("你是 init 对话，按脚本行动。", "开始", "$MEM.model_params")
    init.name = "init"

    # 单客户端准入是真实宿主策略；e2e 共用一个服务被多个 test 的页面轮流打开，
    # 所以这里保持默认（不启用）。
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
    parser.add_argument("--tick", type=float, default=0.5, help="每轮推理的间隔秒数")
    args = parser.parse_args(argv)
    build(args.port, args.host, args.tick)
    return 0


if __name__ == "__main__":
    sys.exit(main())

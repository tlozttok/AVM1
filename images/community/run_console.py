#!/usr/bin/env python3
"""调试面板启动器：加载社区镜像并起 `avm.console` 的控制台。

用法：
    python images/community/run_console.py                 # 默认 127.0.0.1:8700
    python images/community/run_console.py --port 8799

`plm.community` 由镜像的 `plms` 段自行声明，这里不再注册；等价的标准入口是
`python -m avm.console images/community/community.json`，本脚本只多一个默认端口。

打开面板后：
1. 右栏 `$inputs` 的喂输入框里输入社区命令（一次可多行：第一行命令，其余行正文）；
2. 回执写进 `$MEM.outputs`：终端 stdout 会打印，面板的内存索引里也能看该设备；
3. 内存索引里可以逐节点查看 `$MEM.forum`（板块、帖子、评论、审计）。

本启动器不调用 `start_core`：本阶段镜像只有 Python 对话，没有 LLM 调用，
不需要拉上游模型清单，也不需要 API 失败策略。
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMAGE_PATH = os.path.join(HERE, "community.json")

sys.path.insert(0, ROOT)
from avm.core import Core  # noqa: E402
from avm.image import load_image  # noqa: E402
from avm.console import Console, serve  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 讨论社区调试面板")
    parser.add_argument("--port", type=int, default=8700, help="监听端口（默认 8700）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认只绑回环）")
    args = parser.parse_args(argv)

    if not os.path.isfile(IMAGE_PATH):
        print(f"[community] 镜像不存在：{IMAGE_PATH}；先运行 python images/community/build_image.py", file=sys.stderr)
        return 1

    core = load_image(Core(), IMAGE_PATH, host_flags={"console": True})
    console = Console(core)
    server = serve(console, args.host, args.port)
    print(f"[community] 面板 http://{args.host}:{server.server_port}/", file=sys.stderr)
    print("[community] 在右栏 $inputs 喂输入框里输入社区命令（HELP 看用法）", file=sys.stderr)

    console.running = True
    status = 0
    try:
        core.run()
    except KeyboardInterrupt:
        print("[community] 收到中断，写回后退出", file=sys.stderr)
        status = 130
    finally:
        console.running = False
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[community] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[community] 写回失败: {e}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())

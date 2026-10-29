#!/usr/bin/env python3
"""终端启动器：加载社区镜像并运行。

用法：
    python images/community/run_community.py
    printf 'BOARD design\\nNEW code insight 标题 | 正文\\nMAINT\\nexit\\n' | python images/community/run_community.py

`plm.community` 由镜像的 `plms` 段自行声明，这里不再注册；等价的标准入口是
`python -m avm.image images/community/community.json`。

`$inputs` 在终端模式下逐行读 stdin；面板 shell 的每条输入是一条命令，正文可用
" | " 接在同一行。退出时按镜像 `persist_to` 写回 `images/community/out/community_mem.json`。
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMAGE_PATH = os.path.join(HERE, "community.json")

sys.path.insert(0, ROOT)
from avm.boot import boot  # noqa: E402


def main(argv=None) -> int:
    if not os.path.isfile(IMAGE_PATH):
        print(f"[community] 镜像不存在：{IMAGE_PATH}；先运行 python images/community/build_image.py", file=sys.stderr)
        return 1
    # 本阶段镜像只有 Python 对话，不需要拉上游模型清单，也就没有 API 失败策略
    core = boot(IMAGE_PATH, refresh_models=False, api_error_policy=lambda *_a: "exit")
    try:
        core.run()
    except KeyboardInterrupt:
        print("[community] 收到中断，写回后退出", file=sys.stderr)
        return 130
    finally:
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[community] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[community] 写回失败: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

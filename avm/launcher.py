"""AVM 启动器：把已装载的镜像跑起来，并负责收尾（快照 / 写回 / 退出码）。

这个模块只做「宿主侧的生命周期」，不解析镜像内容。拆出来的原因是依赖方向：

    image（解析镜像）  ←  boot（决定用哪种模式启动）
                       ←  launcher（CLI 参数 → 启动 → 收尾）

`boot` 依赖 `image.load_image`，所以 `image` 不能反过来依赖 `boot`。
原先这段 CLI 代码住在 image.py 里，只能靠函数内 import 绕开成环；
搬到这里之后 image.py 的装载职责不再牵扯启动逻辑。

`python -m avm.image` 依然可用：avm/image.py 的 main() 转调本模块。
"""

import argparse
import json
import os
import sys

from .boot import (
    add_core_dump_args, add_mode_arg, add_restore_arg, boot, dump_on_death,
    write_snapshot,
)
from .exceptions import ImageError, RunnerStop, VMMemoryError
from .journal import JournalError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 系统镜像启动器")
    parser.add_argument("image", help="系统镜像 JSON 文件路径")
    parser.add_argument("--checkpoint-in", metavar="PATH",
                        help="调试用：启动时加载单对话检查点（跳过 init，恢复为就绪对话）")
    add_core_dump_args(parser)
    add_restore_arg(parser)
    add_mode_arg(parser)
    args = parser.parse_args(argv)
    try:
        core = boot(
            args.image, mode=args.mode,
            checkpoint_in=args.checkpoint_in, restore=args.restore,
        )
    except (ImageError, JournalError, VMMemoryError, OSError, json.JSONDecodeError) as e:
        print(f"[AVM] 镜像错误: {e}", file=sys.stderr)
        return 1
    if args.no_core_dump:
        core.core_dump_enabled = False
    status = 0
    snapshot_ok = True  # 只有安全点退出才写重启快照
    try:
        core.run()
    except RunnerStop as e:
        print(f"[AVM] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[AVM] 收到中断，写回后退出", file=sys.stderr)
        status = 130
        snapshot_ok = False
    except BaseException as e:
        # 内核死亡：先落现场快照，再保留原始 traceback 往上抛
        dump_on_death(core, e, args.core_dump)
        snapshot_ok = False
        raise
    finally:
        if snapshot_ok:
            write_snapshot(core)
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[AVM] 内存已写回 {persist_to}")
            except OSError as e:
                print(f"[AVM] 写回失败: {e}", file=sys.stderr)
    return status

#!/usr/bin/env python3
"""plm_ledger 宿主启动器。

镜像格式没有 PLM 声明段；本启动器先把镜像外的 LedgerPLM 注册进
PLM_REGISTRY，再按标准流程加载并运行系统镜像。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for path in (HERE, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from avm.exceptions import VMMemoryError
from avm.image import ImageError, load_image
from avm.python_server import PLM_REGISTRY
from plm_ledger_server import LedgerPLM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="运行 plm_ledger 系统镜像（注册镜像外 LedgerPLM）")
    parser.add_argument("image", nargs="?", default=os.path.join(HERE, "plm_ledger.json"),
                        help="镜像 JSON 路径（默认同目录 plm_ledger.json）")
    args = parser.parse_args(argv)

    PLM_REGISTRY["plm.ledger"] = LedgerPLM

    try:
        core = load_image(args.image)
    except (ImageError, VMMemoryError, OSError, ValueError) as e:
        print(f"[plm_ledger] 镜像错误: {e}", file=sys.stderr)
        return 1

    try:
        core.run()
    finally:
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[plm_ledger] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[plm_ledger] 写回失败: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

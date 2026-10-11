#!/usr/bin/env python3
"""启动调试控制台，并给这个进程一份清洗过的代理环境。

本机（以及不少装了代理工具的机器）会在 `NO_PROXY` / `no_proxy` 里写**已经带方括号**的
IPv6 项，例如 `[::1]`。httpx 把这类项变成 `all://*[::1]` 之后解析不了，于是**任何**
httpx 客户端都建不起来——`OpenAI()` 在建客户端时就抛 `InvalidURL: Invalid port: ':1]'`，
栈里完全看不出与代理有关。

这个脚本只改**自己这份**环境变量（父进程、`.env`、系统设置都不动），然后照常进
`python -m avm.console`：

    python scripts/run_console.py images/demo.json

其余参数原样传给控制台。清洗规则：丢掉形如 `[...]` 的项（同一串里不带方括号的 `::1`
通常已经覆盖它）；清洗后先试着建一个 httpx 客户端，建不起来就在启动前把原因说清楚，
而不是等第一轮模型调用时才崩。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # 从任意目录调用都能 import avm

PROXY_VARS = ("NO_PROXY", "no_proxy")


def clean_no_proxy(value: str) -> tuple[str, list[str]]:
    """丢掉已经带方括号的项。返回（清洗后的串，被丢掉的项）。"""
    kept, dropped = [], []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("[") and item.endswith("]"):
            dropped.append(item)
        else:
            kept.append(item)
    return ",".join(kept), dropped


def clean_environment(environ=None) -> list[str]:
    """就地把当前进程的 NO_PROXY / no_proxy 洗干净；返回被丢掉的项（去重）。"""
    env = os.environ if environ is None else environ
    dropped: list[str] = []
    for name in PROXY_VARS:
        raw = env.get(name)
        if not raw:
            continue
        cleaned, gone = clean_no_proxy(raw)
        if gone:
            env[name] = cleaned
            dropped.extend(item for item in gone if item not in dropped)
    return dropped


def client_buildable() -> str | None:
    """试建一个 httpx 客户端；建不起来就返回原因。不联网。"""
    try:
        import httpx

        httpx.Client().close()
    except Exception as e:  # httpx 的异常类型随版本变，这里只要求把原因带出去
        return f"{type(e).__name__}: {e}"
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    dropped = clean_environment()
    if dropped:
        print(f"[run_console] NO_PROXY/no_proxy 里去掉 httpx 解析不了的项：{', '.join(dropped)}",
              file=sys.stderr)

    reason = client_buildable()
    if reason is not None:
        print(f"[run_console] 建不了 httpx 客户端：{reason}", file=sys.stderr)
        print("[run_console] 代理环境变量里还有它解析不了的项，先看 NO_PROXY / no_proxy",
              file=sys.stderr)
        return 1

    from avm.console import main as console_main

    return console_main(argv)


if __name__ == "__main__":
    sys.exit(main())

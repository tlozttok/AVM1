"""运行者提示：内核停在需要人做决定的地方时，从终端收一个选择。

这是**显式安装**的宿主覆盖：把 `prompt_api_error` 传进 `start_core(api_error_policy=...)`
或直接赋给 `Core.api_error_policy`。缺省不安装——内核走内置策略（余额不足暂停 /
上下文超限跳过对话 / 其余重试并暂停）。内核不认识终端，只认识回调的返回值
（retry / reload / exit）。

EOF、非 tty、Ctrl+C 一律按退出处理：停下等一个不会来的选择，比退出更糟。无人值守
策略（等多久、要不要重试）另行决定，不在这里。
"""

import sys


_PROMPT = """[AVM] API 调用失败（第 {attempt} 次）：{error}
  1) 更新环境：改好 .env / 环境变量后选它，内核重建客户端并重试本轮
  2) 继续：不改环境，直接重试本轮
  3) 退出：写回内存后结束
选择 [1/2/3]："""

_CHOICES = {"1": "reload", "2": "retry", "3": "exit"}


def prompt_api_error(error, attempt: int = 1) -> str:
    """终端三选项。返回 retry / reload / exit。"""
    stdin = sys.stdin
    is_tty = bool(getattr(stdin, "isatty", lambda: False)())
    if stdin is None or not is_tty:
        print(f"[AVM] API 调用失败，输入不是终端，按退出处理：{error}", file=sys.stderr)
        return "exit"
    while True:
        try:
            line = input(_PROMPT.format(attempt=attempt, error=error)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[AVM] 按退出处理", file=sys.stderr)
            return "exit"
        if line in _CHOICES:
            return _CHOICES[line]
        print("[AVM] 请输入 1、2 或 3", file=sys.stderr)

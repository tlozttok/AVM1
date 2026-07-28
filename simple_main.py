"""简单 main：单对话 + 设备 demo（时间/随机/休眠）"""

import json
import logging
import os
import sys
from avm.memory import Memory
from avm.memory_device import InputsListDevice, OutputsListDevice
from avm.types import MetaDict, MetaList
from avm.core import Core
import readline

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    class _ColoredFormatter(logging.Formatter):
        _COLORS = {"DEBUG": "\033[90m", "INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
        _RESET = "\033[0m"

        def format(self, record):
            lvl = record.levelname
            color = self._COLORS.get(lvl, "")
            record.levelname = f"{color}{lvl:8}{self._RESET}"
            record.name = f"\033[36m{record.name}{self._RESET}"
            return super().format(record)

    console_fmt = _ColoredFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(console_fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)
        root.addHandler(fh)


SYSTEM_PROMPT = """
你是一个 AI 助手，运行在 AVM（Agent 虚拟机）中。你拥有一块共享内存，可以进行读、写、创建新内存地址等操作。

# 内存结构
- 内存以路径形式访问，如 $MEM.key.subkey
- 字典类型返回值包含 keys 列表和元数据
- 列表类型返回值包含长度和元数据
- 字符串类型直接返回内容
- 内存地址可以用 memory_make 指令创建，类型有 str、dict、list

# 特殊设备（挂载在非 MEM 路径下）
- $time         读取当前时间
- $random       读取一个 0 到 1 之间的随机数
- $sleep        向其写入数字（秒）触发休眠（memory_write $sleep 3），休眠后不返回内容，如需确认，请在休眠前后各读一次 $time
- 工具调用按顺序执行，可以依赖前一个调用的结果
- $inputs       用户输入列表，$inputs.-1 读取最新输入（会等待用户输入）
- $outputs      用户输出列表，写入 $outputs.-1 会将内容显示给用户

# 你的任务
你是一个简洁友好的助手。收到用户消息后认真处理，用 memory_write 给 $outputs 追加输出。

# 重要约束
- 不允许创建子对话或亚对话
"""


def main():
    setup_logging("DEBUG")

    print("=" * 50)
    print("AVM - 简易助手")
    print("=" * 50)

    core = Core()

    mem = core.mem

    # 模型参数
    mem["model_params"] = MetaDict(data={
        "model": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
        "use_tool": "auto",
    }, metadata="模型调用参数")

    # 挂载基本 IO 设备
    mem.mount("inputs", InputsListDevice(data=[], metadata="用户输入列表"))
    mem.mount("outputs", OutputsListDevice(data=[], metadata="对用户的输出列表"))

    # 挂载时间设备
    import time as _time
    from avm.memory_device import MemoryDevice

    class _TimeDevice(MemoryDevice):
        def to_llm_string(self) -> str:
            return _time.strftime("%Y-%m-%d %H:%M:%S")

        def resolve_path(self, path: list):
            if not path:
                return _time.strftime("%Y-%m-%d %H:%M:%S")
            from avm.exceptions import VMMemoryError
            raise VMMemoryError("时间设备不支持子路径")

    mem.mount("time", _TimeDevice())

    # 挂载随机数设备
    import random as _random

    class _RandomDevice(MemoryDevice):
        def to_llm_string(self) -> str:
            return str(_random.random())

        def resolve_path(self, path: list):
            if not path:
                return str(_random.random())
            from avm.exceptions import VMMemoryError
            raise VMMemoryError("随机设备不支持子路径")

    mem.mount("random", _RandomDevice())

    # 挂载休眠设备
    class _SleepDevice(MemoryDevice):
        def to_llm_string(self) -> str:
            return "休眠设备（写入数字触发休眠）"

        def set_value(self, value):
            try:
                secs = float(str(value))
            except ValueError:
                from avm.exceptions import VMMemoryError
                raise VMMemoryError(f"休眠时间必须是数字，got {value!r}")
            _time.sleep(secs)
            logger.info("[SleepDevice] slept %.2fs", secs)

    mem.mount("sleep", _SleepDevice())

    print("[系统] 设备挂载完成：inputs, outputs, time, random, sleep")
    print("[系统] 输入任意消息开始对话，输入 /exit 退出")

    # 启动根对话
    first_input = input("\n[启动] 请说点什么吧: ").strip()
    if not first_input:
        first_input = "你好"

    core.start(SYSTEM_PROMPT, first_input)
    core.run()

    # 主循环：对话结束后重新启动（清洗上下文）
    while True:
        try:
            text = input("\n[输入] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 再见")
            break
        if text == "/exit":
            print("[系统] 再见")
            break
        if not text:
            continue

        core.start(SYSTEM_PROMPT, text)
        core.run()


if __name__ == "__main__":
    main()

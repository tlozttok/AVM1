"""AVM 主程序入口"""
import json
import logging
import os
import sys
from memory import Memory
from memory_device import InputsListDevice, OutputsListDevice
from avm_types import MetaDict, MetaList
from AVM import Core

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "log_level": "INFO",
    "log_file": None,
    "debug": False,
    "prompt": "program/deepseek_simplified.json",
}


def load_config(path: str = "config.json") -> dict:
    """加载配置文件，不存在则返回默认值"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # 用默认值补全缺失的键
        for key, value in DEFAULT_CONFIG.items():
            config.setdefault(key, value)
        return config
    except FileNotFoundError:
        return DEFAULT_CONFIG.copy()


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """配置日志系统"""
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _json_to_meta(value):
    """递归将 JSON 值转换为 MetaDict/MetaList/str
    - dict: 键 "meta" 的值提取为元数据，其余键递归转换后作为数据
    - list: 元素递归转换后作为 MetaList 数据
    - 其他: 原样返回
    """
    if isinstance(value, dict):
        metadata = value.pop("meta", None) if "meta" in value else None
        data = {k: _json_to_meta(v) for k, v in value.items()}
        return MetaDict(data=data, metadata=metadata)
    elif isinstance(value, list):
        data = [_json_to_meta(v) for v in value]
        return MetaList(data=data)
    else:
        return value


def load_json_to_memory(mem: Memory, path: str) -> None:
    """将 JSON 文件加载到 MEM 根层级"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        mem[key] = _json_to_meta(value)


def init_memory(mem: Memory, prompt_path: str) -> None:
    """初始化内存"""
    logger.info("[init_memory] starting...")

    # 1. 加载 JSON 提示词到 MEM
    load_json_to_memory(mem, prompt_path)

    # 2. 创建模型调用参数
    model_params = MetaDict(data={
        "model": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
        "use_tool": "auto"
    },metadata="默认模型调用参数,最好就用这个")
    mem["model_params"] = model_params

    # 3. 挂载伪列表输入/输出设备
    inputs_device = InputsListDevice(data=[], metadata="用户输入列表，使用 $MEM.inputs.-1 读取最后一条（会自动要求用户输入新内容），也可以读取历史消息")
    outputs_device = OutputsListDevice(data=[], metadata="对用户的输出列表，使用 $MEM.outputs.-1 追加写入（会自动输出给用户），也可以读取历史消息")

    mem.mount("inputs", inputs_device)
    mem.mount("outputs", outputs_device)

    logger.info("[init_memory] done")
    print("[系统] 内存初始化完成")
    print("[系统] 已挂载伪列表设备：inputs, outputs")
    print(f"[系统] 已设置参数：model_params")


def main():
    """主函数"""
    config = load_config()
    setup_logging(config["log_level"], config.get("log_file"))

    print("=" * 50)
    print("AVM - Agent 虚拟机")
    print("=" * 50)

    # 创建 Core 实例
    core = Core()
    core.debug = config.get("debug", False)

    # 内存实时监控（通过环境变量 AVM_MEMDUMP 指定输出文件路径）
    mem_dump_file = os.environ.get("AVM_MEMDUMP")
    mem_sock_path = os.environ.get("AVM_MEMSOCK")
    if mem_dump_file:
        core.start_memory_monitor(mem_dump_file, socket_path=mem_sock_path)

    # 初始化内存
    init_memory(core.mem, prompt_path=config["prompt"])

    # 设置初始命令栈
    core.command_stack.append("create 0 -1 $MEM.system $MEM.user $MEM.model_params")
    core.run()


if __name__ == "__main__":
    main()

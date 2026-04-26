"""AVM 主程序入口"""
import json
import logging
from memory import Memory
from memory_device import InputsListDevice, OutputsListDevice
from avm_types import MetaDict
from AVM import Core

logger = logging.getLogger(__name__)


def load_prompts(path: str = "program/deepseek_simplified.json"):
    """从 JSON 文件加载系统提示词和用户提示词"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("system", ""), data.get("user", "")


def init_memory(mem: Memory) -> None:
    """
    初始化内存
    - 加载提示词
    - 创建模型调用参数
    - 挂载伪列表输入/输出设备
    """
    logger.info("[init_memory] starting...")

    # 1. 加载提示词
    system_prompt, user_prompt = load_prompts()
    mem["system_prompt"] = system_prompt
    mem["user_prompt"] = user_prompt

    # 2. 创建模型调用参数
    model_params = MetaDict(data={
        "model": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}},
        "use_tool": "auto"
    })
    mem["model_params"] = model_params

    # 3. 挂载伪列表输入/输出设备
    inputs_device = InputsListDevice(data=[], metadata="用户输入列表，使用 $MEM.inputs.-1 读取最后一条")
    outputs_device = OutputsListDevice(data=[], metadata="用户输出列表，使用 $MEM.outputs.-1 追加写入")

    mem.mount("inputs", inputs_device)
    mem.mount("outputs", outputs_device)

    logger.info("[init_memory] done")
    print("[系统] 内存初始化完成")
    print("[系统] 已挂载伪列表设备：inputs, outputs")
    print(f"[系统] 已设置参数：model_params")


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    print("=" * 50)
    print("AVM - Agent 虚拟机")
    print("=" * 50)

    # 创建 Core 实例
    core = Core()

    # 初始化内存
    init_memory(core.mem)

    # 设置初始命令栈
    core.command_stack.append("create 0 -1 $MEM.system_prompt $MEM.user_prompt $MEM.model_params")
    core.run()


if __name__ == "__main__":
    main()

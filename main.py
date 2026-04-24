"""AVM 主程序入口"""
import threading
import logging
from memory import Memory
from memory_device import StringDevice
from avm_types import MetaDict
from AVM import Core

logger = logging.getLogger(__name__)
class UserInputDevice(StringDevice):
    """
    用户输入设备
    假装是字符串，读取时会阻塞并向界面发送 input 请求
    """

    def __init__(self):
        super().__init__(value="")
        self._lock = threading.Lock()
        self._pending_input = False
        self._input_value = ""

    def get_value(self) -> str:
        """获取输入值（阻塞式）"""
        with self._lock:
            self._pending_input = True

        # 模拟阻塞等待用户输入
        print("\n[用户输入请求] 请输入：", end="", flush=True)
        try:
            user_input = input()
        except EOFError:
            user_input = ""

        with self._lock:
            self._input_value = user_input
            self._pending_input = False
            self._value = user_input

        logger.info("[UserInputDevice] received input: %r", user_input)
        return user_input

    def to_llm_string(self) -> str:
        """返回给 LLM 的字符串表示"""
        if self._pending_input:
            return "[等待用户输入...]"
        return self.get_value()

    def is_pending(self) -> bool:
        """检查是否有待处理的输入"""
        return self._pending_input


class UserOutputDevice(StringDevice):
    """
    用户输出设备
    假装是字符串，写入时会输出到屏幕
    """

    def __init__(self):
        super().__init__(value="")

    def set_value(self, value: str) -> None:
        """设置输出值（同时输出到屏幕）"""
        self._value = value
        print(f"\n[Agent 输出] {value}")

    def to_llm_string(self) -> str:
        """返回给 LLM 的字符串表示"""
        return self._value if self._value else "[空]"


def init_memory(mem: Memory) -> None:
    """
    初始化内存
    - 创建模型调用参数（字典格式）
    - 创建初始提示词
    - 挂载用户输入/输出设备
    """
    logger.info("[init_memory] starting...")
    # 1. 创建模型调用参数
    model_params = MetaDict(data={
        "model": "deepseek-v4-flash",
        "extra_body": {"thinking": {"type": "disabled"}}
    })
    mem["model_params"] = model_params

    # 2. 创建初始提示词
    # 这是一个让 LLM 循环读写用户输入/输出设备的提示词
    system_prompt = """你正在一个系统中。该系统的核心是一个递归子调用执行器和一个共享json式带元数据的复合数据结构。递归子调用执行器是这样的：对于一个LLM对话C，从系统提示词Ps和用户初始提示词Pu开始，进入LLM使用工具调用操作系统-系统返回的循环，此时LLM使用工具调用（TC）可以启动另一个对话C'，并且使C'的非工具调用的输出结果成为TC的返回值（启动C'后执行器会执行C'），C的非工具调用的输出结果也会成为父对话'C的工具调用返回值；直到C的LLM返回没有工具调用的一个结果（这个结果同样会存到'C的消息中），C结束，执行器回到父对话'C。共享json式带元数据的复合数据结构是指，这是一个对所有对话统一可见的数据（内存），由带元数据的字典、带元数据的列表、具体字符串等组成，类似json。路径这样表示：$MEM.path.to.key（列表的key是数字，支持-1这样的索引），这个路径指向字典和列表时，会返回元数据，这样你就能进一步访问。元数据可以修改，用于清楚指明该数据是什么，但是在没有元数据的情况下，字典默认返回键列表，列表默认返回其长度。关于数据的更多描述、分析和思考可以放到一般的内存里。访问字符串时候，就是直接返回内容。该数据结构有一些“伪字符串”，类似linux的伪文件。你可以自己探索一切。不过在这个设置中，系统的重点是读取用户输入和输出到用户。用户输入是一个伪字符串，路径为$MEM.inputs，给用户的输出也是一个伪字符串（可写入），路径为$MEM.outputs。目前你的任务是对话，从$MEM.inputs读取数据，往$MEM.outputs写入数据，这是一个简单的模拟。
"""

    mem["system_prompt"] = system_prompt

    # 3. 系统提示词
    user_prompt = """现在开始循环。"""
    mem["user_prompt"] = user_prompt

    # 4. 用户输入内容模板
    mem["user_input_template"] = "用户输入：$user_input"

    # 5. 挂载用户输入/输出设备
    user_input_device = UserInputDevice()
    user_output_device = UserOutputDevice()

    mem.mount("inputs", user_input_device)
    mem.mount("outputs", user_output_device)

    logger.info("[init_memory] done")
    print("[系统] 内存初始化完成")
    print(f"[系统] 已挂载设备：user_input, user_output")
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

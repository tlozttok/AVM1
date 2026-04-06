"""AVM 主程序入口"""
import threading
from memory import Memory
from memory_device import StringDevice
from avm_types import MetaDict
from AVM import Core


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

        return user_input

    def to_llm_string(self) -> str:
        """返回给 LLM 的字符串表示"""
        if self._pending_input:
            return "[等待用户输入...]"
        return self._value

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
    # 1. 创建模型调用参数
    model_params = MetaDict(data={
        "model": "deepseek-chat",
        "use_tool": True
    })
    mem["model_params"] = model_params

    # 2. 创建初始提示词
    # 这是一个让 LLM 循环读写用户输入/输出设备的提示词
    agent_prompt = """你是一个智能助手。请通过读取用户输入并生成回复来与用户对话。

你可以使用以下命令：
- read $MEM.user_input：读取用户输入
- write $MEM.user_output <你的回复>：向用户输出

请循环执行以下步骤：
1. 读取 $MEM.user_input 获取用户输入
2. 思考并生成回复
3. 将回复写入 $MEM.user_output

现在开始对话。"""

    mem["agent_prompt"] = agent_prompt

    # 3. 系统提示词
    system_prompt = """你是一个有帮助的 AI 助手。请友好、准确地回答用户的问题。"""
    mem["system_prompt"] = system_prompt

    # 4. 用户输入内容模板
    mem["user_input_template"] = "用户输入：$user_input"

    # 5. 挂载用户输入/输出设备
    user_input_device = UserInputDevice()
    user_output_device = UserOutputDevice()

    mem.mount("user_input", user_input_device)
    mem.mount("user_output", user_output_device)

    print("[系统] 内存初始化完成")
    print(f"[系统] 已挂载设备：user_input, user_output")
    print(f"[系统] 已设置参数：model_params")


def main():
    """主函数"""
    print("=" * 50)
    print("AVM - Agent 虚拟机")
    print("=" * 50)

    # 创建 Core 实例
    core = Core()

    # 初始化内存
    init_memory(core.mem)

    # 设置初始命令栈
    core.command_stack.append("create 0 -1 $MEM.system_prompt $MEM.agent_prompt $MEM.model_params ")
    core.run()


if __name__ == "__main__":
    main()

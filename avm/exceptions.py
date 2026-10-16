"""AVM 异常类定义"""


class VMError(Exception):
    """所有虚拟机异常的基类"""
    pass


class VMSyntaxError(VMError):
    """语法错误：未知指令、参数数量错误等"""
    pass


class VMMemoryError(VMError):
    """内存访问基础错误"""
    pass


class MemoryKeyNotFoundError(VMMemoryError):
    """键不存在 — LLM 可以用 memory_make 创建"""
    pass


class MemoryIndexOutOfRangeError(VMMemoryError):
    """列表索引越界 — LLM 可以换索引重试"""
    pass


class MemoryTypeError(VMMemoryError):
    """
    类型不匹配 — 在字符串下找子节点、对列表用非数字索引等。
    此路不通，LLM 应换条路。
    将来可细分为：MemoryStringSubAccessError, MemoryListNonIntIndexError 等。
    """
    pass


class MemoryCircularReferenceError(VMMemoryError):
    """符号链接成环"""
    pass


class VMResourceError(VMError):
    """资源错误：栈溢出、LLM 调用超时等"""
    pass


class APIError(VMError):
    """外部 API 调用失败：网络、鉴权、超时、协议错误。

    与其它异常分开的理由：它要停下来问运行者怎么办（更新环境 / 继续 / 退出），
    不是内核缺陷。
    """
    pass


class RunnerStop(VMError):
    """运行者选择退出。宿主捕获后走干净退出（finally 仍写回 persist_to）。"""
    pass


class KernelPanic(VMError):
    """内核不变量被破坏。

    出现即说明某条边界少了一条判定——不变量是结构量，可判定的输入不该走到这里。
    """
    pass

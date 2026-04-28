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
    """$ 解引用成环"""
    pass


class VMResourceError(VMError):
    """资源错误：栈溢出、LLM 调用超时等"""
    pass


def vm_exception_handler(func):
    """异常处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VMError as e:
            import logging
            logging.error(f"VM 错误：{e}")
            raise
    return wrapper

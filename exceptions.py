"""AVM 异常类定义"""


class VMError(Exception):
    """所有虚拟机异常的基类"""
    pass


class VMSyntaxError(VMError):
    """语法错误：未知指令、参数数量错误等"""
    pass


class VMMemoryError(VMError):
    """内存错误：键不存在、索引越界、类型错误等"""
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

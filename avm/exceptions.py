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


class IllegalOperationError(VMMemoryError):
    """非法操作：输入本身不合法（改参数可重试，内核状态没有变化）"""
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


class MemoryLinkBrokenError(VMMemoryError):
    """链接无效 — 链接的目标不存在（链接节点本身保留，像符号链接一样悬空）"""
    pass


class VMResourceError(VMError):
    """资源错误：栈溢出、LLM 调用超时等"""
    pass


class APIError(VMError):
    """外部 API 调用失败：网络、鉴权、超时、协议错误、余额不足、上下文超限。

    与其它异常分开的理由：它的处置由外部原因决定（暂停 / 跳过对话 / 重试），
    不是内核缺陷。``category`` 由 ``avm.api_errors.classify_api_error`` 填写；
    ``status_code`` 与 ``detail`` 保留分类依据。文本格式保持 ``{类型}: {信息}``，
    分类不改变异常消息。
    """

    def __init__(self, message="", *, category=None, status_code=None, detail=None):
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.detail = dict(detail or {})


class RunnerStop(VMError):
    """运行者选择退出。宿主捕获后走干净退出（finally 仍写回 persist_to）。"""
    pass


class KernelPanic(VMError):
    """内核不变量被破坏。

    出现即说明某条边界少了一条判定——不变量是结构量，可判定的输入不该走到这里。
    """
    pass

"""服务层组合根（L4）：把 LLM 服务与 PLM 服务装到一起。

计划 plan-service-layer-2026-11-17 批次 S5。Core 通过构造参数拿到 `Services`；
默认实例由 `default_services()` 装配。这是唯一同时 import llm 与 plm 的模块，
`avm/llm.py` 与 `avm/plm.py` 互不 import。
"""

from .llm import LLMService
from .plm import PLMService


class Services:
    """服务层容器。测试可注入假实现（`Core(services=...)`）。"""

    def __init__(self, llm=None, plm=None):
        self.llm = llm if llm is not None else LLMService()
        self.plm = plm if plm is not None else PLMService()


def default_services() -> Services:
    return Services()

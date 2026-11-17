"""PLM 服务（L4）：执行器适配、注册表门面、模型目录项与恢复预热。

从 core.py 抽出（计划 plan-service-layer-2026-11-17 批次 S4）。PLM 实现仍在
`python_server.py`；本模块只是服务层门面，Core 只通过它接触 PLM。
"""

from typing import Optional

from .executor import assemble_messages, commit_round, parse_json_args
from .instructions import build_imports, build_tools
from .python_server import (
    HOST_ONLY_PLMS, PLM_REGISTRY, PLMServer, create_plm,
)
from .types import Conversation, MetaDict

_PLM_PARAMS_NOTE = "只有 model 有效；tools 由内核注入；LLM 参数无效"
_PLM_OPERATOR_NOTE = "只能由宿主创建（plm.operator 需要注入操作员通道）；不能由镜像声明"


def _first_line(text: str, limit: int = 72) -> str:
    """取文档串首行，超长截断——目录里的说明要能一眼看完。"""
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


class PLMExecutor:
    """把 PLMServer 适配成与 LMU 相同的 exec 接口，供 Core 统一调度。

    输入：Conversation + para；内部组装消息调用 PLM，
    输出：(result, return_calls, conversation)，与 LMU.exec 形状一致。
    """

    def __init__(self, plm):
        self.plm = plm
        self.last_call: Optional[dict] = None

    def exec(self, conversation: Conversation, para: MetaDict):
        messages = assemble_messages(conversation)

        params = para.to_dict()
        params["tools"] = build_tools(conversation)
        # 覆盖 para 里可能同名出现的字段：白名单只由对话 caps 决定（防 para 伪造）。
        params["imports"] = build_imports(conversation)
        response = self.plm.handle_messages(messages, params)
        result = response.get("content", "") or ""
        return_calls = self._parse_tool_calls(response.get("tool_calls") or [])
        self.last_call = {
            "messages": messages,
            "result": result,
            "tool_calls": return_calls,
        }

        # 提交历史：工具响应 → 用户内容 → 新的 assistant 消息
        tc_list = [
            {"id": t.get("id"), "type": t.get("type", "function"), "function": t.get("function")}
            for t in (response.get("tool_calls") or [])
        ] or None
        commit_round(conversation, result, tool_calls=tc_list)
        return result, return_calls, conversation

    def warm_restore(self, messages: list, params: Optional[dict] = None) -> None:
        """恢复出来的 Python 对话：重放 system 消息并把历史标记为已处理。

        只在 PLM 实现提供 `restore_state` 时生效；跨轮状态必须由程序落在 `$MEM`。
        """
        restore_state = getattr(self.plm, "restore_state", None)
        if not callable(restore_state):
            return
        restore_state(messages, params or {})

    @staticmethod
    def _parse_tool_calls(tool_calls: list) -> list:
        out = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            args, error = parse_json_args(fn.get("arguments", "{}"), fn.get("name"), tc.get("id"))
            if error is not None:
                out.append(error)
                continue
            out.append({"call_id": tc.get("id"), "cmd_type": fn.get("name"), "args": args})
        return out


class PLMService:
    """PLM 侧服务门面：可用名单、执行器创建、宿主独占判定、模型目录项。"""

    def available(self, extra: Optional[dict] = None) -> list:
        """可用 PLM 名（内建注册表在前，镜像声明的按字典序附后）。"""
        extra = extra or {}
        return list(PLM_REGISTRY) + sorted(set(extra) - set(PLM_REGISTRY))

    def is_host_only(self, model) -> bool:
        return isinstance(model, str) and model in HOST_ONLY_PLMS

    def create_executor(self, model: str, extra: Optional[dict] = None, **kwargs) -> PLMExecutor:
        return PLMExecutor(create_plm(model, extra=extra, **kwargs))

    def catalog_entries(self, extra: Optional[dict] = None) -> list:
        """对话可见的 PLM 目录项：内建（除宿主独占）+ 镜像声明的。"""
        extra = extra or {}
        entries = [_catalog_entry(name, cls) for name, cls in PLM_REGISTRY.items()
                   if name not in HOST_ONLY_PLMS]
        entries += [_catalog_entry(name, cls) for name, cls in extra.items()
                    if name not in PLM_REGISTRY]
        return entries


def _catalog_entry(name: str, cls) -> MetaDict:
    entry = MetaDict(
        data={"name": name, "required": "model"},
        metadata=_first_line(getattr(cls, "__doc__", "")) or name,
    )
    entry["params"] = _PLM_OPERATOR_NOTE if name == "plm.operator" else _PLM_PARAMS_NOTE
    return entry


__all__ = [
    "HOST_ONLY_PLMS", "PLM_REGISTRY", "PLMServer", "create_plm",
    "PLMExecutor", "PLMService",
]

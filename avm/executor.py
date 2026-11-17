"""LLM/PLM 执行器共享的执行协议与骨架（L4）。

从 core.py 抽出（计划 plan-service-layer-2026-11-17 批次 S2）：Core 只依赖
`Executor` 协议；LLM 服务与 PLM 服务共用消息组装与提交历史的两段逻辑，避免
`LMU.exec` 与 `PLMExecutor.exec` 两处逐行重复。消息形状仍是 OpenAI chat 投影
——中立 IR 的改动落在这两个函数与 `types.py`（见 discussion-message-ir-2026-11-16）。
"""

import json
from typing import Optional, Protocol, runtime_checkable

from .types import Conversation


@runtime_checkable
class Executor(Protocol):
    """Core 调度看到的执行器形状：一次推理 + 最近调用记录。

    `reset_client()` 是可选能力（API 失败策略的 "reload" 会探测它），
    Python 协议的实现类不要求提供。
    """

    last_call: Optional[dict]

    def exec(self, conversation: Conversation, para) -> tuple:
        ...


def assemble_messages(conversation: Conversation) -> list:
    """组装一次调用要发送的消息列表（不改对话状态）。

    历史 + 待发工具响应 + 待发用户内容；两个执行器原本各写一遍。
    """
    messages = conversation.to_api_messages()
    messages.extend(conversation.user_batch.to_tool_messages())
    user_content = conversation.user_batch.get_user_content()
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def commit_round(conversation: Conversation, result, tool_calls=None,
                 reasoning_content=None) -> None:
    """提交一次成功调用的历史：工具响应 → 用户内容 → 新的 assistant 消息。

    保持"调用成功后才提交"的零提交语义：调用抛错时本函数不会被走到。
    """
    user_content = conversation.user_batch.get_user_content()
    for resp in conversation.user_batch.tool_responses:
        conversation.append_tool_message(resp.content, resp.tool_call_id)
    if user_content:
        conversation.append_user_message(user_content)
    conversation.append_assistant_message(
        result or "", tool_calls=tool_calls, reasoning_content=reasoning_content
    )


def parse_json_args(raw, name, call_id):
    """解析工具调用参数，返回 (args, None) 或 (None, json_error 调用记录)。

    两个执行器原本各写一遍；原始字符串与已解析对象两种输入都接受。
    """
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return args, None
    except ValueError as e:
        return None, {
            "call_id": call_id,
            "cmd_type": "json_error",
            "args": {"error": str(e), "name": name, "raw": raw},
        }

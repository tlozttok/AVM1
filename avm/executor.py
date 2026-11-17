"""LLM/PLM 执行器共享的执行协议与骨架（L4）。

从 core.py 抽出（计划 plan-service-layer-2026-11-17 批次 S2）：Core 只依赖
`Executor` 协议；LLM 服务与 PLM 服务共用消息组装与提交历史的两段逻辑，避免
`LMU.exec` 与 `PLMExecutor.exec` 两处逐行重复。消息形状仍是 OpenAI chat 投影
——中立 IR 的改动落在这两个函数与 `types.py`（见 discussion-message-ir-2026-11-16）。
"""

import json
from typing import Optional, Protocol, runtime_checkable

from .types import AssistantMessage, Conversation, reasoning_part, tool_call_part


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


def commit_round(conversation: Conversation, result="", tool_calls=None,
                 reasoning_content=None, *, assistant=None) -> None:
    """提交一次成功调用的历史：工具响应 → 用户内容 → 新的 assistant 消息。

    保持"调用成功后才提交"的零提交语义：调用抛错时本函数不会被走到。
    `assistant` 给出时直接提交这条 IR 消息（流式路径用）；否则按 chat 投影
    参数构造一条（非流式路径与 PLM 用）。
    """
    user_content = conversation.user_batch.get_user_content()
    for resp in conversation.user_batch.tool_responses:
        conversation.append_tool_message(resp.content, resp.tool_call_id)
    if user_content:
        conversation.append_user_message(user_content)
    if assistant is not None:
        conversation.messages.append(assistant)
        return
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


# ----------------------------------------------------------------------
# 规范流式事件 → assistant IR 消息（计划 plan-ir-multiprotocol-2026-11-17 I2）
# ----------------------------------------------------------------------

def assistant_from_events(events) -> tuple:
    """把统一流式事件组装成 (AssistantMessage, usage, stop_reason)。

    事件形状见计划 §三.3：part_start / text_delta / reasoning_delta /
    tool_call_start / tool_call_args_delta / part_end(可带 opaque) / finish。
    未知事件忽略（适配器可以带自己的扩展事件，组装层不认就当没发生）。
    """
    order = []
    parts = {}
    usage = None
    stop_reason = None
    for event in events:
        etype = event.get("type")
        index = event.get("index")
        if etype == "part_start":
            kind = event.get("part")
            if kind == "text":
                parts[index] = {"type": "text", "text": ""}
            elif kind == "reasoning":
                parts[index] = reasoning_part(
                    event.get("interface") or "chat", text=""
                )
            elif kind == "tool_call":
                parts[index] = tool_call_part(
                    event.get("call_id") or "", event.get("name") or "", ""
                )
            else:
                continue
            order.append(index)
        elif etype == "part_end":
            if index in parts:
                opaque = event.get("opaque")
                if opaque is not None:
                    parts[index]["opaque"] = opaque
        elif etype == "tool_call_start":
            if index not in parts:
                parts[index] = tool_call_part(
                    event.get("call_id") or "", event.get("name") or "", ""
                )
                order.append(index)
        elif etype == "finish":
            usage = event.get("usage")
            stop_reason = event.get("stop_reason")
        elif index not in parts:
            continue
        elif etype == "text_delta":
            parts[index]["text"] = parts[index].get("text", "") + (event.get("text") or "")
        elif etype == "reasoning_delta":
            parts[index]["text"] = parts[index].get("text", "") + (event.get("text") or "")
        elif etype == "tool_call_args_delta":
            parts[index]["args"] = parts[index].get("args", "") + (event.get("args") or "")
    ordered = [parts[i] for i in order if i in parts]
    return AssistantMessage(parts=ordered), usage, stop_reason


def return_calls_from_assistant(message) -> list:
    """assistant 的 tool_call parts → Core 的 return_calls（args 在这里解析）。

    与 LMU._return_calls_from_message 的语义一致：解析失败记 json_error；
    名字原样进 cmd_type，认不认识由 Core 处理指令时判断。
    """
    out = []
    for part in message.parts:
        if part.get("type") != "tool_call":
            continue
        args, error = parse_json_args(part.get("args", ""), part.get("name"), part.get("call_id"))
        if error is not None:
            out.append(error)
            continue
        out.append({"call_id": part.get("call_id"), "cmd_type": part.get("name"),
                    "args": args})
    return out

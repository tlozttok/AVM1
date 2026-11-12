"""API 调用失败分类：纯函数，不依赖 openai/网络，可单测。

分类只有三类，内核据此决定处置：

- ``insufficient_balance``：余额/配额耗尽 → 暂停执行，等运行者处理账户后恢复；
- ``context_limit``：上下文超限 → 跳过该对话（内核级关闭），其他对话继续；
- ``other``：其余外部失败（网络、超时、限流等）→ 重试同一轮，超过上限后暂停。

分类是**尽力而为的字符串匹配**：外部 API 的错误形状随厂商与版本变化，这里只承诺
覆盖已知的可判定文本；判不出来的都落到 ``other``，由重试上限兜底。
"""

import re


INSUFFICIENT_BALANCE = "insufficient_balance"
CONTEXT_LIMIT = "context_limit"
OTHER = "other"

CATEGORIES = (INSUFFICIENT_BALANCE, CONTEXT_LIMIT, OTHER)

# 消息里出现这些片段即视为余额/配额问题（大小写不敏感）。
# 429 rate limit 不含这些片段时归 other：限流不等于余额不足。
_BALANCE_MARKERS = (
    "insufficient_quota",
    "insufficient balance",
    "余额",
    "billing",
    "payment",
    "credit",
)

# 只有 HTTP 400/413/422 才可能是上下文超限：请求体本身被拒。
_CONTEXT_STATUSES = (400, 413, 422)

# 上下文超限的消息标记（大小写不敏感）。
_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "exceeds the maximum",
    "上下文",
    "token limit",
)

# 从消息里解析上下文上限（token 数）。按顺序尝试，取第一个命中。
_CONTEXT_LIMIT_PATTERNS = (
    # "This model's maximum context length is 128000 tokens"
    re.compile(r"maximum\s+context\s+(?:length|window)[^\d]{0,40}([\d,]+)", re.I),
    # "context length is 128000" / "context window: 128000"
    re.compile(r"context\s+(?:length|window)[^\d]{0,40}([\d,]+)", re.I),
    # "128000 tokens maximum" / "token limit 128000"
    re.compile(r"([\d,]+)\s*tokens?\s*(?:maximum|limit)", re.I),
    # "exceeds the maximum of 128000 tokens"
    re.compile(r"exceeds?\s+the\s+maximum[^\d]{0,40}([\d,]+)", re.I),
    # "token limit: 128000"
    re.compile(r"token\s+limit[^\d]{0,40}([\d,]+)", re.I),
)


def _message_of(exc) -> str:
    try:
        text = str(exc)
    except Exception:  # pragma: no cover - 异常对象自己的 __str__ 坏了
        text = ""
    if text:
        return text
    return type(exc).__name__


def _status_code_of(exc):
    """从异常/响应里取 HTTP 状态码；取不到返回 None（尽力而为）。"""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _parse_context_limit(message: str):
    for pattern in _CONTEXT_LIMIT_PATTERNS:
        match = pattern.search(message)
        if match is None:
            continue
        try:
            value = int(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def classify_api_error(exc, status_code=None):
    """把一次外部 API 失败分成 ``(category, detail)``。

    ``exc`` 可以是任意异常对象；``status_code`` 显式给出时优先于异常自带的字段。
    ``detail`` 恒含 ``category`` / ``message`` / ``status_code`` / ``context_limit``。
    """
    message = _message_of(exc)
    status = status_code if status_code is not None else _status_code_of(exc)
    lowered = message.lower()

    category = OTHER
    context_limit = None
    if status == 402 or any(marker in lowered for marker in _BALANCE_MARKERS):
        category = INSUFFICIENT_BALANCE
    elif status in _CONTEXT_STATUSES and any(
        marker in lowered for marker in _CONTEXT_MARKERS
    ):
        category = CONTEXT_LIMIT
        context_limit = _parse_context_limit(message)

    detail = {
        "category": category,
        "message": message,
        "status_code": status,
        "context_limit": context_limit,
    }
    return category, detail

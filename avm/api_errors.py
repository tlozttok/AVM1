"""API 调用失败分类：纯函数，不依赖 openai/网络，可单测。

分类只有三类，内核据此决定处置：

- ``insufficient_balance``：余额/配额耗尽 → 暂停执行，等运行者处理账户后恢复；
- ``context_limit``：上下文超限 → 跳过该对话（内核级关闭），其他对话继续；
- ``other``：其余外部失败（网络、超时、限流等）→ 重试同一轮，超过上限后暂停。

分类是**尽力而为的字符串匹配**：外部 API 的错误形状随厂商与版本变化，这里只承诺
覆盖已知的可判定文本；判不出来的都落到 ``other``，由重试上限兜底。
"""

import re
import sys
import time


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


# ---------------------------------------------------------------------------
# 处置：宿主策略、失败记录、重试上限
#
# 分类（上面那个函数）只回答"这是哪种失败"，纯函数、同样的异常永远得到同样的
# 类别。这里回答"接下来怎么办"——要看宿主装没装策略、这个对话已经连着失败几次、
# 要不要重置执行器的连接。两者分开，是因为内核在同一个 except 分支里既要记帧
# 又要决定走向，混在一起读不出哪句是判断、哪句是副作用。
# ---------------------------------------------------------------------------


def category_and_detail(error):
    """取 ``(category, detail)``：外部执行器已经分类的直接用，否则在这里补分类。

    执行器在抛错前就该把分类放进 APIError；镜像自定义的 PLM 与测试替身可能只抛
    一个普通 APIError，这里按同一套规则再判一次——所以这个函数对"分类过一次"和
    "完全没分类"的异常都成立，不要求调用方先做什么。
    """
    category = getattr(error, "category", None)
    detail = getattr(error, "detail", None)
    if category in CATEGORIES:
        detail = dict(detail or {})
        detail.setdefault("category", category)
        detail.setdefault("message", str(error))
        detail.setdefault("status_code", getattr(error, "status_code", None))
        detail.setdefault("context_limit", None)
        return category, detail
    return classify_api_error(error, status_code=getattr(error, "status_code", None))


def resolve_action(policy, error, attempt, executor) -> str:
    """问宿主 API 失败怎么办，返回 retry / reload / exit。

    没装策略、策略自己出错、返回未知动作，一律按退出处理：停下等一个不会来的
    选择比退出更糟。终端三选项策略在 avm/runner_prompt.py。

    ``policy`` 为 None 也走退出（调用方已先判空，这里兜底让函数自洽，免得别的
    调用点忘了判空就静默继续）。
    """
    if policy is None:
        print(f"[AVM] API 调用失败且没有安装 api_error_policy，按退出处理：{error}",
              file=sys.stderr)
        return "exit"
    try:
        action = policy(error, attempt)
    except Exception as e:
        print(
            f"[AVM] api_error_policy 自身异常（{type(e).__name__}: {e}），按退出处理",
            file=sys.stderr,
        )
        return "exit"
    if action not in ("retry", "reload", "exit"):
        print(f"[AVM] api_error_policy 返回未知动作 {action!r}，按退出处理", file=sys.stderr)
        return "exit"
    if action == "reload":
        reset = getattr(executor, "reset_client", None)
        if callable(reset):
            reset()
    return action


def remember_failure(conv, detail: dict) -> None:
    """把分类信息挂到对话运行态（不写内存树）；成功一轮后由调用方清掉。

    只保留设备契约里的五个字段：外部执行器给的 detail 可能带额外键，设备面不
    因此漂移。时间戳让面板能看出"这次失败发生在什么时候"。
    """
    conv.last_api_error = {
        "category": detail.get("category"),
        "message": detail.get("message", ""),
        "status_code": detail.get("status_code"),
        "context_limit": detail.get("context_limit"),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def retry_limit(configured) -> int:
    """连续失败到几次暂停：非负整数照用，其余（含没配）一律 5。

    bool 是 int 的子类，``True`` 会悄悄变成"只重试 1 次"，所以单独排掉。
    """
    if not isinstance(configured, int) or isinstance(configured, bool) or configured < 0:
        return 5
    return configured

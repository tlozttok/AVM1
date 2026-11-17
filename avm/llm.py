"""LLM 服务（L4）：客户端生命周期、API 参数表、工具调用解析、模型清单原始拉取。

从 core.py 抽出（计划 plan-service-layer-2026-11-17 批次 S3）。本模块是 Core 之外
唯一 import openai/dotenv 的地方；多协议适配器（Responses/Anthropic/Gemini）将落在
这里，Core 不感知协议细节。
"""

import json
import time
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from .api_errors import classify_api_error
from .exceptions import APIError
from .executor import assemble_messages, commit_round
from .instructions import build_tools
from .types import Conversation, MetaDict


class LMU:
    """一次 LLM 推理的调用侧实现（chat completions 投影）。"""

    _client: Optional[OpenAI] = None

    def __init__(self):
        self._client = None
        self.last_call: Optional[dict] = None  # 最近一次 API 调用全文（供监测器写全文文件）

    @property
    def client(self):
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    def reset_client(self) -> None:
        """丢弃缓存的客户端，按当前环境重建。

        运行者选"更新环境"后调用：光重建客户端不够——load_dotenv 默认不覆盖已存在的
        环境变量，第二次读 .env 不会生效，这里显式 override。
        """
        self._client = None
        load_dotenv(override=True)

    _API_PARAM_KEYS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "stop", "stream", "extra_body",
        "seed", "logit_bias", "logprobs", "top_logprobs",
        "n", "response_format", "timeout", "reasoning_effort",
    }
    _NUMERIC_API_PARAMS: set = {
        "temperature", "max_tokens", "top_p", "frequency_penalty",
        "presence_penalty", "n", "seed", "timeout", "logprobs", "top_logprobs",
    }

    def api_param_names(self) -> tuple:
        """para 白名单里允许出现的 API 参数名（模型目录用）。"""
        return tuple(sorted(self._API_PARAM_KEYS))

    def numeric_param_names(self) -> tuple:
        """其中以数值型存储、调用前需要还原的参数名（模型目录用）。"""
        return tuple(sorted(self._NUMERIC_API_PARAMS))

    def _filter_api_params(self, para: dict) -> dict:
        out = {}
        for k, v in para.items():
            if k not in self._API_PARAM_KEYS:
                continue
            # para 以 str 存储在内存中，数值型参数需要还原为数值
            if k in self._NUMERIC_API_PARAMS and isinstance(v, str):
                try:
                    v = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    pass
            out[k] = v
        return out

    def _parse_tool_args(self, tool_call, return_calls: list):
        call_id = tool_call.id
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return_calls.append({
                "call_id": call_id,
                "cmd_type": "json_error",
                "args": {"error": str(e), "name": tool_call.function.name, "raw": tool_call.function.arguments},
            })
            return None

    def _return_calls_from_message(self, message) -> list:
        return_calls = []
        if not message.tool_calls:
            return return_calls

        for tool_call in message.tool_calls:
            call_id = tool_call.id
            name = tool_call.function.name
            args = self._parse_tool_args(tool_call, return_calls)
            if args is None:
                continue

            # 解析器不查指令集：名字原样进 cmd_type。认不认识这个名字由
            # _process_return_calls 在处理指令时判断一次——查表只在一处发生，
            # 解析器因此不必知道指令集，也不会把未知名字改写掉。
            return_calls.append({"call_id": call_id, "cmd_type": name, "args": args})

        return return_calls

    def exec(self, conversation: Conversation, para: MetaDict):
        messages = assemble_messages(conversation)

        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")
        tools = build_tools(conversation)
        model = para.get("model", "gpt-4")

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=use_tool,
                **extra_para
            )
        except Exception as e:
            self.last_call = {
                "model": model,
                "messages": messages,
                "result": None,
                "tool_calls": [],
                "elapsed_ms": (time.time() - start) * 1000,
                "error": f"{type(e).__name__}: {e}",
            }
            # 失败时历史与输入批次都没动过（追加在下面、clear 在 exec 返回之后），
            # 所以从这里重试同一轮是干净的，不需要检查点或回退。
            category, detail = classify_api_error(e)
            raise APIError(
                f"{type(e).__name__}: {e}",
                category=category,
                status_code=detail["status_code"],
                detail=detail,
            ) from e

        choice = response.choices[0]
        message = choice.message
        result = message.content
        # 思维链内容：防御性读取（openai SDK 对非标准响应字段的支持依赖版本）
        reasoning_content = getattr(message, "reasoning_content", None)
        return_calls = self._return_calls_from_message(message)

        tc_list = None
        if message.tool_calls:
            tc_list = [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
        commit_round(conversation, result, tool_calls=tc_list,
                     reasoning_content=reasoning_content)

        self.last_call = {
            "model": model,
            "messages": messages,
            "result": result,
            "reasoning": reasoning_content,
            "tool_calls": return_calls,
            "elapsed_ms": (time.time() - start) * 1000,
            "error": None,
        }
        return result, return_calls, conversation

    def fetch_models_raw(self, timeout: float = 10.0) -> dict:
        """拉一次上游 `/models` 并返回原始 JSON 对象（信封校验留给 Core）。

        任何异常直接抛给调用方；`Core.refresh_model_catalog` 负责收敛成
        source="unavailable" 并打一行 stderr。
        """
        raw = self.client.models.with_raw_response.list(timeout=timeout)
        return _raw_response_json(raw)


def _raw_response_json(raw):
    """从 SDK 的"原始响应"对象里取回 JSON 对象，兼容 openai 各版本。

    - openai 2.x 的 `LegacyAPIResponse`：没有 `.json()`，有 `.http_response`（httpx 响应）与 `.text`；
    - openai 1.x：有 `.json()`；
    - 未来主版本把 `.text`/`.content` 改成方法——两者都按可调用或属性处理。
    """
    http = getattr(raw, "http_response", None)
    json_fn = getattr(http, "json", None) if http is not None else None
    if callable(json_fn):
        return json_fn()
    json_fn = getattr(raw, "json", None)
    if callable(json_fn):
        return json_fn()
    text = getattr(raw, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return json.loads(text)
    content = getattr(raw, "content", None)
    if callable(content):
        content = content()
    if isinstance(content, (bytes, bytearray)):
        return json.loads(bytes(content).decode("utf-8"))
    raise TypeError(f"无法从 {type(raw).__name__} 取原始 JSON 响应体")

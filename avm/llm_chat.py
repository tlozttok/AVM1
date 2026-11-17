"""chat 协议适配器（L4）：OpenAI chat completions 投影的调用侧实现。

计划 plan-ir-multiprotocol-2026-11-17 批次 I2：从 llm.py 抽出。保留非流式
`exec`（行为与拆分前一致，现有 fake-client 测试依赖它）与新增的 `stream`：
后者产出统一流式事件（见 executor.assistant_from_events 与计划 §三.3）。
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
        self.last_request: Optional[dict] = None  # 最近一次请求的 model/messages（流式服务拼 last_call 用）

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
        "presence_penalty", "stop", "extra_body",
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
        """拉一次上游 `/models` 并返回原始 JSON 对象（信封校验由 parse_models 做）。"""
        raw = self.client.models.with_raw_response.list(timeout=timeout)
        return _raw_response_json(raw)

    @staticmethod
    def parse_models(payload) -> list:
        return parse_openai_models(payload)

    # ------------------------------------------------------------------
    # 流式路径（计划 I2）：产出统一事件；组装与提交由 LLMService 负责
    # ------------------------------------------------------------------

    def stream(self, conversation: Conversation, para: MetaDict):
        """流式调用 chat completions，逐条产出规范事件（不提交历史）。"""
        messages = assemble_messages(conversation)
        extra_para = self._filter_api_params(para.to_dict())
        use_tool = para.get("use_tool")
        tools = build_tools(conversation)
        model = para.get("model", "gpt-4")
        self.last_request = {"model": model, "messages": messages}

        start = time.time()
        try:
            chunks = self.client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                tool_choice=use_tool, stream=True, **extra_para,
            )
        except Exception as e:
            self._stream_failed(e, model, messages, start)
            raise

        yield {"type": "start"}
        index = 0
        reasoning_index = text_index = None
        tool_indexes = {}      # provider 的 tool_call index → 本服务的 part index
        finish_reason = None
        usage = None
        try:
            for chunk in chunks:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = _usage_dict(chunk_usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if reasoning_index is None:
                        reasoning_index = index
                        index += 1
                        yield {"type": "part_start", "index": reasoning_index,
                               "part": "reasoning", "interface": "chat"}
                    yield {"type": "reasoning_delta", "index": reasoning_index,
                           "text": reasoning}

                content = getattr(delta, "content", None)
                if content:
                    if reasoning_index is not None:
                        yield {"type": "part_end", "index": reasoning_index}
                        reasoning_index = None
                    if text_index is None:
                        text_index = index
                        index += 1
                        yield {"type": "part_start", "index": text_index, "part": "text"}
                    yield {"type": "text_delta", "index": text_index, "text": content}

                for tool_call in getattr(delta, "tool_calls", None) or []:
                    position = getattr(tool_call, "index", 0)
                    fn = getattr(tool_call, "function", None)
                    if position not in tool_indexes:
                        if reasoning_index is not None:
                            yield {"type": "part_end", "index": reasoning_index}
                            reasoning_index = None
                        if text_index is not None:
                            yield {"type": "part_end", "index": text_index}
                            text_index = None
                        part_index = index
                        index += 1
                        tool_indexes[position] = part_index
                        yield {"type": "tool_call_start", "index": part_index,
                               "call_id": getattr(tool_call, "id", None) or "",
                               "name": getattr(fn, "name", None) or "" if fn else ""}
                    arguments = getattr(fn, "arguments", None) if fn else None
                    if arguments:
                        yield {"type": "tool_call_args_delta",
                               "index": tool_indexes[position], "args": arguments}

            if reasoning_index is not None:
                yield {"type": "part_end", "index": reasoning_index}
            if text_index is not None:
                yield {"type": "part_end", "index": text_index}
            for part_index in tool_indexes.values():
                yield {"type": "part_end", "index": part_index}
            yield {"type": "finish",
                   "stop_reason": _stop_reason(finish_reason),
                   "usage": usage or {}}
        except Exception as e:
            self._stream_failed(e, model, messages, start)
            raise

    def _stream_failed(self, error, model: str, messages: list, start: float) -> None:
        self.last_call = {
            "model": model,
            "messages": messages,
            "result": None,
            "tool_calls": [],
            "elapsed_ms": (time.time() - start) * 1000,
            "error": f"{type(error).__name__}: {error}",
        }
        if isinstance(error, APIError):
            return
        category, detail = classify_api_error(error)
        if category is None:  # pragma: no cover - classify_api_error 恒返回分类
            return
        raise APIError(
            f"{type(error).__name__}: {error}",
            category=category,
            status_code=detail["status_code"],
            detail=detail,
        ) from error


def _stop_reason(finish_reason):
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_calls"
    if finish_reason == "length":
        return "length"
    if finish_reason == "content_filter":
        return "content_filter"
    return "stop"


def _usage_dict(usage) -> dict:
    return {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


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


def parse_openai_models(payload) -> list:
    """OpenAI 风格 `/models` 信封 → 目录条目列表（除 id 外全部可选、忽略未知字段）。

    Core 只负责把它写进 $MEM.model_catalog；信封校验与字段归一化在这里。
    """
    if (not isinstance(payload, dict) or payload.get("object") != "list"
            or not isinstance(payload.get("data"), list)):
        raise ValueError("上游 /models 的信封不是 OpenAI 形状（object=list 且 data 是数组）")
    entries = []
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        entry = {"id": model_id}
        for key in ("name", "owned_by", "context_window", "max_output_tokens",
                    "input_modalities", "output_modalities"):
            value = item.get(key)
            if value is None:
                continue
            entry[key] = (", ".join(str(v) for v in value)
                          if isinstance(value, list) else str(value))
        effort = item.get("effort")
        if isinstance(effort, dict):
            levels = effort.get("supported_levels")
            if isinstance(levels, list):
                entry["reasoning_effort_levels"] = ", ".join(str(v) for v in levels)
            default_level = effort.get("default_level")
            if isinstance(default_level, str) and default_level:
                entry["default_effort_level"] = default_level
        entries.append(entry)
    return entries

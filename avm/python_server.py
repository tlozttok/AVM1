
import contextlib
import hashlib
import io
import json
import math
import re
from typing import List
from abc import ABC, abstractmethod

from .types import (
    Conversation, MetaDict, MetaList, Role, message_to_api_dict, Message,
    SystemMessage, UserMessage, AssistantMessage, ToolMessage,
)
from .sandbox import safe_globals, exec_safe, check_code


def message_list_from_api_dict(messages: List[dict]) -> List[Message]:
    """把 OpenAI 格式的原始消息列表转换成 Message 列表（message_to_api_dict 的逆操作）"""
    result = []
    for d in messages:
        role = d.get("role")
        content = d.get("content", "")
        if role == "assistant" and content is None:
            # assistant 无内容（仅工具调用）时不出现，只有后续的 tool 消息
            continue
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(UserMessage(content=content))
        elif role == "assistant":
            result.append(AssistantMessage(
                content=content or "",
                tool_calls=d.get("tool_calls"),
                reasoning_content=d.get("reasoning_content"),
            ))
        elif role == "tool":
            result.append(ToolMessage(content=content, tool_call_id=d.get("tool_call_id", "")))
        else:
            result.append(Message(content=content, role=role or ""))
    return result


class PLMServer(ABC):
    """Python 执行器：OpenAI 兼容的 chat 服务器。

    每次调用收到完整的原始消息列表（历史只增不减）；
    通过比较上一次输入与本次输入，取出最新新增的输入消息列表。
    约束：
    - Python 程序输出是确定性的，AI 端角色消息（assistant）不处理；
    - 用户端角色消息有三种：system、user、tool（tool 返回值也是用户端消息）；
    - Python 程序与 AI 一样，只能通过设备访问外部（网络/OS/IO 被封锁），不直接触碰系统。
    """

    def __init__(self):
        self._prev_input: List[dict] = []
        self._tools: dict = {}  # 工具定义：name → {"description", "parameters"}
        self._call_records: dict = {}  # 基类保证的函数调用记录：call_id → {"name", "arguments"}，用于返回与调用配对

    @staticmethod
    def call_id_for(message: str) -> str:
        """确定性工具调用 id：对发来的消息做 sha256 编码（同一消息必然同一 id）。"""
        return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]

    def _new_input_messages(self, raw_messages: List[dict]) -> List[dict]:
        """比较上一次输入与本次输入，返回本次新增的消息（共同前缀之后的切片）"""
        prev = self._prev_input
        i = 0
        while i < len(prev) and i < len(raw_messages) and prev[i] == raw_messages[i]:
            i += 1
        return raw_messages[i:]

    def _capture_tools(self, params: dict) -> None:
        """解析工具定义（OpenAI 格式的 tools 列表，通常来自 params），按名字存入实例字段。"""
        self._tools = {}
        for t in params.get("tools") or []:
            fn = t.get("function") or {}
            name = fn.get("name")
            if name:
                self._tools[name] = {
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                }

    def _dispatch(self, msg) -> None:
        """按消息类型派发执行；system 不返回，user/tool 可以返回消息。"""
        if isinstance(msg, SystemMessage):
            self._execute_system_message(msg.content)
            return None
        if isinstance(msg, UserMessage):
            return self._execute_user_message(msg.content)
        if isinstance(msg, ToolMessage):
            return self._execute_tool_message(msg.content, msg.tool_call_id)
        return None

    @abstractmethod
    def _execute_system_message(self, content: str):
        """抽象：处理系统消息（导入模块、设置可复用背景等），由子类实现"""
        raise NotImplementedError

    @abstractmethod
    def _execute_user_message(self, content: str):
        """抽象：处理用户消息（解析指令字符串、处理并产出结果），由子类实现。
        返回一条原始消息（字符串或 Message），入口 handle_messages 会把它包装成 Response。"""
        raise NotImplementedError

    @abstractmethod
    def _execute_tool_message(self, content: str, call_id: str):
        """抽象：处理工具返回消息（content + call_id），由子类实现。
        返回一条原始消息（字符串或 Message），入口 handle_messages 会把它包装成 Response。"""
        raise NotImplementedError

    def make_response(self, message=None, tool_calls=None) -> dict:
        """把消息包装成 OpenAI 兼容的 Response 形式（{"content": ..., "tool_calls": ...}）。

        message 可以是字符串、Message 对象，或已经是 Response 的字典（直接透传）；
        tool_calls 结构与 OpenAI 一致：
        [{"id", "type", "function": {"name", "arguments"}}]。
        """
        if isinstance(message, dict):
            response = dict(message)
            if tool_calls:
                response["tool_calls"] = tool_calls
            return response
        if isinstance(message, Message):
            content = message.content
            if tool_calls is None:
                tool_calls = getattr(message, "tool_calls", None)
        else:
            content = message if message is not None else ""
        response = {"content": content}
        if tool_calls:
            response["tool_calls"] = tool_calls
        return response

    def parse_message(self, content: str) -> dict:
        """解析指令消息（JSON：from/to/icc_id/content）；失败返回带 __error__ 的字典。"""
        try:
            msg = json.loads(content)
        except (ValueError, TypeError):
            return {"__raw__": content, "__error__": "JSON 解析失败"}
        if not isinstance(msg, dict):
            return {"__raw__": content, "__error__": "不是对象"}
        return msg

    def handle_messages(self, raw_messages, params):
        # 解析工具定义，放到实例字段（供程序了解可用工具）
        self._capture_tools(params)
        # 通过比较上一次的输入和这一次的输入，获取最新的输入消息列表
        new_messages = message_list_from_api_dict(self._new_input_messages(raw_messages))
        self._prev_input = raw_messages
        # 逐个执行；只有最后一个消息的返回被保留（程序确定性：同样输入序列只有同样输出）
        last_result = None
        for i, msg in enumerate(new_messages):
            result = self._dispatch(msg)
            if i == len(new_messages) - 1:
                last_result = result
        response = self.make_response(last_result)
        # 基类保证：Response 里的 tool_calls 记入调用记录，供工具返回与调用行为配对
        for tc in response.get("tool_calls") or []:
            fn = tc.get("function") or {}
            self._call_records[tc.get("id")] = {
                "name": fn.get("name"),
                "arguments": fn.get("arguments"),
            }
        return response


class SimplePLM(PLMServer):
    """最简单的 PLM：计算器。只接受一行表达式，eval 求值后直接返回。

    固定行为模式：
    - 用户提示词：消息为 JSON 信封（from/to/icc_id/content），取 content 字段作表达式，
      eval 求值（受限命名空间：无内置函数，提供 math）；
    - 信封带 icc_id 时，构造 return_result 工具调用把结果投递给发起者（工具调用是具体实现的行为）；
    - 系统 / 工具消息：计算器无状态，不处理。
    """

    def _execute_system_message(self, content: str):
        pass

    def _execute_user_message(self, content: str):
        msg = self.parse_message(content)
        expr = content
        icc_id = None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            expr = msg["content"]
            icc_id = msg.get("icc_id")
        try:
            # eval 模式天然只接受单个表达式；受限命名空间（安全 builtins + math）
            check_code(expr, mode="eval")
            value = eval(compile(expr, "<expr>", "eval"), safe_globals({"math": math}), {})
            result = str(value)
        except Exception as e:
            result = f"错误: {type(e).__name__}: {e}"
        if icc_id:
            # 返回工具：结果通过 return_result 投递给发起者（id = 确定性 hash(发来的消息)）
            tool_calls = [{
                "id": self.call_id_for(content),
                "type": "function",
                "function": {
                    "name": "return_result",
                    "arguments": json.dumps({"content": result, "icc_id": icc_id}),
                },
            }]
            return self.make_response(result, tool_calls=tool_calls)
        return result

    def _execute_tool_message(self, content: str, call_id: str):
        return ""


class PythonPLM(PLMServer):
    """图灵完备 PLM：Jupyter 笔记本式——系统/用户提示词都执行进同一个持久命名空间，
    stdout 全部捕获并按顺序拼接为整体返回。

    固定行为模式：
    - 系统提示词：setup 代码，受限执行进持久命名空间（跨轮次保留）；
    - 用户提示词：代码，也执行进同一个命名空间（与 setup 共享变量、可读写）；
    - 多个 print 全部执行，输出按顺序拼接为一个整体作为返回；
    - 信封带 icc_id 时自动构造 return_result 返回；
    - 沙箱：安全 builtins 白名单 + AST 检查（禁 import、禁双下划线属性、禁危险调用）。
    """

    def __init__(self):
        super().__init__()
        self.namespace = safe_globals({"math": math, "json": json, "re": re})

    def _execute_system_message(self, content: str):
        exec_safe(content, self.namespace)

    def _execute_user_message(self, content: str):
        msg = self.parse_message(content)
        code = content
        icc_id = None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            code = msg["content"]
            icc_id = msg.get("icc_id")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec_safe(code, self.namespace)
            result = buf.getvalue()
        except Exception as e:
            result = f"错误: {type(e).__name__}: {e}"
        if icc_id:
            tool_calls = [{
                "id": self.call_id_for(content),
                "type": "function",
                "function": {
                    "name": "return_result",
                    "arguments": json.dumps({"content": result, "icc_id": icc_id}),
                },
            }]
            return self.make_response(result, tool_calls=tool_calls)
        return result

    def _execute_tool_message(self, content: str, call_id: str):
        return ""


PLM_REGISTRY = {
    "plm.simple": SimplePLM,
    "plm.python": PythonPLM,
}


def create_plm(model: str) -> PLMServer:
    """按 model 参数创建 PLM 实例（注册表在本模块）。"""
    try:
        return PLM_REGISTRY[model]()
    except KeyError:
        raise ValueError(f"未知 PLM 类型: {model}（可用: {list(PLM_REGISTRY)}）")
        

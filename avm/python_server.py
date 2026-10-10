
import contextlib
import inspect
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

    仅测试调试，暂存；不推荐使用，不维护。

    固定行为模式：
    - 用户提示词：消息内容直接作为一行表达式，eval 求值（受限命名空间：无内置函数，提供 math）；
    - 系统 / 工具消息：计算器无状态，不处理。
    """

    def _execute_system_message(self, content: str):
        pass

    def _execute_user_message(self, content: str):
        expr = content
        try:
            # eval 模式天然只接受单个表达式；受限命名空间（安全 builtins + math）
            check_code(expr, mode="eval")
            value = eval(compile(expr, "<expr>", "eval"), safe_globals({"math": math}), {})
            result = str(value)
        except Exception as e:
            result = f"错误: {type(e).__name__}: {e}"
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
    - 沙箱：安全 builtins 白名单 + AST 检查（禁 import、禁双下划线属性、禁危险调用）。
    """

    def __init__(self):
        super().__init__()
        self.namespace = safe_globals({"math": math, "json": json, "re": re})

    def _execute_system_message(self, content: str):
        exec_safe(content, self.namespace)

    def _execute_user_message(self, content: str):
        code = content
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec_safe(code, self.namespace)
            result = buf.getvalue()
        except Exception as e:
            result = f"错误: {type(e).__name__}: {e}"
        return result

    def _execute_tool_message(self, content: str, call_id: str):
        return ""


class FunctionPLM(PLMServer):
    """纯函数 PLM：system 定义 handle(input)，user 传入 JSON，返回 JSON。

    固定行为模式：
    - system 消息：定义可调用 handle(input) 的 Python 代码，编译一次；
    - user 消息：JSON 文本，json.loads 后作为 input 传入 handle；
    - 每次调用全新受限命名空间；无跨调用状态、无 tool_calls、stdout 丢弃；
    - 返回值必须可 JSON 序列化，json.dumps 后作为 content 返回。
    """

    def __init__(self):
        super().__init__()
        self._compiled = None
        self._parse_error = None

    def _execute_system_message(self, content: str):
        try:
            check_code(content)
            self._compiled = compile(content, "<function>", "exec")
            self._parse_error = None
        except Exception as e:
            self._compiled = None
            self._parse_error = f"{type(e).__name__}: {e}"

    def _execute_user_message(self, content: str):
        if self._compiled is None:
            detail = self._parse_error or "未收到 system 定义"
            return f"错误: FunctionPLM: 定义编译失败: {detail}"
        try:
            input_data = json.loads(content)
        except Exception as e:
            return f"错误: FunctionPLM: {type(e).__name__}: {e}"
        try:
            namespace = safe_globals({"math": math, "json": json, "re": re})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(self._compiled, namespace)
                handle = namespace.get("handle")
                if not callable(handle):
                    raise TypeError("handle 未定义或不可调用")
                result = handle(input_data)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"错误: FunctionPLM: {type(e).__name__}: {e}"

    def _execute_tool_message(self, content: str, call_id: str):
        return ""


class _MachineParseError(Exception):
    """状态机定义解析错误，携带行号信息。"""


class StateMachinePLM(PLMServer):
    """状态机 PLM：system 是状态机定义 DSL，user 是 JSON 事件。

    状态与变量只活在本实例内；tool 消息追加到 last 供 guard 使用。
    """

    _VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self):
        super().__init__()
        self._initial = None
        self._states = {}
        self._state = None
        self._vars = {}
        self._parse_error = None
        self._last_tool_output = ""
        self._fsm_counter = 0

    # ------------------------------------------------------------------
    # system 消息：解析 DSL
    # ------------------------------------------------------------------

    def _execute_system_message(self, content: str):
        try:
            self._parse_machine(content)
            self._parse_error = None
            self._last_tool_output = ""
        except Exception as e:
            self._initial = None
            self._states = {}
            self._state = None
            self._vars = {}
            self._parse_error = str(e)

    def _parse_machine(self, content: str):
        initial = None
        states = {}
        current_state = None
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            lineno = i + 1
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            if line.startswith("initial:"):
                name = line.split(":", 1)[1].strip()
                if not name:
                    raise _MachineParseError(f"第 {lineno} 行: initial 缺少状态名")
                initial = name
                i += 1
                continue
            if line.startswith("state "):
                rest = line[6:].strip()
                if not rest.endswith(":"):
                    raise _MachineParseError(f"第 {lineno} 行: state 行必须以 : 结尾")
                name = rest[:-1].strip()
                if not name:
                    raise _MachineParseError(f"第 {lineno} 行: state 缺少名称")
                if name in states:
                    raise _MachineParseError(f"第 {lineno} 行: state {name} 重复定义")
                states[name] = []
                current_state = name
                i += 1
                continue
            if line.startswith("on "):
                if current_state is None:
                    raise _MachineParseError(f"第 {lineno} 行: 转移行出现在任何 state 之前")
                text = line
                # 可选 response JSON 允许跨多行：持续吸收后续行，直到 JSON 可解析。
                if "|" in text:
                    while True:
                        resp_text = text.partition("|")[2].strip()
                        try:
                            json.loads(resp_text)
                            break
                        except Exception:
                            if i + 1 >= len(lines):
                                break
                            nxt = lines[i + 1].strip()
                            if not nxt or nxt.startswith(("initial:", "state ", "on ")):
                                break
                            i += 1
                            text = text + "\n" + lines[i]
                states[current_state].append(self._parse_transition(text, lineno))
                i += 1
                continue
            raise _MachineParseError(f"第 {lineno} 行: 无法识别的语句: {line}")

        if initial is None:
            raise _MachineParseError("缺少 initial: 声明")
        if initial not in states:
            raise _MachineParseError(f"initial 状态 {initial} 未声明")
        for state_name, transitions in states.items():
            for t in transitions:
                if t["next"] not in states:
                    raise _MachineParseError(
                        f"第 {t['line']} 行: 转移目标 {t['next']} 未声明"
                    )
        self._initial = initial
        self._states = states
        self._state = initial
        self._vars = {}

    def _parse_transition(self, line: str, lineno: int) -> dict:
        text = line[3:].strip()
        if "->" not in text:
            raise _MachineParseError(f"第 {lineno} 行: 转移行缺少 ->")
        src, _, dest = text.rpartition("->")
        src = src.strip()
        dest = dest.strip()
        if not src:
            raise _MachineParseError(f"第 {lineno} 行: 转移行缺少事件名")
        if not dest:
            raise _MachineParseError(f"第 {lineno} 行: 转移行缺少目标状态")

        # 解析目标状态与可选 response JSON
        next_state = dest
        content = ""
        tools = []
        if "|" in dest:
            left, _, resp_text = dest.partition("|")
            next_state = left.strip()
            resp_text = resp_text.strip()
            try:
                response = json.loads(resp_text)
            except Exception as e:
                raise _MachineParseError(f"第 {lineno} 行: response JSON 解析失败: {e}")
            if not isinstance(response, dict):
                raise _MachineParseError(f"第 {lineno} 行: response 必须是 JSON 对象")
            content = response.get("content", "")
            if content is None:
                content = ""
            if not isinstance(content, str):
                raise _MachineParseError(f"第 {lineno} 行: response.content 必须是字符串")
            tools = response.get("tools") or []
            if not isinstance(tools, list):
                raise _MachineParseError(f"第 {lineno} 行: response.tools 必须是数组")
            checked = []
            for tool in tools:
                if not isinstance(tool, dict):
                    raise _MachineParseError(f"第 {lineno} 行: response.tools 每项必须是对象")
                name = tool.get("name")
                arguments = tool.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise _MachineParseError(f"第 {lineno} 行: tool.name 必须是非空字符串")
                if not isinstance(arguments, dict):
                    raise _MachineParseError(f"第 {lineno} 行: tool.arguments 必须是对象")
                checked.append({"name": name, "arguments": arguments})
            tools = checked

        # 解析 set 块
        sets = []
        brace = src.find("{")
        if brace != -1:
            end = src.rfind("}")
            if end < brace:
                raise _MachineParseError(f"第 {lineno} 行: set 块缺少 }}")
            block = src[brace + 1:end]
            src = (src[:brace] + " " + src[end + 1:]).strip()
            for raw_stmt in block.split(";"):
                stmt = raw_stmt.strip()
                if not stmt:
                    continue
                if not stmt.startswith("set "):
                    raise _MachineParseError(f"第 {lineno} 行: set 块语句必须以 set 开头")
                assignment = stmt[4:].strip()
                var, sep, expr = assignment.partition("=")
                var = var.strip()
                expr = expr.strip()
                if not sep or not var or not expr:
                    raise _MachineParseError(f"第 {lineno} 行: set 语句格式应为 set <var> = <expr>")
                if not self._VAR_NAME_RE.match(var):
                    raise _MachineParseError(f"第 {lineno} 行: 非法变量名 {var!r}")
                sets.append((var, expr))

        # 解析事件与可选 guard
        event, _, rest = src.partition(" ")
        rest = rest.strip()
        guard = None
        if rest:
            if not rest.startswith("if "):
                raise _MachineParseError(f"第 {lineno} 行: 事件后只能是 if guard 或 set 块")
            guard = rest[3:].strip()
            if not guard:
                raise _MachineParseError(f"第 {lineno} 行: if 后缺少 guard 表达式")

        if self._tools:
            for tool in tools:
                if tool["name"] not in self._tools:
                    raise _MachineParseError(
                        f"第 {lineno} 行: 未知工具 {tool['name']}（可用: {sorted(self._tools)}）"
                    )

        return {
            "event": event,
            "guard": guard,
            "sets": sets,
            "next": next_state,
            "content": content,
            "tools": tools,
            "line": lineno,
        }

    # ------------------------------------------------------------------
    # user / tool 消息
    # ------------------------------------------------------------------

    def _error(self, message: str) -> str:
        return f"错误: StateMachinePLM: {message}"

    def _eval_expr(self, expr: str, context: dict):
        check_code(expr, mode="eval")
        compiled = compile(expr, "<fsm>", "eval")
        return eval(compiled, safe_globals({"math": math}), dict(context))

    def _execute_user_message(self, content: str):
        if self._parse_error is not None:
            return self._error(f"定义解析失败: {self._parse_error}")
        try:
            msg = json.loads(content)
        except Exception as e:
            return self._error(f"{type(e).__name__}: {e}")
        if not isinstance(msg, dict):
            return self._error("用户消息必须是 JSON 对象")
        event_name = msg.get("event")
        if not isinstance(event_name, str):
            return self._error("用户消息必须包含字符串 event")
        event_data = msg.get("data", {})
        if not isinstance(event_data, dict):
            return self._error("event.data 必须是对象")

        context = dict(self._vars)
        context.update({
            "event_name": event_name,
            "event_data": event_data,
            "last": self._last_tool_output,
        })

        chosen = None
        for transition in self._states.get(self._state, []):
            if transition["event"] != event_name and transition["event"] != "*":
                continue
            if transition["guard"]:
                try:
                    if not self._eval_expr(transition["guard"], context):
                        continue
                except Exception as e:
                    self._last_tool_output = ""
                    return self._error(f"{type(e).__name__}: {e}")
            chosen = transition
            break

        if chosen is None:
            self._last_tool_output = ""
            return self._error(f"state {self._state} 无匹配事件 {event_name}")

        # set 事务化：先在临时上下文求值，全部成功后再写回。
        try:
            next_vars = dict(self._vars)
            eval_context = dict(context)
            for var, expr in chosen["sets"]:
                value = self._eval_expr(expr, eval_context)
                eval_context[var] = value
                next_vars[var] = value
        except Exception as e:
            self._last_tool_output = ""
            return self._error(f"{type(e).__name__}: {e}")

        # 工具名校验（若已捕获 tools 定义）
        try:
            for tool in chosen["tools"]:
                if tool["name"] not in self._tools:
                    raise ValueError(f"未知工具 {tool['name']}")
        except Exception as e:
            self._last_tool_output = ""
            return self._error(f"{type(e).__name__}: {e}")

        # 提交状态
        self._vars = next_vars
        self._state = chosen["next"]

        tool_calls = []
        for tool in chosen["tools"]:
            call_id = f"fsm-{self._fsm_counter}"
            self._fsm_counter += 1
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": json.dumps(tool["arguments"], ensure_ascii=False),
                },
            })

        self._last_tool_output = ""
        response = {"content": chosen["content"]}
        if tool_calls:
            response["tool_calls"] = tool_calls
        return response

    def _execute_tool_message(self, content: str, call_id: str):
        if self._last_tool_output:
            self._last_tool_output += "\n"
        self._last_tool_output += content
        return ""


# CommandPLM 注入程序的工具函数：yield 一次工具请求，返回 Core 的工具响应字符串。
# 程序侧用法：content = yield from memory_read("$MEM.x")

def _tool_memory_read(ref):
    return (yield ("memory_read", {"ref": ref}))


def _tool_memory_write(ref, content):
    return (yield ("memory_write", {"ref": ref, "content": content}))


def _tool_memory_make(ref, key, mem_type):
    return (yield ("memory_make", {"ref": ref, "key": key, "mem_type": mem_type}))


def _tool_send_instruction(cid, content):
    return (yield ("send_instruction", {"cid": cid, "content": content}))


def _tool_close_conversation(cid):
    return (yield ("close_conversation", {"cid": cid}))


# 注入程序的工具函数集合（命令执行子集，不开放创建对话等重指令）
COMMAND_TOOLS = {
    "memory_read": _tool_memory_read,
    "memory_write": _tool_memory_write,
    "memory_make": _tool_memory_make,
    "send_instruction": _tool_send_instruction,
    "close_conversation": _tool_close_conversation,
}

_NO_RESUME = object()  # 生成器首次推进（next）与续跑（send）的区分哨兵


class CommandPLM(PLMServer):
    """命令行 PLM：system 是命令处理程序，user 是一行命令与参数，IO 都是字符串。

    固定行为模式：
    - system 消息：PLM 程序（定义 `handle(cmd)` 及辅助代码），受限检查后编译一次；
    - user 消息：命令与参数（整行字符串），整串作为 `cmd` 传入 `handle`；程序自己解析参数；
    - 每次命令在全新受限命名空间中执行（无跨命令状态；要持久化就用工具写 `$MEM`）；
    - `handle` 返回字符串即命令输出（返回 None 视为空输出，其它类型报错）；
    - `handle` 也可以是生成器：注入的工具函数用 `yield from` 产生工具调用，Core 执行后
      把工具响应字符串送回生成器，程序从断点继续（一条命令内可完成「读→算→写」往返）；
    - 注入的工具函数（对应 AVM 指令子集）：memory_read(ref) / memory_write(ref, content)
      / memory_make(ref, key, mem_type) / send_instruction(cid, content) / close_conversation(cid)；
      工具响应原样返回（包括 `Error: ...` 文本），由程序决定如何处理；
    - 命令行沙箱：安全 builtins 白名单 + AST 检查（禁 import、禁双下划线属性、禁危险调用）；
    - stdout 捕获后丢弃（print 不进入返回值），输出就是 `handle` 的返回值；
    - 工具调用 id 形如 `cmd-<n>`，每条命令内递增；一条命令最多挂起一个工具调用。
    """

    def __init__(self):
        super().__init__()
        self._compiled = None
        self._parse_error = None
        self._gen = None  # 挂起在工具调用上的生成器
        self._pending_call_id = None
        self._call_seq = 0

    # ------------------------------------------------------------------
    # system 消息：编译程序
    # ------------------------------------------------------------------

    def _execute_system_message(self, content: str):
        self._gen = None
        self._pending_call_id = None
        self._compiled = None
        self._parse_error = None
        try:
            check_code(content)
            self._compiled = compile(content, "<command>", "exec")
        except Exception as e:
            self._parse_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # user / tool 消息：执行命令、续跑生成器
    # ------------------------------------------------------------------

    def _error(self, message: str) -> str:
        return f"错误: CommandPLM: {message}"

    def _fresh_namespace(self) -> dict:
        extra = {"math": math, "json": json, "re": re}
        extra.update(COMMAND_TOOLS)
        return safe_globals(extra)

    def _abandon(self) -> None:
        """丢弃挂起的生成器（其剩余步骤不再执行）。"""
        gen = self._gen
        self._gen = None
        self._pending_call_id = None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass

    def _output(self, value):
        """命令输出：字符串原样返回，None 视为空输出，其它类型报错。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return self._error(f"handle 必须返回字符串，得到 {type(value).__name__}")

    @staticmethod
    def _as_tool_call(yielded):
        """校验生成器 yield 出的工具请求，返回 (name, args) 或 None。"""
        if not isinstance(yielded, (tuple, list)) or len(yielded) != 2:
            return None
        name, args = yielded
        if not isinstance(name, str) or name not in COMMAND_TOOLS:
            return None
        if not isinstance(args, dict):
            return None
        return name, args

    def _drive(self, gen, sent):
        """推进生成器一步：产出工具调用，或结束并返回命令输出。"""
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                yielded = next(gen) if sent is _NO_RESUME else gen.send(sent)
        except StopIteration as stop:
            self._abandon()
            return self._output(stop.value)
        except Exception as e:
            self._abandon()
            return self._error(f"{type(e).__name__}: {e}")
        parsed = self._as_tool_call(yielded)
        if parsed is None:
            self._abandon()
            return self._error(
                f"非法工具调用: {yielded!r}（可用: {sorted(COMMAND_TOOLS)}）"
            )
        name, args = parsed
        call_id = f"cmd-{self._call_seq}"
        self._call_seq += 1
        self._gen = gen
        self._pending_call_id = call_id
        return self.make_response("", tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }])

    def _execute_user_message(self, content: str):
        if self._compiled is None:
            detail = self._parse_error or "未收到 system 定义"
            return self._error(f"程序编译失败: {detail}")
        if self._gen is not None:
            # 上一条命令仍挂起在工具调用上：丢弃，开始新命令（确定性边界）
            self._abandon()
        namespace = self._fresh_namespace()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(self._compiled, namespace)
                handle = namespace.get("handle")
                if not callable(handle):
                    raise TypeError("handle 未定义或不可调用")
                result = handle(content)
        except Exception as e:
            self._abandon()
            return self._error(f"{type(e).__name__}: {e}")
        if inspect.isgenerator(result):
            return self._drive(result, _NO_RESUME)
        return self._output(result)

    def _execute_tool_message(self, content: str, call_id: str):
        if self._gen is None:
            # 没有挂起的命令：与其他 PLM 一致，忽略
            return ""
        expected = self._pending_call_id
        if call_id != expected:
            self._abandon()
            return self._error(f"工具响应与挂起的调用不匹配: {call_id!r}（等待 {expected!r}）")
        gen = self._gen
        self._pending_call_id = None
        return self._drive(gen, content)


class OperatorPLM(PLMServer):
    """操作员执行器：这个对话的"推理"由操作员完成。

    它不是模型：不产生内容，也不解释提示词。每次推理阻塞等待控制台提交一个
    动作，把动作原样作为一次工具调用返回；收到结束动作时返回不带工具调用的
    结果，对话据此结束本轮交互，Core 把执行权让回就绪队列。

    动作形状（控制台保证）：
        {"tool": <工具名>, "arguments": {...}}   一次工具调用
        {"end": True}                            结束本轮交互

    工具名原样透传，不在这里校验：面板要能造出模型造得出的各种输入，包括未知
    工具名，以便观察 Core 的错误路径。
    """

    def __init__(self, channel=None):
        super().__init__()
        self._channel = channel  # 由控制台注入；鸭子类型，只需要 wait_action / publish
        self._seq = 0

    def handle_messages(self, raw_messages, params):
        """覆写基类的"确定性程序"语义。

        基类逐条执行新增消息、只保留最后一条的返回值，那套是给无状态程序用的。
        操作员执行器的每轮动作只有一个：把新增消息转给控制台显示，然后等下一个
        动作。逐个等待会让一次推理内的多条消息各等一次动作。
        """
        channel = self._require_channel()
        new_messages = message_list_from_api_dict(self._new_input_messages(raw_messages))
        self._prev_input = raw_messages
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                channel.publish("tool_result", {
                    "call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif isinstance(msg, UserMessage):
                channel.publish("user_message", {"content": msg.content})
        return self._next_action()

    def _require_channel(self):
        if self._channel is None:
            raise ValueError(
                "操作员对话必须由控制台创建：plm.operator 需要一个已绑定的操作员通道"
                "（见 Core.start_operator_conversation）"
            )
        return self._channel

    def _next_action(self):
        """等一个形状合法的动作；形状不合法就报回控制台并继续等。"""
        channel = self._require_channel()
        while True:
            action = channel.wait_action()
            if not isinstance(action, dict):
                channel.publish("action_error", {
                    "reason": f"动作必须是对象，得到 {type(action).__name__}",
                })
                continue
            if action.get("end"):
                return {"content": ""}
            tool = action.get("tool")
            arguments = action.get("arguments", {})
            if not isinstance(tool, str) or not tool:
                channel.publish("action_error", {"reason": "动作缺少字符串字段 tool"})
                continue
            if not isinstance(arguments, dict):
                channel.publish("action_error", {"reason": "arguments 必须是对象"})
                continue
            # TODO(issue): 计划 C1 原本要求在这里拒绝九条之外的工具名。实现时改成
            # 原样透传——面板的意义是造出模型造得出的输入，包括未知工具名，用来观察
            # Core 的错误路径；拒绝掉就测不到那条路径。未知名字的提示改由控制台给出。
            self._seq += 1
            call_id = f"op-{self._seq}"
            channel.publish("tool_call", {
                "call_id": call_id, "tool": tool, "arguments": arguments,
            })
            return {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            }

    # handle_messages 已覆写，以下三条抽象方法不会被走到；实现为同样的动作等待。

    def _execute_system_message(self, content):
        return None

    def _execute_user_message(self, content):
        return self._next_action()

    def _execute_tool_message(self, content, call_id):
        return self._next_action()


PLM_REGISTRY = {
    "plm.simple": SimplePLM,      # 仅测试调试，暂存；不推荐使用，不维护
    "plm.python": PythonPLM,      # 图灵完备兜底
    "plm.function": FunctionPLM,
    "plm.machine": StateMachinePLM,
    "plm.command": CommandPLM,    # 字符串 IO 命令行：setup 是程序，user 是命令与参数
    "plm.operator": OperatorPLM,  # 操作员：调试控制台持有的假对话（需注入通道）
}


def create_plm(model: str, **kwargs) -> PLMServer:
    """按 model 参数创建 PLM 实例（注册表在本模块）。"""
    cls = PLM_REGISTRY.get(model)
    if cls is None:
        raise ValueError(f"未知 PLM 类型: {model}（可用: {list(PLM_REGISTRY)}）")
    return cls(**kwargs)
        

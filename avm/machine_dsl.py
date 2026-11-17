"""状态机 DSL 的编译：文本 → (初始状态, 状态表)。

语法就是几行文本：

    initial: 待命
    state 待命:
      on 输入 -> 处理 | {"content": "收到"}
      on 超时 -> 待命

这里只做一件事：把文本读成结构，读不动就带着行号报错。运行期（当前状态、变量、
guard 求值、工具消息）在 python_server.StateMachinePLM 里——那边不该为了看懂
状态转移表而先读一遍语法分析。

两条容易写坏的规则都在这层的注释里：response 的 JSON 可以跨行（判停要同时看
"JSON 能解析"与"下一行是不是新语句"），以及同一行有多处错误时报哪一条取决于
求值顺序（顺序即语义，别重排）。
"""

import json
import re


class MachineParseError(Exception):
    """状态机 DSL 解析失败（消息里带行号）。"""


class MachineParser:
    """DSL 解析器。check_tool(name) 返回 None 表示允许，否则返回错误原因。"""

    _VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, check_tool=None):
        self.check_tool = check_tool

    def parse(self, content: str) -> tuple:
        """把整段 DSL 读成 (初始状态名, 状态表)。任何不合语法的地方都带行号抛出。"""
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
                initial = self._parse_initial(line, lineno)
                i += 1
                continue
            if line.startswith("state "):
                current_state = self._parse_state_head(line, lineno, states)
                i += 1
                continue
            if line.startswith("on "):
                if current_state is None:
                    raise MachineParseError(f"第 {lineno} 行: 转移行出现在任何 state 之前")
                text, i = self._absorb_transition(lines, i)
                states[current_state].append(self._parse_transition(text, lineno))
                i += 1
                continue
            raise MachineParseError(f"第 {lineno} 行: 无法识别的语句: {line}")

        if initial is None:
            raise MachineParseError("缺少 initial: 声明")
        if initial not in states:
            raise MachineParseError(f"initial 状态 {initial} 未声明")
        self._check_transition_targets(states)
        # 只返回结果，不碰运行期状态：解析失败时调用方一个字都不用回滚。
        return initial, states

    @staticmethod
    def _parse_initial(line: str, lineno: int) -> str:
        """`initial: 名字`"""
        name = line.split(":", 1)[1].strip()
        if not name:
            raise MachineParseError(f"第 {lineno} 行: initial 缺少状态名")
        return name

    @staticmethod
    def _parse_state_head(line: str, lineno: int, states: dict) -> str:
        """`state 名字:` —— 建一张空的转移表，返回当前状态名。"""
        rest = line[6:].strip()
        if not rest.endswith(":"):
            raise MachineParseError(f"第 {lineno} 行: state 行必须以 : 结尾")
        name = rest[:-1].strip()
        if not name:
            raise MachineParseError(f"第 {lineno} 行: state 缺少名称")
        if name in states:
            raise MachineParseError(f"第 {lineno} 行: state {name} 重复定义")
        states[name] = []
        return name

    @staticmethod
    def _absorb_transition(lines: list, i: int) -> tuple:
        """从第 i 行起吸收一条 `on ...`，返回 (合并后的文本, 最后消费到的行号)。

        response JSON 允许跨行，所以判停有两个条件：JSON 能解析了，或者下一行
        看起来已经是新语句（initial:/state/on）。只看其中一个都不行——
        只等 JSON 会把后面的 state 行一起吞掉，只看下一行会截断合法的多行 JSON。
        """
        text = lines[i].strip()  # 首行必须去缩进：_parse_transition 直接按 "on " 切前缀
        if "|" not in text:
            return text, i
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
        return text, i

    @staticmethod
    def _check_transition_targets(states: dict) -> None:
        """全部转移写完后统一检查：去向必须是已声明的状态（拼错的越早报越好）。"""
        for transitions in states.values():
            for t in transitions:
                if t["next"] not in states:
                    raise MachineParseError(
                        f"第 {t['line']} 行: 转移目标 {t['next']} 未声明"
                    )

    def _parse_transition(self, line: str, lineno: int) -> dict:
        text = line[3:].strip()
        if "->" not in text:
            raise MachineParseError(f"第 {lineno} 行: 转移行缺少 ->")
        src, _, dest = text.rpartition("->")
        src = src.strip()
        dest = dest.strip()
        if not src:
            raise MachineParseError(f"第 {lineno} 行: 转移行缺少事件名")
        if not dest:
            raise MachineParseError(f"第 {lineno} 行: 转移行缺少目标状态")

        # 顺序即语义：先 dest（含 response JSON），再 src（set 块、事件与 guard），
        # 最后才对照工具声明。同一行有多处错误时报哪一条取决于这个顺序，别重排。
        next_state, content, tools = self._parse_dest(dest, lineno)
        sets, event, guard = self._parse_src(src, lineno)
        self._check_tools_declared(tools, lineno)
        return {
            "event": event,
            "guard": guard,
            "sets": sets,
            "next": next_state,
            "content": content,
            "tools": tools,
            "line": lineno,
        }

    def _parse_dest(self, dest: str, lineno: int) -> tuple:
        """目标状态与可选 response JSON（`状态 | {"content": ...}`）。"""
        if "|" not in dest:
            return dest, "", []
        next_state, _, resp_text = dest.partition("|")
        content, tools = self._parse_response(resp_text.strip(), lineno)
        return next_state.strip(), content, tools

    def _parse_response(self, resp_text: str, lineno: int) -> tuple:
        """response 对象 → (content, tools)。content 缺省为空串，None 也当空串。"""
        try:
            response = json.loads(resp_text)
        except Exception as e:
            raise MachineParseError(f"第 {lineno} 行: response JSON 解析失败: {e}")
        if not isinstance(response, dict):
            raise MachineParseError(f"第 {lineno} 行: response 必须是 JSON 对象")
        content = response.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise MachineParseError(f"第 {lineno} 行: response.content 必须是字符串")
        return content, self._parse_response_tools(response.get("tools") or [], lineno)

    def _parse_response_tools(self, tools: list, lineno: int) -> list:
        """response.tools 每项规范成 {name, arguments}，其余键丢弃。"""
        if not isinstance(tools, list):
            raise MachineParseError(f"第 {lineno} 行: response.tools 必须是数组")
        checked = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise MachineParseError(f"第 {lineno} 行: response.tools 每项必须是对象")
            name = tool.get("name")
            arguments = tool.get("arguments", {})
            if not isinstance(name, str) or not name:
                raise MachineParseError(f"第 {lineno} 行: tool.name 必须是非空字符串")
            if not isinstance(arguments, dict):
                raise MachineParseError(f"第 {lineno} 行: tool.arguments 必须是对象")
            checked.append({"name": name, "arguments": arguments})
        return checked

    def _parse_src(self, src: str, lineno: int) -> tuple:
        """来源侧：`事件 { set ... } if guard` → (sets, 事件名, guard)。"""
        src, sets = self._parse_set_block(src, lineno)
        event, _, rest = src.partition(" ")
        rest = rest.strip()
        guard = None
        if rest:
            if not rest.startswith("if "):
                raise MachineParseError(f"第 {lineno} 行: 事件后只能是 if guard 或 set 块")
            guard = rest[3:].strip()
            if not guard:
                raise MachineParseError(f"第 {lineno} 行: if 后缺少 guard 表达式")
        return sets, event, guard

    def _parse_set_block(self, src: str, lineno: int) -> tuple:
        """摘出 `{ set a = 1; set b = 2 }` 块，返回 (去掉块的 src, [(var, expr)])。"""
        brace = src.find("{")
        if brace == -1:
            return src, []
        end = src.rfind("}")
        if end < brace:
            raise MachineParseError(f"第 {lineno} 行: set 块缺少 }}")
        block = src[brace + 1:end]
        src = (src[:brace] + " " + src[end + 1:]).strip()
        sets = []
        for raw_stmt in block.split(";"):
            stmt = raw_stmt.strip()
            if not stmt:
                continue
            if not stmt.startswith("set "):
                raise MachineParseError(f"第 {lineno} 行: set 块语句必须以 set 开头")
            assignment = stmt[4:].strip()
            var, sep, expr = assignment.partition("=")
            var = var.strip()
            expr = expr.strip()
            if not sep or not var or not expr:
                raise MachineParseError(f"第 {lineno} 行: set 语句格式应为 set <var> = <expr>")
            if not self._VAR_NAME_RE.match(var):
                raise MachineParseError(f"第 {lineno} 行: 非法变量名 {var!r}")
            sets.append((var, expr))
        return src, sets

    def _check_tools_declared(self, tools: list, lineno: int) -> None:
        """声明过 tools 后，转移里只能引用已声明的工具。

        谁会"允许"什么由 check_tool 决定（None 表示允许，否则给出原因），
        解析器不关心工具表长什么样——它只编译文本。
        """
        if self.check_tool is None:
            return
        for tool in tools:
            reason = self.check_tool(tool["name"])
            if reason:
                raise MachineParseError(f"第 {lineno} 行: {reason}")


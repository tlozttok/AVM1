"""base 镜像的构造器 PLM（`plm.builder`）：把创建内存节点的多步指令打包成一次请求。

这个执行器不是 AI，是内存树的确定性构造服务。作者把一份 JSON 规格交给它，
它按「探测 → 构造/更新 → 直通指令 → 验证」的顺序逐个发出工具调用，最后返回
一份 JSON 摘要。一次规格可以创建 settingup/python 程序节点、handler 节点或
para 节点，也可以顺带执行任意打包指令。

用户消息（JSON）：

    {
      "mode": "create" | "update",       # 缺省 create
      "ops": [
        {"type": "settingup", "path": "$MEM.programs.hello", "name": "hello",
         "content": "...", "signature": "...", "meta": "一句描述"},
        {"type": "python", "path": "$MEM.programs.py", "name": "py", "content": "..."},
        {"type": "handler", "path": "$MEM.handlers.h", "name": "h",
         "content": "def handle(event): ...",
         "condition": {"events": ["memory_write"], "phase": ["before"]}},
        {"type": "para", "path": "$MEM.para.python",
         "fields": {"model": "plm.python", "temperature": "0.2"}}
      ],
      "calls": [{"tool": "memory_read", "arguments": {"ref": "$MEM.x"}}]
    }

节点合法性（与 Core 的校验一致）：
- 程序节点：dict + `ctrl.type` 为 settingup/python + 子字段 `name`（可选）、
  `content`（必填）、`signature`（可选）；
- handler 节点：dict + `ctrl.type="handler"` + `name`（可选）、`content`（必填）、
  `condition`（必填，JSON 字符串；规格里给 dict/list 会自动 json.dumps）；
- para 节点：dict + `ctrl.type="para"` + `fields` 的所有键值转成 str 子节点。

create 模式不覆盖已存在的节点（记错误并跳过）；update 模式只改已有节点，
缺失的目标记错误并跳过。`calls` 是直通通道：规格里给什么工具与参数就发什么，
结果原样收进摘要。

错误不抛给 Core：非法规格直接返回 `错误: BuilderPLM: ...` 且不产生工具调用；
构造期单点失败记进摘要的 errors，继续做其余步骤。system 消息被忽略
（程序节点里的说明文字不是 JSON，也不应该报错）。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from avm.python_server import PLMServer

# 构造器能直通的指令名（Core 的完整指令集，见 avm/core.py 的指令类）。
INSTRUCTION_TOOLS = frozenset({
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
    "create_cmd",
    "create_sub",
    "send_instruction",
    "close_conversation",
    "listen_event",
})

NODE_TYPES = ("settingup", "python", "handler", "para")

_ERROR_PREFIXES = ("Error:", "Illegal:")

# 每种节点允许在规格里出现的字段。type/path/meta 是所有类型共有的。
_FIELD_KEYS = {
    "settingup": frozenset({"name", "content", "signature"}),
    "python": frozenset({"name", "content", "signature"}),
    "handler": frozenset({"name", "content", "condition"}),
    "para": frozenset({"fields"}),
}


class _SpecError(Exception):
    """规格非法：只给调用者看的 `错误: ...`，不产生任何工具调用。"""


class NodeBuilderPLM(PLMServer):
    """构造器执行器：一次用户消息驱动一整条多步构造链。"""

    def __init__(self):
        super().__init__()
        self._gen = None
        self._pending: Optional[str] = None
        self._call_seq = 0

    # ------------------------------------------------------------------
    # PLMServer 接口
    # ------------------------------------------------------------------

    def _execute_system_message(self, content: str):
        """system 消息只是程序节点的说明文字：忽略内容，不报错。"""
        self._abandon()

    def _execute_user_message(self, content: str):
        self._abandon()
        try:
            spec = self._parse_spec(content)
        except _SpecError as e:
            return f"错误: BuilderPLM: {e}"
        except Exception as e:  # 解析器自身的异常同样不能打穿 Core
            return f"错误: BuilderPLM: 解析异常 {type(e).__name__}: {e}"
        try:
            self._gen = self._run(spec)
            return self._drive(None)
        except _SpecError as e:
            self._abandon()
            return f"错误: BuilderPLM: {e}"
        except Exception as e:
            self._abandon()
            return f"错误: BuilderPLM: 内部异常 {type(e).__name__}: {e}"

    def _execute_tool_message(self, content: str, call_id: str):
        if self._gen is None:
            return ""
        if self._pending is None or call_id != self._pending:
            self._abandon()
            return "错误: BuilderPLM: 工具响应与挂起的调用不匹配"
        self._pending = None
        try:
            return self._drive(content)
        except _SpecError as e:
            self._abandon()
            return f"错误: BuilderPLM: {e}"
        except Exception as e:
            self._abandon()
            return f"错误: BuilderPLM: 内部异常 {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # 生成器驱动：一个工具调用一次挂起
    # ------------------------------------------------------------------

    def _abandon(self) -> None:
        """丢弃未跑完的生成器（新请求到来、调用不匹配或异常收尾时）。"""
        gen = self._gen
        self._gen = None
        self._pending = None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass

    def _drive(self, sent: Optional[str]) -> str:
        try:
            if sent is None:
                request = next(self._gen)
            else:
                request = self._gen.send(sent)
        except StopIteration as stop:
            self._gen = None
            self._pending = None
            value = stop.value
            return "" if value is None else str(value)
        if not isinstance(request, (tuple, list)) or len(request) != 2:
            raise _SpecError(f"内部错误：非法的工具请求 {request!r}")
        name, args = request
        if not isinstance(name, str) or name not in INSTRUCTION_TOOLS:
            raise _SpecError(f"内部错误：非法的工具名 {name!r}")
        if not isinstance(args, dict):
            raise _SpecError("内部错误：工具参数必须是对象")
        call_id = f"builder-{self._call_seq}"
        self._call_seq += 1
        self._pending = call_id
        return self.make_response(
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        )

    # ------------------------------------------------------------------
    # 规格解析
    # ------------------------------------------------------------------

    def _parse_spec(self, content: str) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise _SpecError("空规格：用户消息必须是 JSON 对象")
        try:
            spec = json.loads(content)
        except ValueError as e:
            raise _SpecError(f"规格不是合法 JSON：{e}")
        if not isinstance(spec, dict):
            raise _SpecError(f"规格必须是 JSON 对象，得到 {type(spec).__name__}")
        unknown = sorted(set(spec) - {"mode", "ops", "calls"})
        if unknown:
            raise _SpecError(f"规格含未知字段：{unknown}")
        mode = spec.get("mode", "create")
        if mode not in ("create", "update"):
            raise _SpecError(f"mode 必须是 create/update，得到 {mode!r}")
        raw_ops = spec.get("ops", [])
        if not isinstance(raw_ops, list):
            raise _SpecError(f"ops 必须是数组，得到 {type(raw_ops).__name__}")
        ops = [self._parse_op(index, raw) for index, raw in enumerate(raw_ops)]
        raw_calls = spec.get("calls", [])
        if not isinstance(raw_calls, list):
            raise _SpecError(f"calls 必须是数组，得到 {type(raw_calls).__name__}")
        calls = [self._parse_call(index, raw) for index, raw in enumerate(raw_calls)]
        return {"mode": mode, "ops": ops, "calls": calls}

    def _parse_op(self, index: int, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise _SpecError(f"ops[{index}] 必须是对象，得到 {type(raw).__name__}")
        node_type = raw.get("type")
        if node_type not in NODE_TYPES:
            raise _SpecError(f"ops[{index}].type 必须是 {list(NODE_TYPES)} 之一，得到 {node_type!r}")
        unknown = sorted(set(raw) - {"type", "path", "meta"} - _FIELD_KEYS[node_type])
        if unknown:
            raise _SpecError(f"ops[{index}]（type={node_type}）含未知字段：{unknown}")
        path = raw.get("path")
        self._path_segments(path, f"ops[{index}].path")
        meta = raw.get("meta")
        if meta is not None and not isinstance(meta, str):
            raise _SpecError(f"ops[{index}].meta 必须是字符串")
        fields: List[Tuple[str, str]] = []
        if node_type in ("settingup", "python"):
            fields.extend(self._parse_program_fields(index, raw))
        elif node_type == "handler":
            fields.extend(self._parse_handler_fields(index, raw))
        else:
            fields.extend(self._parse_para_fields(index, raw))
        return {"type": node_type, "path": path, "fields": fields, "meta": meta}

    def _parse_program_fields(self, index: int, raw: Dict[str, Any]) -> List[Tuple[str, str]]:
        fields: List[Tuple[str, str]] = []
        name = raw.get("name")
        if name is not None:
            if not isinstance(name, str):
                raise _SpecError(f"ops[{index}].name 必须是字符串")
            fields.append(("name", name))
        content = raw.get("content")
        if not isinstance(content, str):
            raise _SpecError(f"ops[{index}].content 必须是字符串（必填）")
        fields.append(("content", content))
        signature = raw.get("signature")
        if signature is not None:
            if not isinstance(signature, str):
                raise _SpecError(f"ops[{index}].signature 必须是字符串")
            fields.append(("signature", signature))
        return fields

    def _parse_handler_fields(self, index: int, raw: Dict[str, Any]) -> List[Tuple[str, str]]:
        fields: List[Tuple[str, str]] = []
        name = raw.get("name")
        if name is not None:
            if not isinstance(name, str):
                raise _SpecError(f"ops[{index}].name 必须是字符串")
            fields.append(("name", name))
        content = raw.get("content")
        if not isinstance(content, str):
            raise _SpecError(f"ops[{index}].content 必须是字符串（必填）")
        fields.append(("content", content))
        if "condition" not in raw:
            raise _SpecError(f"ops[{index}].condition 必填（JSON 字符串或对象）")
        condition = raw["condition"]
        if isinstance(condition, str):
            try:
                json.loads(condition)
            except ValueError as e:
                raise _SpecError(f"ops[{index}].condition 不是合法 JSON：{e}")
            condition_text = condition
        elif isinstance(condition, (dict, list)):
            condition_text = json.dumps(condition, ensure_ascii=False)
        else:
            raise _SpecError(
                f"ops[{index}].condition 必须是 JSON 字符串/对象/数组，"
                f"得到 {type(condition).__name__}"
            )
        fields.append(("condition", condition_text))
        return fields

    def _parse_para_fields(self, index: int, raw: Dict[str, Any]) -> List[Tuple[str, str]]:
        values = raw.get("fields")
        if not isinstance(values, dict):
            raise _SpecError(f"ops[{index}].fields 必须是对象（键 → 值），必填")
        fields: List[Tuple[str, str]] = []
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise _SpecError(f"ops[{index}].fields 的键必须是非空字符串，得到 {key!r}")
            if "." in key:
                raise _SpecError(
                    f"ops[{index}].fields 的键不能包含点号（点号是路径分隔符）：{key!r}"
                )
            fields.append((key, self._scalar_text(index, key, value)))
        return fields

    @staticmethod
    def _scalar_text(index: int, key: str, value: Any) -> str:
        """para 字段值转成 str 子节点；非字符串按 JSON 字面量转换（0.2 → "0.2"）。"""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return json.dumps(value, ensure_ascii=False)
        raise _SpecError(
            f"ops[{index}].fields[{key!r}] 必须是字符串/数值/布尔，"
            f"得到 {type(value).__name__}"
        )

    def _parse_call(self, index: int, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise _SpecError(f"calls[{index}] 必须是对象，得到 {type(raw).__name__}")
        unknown = sorted(set(raw) - {"tool", "arguments"})
        if unknown:
            raise _SpecError(f"calls[{index}] 含未知字段：{unknown}")
        tool = raw.get("tool")
        if not isinstance(tool, str) or not tool:
            raise _SpecError(f"calls[{index}].tool 必须是非空字符串")
        if tool not in INSTRUCTION_TOOLS:
            raise _SpecError(
                f"calls[{index}].tool {tool!r} 不是可用指令"
                f"（可用：{sorted(INSTRUCTION_TOOLS)}）"
            )
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _SpecError(f"calls[{index}].arguments 必须是对象")
        return {"tool": tool, "arguments": arguments}

    @staticmethod
    def _path_segments(path: Any, subject: str) -> List[str]:
        if not isinstance(path, str) or not path:
            raise _SpecError(f"{subject} 必须是非空字符串")
        if not path.startswith("$"):
            raise _SpecError(f"{subject} 必须以 $ 开头，得到 {path!r}")
        parts = path[1:].split(".")
        if any(part == "" for part in parts):
            raise _SpecError(f"{subject} 不能有空段，得到 {path!r}")
        if len(parts) < 2:
            raise _SpecError(f"{subject} 至少要写到根下第一层（不能是根本身），得到 {path!r}")
        return parts

    # ------------------------------------------------------------------
    # 构造流程
    # ------------------------------------------------------------------

    def _run(self, spec: Dict[str, Any]):
        mode = spec["mode"]
        ops = spec["ops"]
        summary: Dict[str, Any] = {
            "ok": False,
            "mode": mode,
            "built": [],
            "updated": [],
            "skipped": [],
            "errors": [],
            "calls": [],
            "verified": {},
        }
        exists: Dict[str, bool] = {}
        yield from self._probe(ops, mode, exists)
        for op in ops:
            if mode == "create":
                yield from self._create_op(op, exists, summary)
            else:
                yield from self._update_op(op, exists, summary)
        for call in spec["calls"]:
            response = yield (call["tool"], call["arguments"])
            summary["calls"].append({
                "tool": call["tool"],
                "arguments": call["arguments"],
                "result": response,
            })
            if self._is_error(response):
                summary["errors"].append(f"call {call['tool']}: {response}")
        summary["verified"] = yield from self._verify(ops)
        summary["ok"] = (
            not summary["errors"]
            and all(entry["ok"] for entry in summary["verified"].values())
        )
        return json.dumps(summary, ensure_ascii=False)

    def _probe(self, ops: List[Dict[str, Any]], mode: str, exists: Dict[str, bool]):
        """探测 path、祖先链与（update 模式下的）字段路径，记录存在性。"""
        for op in ops:
            refs = [op["path"]] + self._ancestors(op["path"])
            if mode == "update":
                refs += [f"{op['path']}.{field}" for field, _value in op["fields"]]
            for ref in refs:
                if ref not in exists:
                    exists[ref] = yield from self._exists(ref)

    def _create_op(self, op: Dict[str, Any], exists: Dict[str, bool],
                   summary: Dict[str, Any]):
        path = op["path"]
        node_type = op["type"]
        if exists.get(path):
            summary["errors"].append(f"{path}: create 模式下目标已存在，跳过（不覆盖）")
            summary["skipped"].append(path)
            return
        # 缺失的祖先自顶向下建出来；中途失败就整条 op 放弃。
        for ancestor in self._ancestors(path)[1:]:
            if exists.get(ancestor):
                continue
            response = yield ("memory_make", {
                "ref": self._parent_of(ancestor),
                "key": self._key_of(ancestor),
                "mem_type": "dict",
            })
            if self._is_error(response):
                summary["errors"].append(f"{ancestor}: 建祖先失败：{response}")
                return
            exists[ancestor] = True
        response = yield ("memory_make", {
            "ref": self._parent_of(path),
            "key": self._key_of(path),
            "mem_type": "dict",
        })
        if self._is_error(response):
            summary["errors"].append(f"{path}: 建节点失败：{response}")
            return
        exists[path] = True
        errors: List[str] = []
        yield from self._write_fields(op, exists, errors)
        yield from self._set_type(op, errors)
        yield from self._write_meta(op, errors)
        if errors:
            summary["errors"].extend(errors)
        else:
            summary["built"].append(path)

    def _update_op(self, op: Dict[str, Any], exists: Dict[str, bool],
                   summary: Dict[str, Any]):
        path = op["path"]
        if not exists.get(path):
            summary["errors"].append(f"{path}: update 模式下目标不存在，跳过")
            summary["skipped"].append(path)
            return
        errors: List[str] = []
        yield from self._write_fields(op, exists, errors)
        yield from self._set_type(op, errors)
        yield from self._write_meta(op, errors)
        if errors:
            summary["errors"].extend(errors)
        else:
            summary["updated"].append(path)

    def _write_fields(self, op: Dict[str, Any], exists: Dict[str, bool],
                      errors: List[str]):
        """字段存在就写，缺失先 memory_make 成 str 子节点再写。"""
        path = op["path"]
        for field, value in op["fields"]:
            ref = f"{path}.{field}"
            if not exists.get(ref):
                response = yield ("memory_make", {
                    "ref": path,
                    "key": field,
                    "mem_type": "str",
                })
                if self._is_error(response):
                    errors.append(f"{ref}: 建字段失败：{response}")
                    continue
                exists[ref] = True
            response = yield ("memory_write", {"ref": ref, "content": value})
            if self._is_error(response):
                errors.append(f"{ref}: 写字段失败：{response}")

    def _set_type(self, op: Dict[str, Any], errors: List[str]):
        path = op["path"]
        response = yield ("edit_metadata", {
            "ref": path,
            "type": "set",
            "key": "type",
            "value": op["type"],
        })
        if self._is_error(response):
            errors.append(f"{path}: 写 ctrl.type 失败：{response}")

    def _write_meta(self, op: Dict[str, Any], errors: List[str]):
        if op["meta"] is None:
            return
        response = yield ("memory_write", {"ref": op["path"], "content": op["meta"]})
        if self._is_error(response):
            errors.append(f"{op['path']}: 写描述失败：{response}")

    def _verify(self, ops: List[Dict[str, Any]]):
        """对每个 op 的 path 读一次 ctrl，检查 type 与规格一致。"""
        verified: Dict[str, Any] = {}
        for op in ops:
            path = op["path"]
            expected = op["type"]
            entry: Dict[str, Any] = {"expected": expected, "actual": None, "ok": False}
            response = yield ("edit_metadata", {"ref": path, "type": "get"})
            if self._is_error(response):
                entry["error"] = response
            else:
                try:
                    ctrl = json.loads(response)
                except ValueError as e:
                    entry["error"] = f"ctrl 不是合法 JSON：{e}"
                else:
                    actual = ctrl.get("type") if isinstance(ctrl, dict) else None
                    entry["actual"] = actual
                    entry["ok"] = (actual == expected)
            verified[path] = entry
        return verified

    # ------------------------------------------------------------------
    # 小助手
    # ------------------------------------------------------------------

    @staticmethod
    def _exists(ref: str):
        response = yield ("memory_read", {"ref": ref})
        return not NodeBuilderPLM._is_error(response)

    @staticmethod
    def _ancestors(path: str) -> List[str]:
        """返回根到父的路径链（含根），不含目标本身。"""
        parts = path[1:].split(".")
        root = "$" + parts[0]
        return [root] + [
            root + "." + ".".join(parts[1:index]) for index in range(2, len(parts))
        ]

    @staticmethod
    def _parent_of(path: str) -> str:
        return path.rsplit(".", 1)[0]

    @staticmethod
    def _key_of(path: str) -> str:
        return path.rsplit(".", 1)[1]

    @staticmethod
    def _is_error(response: Any) -> bool:
        return isinstance(response, str) and response.startswith(_ERROR_PREFIXES)

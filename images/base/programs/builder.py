"""构造器命令程序（`plm.command`）：把创建内存节点的多步指令打包成一次请求。

这个程序不是 AI，是内存树的确定性构造服务。作者把一份 JSON 规格作为命令交给
它，它按「探测 → 构造/更新 → 直通指令 → 验证」的顺序逐个调用注入的工具，
最后返回一份 JSON 摘要。一次规格可以创建 settingup/python 程序节点、handler
节点或 para 节点，也可以顺带执行打包指令。

命令（用户消息）是 JSON 对象：

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
结果原样收进摘要（可用的工具是本命令程序注入的工具集合，见 `_CALL_TOOLS`）。

错误不抛给 Core：非法规格直接返回 `错误: builder: ...` 且不产生工具调用；
构造期单点失败记进摘要的 errors，继续做其余步骤。
"""

# 本文件就是 plm.command 的 system 消息：运行时把整份源码编译进受限命名空间，
# 每条命令调用一次 handle(cmd)。沙箱只注入 math/json/re 与工具函数，不能 import；
# 工具函数都是生成器，必须 `yield from` 才会挂起并把响应送回来。

# 构造器自己能直通调用的工具：plm.command 注入的命令执行子集。
_CALL_TOOLS = frozenset({
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
    "send_instruction",
    "close_conversation",
})

_NODE_TYPES = ("settingup", "python", "handler", "para")

_FIELD_KEYS = {
    "settingup": frozenset({"name", "content", "signature"}),
    "python": frozenset({"name", "content", "signature"}),
    "handler": frozenset({"name", "content", "condition"}),
    "para": frozenset({"fields"}),
}

_ERROR_PREFIXES = ("Error:", "Illegal:")
_ERROR_HEAD = "错误: builder: "


def _error(message):
    return _ERROR_HEAD + message


def _is_error(response):
    return isinstance(response, str) and response.startswith(_ERROR_PREFIXES)


def _exists(ref):
    response = yield from memory_read(ref)
    return not _is_error(response)


def _ancestors(path):
    """返回根到父的路径链（含根），不含目标本身。"""
    parts = path[1:].split(".")
    root = "$" + parts[0]
    return [root] + [
        root + "." + ".".join(parts[1:index]) for index in range(2, len(parts))
    ]


def _parent_of(path):
    return path.rsplit(".", 1)[0]


def _key_of(path):
    return path.rsplit(".", 1)[1]


def _path_segments(path, subject):
    if not isinstance(path, str) or not path:
        raise ValueError(subject + " 必须是非空字符串")
    if not path.startswith("$"):
        raise ValueError(subject + " 必须以 $ 开头，得到 " + repr(path))
    parts = path[1:].split(".")
    if any(part == "" for part in parts):
        raise ValueError(subject + " 不能有空段，得到 " + repr(path))
    if len(parts) < 2:
        raise ValueError(subject + " 至少要写到根下第一层（不能是根本身），得到 " + repr(path))
    return parts


def _scalar_text(index, key, value):
    """para 字段值转成 str 子节点；非字符串按 JSON 字面量转换（0.2 → "0.2"）。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(
        "ops[" + str(index) + "].fields[" + repr(key) + "] 必须是字符串/数值/布尔"
    )


def _parse_program_fields(index, raw):
    fields = []
    name = raw.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise ValueError("ops[" + str(index) + "].name 必须是字符串")
        fields.append(("name", name))
    content = raw.get("content")
    if not isinstance(content, str):
        raise ValueError("ops[" + str(index) + "].content 必须是字符串（必填）")
    fields.append(("content", content))
    signature = raw.get("signature")
    if signature is not None:
        if not isinstance(signature, str):
            raise ValueError("ops[" + str(index) + "].signature 必须是字符串")
        fields.append(("signature", signature))
    return fields


def _parse_handler_fields(index, raw):
    fields = []
    name = raw.get("name")
    if name is not None:
        if not isinstance(name, str):
            raise ValueError("ops[" + str(index) + "].name 必须是字符串")
        fields.append(("name", name))
    content = raw.get("content")
    if not isinstance(content, str):
        raise ValueError("ops[" + str(index) + "].content 必须是字符串（必填）")
    fields.append(("content", content))
    if "condition" not in raw:
        raise ValueError("ops[" + str(index) + "].condition 必填（JSON 字符串或对象）")
    condition = raw["condition"]
    if isinstance(condition, str):
        try:
            json.loads(condition)
        except ValueError as e:
            raise ValueError(
                "ops[" + str(index) + "].condition 不是合法 JSON：" + str(e)
            )
        condition_text = condition
    elif isinstance(condition, (dict, list)):
        condition_text = json.dumps(condition, ensure_ascii=False)
    else:
        raise ValueError(
            "ops[" + str(index) + "].condition 必须是 JSON 字符串/对象/数组"
        )
    fields.append(("condition", condition_text))
    return fields


def _parse_para_fields(index, raw):
    values = raw.get("fields")
    if not isinstance(values, dict):
        raise ValueError("ops[" + str(index) + "].fields 必须是对象（键 → 值），必填")
    fields = []
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                "ops[" + str(index) + "].fields 的键必须是非空字符串，得到 " + repr(key)
            )
        if "." in key:
            raise ValueError(
                "ops[" + str(index) + "].fields 的键不能包含点号（点号是路径分隔符）："
                + repr(key)
            )
        fields.append((key, _scalar_text(index, key, value)))
    return fields


def _parse_op(index, raw):
    subject = "ops[" + str(index) + "]"
    if not isinstance(raw, dict):
        raise ValueError(subject + " 必须是对象")
    node_type = raw.get("type")
    if node_type not in _NODE_TYPES:
        raise ValueError(
            subject + ".type 必须是 " + repr(list(_NODE_TYPES)) + " 之一，得到 "
            + repr(node_type)
        )
    unknown = sorted(set(raw) - {"type", "path", "meta"} - _FIELD_KEYS[node_type])
    if unknown:
        raise ValueError(
            subject + "（type=" + str(node_type) + "）含未知字段：" + repr(unknown)
        )
    path = raw.get("path")
    _path_segments(path, subject + ".path")
    meta = raw.get("meta")
    if meta is not None and not isinstance(meta, str):
        raise ValueError(subject + ".meta 必须是字符串")
    if node_type in ("settingup", "python"):
        fields = _parse_program_fields(index, raw)
    elif node_type == "handler":
        fields = _parse_handler_fields(index, raw)
    else:
        fields = _parse_para_fields(index, raw)
    return {"type": node_type, "path": path, "fields": fields, "meta": meta}


def _parse_call(index, raw):
    subject = "calls[" + str(index) + "]"
    if not isinstance(raw, dict):
        raise ValueError(subject + " 必须是对象")
    unknown = sorted(set(raw) - {"tool", "arguments"})
    if unknown:
        raise ValueError(subject + " 含未知字段：" + repr(unknown))
    tool = raw.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError(subject + ".tool 必须是非空字符串")
    if tool not in _CALL_TOOLS:
        raise ValueError(
            subject + ".tool " + repr(tool) + " 不是本命令程序可调用的指令"
            "（可用：" + repr(sorted(_CALL_TOOLS)) + "）"
        )
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(subject + ".arguments 必须是对象")
    return {"tool": tool, "arguments": arguments}


def _parse_spec(cmd):
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("空规格：用户消息必须是 JSON 对象")
    try:
        spec = json.loads(cmd)
    except ValueError as e:
        raise ValueError("规格不是合法 JSON：" + str(e))
    if not isinstance(spec, dict):
        raise ValueError("规格必须是 JSON 对象")
    unknown = sorted(set(spec) - {"mode", "ops", "calls"})
    if unknown:
        raise ValueError("规格含未知字段：" + repr(unknown))
    mode = spec.get("mode", "create")
    if mode not in ("create", "update"):
        raise ValueError("mode 必须是 create/update，得到 " + repr(mode))
    raw_ops = spec.get("ops", [])
    if not isinstance(raw_ops, list):
        raise ValueError("ops 必须是数组")
    ops = [_parse_op(index, raw) for index, raw in enumerate(raw_ops)]
    raw_calls = spec.get("calls", [])
    if not isinstance(raw_calls, list):
        raise ValueError("calls 必须是数组")
    calls = [_parse_call(index, raw) for index, raw in enumerate(raw_calls)]
    return {"mode": mode, "ops": ops, "calls": calls}


def _probe(ops, mode, exists):
    """探测 path、祖先链与（update 模式下的）字段路径，记录存在性。"""
    for op in ops:
        refs = [op["path"]] + _ancestors(op["path"])
        if mode == "update":
            refs += [op["path"] + "." + field for field, _value in op["fields"]]
        for ref in refs:
            if ref not in exists:
                exists[ref] = yield from _exists(ref)


def _write_fields(op, exists, errors):
    """字段存在就写，缺失先 memory_make 成 str 子节点再写。"""
    path = op["path"]
    for field, value in op["fields"]:
        ref = path + "." + field
        if not exists.get(ref):
            response = yield from memory_make(path, field, "str")
            if _is_error(response):
                errors.append(ref + ": 建字段失败：" + str(response))
                continue
            exists[ref] = True
        response = yield from memory_write(ref, value)
        if _is_error(response):
            errors.append(ref + ": 写字段失败：" + str(response))


def _set_type(op, errors):
    path = op["path"]
    response = yield from edit_metadata(path, "set", key="type", value=op["type"])
    if _is_error(response):
        errors.append(path + ": 写 ctrl.type 失败：" + str(response))


def _write_meta(op, errors):
    if op["meta"] is None:
        return
    response = yield from memory_write(op["path"], op["meta"])
    if _is_error(response):
        errors.append(op["path"] + ": 写描述失败：" + str(response))


def _create_op(op, exists, summary):
    path = op["path"]
    if exists.get(path):
        summary["errors"].append(path + ": create 模式下目标已存在，跳过（不覆盖）")
        summary["skipped"].append(path)
        return
    # 缺失的祖先自顶向下建出来；中途失败就整条 op 放弃。
    for ancestor in _ancestors(path)[1:]:
        if exists.get(ancestor):
            continue
        response = yield from memory_make(
            _parent_of(ancestor), _key_of(ancestor), "dict"
        )
        if _is_error(response):
            summary["errors"].append(ancestor + ": 建祖先失败：" + str(response))
            return
        exists[ancestor] = True
    response = yield from memory_make(_parent_of(path), _key_of(path), "dict")
    if _is_error(response):
        summary["errors"].append(path + ": 建节点失败：" + str(response))
        return
    exists[path] = True
    errors = []
    yield from _write_fields(op, exists, errors)
    yield from _set_type(op, errors)
    yield from _write_meta(op, errors)
    if errors:
        summary["errors"].extend(errors)
    else:
        summary["built"].append(path)


def _update_op(op, exists, summary):
    path = op["path"]
    if not exists.get(path):
        summary["errors"].append(path + ": update 模式下目标不存在，跳过")
        summary["skipped"].append(path)
        return
    errors = []
    yield from _write_fields(op, exists, errors)
    yield from _set_type(op, errors)
    yield from _write_meta(op, errors)
    if errors:
        summary["errors"].extend(errors)
    else:
        summary["updated"].append(path)


def _verify(ops):
    """对每个 op 的 path 读一次 ctrl，检查 type 与规格一致。"""
    verified = {}
    for op in ops:
        path = op["path"]
        expected = op["type"]
        entry = {"expected": expected, "actual": None, "ok": False}
        response = yield from edit_metadata(path, "get")
        if _is_error(response):
            entry["error"] = str(response)
        else:
            try:
                ctrl = json.loads(response)
            except ValueError as e:
                entry["error"] = "ctrl 不是合法 JSON：" + str(e)
            else:
                actual = ctrl.get("type") if isinstance(ctrl, dict) else None
                entry["actual"] = actual
                entry["ok"] = actual == expected
        verified[path] = entry
    return verified


def _run(spec):
    mode = spec["mode"]
    ops = spec["ops"]
    summary = {
        "ok": False,
        "mode": mode,
        "built": [],
        "updated": [],
        "skipped": [],
        "errors": [],
        "calls": [],
        "verified": {},
    }
    exists = {}
    yield from _probe(ops, mode, exists)
    for op in ops:
        if mode == "create":
            yield from _create_op(op, exists, summary)
        else:
            yield from _update_op(op, exists, summary)
    for call in spec["calls"]:
        # 直通通道：原样 yield 工具名与参数，不经过注入函数的参数绑定，
        # 让 Core 的参数校验决定 Error/Illegal（与原构造器一致）。
        response = yield (call["tool"], call["arguments"])
        summary["calls"].append({
            "tool": call["tool"],
            "arguments": call["arguments"],
            "result": response,
        })
        if _is_error(response):
            summary["errors"].append("call " + call["tool"] + ": " + str(response))
    summary["verified"] = yield from _verify(ops)
    summary["ok"] = (
        not summary["errors"]
        and all(entry["ok"] for entry in summary["verified"].values())
    )
    return json.dumps(summary, ensure_ascii=False)


def handle(cmd):
    """命令入口：cmd 是 JSON 规格字符串；返回摘要 JSON 或 `错误: builder: ...`。

    沙箱里不能定义异常类（没有 __build_class__），所以规格错误统一用 ValueError：
    handle 把 ValueError 当作"只给调用者看的规格错误"，其它异常才是内部异常。
    """
    try:
        spec = _parse_spec(cmd)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        return _error("解析异常 " + repr(e))
    try:
        return (yield from _run(spec))
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        return _error("内部异常 " + repr(e))

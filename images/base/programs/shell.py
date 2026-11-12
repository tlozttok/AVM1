"""统一命令入口（`plm.command` 程序）：base 镜像的 shell。

作者把一行命令交给这个对话，它按第一个空格切成「命令名 + 参数」，从
`$MEM.sys.shell.commands.<名字>` 读出注册的 `program` / `para` 引用，把参数原样
写进 `$MEM.sys.shell.args`，再用 `create_sub` 把命令同步分发给对应的命令程序。
亚对话完成时结果自动写回本对话，这里把那条工具响应原样返回（例如 builder 的
JSON 摘要）；shell 自己不做解析、不做包装。

命令：

    help              列出已注册命令、摘要与 shell 调用形式
    help <名字>       先给 shell 调用形式，再把 `help` 分发给命令程序由其回答用法
    <名字> <参数...>  分发；第一个空格之后的整段参数原样传给子命令

注册表只存引用、不存 cid：命令程序按需现建现用。新增命令 = 建一个
`ctrl.type="python"` 的命令程序节点 + 在 `$MEM.sys.shell.commands` 下登记同名
条目（`program`、`para`、可选 `summary` / `usage` 四个 str 子节点），不需要改
这个程序、也不需要重启。约定：
- `summary`：`help` 列表的一行说明；
- `usage`：**在 shell 里怎么调用**的命令行示例（含 `help <名字>` 的头部）；
- 命令程序处理 `help` 输入，回答自己的参数格式（详情）。
shell 自己不猜用法，只把列表、调用形式与 help 转发串起来。

失败一律返回 `错误: shell: ...`（不抛穿 Core）：空命令、未知命令、args 写入失败、
程序内部异常。子命令自己的 Error/Illegal 响应按「工具响应原样返回」直接返回。

注意：plm.command 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""

_COMMANDS_REF = "$MEM.sys.shell.commands"
_ARGS_REF = "$MEM.sys.shell.args"
_ERROR_HEAD = "错误: shell: "
_ERROR_PREFIXES = ("Error:", "Illegal:")


def _error(message):
    return _ERROR_HEAD + message


def _is_error(response):
    return isinstance(response, str) and response.startswith(_ERROR_PREFIXES)


def _unknown(name):
    return _error("未知命令 '" + name + "'（用 help 查看）")


def _split(cmd):
    """切出（命令名, 参数）。空命令抛 ValueError；参数保留第一个空格后的原样。"""
    if not isinstance(cmd, str):
        raise ValueError("空命令")
    # 只去掉行首空白：行尾空白也属于参数，整段原样传给子命令。
    text = cmd.lstrip()
    if not text:
        raise ValueError("空命令")
    name, _, args = text.partition(" ")
    return name, args


def _read_str(ref):
    """读一个 str 节点；读取失败（Error/Illegal）返回 None。"""
    response = yield from memory_read(ref)
    if _is_error(response):
        return None
    return str(response)


def _registered(name):
    """返回 (program, para) 引用；读不到或内容为空时返回 None。"""
    program = yield from _read_str(_COMMANDS_REF + "." + name + ".program")
    para = yield from _read_str(_COMMANDS_REF + "." + name + ".para")
    if program is None or para is None:
        return None
    if not program.strip() or not para.strip():
        return None
    return program, para


def _command_names():
    """从 `$MEM.sys.shell.commands` 的展示文本里取已注册命令名。

    内存读取给 dict 的文本表示是 `dict[keys=['a', 'b'],metadata=...]`；命令名是
    内存键（不含点号，否则无法作为路径），这里只解析 keys 段。名字里含引号的
    极端命令名会列不出来（这种名字不适合当命令）；`help <名字>` 直接按路径读，
    不受影响。
    """
    listing = yield from memory_read(_COMMANDS_REF)
    text = str(listing)
    if _is_error(text):
        return []
    match = re.search(r"dict\[keys=\[(.*?)\],metadata=", text, re.S)
    if match is None:
        return []
    body = match.group(1)
    pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", body)
    return [first or second for first, second in pairs]


def _help(args):
    """内建 help：列表读注册表摘要；详情把 `help` 分发给命令程序自己回答。"""
    name = args.strip()
    if not name:
        names = yield from _command_names()
        if not names:
            return "已注册命令：（无）"
        lines = ["命令格式：<名字> <参数...>（第一个空格后的内容原样传给命令程序）",
                 "已注册命令（" + str(len(names)) + "）："]
        for command in names:
            summary = yield from _read_str(
                _COMMANDS_REF + "." + command + ".summary"
            )
            usage = yield from _read_str(
                _COMMANDS_REF + "." + command + ".usage"
            )
            extra = summary.strip() if summary is not None else ""
            lines.append("  " + command + ("  " + extra if extra else ""))
            if usage is not None and usage.strip():
                lines.append("    调用：" + usage.strip())
        lines.append("用 help <名字> 查看详细用法")
        return "\n".join(lines)
    command = name.split()[0]
    refs = yield from _registered(command)
    if refs is None:
        return _unknown(command)
    program, para = refs
    usage = yield from _read_str(_COMMANDS_REF + "." + command + ".usage")
    response = yield from memory_write(_ARGS_REF, "help")
    if _is_error(response):
        return _error("写入参数失败：" + str(response))
    result = yield from create_sub(program, _ARGS_REF, para)
    head = ""
    if usage is not None and usage.strip():
        head = "在 shell 里调用：" + usage.strip() + "\n\n"
    return head + str(result)


def _dispatch(name, args):
    """按注册表同步分发：写参数 → create_sub → 原样返回亚对话结果。"""
    refs = yield from _registered(name)
    if refs is None:
        return _unknown(name)
    program, para = refs
    response = yield from memory_write(_ARGS_REF, args)
    if _is_error(response):
        return _error("写入参数失败：" + str(response))
    result = yield from create_sub(program, _ARGS_REF, para)
    return str(result)


def handle(cmd):
    """命令入口：cmd 是整行命令字符串；返回子命令输出或 `错误: shell: ...`。"""
    try:
        name, args = _split(cmd)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        return _error("内部异常 " + repr(e))
    try:
        if name == "help":
            return (yield from _help(args))
        return (yield from _dispatch(name, args))
    except Exception as e:
        return _error("内部异常 " + repr(e))

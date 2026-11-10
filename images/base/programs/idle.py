"""base 镜像的空闲处理程序（guest `idle` 角色，`plm.python`）。

内核在"活跃为空、就绪为空、无待派发中断"时向 idle 角色挂中断。程序做三件事：

1. 把 `$MEM.sys.idle.state` 从 `"waiting"` 改回 `"active"`（OS 自己负责离开等待态）；
2. 看 `$MEM.sys.inbox` 有没有 `status == "new"` 的输入；
3. 有输入就按名字找到服务对话 `panel_shell`（cid 不持久化，运行期查
   `$MEM.conversations`），发一条 `yield from panel_shell()` 唤醒它；
   没有输入就写 `$MEM.sys.idle.state = "waiting"` 后结束本轮，交给内核等待宿主中断。

输入刚由保留对话写入、而 idle 先被唤醒的竞态由宿主处理（`/api/input` 先挂 idle
再挂 operator，保证面板先写完 inbox）。这里只按当前内存做判断。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""


def _ok(response):
    text = str(response)
    return not (text.startswith("Error:") or text.startswith("Illegal:"))


def _list_len(text):
    match = re.search(r"len=(\d+)", str(text))
    if match is None:
        return 0
    return int(match.group(1))


def _conversations():
    listing = yield from memory_read("$MEM.conversations")
    rows = []
    state = ""
    for line in str(listing).splitlines():
        text = line.strip()
        if text in ("active:", "ready:", "dormant:", "finished:"):
            state = text[:-1]
            continue
        match = re.search(r"cid=(\d+)\s+(\S+)", text)
        if match is not None:
            rows.append((int(match.group(1)), match.group(2), state))
    return rows


def _find_cid(name):
    rows = yield from _conversations()
    for cid, identity, state in rows:
        if identity == name and state != "finished":
            return cid
    return None


def _pending_inputs():
    summary = yield from memory_read("$MEM.sys.inbox")
    count = _list_len(summary)
    pending = 0
    for index in range(count):
        raw = yield from memory_read("$MEM.sys.inbox." + str(index))
        text = str(raw)
        if text.startswith("Error:"):
            continue
        try:
            entry = json.loads(text)
        except Exception:
            continue
        if isinstance(entry, dict) and entry.get("status") == "new":
            pending += 1
    return pending


def idle():
    yield from memory_write("$MEM.sys.idle.state", "active")

    pending = yield from _pending_inputs()
    if pending > 0:
        shell = yield from _find_cid("panel_shell")
        if shell is not None:
            response = yield from send_instruction(shell, "yield from panel_shell()")
            if _ok(response):
                return
            notice = "idle: 唤醒 panel_shell 失败: " + str(response)
        else:
            notice = ("idle: inbox 有 " + str(pending)
                      + " 条新输入，但 panel_shell 不存在")
        yield from memory_write("$MEM.sys.outbox.-1", notice)
        yield from memory_write("$MEM.outputs.out", notice)

    # 没有可交给服务对话的工作：声明"在等待"，由内核停在安全点等宿主中断。
    yield from memory_write("$MEM.sys.idle.state", "waiting")

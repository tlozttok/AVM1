"""base 镜像的全局设备事件钟 B（`plm.python`，由 OS 监督程序启停）。

与 clock_a 互相激活：clock_b 不计数，只把下一次激活投回 clock_a，形成两个
PLM 的乒乓。这样"设备事件钟"是两只会休眠的普通对话，不是内核周期定时器；
supervisor 写 `$MEM.sys.clock.state = "stopped"` 后两边都不再投递，钟就停了。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""


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


def clock_b():
    state = str((yield from memory_read("$MEM.sys.clock.state"))).strip()
    if state != "running":
        return
    peer = yield from _find_cid("clock_a")
    if peer is not None:
        yield from send_instruction(peer, "yield from clock_a()")

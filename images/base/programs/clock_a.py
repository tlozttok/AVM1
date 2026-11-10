"""base 镜像的全局设备事件钟 A（`plm.python`，由 OS 监督程序启停）。

两个互相激活的 PLM 做乒乓：clock_a 每跑一次算一个逻辑 tick 并写 `$MEM.sys.clock`，
然后把下一次激活投给 clock_b；clock_b 只把激活投回 clock_a。tick 是普通内存
写入（进日志、可重放），逻辑时间与关机超时都建立在这套 tick 上。

- 启动：supervisor 写 `$MEM.sys.clock.state = "running"`、设 target，然后唤醒
  clock_a；clock_a/clock_b 自身不主动启动。
- 超时：tick 跨过 target 时把 `shutdown_timeout` 通知投给 supervisor（按运行期
  视图找 `guest-shutdown`），并把 target 后移一个 interval；发多少次都不等于
  断电，只有 supervisor 写 off 才会关机。
- 停止：任何一方发现 `state != "running"` 就不再投递下一次激活，两个 PLM 休眠。

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


def _int_at(ref, default=0):
    raw = yield from memory_read(ref)
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def clock_a():
    state = str((yield from memory_read("$MEM.sys.clock.state"))).strip()
    if state != "running":
        return

    tick = (yield from _int_at("$MEM.sys.clock.tick", 0)) + 1
    yield from memory_write("$MEM.sys.clock.tick", str(tick))

    target = yield from _int_at("$MEM.sys.clock.target", 0)
    interval = yield from _int_at("$MEM.sys.clock.interval", 5)
    if interval < 1:
        interval = 1
    if target <= 0:
        target = tick + interval
        yield from memory_write("$MEM.sys.clock.target", str(target))

    if tick >= target:
        count = (yield from _int_at("$MEM.sys.clock.timeouts", 0)) + 1
        yield from memory_write("$MEM.sys.clock.timeouts", str(count))
        yield from memory_write("$MEM.sys.clock.target", str(tick + interval))
        notice = "clock: 设备事件钟超时 #" + str(count) + "（tick=" + str(tick) + "）"
        yield from memory_write("$MEM.sys.outbox.-1", notice)
        supervisor = yield from _find_cid("guest-shutdown")
        if supervisor is not None:
            yield from send_instruction(supervisor, "yield from shutdown()")

    state = str((yield from memory_read("$MEM.sys.clock.state"))).strip()
    if state != "running":
        return
    peer = yield from _find_cid("clock_b")
    if peer is not None:
        yield from send_instruction(peer, "yield from clock_b()")

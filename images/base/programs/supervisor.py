"""base 镜像的 OS 监督程序（guest `shutdown` 角色，`plm.python`）。

宿主在安全点看到 `$MEM.sys.power.request == "shutdown"` 后，用 `shutdown` 中断
把它唤醒（入口 `yield from shutdown()`）。它判断"应用是否全部收尾"：

- 收尾 = 对话结束本轮、不再排工作、进入休眠。运行期视图 `$MEM.conversations`
  里状态为 active / ready 的**非 OS 对话**都算未收尾；OS 自己的对话
  （init、guest-boot、guest-idle、guest-shutdown、panel_shell、clock_a、clock_b）
  不是应用，不阻塞关机。
- 有未收尾对话：确保设备事件钟在跑（两个互相激活的 PLM，超时通知会再唤醒本程序），
  往 outbox/outputs 写一条等待/超时记录，然后结束本轮；**不写 off**。
- 全部收尾：停钟 → 写 `$MEM.sys.boot.state = "clean"` → 写
  `$MEM.sys.power.state = "off"` → 结束本轮。宿主看到 off 后写快照并退出。

OS 软件缺席或一直收不尾时，关机停在挂起；操作员可以走宿主的强制退出。
超时次数、间隔由 `$MEM.sys.clock.interval` 等 OS 约定节点配置；发多少次超时都
不会自动断电，只有这里写 off 才会。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""

_OS_NAMES = (
    "init", "guest-boot", "guest-idle", "guest-shutdown",
    "panel_shell", "clock_a", "clock_b",
)


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


def _start_clock():
    tick = yield from _int_at("$MEM.sys.clock.tick", 0)
    interval = yield from _int_at("$MEM.sys.clock.interval", 5)
    if interval < 1:
        interval = 1
    yield from memory_write("$MEM.sys.clock.timeouts", "0")
    yield from memory_write("$MEM.sys.clock.target", str(tick + interval))
    yield from memory_write("$MEM.sys.clock.state", "running")
    clock_a = yield from _find_cid("clock_a")
    if clock_a is not None:
        response = yield from send_instruction(clock_a, "yield from clock_a()")
        if not str(response).startswith(("Error:", "Illegal:")):
            return
    # clock_a 缺席：状态留着 running，下一轮超时唤醒本程序时会再尝试。
    print("shutdown: 设备事件钟无法启动（clock_a 不存在或投递失败）")


def shutdown():
    request = yield from memory_read("$MEM.sys.power.request")
    if str(request).strip() != "shutdown":
        # 不是关机中断的内容：不执行断电语义。
        return

    rows = yield from _conversations()
    waiting = []
    for cid, identity, state in rows:
        if identity in _OS_NAMES:
            continue
        if state in ("active", "ready"):
            waiting.append(identity + "(cid=" + str(cid) + ")")

    clock_state = str((yield from memory_read("$MEM.sys.clock.state"))).strip()

    if waiting:
        if clock_state != "running":
            yield from _start_clock()
            notice = ("shutdown: 等待 " + str(len(waiting)) + " 个对话收尾: "
                      + ",".join(waiting))
        else:
            timeouts = yield from _int_at("$MEM.sys.clock.timeouts", 0)
            notice = ("shutdown: 超时 #" + str(timeouts) + "，仍有 "
                      + str(len(waiting)) + " 个对话未收尾: " + ",".join(waiting))
        yield from memory_write("$MEM.sys.outbox.-1", notice)
        yield from memory_write("$MEM.outputs.out", notice)
        return

    # 全部收尾：停钟 → 写 clean → 记录一条总结 → 最后写断电标记。off 必须最后写：
    # 宿主在下一个安全点看到 off 就退出，这之后的指令不会再有执行机会。
    yield from memory_write("$MEM.sys.clock.state", "stopped")
    yield from memory_write("$MEM.sys.clock.target", "0")
    yield from memory_write("$MEM.sys.boot.state", "clean")
    summary = "shutdown: 全部对话已收尾，写 clean 并断电"
    yield from memory_write("$MEM.sys.outbox.-1", summary)
    yield from memory_write("$MEM.outputs.out", summary)
    yield from memory_write("$MEM.sys.power.state", "off")

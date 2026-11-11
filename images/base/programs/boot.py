"""base 镜像的引导程序（`plm.python`）。

镜像 `init` 入口是 `yield from init()`：只保留一条空回合，真正的引导在 guest
`boot` 角色里（`meta.guest_programs.boot`，内核对话入口是 `yield from boot()`）。

`boot()` 做六件事，然后结束本轮（对话休眠）：

1. 把 `$MEM.sys.boot.state` 从旧值改成 `"running"`，并把旧值记到 `prior`；
2. 递增 `$MEM.sys.boot.id`（每次上电/恢复一次）；
3. 把 `$MEM.sys.power.state` 复位成 `"on"`（resume 恢复出来的 `off` 是上一次
   关机的存量值，不复位的话 supervisor 再写 off 就没有"非 off → off"的边沿）；
4. 按 `$MEM.sys.boot.manifest` 创建缺失的常驻对话，名字 → cid 写 registry；
5. 创建缺失的 OS 服务对话 `clock_a` / `clock_b`；服务 cid 不写持久 MEM，运行期由
   OS 程序按名字查 `$MEM.conversations`（运行期视图）；
6. 写 `$MEM.sys.boot.report`（boot id、上次状态、宿主给的 last_exit、本次新建的
   服务），并往 `$MEM.sys.outbox` 与 `$MEM.outputs` 写一条启动记录。

resume 时宿主用中断唤醒 boot，把 last_exit 放进入口 payload（形如
`yield from boot('normal')`）；冷启动用默认 `"unknown"`。正常关机由 supervisor
写 `clean`，boot 只读旧值、不写 clean。空闲与关机都走中断：没有空闲对话可跑时
面板对话被 `idle` 中断唤醒并阻塞等待操作员，关机由宿主挂 `shutdown` 中断唤醒
supervisor，boot 不参与这两条路径。

这个节点本身是内存数据：作者在系统内改写它，下一次启动就运行新内容。
清单条目：`{"name": "...", "program": "$MEM...", "para": "$MEM..."}`，
`program` 指向 settingup/python 程序节点，`para` 指向 para 节点。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""


def _ok(response):
    text = str(response)
    return not (text.startswith("Error:") or text.startswith("Illegal:"))


def _int_at(ref, default=0):
    raw = yield from memory_read(ref)
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _conversations():
    """运行期视图：当前对话的 (cid, 名字, 状态)。名字是对话 identity。"""
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


def _live_cid(cid):
    rows = yield from _conversations()
    for row_cid, _identity, state in rows:
        if row_cid == cid and state != "finished":
            return True
    return False


def _ensure_service(name, program_ref, created, existing):
    cid = yield from _find_cid(name)
    if cid is not None:
        existing.append(name)
        return
    response = yield from create_cmd(program_ref, "$MEM.para.python")
    if _ok(response):
        created.append(name)
    else:
        print("boot: 创建服务 " + name + " 失败: " + str(response))


def init():
    """最小的镜像 init 入口；真正的引导由 guest boot 角色执行。"""
    yield from ()


def boot(last_exit="unknown"):
    prior = yield from memory_read("$MEM.sys.boot.state")
    yield from memory_write("$MEM.sys.boot.prior", str(prior))
    yield from memory_write("$MEM.sys.boot.state", "running")
    boot_id = (yield from _int_at("$MEM.sys.boot.id", 0)) + 1
    yield from memory_write("$MEM.sys.boot.id", str(boot_id))

    # 新上电：复位断电寄存器。resume 恢复出来的 state=off 是上一次关机的存量值，
    # 不复位的话 supervisor 再写 off 就没有"非 off → off"的边沿，第二次关机不生效。
    yield from memory_write("$MEM.sys.power.state", "on")

    # 服务对话只建一次：先按运行期视图找，找到就不再建（resume 时 cid 已在快照里）。
    os_created = []
    os_existing = []
    for name, program_ref in (
        ("clock_a", "$MEM.sys.programs.clock_a"),
        ("clock_b", "$MEM.sys.programs.clock_b"),
    ):
        yield from _ensure_service(name, program_ref, os_created, os_existing)

    raw = str((yield from memory_read("$MEM.sys.boot.manifest"))).strip()
    specs = []
    manifest_error = ""
    if raw:
        try:
            specs = json.loads(raw)
        except Exception as e:
            manifest_error = "manifest 不是合法 JSON: " + str(e)
            specs = []
    if not isinstance(specs, list):
        manifest_error = manifest_error or "manifest 必须是 JSON 数组"
        specs = []

    created = []
    skipped = []
    failed = []
    for spec in specs:
        if not isinstance(spec, dict):
            failed.append("条目不是对象")
            continue
        name = spec.get("name")
        program = spec.get("program")
        para = spec.get("para")
        if not isinstance(name, str) or not name:
            failed.append("条目缺少 name: " + str(spec))
            continue
        if not isinstance(program, str) or not isinstance(para, str):
            failed.append(name + ": program/para 必须是引用字符串")
            continue
        registered = yield from memory_read("$MEM.sys.registry." + name)
        if _ok(registered):
            cid = str(registered).strip()
            try:
                live = bool(cid) and (yield from _live_cid(int(cid)))
            except Exception:
                live = False
            if live:
                skipped.append(name)
                continue
        resp = yield from create_cmd(program, para)
        if not _ok(resp):
            failed.append(name + ": " + str(resp))
            continue
        match = re.search(r"cid=(\d+)", str(resp))
        if match is None:
            failed.append(name + ": 回执里没有 cid: " + str(resp))
            continue
        cid = match.group(1)
        yield from memory_make("$MEM.sys.registry", name, "str")  # 已存在时返回 Illegal，由下面覆盖
        write_resp = yield from memory_write("$MEM.sys.registry." + name, cid)
        if not _ok(write_resp):
            failed.append(name + ": 写 registry 失败: " + str(write_resp))
            continue
        created.append(name)

    report = (
        "boot_id=" + str(boot_id) + " prior=" + str(prior)
        + " last_exit=" + str(last_exit)
        + " services=" + (",".join(os_existing + os_created) or "-")
        + " manifest_created=" + str(len(created))
        + " manifest_skipped=" + str(len(skipped))
        + " manifest_failed=" + str(len(failed))
    )
    yield from memory_write("$MEM.sys.boot.report", report)
    summary = ("boot: 创建 " + str(len(created)) + " 个常驻对话，"
               + str(len(failed)) + " 个失败；上次状态=" + str(prior)
               + "；last_exit=" + str(last_exit))
    if manifest_error:
        summary = summary + "；" + manifest_error
    if os_created:
        summary = summary + "；OS 服务=" + ",".join(os_created)
    yield from memory_write("$MEM.sys.outbox.-1", summary)
    yield from memory_write("$MEM.outputs.out", summary)
    return

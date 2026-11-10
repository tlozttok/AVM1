"""base 镜像的最小引导程序（`plm.python`）。

镜像 `init` 指向 `$MEM.sys.programs.boot`，user 是 `boot()`。
它只做四件事，然后结束本轮（对话休眠）：

1. 把 `$MEM.sys.boot.state` 从旧值改成 `"running"`，并把旧值记到 `prior`；
2. 读 `$MEM.sys.boot.manifest`（JSON 字符串），按清单创建缺失的常驻对话；
3. 把 `名字 → cid` 写进 `$MEM.sys.registry`；
4. 往 `$MEM.sys.outbox` 与 `$MEM.outputs` 写一条启动记录。

这个节点本身是内存数据：作者在系统内改写它，下一次启动就运行新内容。
清单条目：`{"name": "...", "program": "$MEM...", "para": "$MEM..."}`，
`program` 指向 settingup/python 程序节点，`para` 指向 para 节点。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""


def _ok(response):
    text = str(response)
    return not (text.startswith("Error:") or text.startswith("Illegal:"))


def boot():
    prior = yield from memory_read("$MEM.sys.boot.state")
    yield from memory_write("$MEM.sys.boot.prior", str(prior))
    yield from memory_write("$MEM.sys.boot.state", "running")

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

    summary = ("boot: 创建 " + str(len(created)) + " 个常驻对话，"
               + str(len(failed)) + " 个失败；上次状态=" + str(prior))
    if manifest_error:
        summary = summary + "；" + manifest_error
    yield from memory_write("$MEM.sys.outbox.-1", summary)
    yield from memory_write("$MEM.outputs.out", summary)
    return

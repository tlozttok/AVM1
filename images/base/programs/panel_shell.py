"""base 镜像的面板服务对话（`plm.python`，由 guest boot 在冷启动时创建）。

它不是内核的保留对话：是 OS 在系统内创建的普通服务对话，名字 `panel_shell`
由 boot 的 `create_cmd` 写进对话身份。idle 处理程序按运行期视图找到它，投递
`yield from panel_shell()` 唤醒；它消费 `$MEM.sys.inbox` 里 `status == "new"`
的条目，把结果追加到 `$MEM.sys.outbox` 与 `$MEM.outputs`，再把条目标记完成。

不做阻塞读：没有输入就结束本轮，由 idle 程序写"等待"并交给内核。

注意：plm.python 的沙箱禁止 import；`json` / `re` / `math` 由运行时注入。
"""


def _list_len(text):
    match = re.search(r"len=(\d+)", str(text))
    if match is None:
        return 0
    return int(match.group(1))


def panel_shell():
    summary = yield from memory_read("$MEM.sys.inbox")
    count = _list_len(summary)
    processed = 0
    for index in range(count):
        raw = yield from memory_read("$MEM.sys.inbox." + str(index))
        text = str(raw)
        if text.startswith("Error:"):
            continue
        try:
            entry = json.loads(text)
        except Exception:
            continue
        if not isinstance(entry, dict) or entry.get("status") != "new":
            continue
        content = entry.get("text")
        if not isinstance(content, str):
            content = str(content)
        line = "panel_shell: " + content
        entry["status"] = "done"
        entry["output"] = line
        yield from memory_write(
            "$MEM.sys.inbox." + str(index), json.dumps(entry, ensure_ascii=False)
        )
        yield from memory_write("$MEM.sys.outbox.-1", line)
        yield from memory_write("$MEM.outputs.out", line)
        processed += 1

    if processed:
        yield from memory_write(
            "$MEM.sys.outbox.-1", "panel_shell: 本轮处理 " + str(processed) + " 条输入"
        )

"""AVM 讨论社区的面板 shell（`plm.python` 程序）。

它挂在镜像 0 号对话上：循环读 `$MEM.inputs.next`（控制台托管时来自面板输入框，
终端模式下来自 stdin），把每条输入当作一条 `plm.community` 请求，经 `create_sub`
交给社区服务，并把回执写进 `$MEM.outputs.out`。

输入格式：
- 第一行是命令，其余行是正文；
- 单行输入可以用 " | " 分隔命令与正文，例如：
  NEW code insight 标题 | 正文
- exit / quit / stop 结束 shell（控制台随之可以退出）。

这个 shell 是验收脚手架：AI 角色与 conductor 接入后，0 号对话会换成正式入口。
"""

WELCOME = """\
AVM 讨论社区 · 面板 shell（验收脚手架）
输入命令，回车后执行；回执写到 $MEM.outputs，也可在内存树里看 $MEM.forum。
  BOARD design            板块视图
  POST P000001            单帖视图
  NEW code insight 标题 | 正文
  COMMENT P000001 | 评论内容
  REPLY C000001 | 回复内容
  QUOTE P000001 标题 | 正文
  MERGE P000001 P000002 / DELETE C000001 理由 / TAG / SOLVE / MAINT / CHECK
  HELP                    完整命令说明
exit 结束 shell。
"""


def start():
    memory_write("$MEM.outputs.out", WELCOME)
    while True:
        raw = memory_read("$MEM.inputs.next")
        text = raw if isinstance(raw, str) else str(raw)
        text = text.strip()
        if text == "":
            memory_write("$MEM.outputs.out", "（空输入 / EOF：面板 shell 结束）")
            return
        if text in ("exit", "quit", "stop"):
            memory_write("$MEM.outputs.out", "面板 shell 结束。")
            return
        lines = text.split("\n")
        head = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if body == "" and " | " in head:
            left, _, right = head.partition(" | ")
            head = left.strip()
            body = right.strip()
        author = ""
        if head.startswith("@"):
            at, _, rest = head.partition(" ")
            author = at[1:].strip()
            head = rest.strip()
        if head == "":
            memory_write("$MEM.outputs.out", "（只有正文没有命令，忽略）")
            continue
        request = head + ("\n" + body if body else "")
        if author != "":
            request = "AS " + author + "\n" + request
        memory_write("$MEM.run.svc_request.panel", request)
        reply = create_sub(
            "$MEM.programs.svc_panel", "$MEM.run.svc_request.panel", "$MEM.para.svc"
        )
        memory_write("$MEM.outputs.out", "> " + head)
        memory_write("$MEM.outputs.out", reply if isinstance(reply, str) else str(reply))

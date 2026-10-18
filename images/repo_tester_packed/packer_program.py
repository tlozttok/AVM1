"""repo_tester_packed 的打包器程序（plm.python）。

这份源码的正文就是镜像里 `$MEM.packer` 节点的 content：0 号对话是 plm.python，system 是它，
user 消息是文件列表（`images/repo_tester_packed/file_list.txt`），也就是最后一行那句调用。

流程：逐页经 `$MEM.AVMRepo` 读文档 → 拼系统提示词 → 写 `$MEM.system.content` →
`create_cmd` 出测试对话 → `send_instruction` 把 `$MEM.user` 投给它 → 写打包报告 → 关闭自己。

沙箱约束（见 avm/sandbox.py）：禁 import / open / 双下划线属性；工具函数由 plm.python 注入，
名字与参数同 9 条指令。写入类调用在重放时按记录返回，不会重复执行。
"""

SKELETON = """你是这个系统中的一个对话。系统提供内存树和一组工具：memory_read、memory_write、memory_make、edit_metadata、create_cmd、create_sub、send_instruction、close_conversation、listen_event。你是这个系统的测试者：在用户指导下，依据仓库里的实际代码，对这个系统做测试。内存里有一个只读设备 $MEM.AVMRepo，它自己能说明怎么用它读文件：用 memory_read 读 $MEM.AVMRepo，照它写的用法用它，你就能读到 AVM 的源码、规范与测试。发现代码与行为不一致、代码没覆盖的边界、读不懂或设备自述没说清的地方，都报给用户。你只读代码、只测系统，不修改代码与镜像。

你的提示词里已经打包了这些仓库文档（启动时组装好，不要再用 $MEM.AVMRepo 读一遍；清单见 $MEM.packed_docs）：{files}。docs/runtime.md 不在打包里，机制细节需要时用 $MEM.AVMRepo 读它。"""

# 打包器自己的 cid：它是镜像的 init（0 号对话）
PACKER_CID = 0


def read_doc(path):
    """经设备逐页读一份文档，只留正文（去掉页头与"下一页/末页"提示）。"""
    page = 0
    parts = []
    while True:
        ref = '$MEM.AVMRepo.read."%s".%d' % (path, page)
        resp = str(memory_read(ref))
        if not resp.startswith("read "):
            raise RuntimeError("读 %s 失败：%s" % (path, resp))
        lines = resp.split("\n")
        parts.append("\n".join(lines[1:-1]))
        if "（末页）" in resp:
            return "\n".join(parts)
        page = page + 1


def pack(*files):
    """组装提示词、建测试对话、启动它，然后关闭打包器自己。"""
    parts = []
    report = []
    for path in files:
        text = read_doc(path)
        parts.append("### " + path)
        parts.append(text)
        report.append("%s（%d 字符）" % (path, len(text)))
    content = SKELETON.replace("{files}", "、".join(files)) + "\n\n" + "\n".join(parts)

    memory_write("$MEM.system.content", content)
    memory_write("$MEM.packed_docs", "\n".join(report))

    created = str(create_cmd("$MEM.system", "$MEM.model_params"))
    tester_cid = int(created.rsplit("=", 1)[1])
    start = str(memory_read("$MEM.user"))
    send_instruction(tester_cid, start)

    memory_write(
        "$MEM.packer_report",
        "已把 %d 份文档（%d 字符）写进 $MEM.system.content，测试对话 cid=%d 已启动，打包器关闭"
        % (len(files), len(content), tester_cid),
    )
    close_conversation(PACKER_CID)

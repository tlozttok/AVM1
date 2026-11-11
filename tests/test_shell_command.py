"""base 镜像 shell（统一命令入口，`plm.command`）测试。

覆盖：冷启动按名字创建 shell 服务；内建 `help` / `help <名字>`；`builder <JSON>`
经 shell 的 `create_sub` 同步分发并返回 JSON 摘要、节点真的建出来；未知命令与
空命令的错误文本；`$MEM.sys.shell.args` 暂存内容；在注册表里新增一条命令后
shell 能分发它（证明入口可扩展）。

全程 MockLMU / 脚本化调度：shell 与 builder 都是确定性 `plm.command` 程序，
不调真实 API。
"""

import json
import os
import re

from avm.core import Core
from avm.image import load_image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = os.path.join(REPO, "images", "base", "base.json")


class NoLMU:
    """base 程序不应调用 LLM；被调用即失败。"""

    def exec(self, conv, para):
        raise AssertionError("shell 测试不应调用 LLM")


def prepared_image(tmp_path, *, mutate=None):
    """复制 base 镜像到 tmp，把设备文件改成绝对路径；可再改 mem。"""
    with open(BASE_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    base_dir = os.path.dirname(BASE_IMAGE)
    for dev in image.get("devices") or []:
        file_path = dev.get("file")
        if isinstance(file_path, str) and not os.path.isabs(file_path):
            dev["file"] = os.path.normpath(os.path.join(base_dir, file_path))
    if mutate is not None:
        mutate(image)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_core(image_path):
    core = load_image(Core(), image_path)
    core.lmu = NoLMU()
    return core


def boot_cold(core):
    """手动推进冷启动：init → boot，不进入 Core.run 的调度循环。"""
    if core._active_cid is None:
        core._pick_next_active()
    for _ in range(500):
        if core._active_cid is None:
            break
        core.advance_conversation()
    assert core._active_cid is None, "冷启动程序没有在合理回合内结束"


def pump(core, limit=3000):
    """把就绪/活跃对话一路推进到全部休眠（含 shell 分发出的亚对话）。"""
    rounds = 0
    while rounds < limit:
        if core._active_cid is None:
            core._pick_next_active()
        if core._active_cid is None:
            return rounds
        core.advance_conversation()
        rounds += 1
    raise AssertionError("调度没有在回合上限内停下")


def shell_conversation(core):
    """按运行期视图 `$MEM.conversations` 里的名字找 shell 服务对话。"""
    listing = str(core.unwrap("$MEM.conversations"))
    match = re.search(r"cid=(\d+)\s+shell\b", listing)
    assert match is not None, "运行期视图里没有 shell 服务（应带 cid）"
    cid = int(match.group(1))
    conv = core.get_conversation(cid)
    assert conv is not None and conv.name == "shell"
    return conv


def send(core, conv, command):
    """向对话投递一条命令并跑到它休眠；返回最后一条 assistant 消息。"""
    conv.append_user_message(command)
    if conv.cid in core._dormant_cids:
        core._dormant_cids.remove(conv.cid)
    core._active_cid = conv.cid
    pump(core)
    assert conv.cid in core._dormant_cids, "命令执行完后对话没有休眠"
    return conv.messages[-1].content


def cold_shell(image_path=BASE_IMAGE):
    core = load_core(image_path)
    boot_cold(core)
    return core, shell_conversation(core)


def test_cold_boot_creates_shell_service():
    core = load_core(BASE_IMAGE)
    boot_cold(core)

    conv = shell_conversation(core)
    assert conv.is_python, "shell 必须是 plm.command 程序对话"
    assert conv.para_ref == "$MEM.para.shell"
    assert conv.cid in core._dormant_cids, "服务对话创建后应休眠等命令"
    # cid 不写持久 MEM：shell 是 OS 服务，运行期按名字查。
    assert dict(core.unwrap("$MEM.sys.registry", for_llm=False)) == {}

    program = core.unwrap("$MEM.sys.programs.shell", for_llm=False)
    assert program.get_ctrl() == {"type": "python"}
    assert program.get("name") == "shell"
    assert program.get("para_ref") == "$MEM.para.shell"
    assert "def handle(cmd)" in program.get("content")
    # 程序正文就是 system 消息：对话里能读到同一份源码。
    assert conv.messages[0].content == program.get("content")

    para = core.unwrap("$MEM.para.shell", for_llm=False)
    assert para.get_ctrl() == {"type": "para"}
    assert para.get("model") == "plm.command"

    # 分区预置：args 是空 str，commands 预注册 builder 的 program/para 引用。
    assert core.unwrap("$MEM.sys.shell.args", for_llm=False) == ""
    assert core.unwrap(
        "$MEM.sys.shell.commands.builder.program", for_llm=False
    ) == "$MEM.sys.programs.builder"
    assert core.unwrap(
        "$MEM.sys.shell.commands.builder.para", for_llm=False
    ) == "$MEM.para.builder"


def test_help_lists_registered_commands():
    core, shell = cold_shell()
    output = send(core, shell, "help")
    assert "builder" in output
    assert "help <名字>" in output


def test_help_shows_registered_refs():
    core, shell = cold_shell()
    output = send(core, shell, "help builder")
    assert output.startswith("builder:")
    assert "$MEM.sys.programs.builder" in output
    assert "$MEM.para.builder" in output


def test_builder_command_is_dispatched_through_shell():
    core, shell = cold_shell()
    before = set(core._conv_by_cid)
    spec = {
        "mode": "create",
        "ops": [
            {
                "type": "python",
                "path": "$MEM.workshop.hello",
                "name": "hello",
                "content": "def handle(cmd):\n    return cmd\n",
            },
        ],
    }
    command = "builder " + json.dumps(spec, ensure_ascii=False)
    output = send(core, shell, command)

    summary = json.loads(output)
    assert summary["ok"] is True
    assert summary["errors"] == []
    assert summary["built"] == ["$MEM.workshop.hello"]
    node = core.unwrap("$MEM.workshop.hello", for_llm=False)
    assert node.get_ctrl() == {"type": "python"}
    assert core.unwrap("$MEM.workshop.hello.content", for_llm=False).startswith("def handle")
    # 参数整段原样写进暂存位；亚对话用完即删，没有留下新个体。
    assert core.unwrap("$MEM.sys.shell.args", for_llm=False) == command[len("builder "):]
    assert set(core._conv_by_cid) - before == set()


def test_unknown_and_empty_commands_report_errors():
    core, shell = cold_shell()
    before = set(core._conv_by_cid)

    assert send(core, shell, "no_such_command") == (
        "错误: shell: 未知命令 'no_such_command'（用 help 查看）"
    )
    assert send(core, shell, "") == "错误: shell: 空命令"
    assert send(core, shell, "   ") == "错误: shell: 空命令"
    assert set(core._conv_by_cid) == before, "错误路径不应创建亚对话"
    assert core.unwrap("$MEM.sys.shell.args", for_llm=False) == ""


def _add_command(image, name, content):
    """在镜像里加一条命令：程序节点 + para 档 + 注册表登记。"""
    programs = image["mem"]["sys"]["value"]["programs"]["value"]
    paras = image["mem"]["para"]["value"]
    commands = image["mem"]["sys"]["value"]["shell"]["value"]["commands"]["value"]
    paras[name] = {
        "kind": "dict",
        "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "plm.command"}},
    }
    programs[name] = {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": content},
            "para_ref": {"kind": "str", "value": "$MEM.para." + name},
            "name": {"kind": "str", "value": name},
        },
    }
    commands[name] = {
        "kind": "dict",
        "value": {
            "program": {"kind": "str", "value": "$MEM.sys.programs." + name},
            "para": {"kind": "str", "value": "$MEM.para." + name},
        },
    }


def test_registered_command_can_be_added_and_dispatched(tmp_path):
    image = prepared_image(
        tmp_path,
        mutate=lambda img: _add_command(
            img, "echo", 'def handle(cmd):\n    return "echo:" + cmd\n'
        ),
    )
    core, shell = cold_shell(image)

    assert "echo" in send(core, shell, "help")
    refs = send(core, shell, "help echo")
    assert "$MEM.sys.programs.echo" in refs and "$MEM.para.echo" in refs

    output = send(core, shell, "echo a b")
    assert output == "echo:a b"
    assert core.unwrap("$MEM.sys.shell.args", for_llm=False) == "a b"

"""base 母板镜像的冒烟测试（MockLMU / 无 LLM 调用）。

验收（plan-base-image-2026-11-08）：
1. 冷启动：boot 程序写 running、manifest 为空、registry 为空、写启动记录，然后休眠；
2. 清单：manifest 有服务时 boot 程序按清单创建对话并写 registry；
3. 输入落点：面板（操作员对话）写 `$MEM.sys.inbox`，日志可重放；
4. 装载设备：经 loader 装入临时设备，日志重放后设备仍在；
5. 无阻塞：base 的程序里没有 `$MEM.inputs.next` 阻塞读。
"""

import hashlib
import json
import os

from avm.core import Core
from avm.image import load_image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = os.path.join(REPO, "images", "base", "base.json")


class NoLMU:
    """base 的 Python 程序不应调用 LLM；被调用即失败。"""

    def exec(self, conv, para):
        raise AssertionError("base 冒烟不应调用 LLM")


def prepared_image(tmp_path, *, mutate=None):
    """复制 base 镜像到 tmp，把设备文件改成绝对路径；可再改 mem。"""
    with open(BASE_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    for dev in image.get("devices") or []:
        file_path = dev.get("file")
        if isinstance(file_path, str) and not os.path.isabs(file_path):
            dev["file"] = os.path.normpath(
                os.path.join(os.path.dirname(BASE_IMAGE), file_path)
            )
    if mutate is not None:
        mutate(image)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_and_run(image_path):
    core = load_image(Core(), image_path)
    core.lmu = NoLMU()
    core.run()
    return core


def test_cold_boot_writes_state_and_report():
    core = load_image(Core(), BASE_IMAGE)
    core.lmu = NoLMU()
    core.run()

    assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "running"
    assert core.unwrap("$MEM.sys.boot.prior", for_llm=False) == "clean"
    assert dict(core.unwrap("$MEM.sys.registry", for_llm=False)) == {}
    assert list(core.unwrap("$MEM.sys.outbox", for_llm=False))
    assert core._active_cid is None
    assert 0 in core._dormant_cids


def test_manifest_creates_service_and_registry(tmp_path):
    program = {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {"content": {"kind": "str", "value": "def handle(cmd):\n    return 'ok'\n"}},
    }
    command_para = {
        "kind": "dict",
        "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "plm.command"}},
    }
    manifest = json.dumps([
        {"name": "echo", "program": "$MEM.sys.programs.echo", "para": "$MEM.para.echo"},
    ])

    def mutate(image):
        mem = image["mem"]
        mem["para"]["value"]["echo"] = command_para
        mem["sys"]["value"]["programs"]["value"]["echo"] = program
        mem["sys"]["value"]["boot"]["value"]["manifest"]["value"] = manifest

    core = load_and_run(prepared_image(tmp_path, mutate=mutate))
    registry = core.unwrap("$MEM.sys.registry", for_llm=False)
    assert registry["echo"] == "1"
    child = core.get_conversation(1)
    assert child is not None and child.is_python
    assert 1 in core._dormant_cids


def test_inbox_write_is_journaled_and_replayable(tmp_path):
    from avm.console import OperatorChannel

    def mutate(image):
        pass

    image = prepared_image(tmp_path, mutate=mutate)
    core = load_image(Core(), image)
    core.lmu = NoLMU()
    core.attach_journal(core.journal_path)
    core.run()  # boot

    channel = OperatorChannel()
    channel.submit_action({
        "tool": "memory_write",
        "arguments": {
            "ref": "$MEM.sys.inbox.-1",
            "content": json.dumps({"id": "m1", "target": "echo", "text": "hi"}),
        },
    })
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    core._active_cid = panel.cid
    core.advance_conversation()

    restored = load_image(Core(), image)
    restored.lmu = NoLMU()
    restored.attach_journal(restored.journal_path)
    assert restored.replay_journal() >= 1
    inbox = list(restored.unwrap("$MEM.sys.inbox", for_llm=False))
    assert inbox and json.loads(inbox[0])["id"] == "m1"


def _run_panel_input(core, channel, panel, text):
    """让保留对话把一条 inbox 追加动作跑完（写 + 结束本轮）。"""
    channel.submit_actions([
        {
            "tool": "memory_write",
            "arguments": {
                "ref": "$MEM.sys.inbox.-1",
                "content": json.dumps(
                    {"id": text, "target": None, "text": text, "status": "new"},
                    ensure_ascii=False,
                ),
            },
        },
        {"end": True},
    ])
    # 模拟中断派发激活保留对话：先移出休眠队列，否则 Core 把它当成"本轮自己让出"
    if panel.cid in core._dormant_cids:
        core._dormant_cids.remove(panel.cid)
    core._active_cid = panel.cid
    for _ in range(10):
        if core._active_cid != panel.cid:
            return
        core.advance_conversation()
    raise AssertionError("保留对话没有让出执行权")


def test_inbox_survives_snapshot_and_journal_tail(tmp_path):
    """正式输入可重放：快照 + 日志尾部恢复后 inbox 内容一致，保留对话仍隐藏。"""
    from avm.console import OperatorChannel
    from avm.core import PANEL_CID
    from avm.snapshot import snapshot

    image = prepared_image(tmp_path)
    core = load_image(Core(), image)
    core.lmu = NoLMU()
    core.attach_journal(core.journal_path)
    core.run()  # boot

    channel = OperatorChannel()
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    _run_panel_input(core, channel, panel, "first")

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps(snapshot(core), ensure_ascii=False), encoding="utf-8")
    _run_panel_input(core, channel, panel, "second")

    restored = load_image(Core(), image, restore=True)
    restored.lmu = NoLMU()
    restored.attach_journal(restored.journal_path)
    restored.restore_from(str(snap_path))
    assert restored.replay_journal() >= 1

    inbox = list(restored.unwrap("$MEM.sys.inbox", for_llm=False))
    assert [json.loads(item)["text"] for item in inbox] == ["first", "second"]
    conv = restored.get_conversation(PANEL_CID)
    assert conv is not None
    assert conv.attrs.get("host") == "console" and conv.attrs.get("hidden") == "1"
    assert restored.is_hidden_cid(PANEL_CID)
    assert f"cid={PANEL_CID} " not in restored.unwrap("$MEM.conversations")


def test_loader_mounts_device_and_replay_keeps_it(tmp_path):
    from avm.console import OperatorChannel

    device_source = tmp_path / "echo_dev.py"
    device_source.write_text(
        "from avm.memory_device import MemoryDevice\n"
        "class EchoDevice(MemoryDevice):\n"
        "    def __init__(self, value=''):\n"
        "        self._value = str(value)\n"
        "    def pretend_as_type(self):\n"
        "        return 'str'\n"
        "    def to_llm_string(self):\n"
        "        return self._value\n"
        "    def set_value(self, value):\n"
        "        self._value = str(value)\n",
        encoding="utf-8",
    )
    image = prepared_image(tmp_path)
    core = load_image(Core(), image)
    core.lmu = NoLMU()
    core.attach_journal(core.journal_path)
    core.run()

    with open(device_source, "rb") as f:
        approved = hashlib.sha256(f.read()).hexdigest()
    command = json.dumps({
        "action": "load_device",
        "approved_sha256": approved,
        "path": "echo",
        "file": str(device_source),
        "class": "EchoDevice",
        "args": {"value": "初始"},
    })
    channel = OperatorChannel()
    channel.submit_action({
        "tool": "memory_write",
        "arguments": {"ref": "$MEM.loader", "content": command},
    })
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    core._active_cid = panel.cid
    core.advance_conversation()
    assert "echo" in core.mem._devices

    restored = load_image(Core(), image)
    restored.attach_journal(restored.journal_path)
    restored.replay_journal()
    assert "echo" in restored.mem._devices
    assert "echo" in restored.device_specs


def test_base_programs_have_no_blocking_input_read():
    with open(BASE_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    boot = image["mem"]["sys"]["value"]["programs"]["value"]["boot"]["value"]["content"]["value"]
    assert "$MEM.inputs.next" not in boot

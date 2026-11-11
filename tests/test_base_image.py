"""base 母板镜像的冒烟测试（MockLMU / 无 LLM 调用）。

验收（空闲与关机按作者 2026-11-11 的口径重写）：
1. 冷启动：boot 程序写 running / boot id / report、创建 clock 服务；空闲时执行权
   交给宿主面板对话（cid -1，阻塞等操作员），不是 Core 自己等待，也不读任何
   `$MEM.sys.idle.state` 之类节点；
2. 清单：manifest 有服务时 boot 程序按清单创建对话并写 registry；
3. 面板写入：面板对话的普通写入进日志、可重放，快照 + 日志尾部恢复后仍在；
4. 装载设备：经 loader 装入临时设备，日志重放后设备仍在；
5. 无阻塞：base 的程序里都没有 `$MEM.inputs.next` 阻塞读；
6. 约定清理：base 不再有 inbox / idle.state / power.request 节点。
"""

import hashlib
import json
import os
import threading
import time

from avm.console import Console, OperatorChannel
from avm.core import Core, PANEL_CID
from avm.image import load_image
from avm.snapshot import snapshot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = os.path.join(REPO, "images", "base", "base.json")
PY_PARA = {
    "kind": "dict", "ctrl": {"type": "para"},
    "value": {"model": {"kind": "str", "value": "plm.python"}},
}


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


def load_core(image_path, journal=False):
    core = load_image(Core(), image_path)
    core.lmu = NoLMU()
    if journal:
        core.attach_journal(core.journal_path)
    return core


def start_run(core):
    """在后台线程跑 Core.run（空闲时会停在面板对话的阻塞推理里）。"""
    errors = []

    def target():
        try:
            core.run()
        except BaseException as e:  # noqa: BLE001 - 测试要把内核异常带回主线程
            errors.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, errors


def wait_for(predicate, timeout=10, message="条件没有满足"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def stop_run(core, thread, console=None, timeout=10):
    """请求停止；面板正阻塞时先结束它这一轮，让 Core 回到安全点。"""
    if console is not None and thread.is_alive():
        console.channel.submit_action({"end": True})
    core.request_stop()
    thread.join(timeout)
    assert not thread.is_alive(), "run() 没有在停止请求后返回"


def boot_cold(core):
    """手动推进冷启动：init → boot，不进入 Core.run 的调度循环。

    每个 `advance_conversation` 只执行程序里的一次工具调用，boot 程序有很多次
    （读状态、建服务、写报告、写 outbox），所以要一直推进到没有活跃对话为止。
    """
    if core._active_cid is None:
        core._pick_next_active()
    for _ in range(500):
        if core._active_cid is None:
            break
        core.advance_conversation()
    assert core._active_cid is None, "冷启动程序没有在合理回合内结束"


def _panel_actions(core, channel, panel, actions):
    """让保留对话把一批动作跑完（批里要带 end，否则它会阻塞等下一个动作）。"""
    channel.submit_actions(actions)
    if panel.cid in core._dormant_cids:
        core._dormant_cids.remove(panel.cid)
    core._active_cid = panel.cid
    for _ in range(20):
        if core._active_cid != panel.cid:
            return
        core.advance_conversation()
    raise AssertionError("保留对话没有让出执行权")


def _panel_write(core, channel, panel, content):
    _panel_actions(core, channel, panel, [
        {"tool": "memory_write",
         "arguments": {"ref": "$MEM.sys.outbox.-1", "content": content}},
        {"end": True},
    ])


def test_cold_boot_and_idle_is_panel_blocking():
    core = load_core(BASE_IMAGE)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_for(lambda: core._active_cid == console.panel.cid,
                 message="空闲时执行权没有交给面板对话")
        assert thread.is_alive(), "面板对话阻塞保持活跃，宿主应保持运行"
        assert not errors

        assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "running"
        assert core.unwrap("$MEM.sys.boot.prior", for_llm=False) == "clean"
        assert core.unwrap("$MEM.sys.boot.id", for_llm=False) == "1"
        report = core.unwrap("$MEM.sys.boot.report", for_llm=False)
        assert "prior=clean" in report and "boot_id=1" in report
        assert dict(core.unwrap("$MEM.sys.registry", for_llm=False)) == {}
        assert list(core.unwrap("$MEM.sys.outbox", for_llm=False))
        assert 0 in core._dormant_cids
        assert core.guest_roles["boot"]["cid"] == 1
        assert core.guest_roles["shutdown"]["cid"] == 2
        assert "idle" not in core.guest_roles
        assert core.interrupts.vectors["idle"] == console.panel.cid
        assert core.interrupts.vectors["shutdown"] == 2

        for name in ("clock_a", "clock_b"):
            assert any(conv.name == name for conv in core._conv_by_cid.values()), \
                name + " 服务对话没创建"
    finally:
        stop_run(core, thread, console)
    assert errors == []


def test_manifest_creates_service_and_registry(tmp_path):
    program = {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": "def handle(cmd):\n    return 'ok'\n"},
            "para_ref": {"kind": "str", "value": "$MEM.para.python"},
        },
    }
    manifest = json.dumps([
        {"name": "echo", "program": "$MEM.sys.programs.echo", "para": "$MEM.para.echo"},
    ])

    def mutate(image):
        mem = image["mem"]
        mem["para"]["value"]["echo"] = PY_PARA
        mem["sys"]["value"]["programs"]["value"]["echo"] = program
        mem["sys"]["value"]["boot"]["value"]["manifest"]["value"] = manifest

    core = load_core(prepared_image(tmp_path, mutate=mutate))
    boot_cold(core)
    registry = core.unwrap("$MEM.sys.registry", for_llm=False)
    cid = int(registry["echo"])
    child = core.get_conversation(cid)
    assert child is not None and child.is_python
    assert cid in core._dormant_cids


def test_manifest_service_is_not_duplicated_on_later_boot(tmp_path):
    program = {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": "def handle(cmd):\n    return 'ok'\n"},
            "para_ref": {"kind": "str", "value": "$MEM.para.python"},
        },
    }
    manifest = json.dumps([
        {"name": "echo", "program": "$MEM.sys.programs.echo", "para": "$MEM.para.echo"},
    ])

    def mutate(image):
        mem = image["mem"]
        mem["para"]["value"]["echo"] = PY_PARA
        mem["sys"]["value"]["programs"]["value"]["echo"] = program
        mem["sys"]["value"]["boot"]["value"]["manifest"]["value"] = manifest

    core = load_core(prepared_image(tmp_path, mutate=mutate))
    boot_cold(core)
    cid = int(core.unwrap("$MEM.sys.registry.echo", for_llm=False))
    before = sorted(core._conv_by_cid)
    # 再跑一次 boot（模拟 resume 时的唤醒）：registry 已有且对话还活着 → 不建新对话
    core.interrupts.raise_("boot", payload="yield from boot('normal')")
    core.interrupts.dispatch()
    for _ in range(500):  # boot 程序剩余回合跑完
        if core._active_cid is None:
            break
        core.advance_conversation()
    assert int(core.unwrap("$MEM.sys.registry.echo", for_llm=False)) == cid
    assert sorted(core._conv_by_cid) == before, "已有 manifest 服务不应在重启时重复创建"


def test_panel_write_is_journaled_and_replayable(tmp_path):
    image = prepared_image(tmp_path)
    core = load_core(image, journal=True)
    boot_cold(core)

    channel = OperatorChannel()
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    _panel_write(core, channel, panel, "panel-写入")

    restored = load_core(image, journal=True)
    assert restored.replay_journal() >= 1
    outbox = list(restored.unwrap("$MEM.sys.outbox", for_llm=False))
    assert outbox and outbox[-1] == "panel-写入"


def test_panel_write_survives_snapshot_and_journal_tail(tmp_path):
    """面板写入可重放：快照 + 日志尾部恢复后内容一致，保留对话仍隐藏。"""
    image = prepared_image(tmp_path)
    core = load_core(image, journal=True)
    boot_cold(core)

    channel = OperatorChannel()
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    _panel_write(core, channel, panel, "first")

    snap_path = tmp_path / "snap.json"
    snap_path.write_text(
        json.dumps(snapshot(core), ensure_ascii=False), encoding="utf-8")
    _panel_write(core, channel, panel, "second")

    restored = load_core(image, journal=True)
    restored.restore_from(str(snap_path))
    assert restored.replay_journal() >= 1

    outbox = list(restored.unwrap("$MEM.sys.outbox", for_llm=False))
    assert [item for item in outbox if item in ("first", "second")] == ["first", "second"]
    conv = restored.get_conversation(PANEL_CID)
    assert conv is not None
    assert conv.attrs.get("host") == "console" and conv.attrs.get("hidden") == "1"
    assert restored.is_hidden_cid(PANEL_CID)
    assert f"cid={PANEL_CID} " not in restored.unwrap("$MEM.conversations")


def test_loader_mounts_device_and_replay_keeps_it(tmp_path):
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
    core = load_core(image, journal=True)
    boot_cold(core)

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
    panel = core.start_operator_conversation(channel, para_ref="$MEM.para.python")
    _panel_actions(core, channel, panel, [
        {"tool": "memory_write",
         "arguments": {"ref": "$MEM.loader", "content": command}},
        {"end": True},
    ])
    assert "echo" in core.mem._devices

    restored = load_image(Core(), image)
    restored.attach_journal(restored.journal_path)
    restored.replay_journal()
    assert "echo" in restored.mem._devices
    assert "echo" in restored.device_specs


def test_base_programs_have_no_blocking_input_read():
    with open(BASE_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    programs = image["mem"]["sys"]["value"]["programs"]["value"]
    assert set(programs) == {"boot", "supervisor", "clock_a", "clock_b"}
    for name, node in programs.items():
        content = node["value"]["content"]["value"]
        assert "$MEM.inputs.next" not in content, f"程序 {name} 含阻塞读"


def test_base_image_drops_old_os_conventions():
    with open(BASE_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    sys_mem = image["mem"]["sys"]["value"]
    assert "inbox" not in sys_mem, "输入路径已冻结：base 不预置 inbox"
    assert "idle" not in sys_mem, "空闲由宿主面板承担：base 不预置 idle 约定节点"
    assert "request" not in sys_mem["power"]["value"], "关机走中断：不再有 power.request"
    assert "state" in sys_mem["power"]["value"]
    assert image["meta"]["guest_programs"] == {
        "boot": "$MEM.sys.programs.boot",
        "shutdown": "$MEM.sys.programs.supervisor",
    }

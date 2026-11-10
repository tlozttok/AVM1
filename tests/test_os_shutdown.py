"""OS 层优雅关机与空闲等待的验收（plan-os-shutdown-2026-11-10）。

覆盖：
1. base 冷启动创建 guest 程序与服务对话，空闲时 idle 写 waiting、宿主等待不 panic；
2. 面板输入唤醒 panel_shell，inbox 被消费，outbox/outputs 出现输出；
3. 优雅关机：supervisor 写 clean/off，宿主写快照退出；resume 后 boot 读到
   last_exit 并写 report；
4. 顽固对话：超时只发通知，supervisor 不自动断电，进程保持运行；
5. 顽固对话收尾后由下一次超时重新唤醒 supervisor，正常关机；
6. 脏关闭（日志比快照长）resume 时 boot 报告 last_exit=forced。

全程离线：base 程序都是 plm.python；LLM 执行器换成会报错的 NoLMU。
"""

import json
import os
import threading
import time

from avm.boot import boot, write_snapshot
from avm.console import Console
from avm.core import Core
from avm.image import load_image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMAGE = os.path.join(REPO, "images", "base", "base.json")


class NoLMU:
    """OS 程序不应调用 LLM；被调用即失败。"""

    def exec(self, conv, para):
        raise AssertionError("OS 测试不应调用 LLM")


# ----------------------------------------------------------------------
# 夹具
# ----------------------------------------------------------------------

COMMON_HELPER = '''def _find_cid(name):
    listing = yield from memory_read("$MEM.conversations")
    for line in str(listing).splitlines():
        m = re.search(r"cid=(\\d+)\\s+(\\S+)", line)
        if m is not None and m.group(2) == name:
            return int(m.group(1))
    return None

'''

APP_A = COMMON_HELPER + '''def app_a():
    peer = yield from _find_cid("app_b")
    if peer is not None:
        yield from send_instruction(peer, "yield from app_b()")
'''

APP_B = COMMON_HELPER + '''def app_b():
    peer = yield from _find_cid("app_a")
    if peer is not None:
        yield from send_instruction(peer, "yield from app_a()")
'''


def _py_program(source, name):
    return {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": source},
            "para_ref": {"kind": "str", "value": "$MEM.para.python"},
            "name": {"kind": "str", "value": name},
        },
    }


def _add_stubborn_apps(image, interval="2"):
    """加两个互相投递的 Python 应用，并把设备事件钟间隔调小。"""
    programs = image["mem"]["sys"]["value"]["programs"]["value"]
    programs["app_a"] = _py_program(APP_A, "app_a")
    programs["app_b"] = _py_program(APP_B, "app_b")
    clock = image["mem"]["sys"]["value"]["clock"]["value"]
    clock["interval"]["value"] = interval


def prepared_image(tmp_path, *, mutate=None):
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


def load_core(image_path):
    core = load_image(Core(), image_path)
    core.lmu = NoLMU()
    core.attach_journal(core.journal_path)
    return core


def start_run(core):
    errors = []

    def target():
        try:
            core.run()
        except BaseException as e:  # noqa: BLE001 - 把内核异常带回主线程
            errors.append(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, errors


def wait_for(predicate, timeout=15, message="条件没有满足"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def stop_run(core, thread, timeout=10):
    core.request_stop()
    thread.join(timeout)
    assert not thread.is_alive(), "run() 没有在停止请求后返回"


def wait_idle_waiting(core, timeout=15):
    wait_for(
        lambda: core.unwrap("$MEM.sys.idle.state", for_llm=False) == "waiting",
        timeout=timeout,
        message="系统没有进入 OS 声明的空闲等待",
    )


def find_conversation(core, name):
    for cid, conv in core._conv_by_cid.items():
        if conv.name == name:
            return cid
    return None


def memory_int(core, ref):
    return int(str(core.unwrap(ref, for_llm=False)).strip() or "0")


def finish_panel(console):
    """提交 end，并保证保留对话能醒来处理（它可能正休眠）。"""
    console.channel.submit_action({"end": True})
    console.core.interrupts.raise_("operator")


def create_app(console, core, name):
    console.submit_action(
        "create_cmd",
        {"system_ref": "$MEM.sys.programs." + name, "para_ref": "$MEM.para.python"},
    )
    wait_for(lambda: find_conversation(core, name) is not None,
             message="服务对话 " + name + " 没有创建")
    return find_conversation(core, name)


def start_stubborn_apps(core, console):
    """创建两个互相投递的应用并让它们跑起来；返回 (cid_a, cid_b)。"""
    cid_a = create_app(console, core, "app_a")
    cid_b = create_app(console, core, "app_b")
    console.submit_action(
        "send_instruction", {"cid": cid_a, "content": "yield from app_a()"}
    )
    finish_panel(console)
    wait_for(
        lambda: cid_a in core._ready_cids or cid_b in core._ready_cids
        or core._active_cid in (cid_a, cid_b),
        message="顽固应用没有开始互相投递",
    )
    return cid_a, cid_b


# ----------------------------------------------------------------------
# 1. 冷启动：guest 程序、服务对话、空闲等待
# ----------------------------------------------------------------------

def test_cold_boot_creates_os_programs_and_waits_idle(tmp_path):
    core = load_core(prepared_image(tmp_path))
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        assert thread.is_alive(), "OS 声明的空闲等待不应让宿主 panic"
        assert not errors
        assert core.guest_roles["boot"]["cid"] == 1
        assert core.guest_roles["idle"]["cid"] == 2
        assert core.guest_roles["shutdown"]["cid"] == 3
        for name in ("panel_shell", "clock_a", "clock_b"):
            assert find_conversation(core, name) is not None, name + " 服务对话没创建"
        assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "running"
        assert core.unwrap("$MEM.sys.boot.id", for_llm=False) == "1"
        report = core.unwrap("$MEM.sys.boot.report", for_llm=False)
        assert "prior=clean" in report and "last_exit=unknown" in report
        assert list(core.unwrap("$MEM.sys.outbox", for_llm=False))
    finally:
        stop_run(core, thread)
    assert errors == []


# ----------------------------------------------------------------------
# 2. 输入：idle 唤醒 panel_shell，inbox 消费，outbox/outputs 输出
# ----------------------------------------------------------------------

def test_input_wakes_panel_shell_and_consumes_inbox(tmp_path):
    core = load_core(prepared_image(tmp_path))
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        console.submit_input("喂给系统的第一句")
        wait_for(
            lambda: any("喂给系统的第一句" in e["text"]
                        for e in core.io_bus.drain_io()),
            message="输出流里没有 panel_shell 的处理结果",
        )
        entry = json.loads(list(core.unwrap("$MEM.sys.inbox", for_llm=False))[0])
        assert entry["status"] == "done", "inbox 条目没有被消费标记"
        assert any("panel_shell: 喂给系统的第一句" in item
                   for item in core.unwrap("$MEM.sys.outbox", for_llm=False))
        assert thread.is_alive() and not errors
        # 处理完再次空闲：idle 自己把状态写回 waiting，下一次输入可以再次唤醒
        wait_idle_waiting(core)
        console.submit_input("第二句")
        wait_for(
            lambda: any("第二句" in e["text"] for e in core.io_bus.drain_io()),
            message="第二次空闲后 panel_shell 没有被再次唤醒",
        )
        entries = [json.loads(item)
                   for item in core.unwrap("$MEM.sys.inbox", for_llm=False)]
        assert [entry["status"] for entry in entries] == ["done", "done"]
    finally:
        stop_run(core, thread)
    assert errors == []


# ----------------------------------------------------------------------
# 3. 优雅关机 + 正常 resume
# ----------------------------------------------------------------------

def test_graceful_shutdown_writes_clean_and_off_then_resume_reports(tmp_path):
    image_path = prepared_image(tmp_path)
    core = load_core(image_path)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        console.request_shutdown()
        wait_for(lambda: not thread.is_alive(), message="优雅关机后宿主没有退出")
        assert errors == []
    finally:
        if thread.is_alive():
            stop_run(core, thread)

    assert core.unwrap("$MEM.sys.power.state", for_llm=False) == "off"
    assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "clean"
    assert core.unwrap("$MEM.sys.clock.state", for_llm=False) == "stopped"
    assert any("shutdown:" in item
               for item in core.unwrap("$MEM.sys.outbox", for_llm=False))
    assert write_snapshot(core), "正常退出应写重启快照"

    resumed = boot(image_path, mode="resume", refresh_models=False)
    resumed.lmu = NoLMU()
    resumed_thread, resumed_errors = start_run(resumed)
    try:
        wait_for(
            lambda: "last_exit=normal"
            in str(resumed.unwrap("$MEM.sys.boot.report", for_llm=False)),
            message="resume 后 boot 没有读到 last_exit=normal",
        )
        assert resumed.unwrap("$MEM.sys.boot.prior", for_llm=False) == "clean"
        assert resumed.unwrap("$MEM.sys.boot.state", for_llm=False) == "running"
        assert resumed.recovery["last_exit"] == "normal"
        assert not resumed_errors
        assert resumed_thread.is_alive(), "resume 后系统应在空闲等待"
    finally:
        stop_run(resumed, resumed_thread)
    assert resumed_errors == []


# ----------------------------------------------------------------------
# 3b. resume 后可以再次关机：boot 清掉上次的 power.request/state 边沿
# ----------------------------------------------------------------------

def test_resume_can_request_shutdown_again(tmp_path):
    image_path = prepared_image(tmp_path)
    core = load_core(image_path)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        console.request_shutdown()
        wait_for(lambda: not thread.is_alive(), message="第一次优雅关机没有退出")
        assert errors == []
    finally:
        if thread.is_alive():
            stop_run(core, thread)

    assert write_snapshot(core)
    # 控制台重启默认走 auto（有快照就恢复）；boot 唤醒条件必须覆盖这条路径。
    resumed = boot(image_path, refresh_models=False)
    assert resumed.recovery["mode"] == "auto"
    assert resumed.recovery["source"] == "snapshot"
    resumed.lmu = NoLMU()
    resumed_thread, resumed_errors = start_run(resumed)
    try:
        wait_for(
            lambda: "last_exit=normal"
            in str(resumed.unwrap("$MEM.sys.boot.report", for_llm=False)),
            message="resume 后 boot 没有跑完",
        )
        # boot 必须把上次关机留下的 request=shutdown / state=off 复位，
        # 否则宿主看不到"新的关机请求 → 新的 off"边沿。
        assert resumed.unwrap("$MEM.sys.power.request", for_llm=False) == ""
        assert resumed.unwrap("$MEM.sys.power.state", for_llm=False) == "on"
        console2 = Console(resumed, mount_io_devices=False)
        console2.request_shutdown()
        wait_for(lambda: not resumed_thread.is_alive(),
                 message="resume 后的第二次关机没有退出")
        assert resumed_errors == []
        assert resumed.unwrap("$MEM.sys.power.state", for_llm=False) == "off"
        assert resumed.unwrap("$MEM.sys.boot.state", for_llm=False) == "clean"
    finally:
        if resumed_thread.is_alive():
            stop_run(resumed, resumed_thread)


# ----------------------------------------------------------------------
# 4. 顽固对话：超时通知不断电，进程保持运行
# ----------------------------------------------------------------------

def test_stubborn_conversation_is_not_powered_off_after_timeout(tmp_path):
    image_path = prepared_image(tmp_path, mutate=_add_stubborn_apps)
    core = load_core(image_path)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        start_stubborn_apps(core, console)
        console.request_shutdown()
        wait_for(
            lambda: memory_int(core, "$MEM.sys.clock.timeouts") >= 1,
            message="设备事件钟没有发出超时通知",
        )
        assert thread.is_alive(), "顽固对话未收尾时进程必须保持运行"
        assert core.unwrap("$MEM.sys.power.state", for_llm=False) != "off"
        assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "running"
        assert any("未收尾" in item or "等待" in item
                   for item in core.unwrap("$MEM.sys.outbox", for_llm=False))
        assert not errors
    finally:
        stop_run(core, thread)
    assert errors == []


# ----------------------------------------------------------------------
# 5. 顽固对话收尾后：下一次超时唤醒 supervisor，正常关机
# ----------------------------------------------------------------------

def test_timeout_then_apps_close_leads_to_clean_shutdown(tmp_path):
    image_path = prepared_image(tmp_path, mutate=_add_stubborn_apps)
    core = load_core(image_path)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        cid_a, cid_b = start_stubborn_apps(core, console)
        console.request_shutdown()
        wait_for(
            lambda: memory_int(core, "$MEM.sys.clock.timeouts") >= 1,
            message="设备事件钟没有发出超时通知",
        )
        assert thread.is_alive() and core.unwrap("$MEM.sys.power.state",
                                                  for_llm=False) != "off"

        # 应用收尾（内核级关闭），让下一次超时唤醒 supervisor 时全部已休眠
        console.submit_action("close_conversation", {"cid": cid_a})
        console.submit_action("close_conversation", {"cid": cid_b})
        finish_panel(console)
        wait_for(lambda: not thread.is_alive(), message="全部收尾后没有正常关机")
        assert errors == []
    finally:
        if thread.is_alive():
            stop_run(core, thread)

    assert core.unwrap("$MEM.sys.power.state", for_llm=False) == "off"
    assert core.unwrap("$MEM.sys.boot.state", for_llm=False) == "clean"
    assert core.unwrap("$MEM.sys.clock.state", for_llm=False) == "stopped"


# ----------------------------------------------------------------------
# 6. 脏关闭：resume 时 boot 读到 last_exit=forced
# ----------------------------------------------------------------------

def test_resume_after_dirty_shutdown_reports_forced(tmp_path):
    image_path = prepared_image(tmp_path)
    core = load_core(image_path)
    console = Console(core, mount_io_devices=True)
    thread, errors = start_run(core)
    try:
        wait_idle_waiting(core)
        assert write_snapshot(core), "先拍一个基准快照"
        # 快照之后继续写日志（消费一条输入），然后不写快照直接停
        console.submit_input("脏关闭前的动作")
        wait_for(
            lambda: any("脏关闭前的动作" in e["text"]
                        for e in core.io_bus.drain_io()),
            message="输入没有被处理，日志没有继续增长",
        )
        wait_idle_waiting(core)
        stop_run(core, thread)
        assert errors == []
    finally:
        if thread.is_alive():
            stop_run(core, thread)

    journal_last = core.journal.last_seq()
    assert journal_last > core._last_snapshot_seq, "日志没有长于快照，无法模拟脏关闭"

    resumed = boot(image_path, mode="resume", refresh_models=False)
    resumed.lmu = NoLMU()
    resumed_thread, resumed_errors = start_run(resumed)
    try:
        wait_for(
            lambda: "last_exit=forced"
            in str(resumed.unwrap("$MEM.sys.boot.report", for_llm=False)),
            message="脏关闭 resume 后 boot 没有报告 last_exit=forced",
        )
        assert resumed.recovery["last_exit"] == "forced"
        assert resumed.unwrap("$MEM.sys.boot.prior", for_llm=False) == "running"
        assert not resumed_errors
    finally:
        stop_run(resumed, resumed_thread)

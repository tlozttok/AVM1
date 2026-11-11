"""VM 层启动生命周期（plan-vm-lifecycle-2026-11-10）验收测试。

覆盖：三种启动模式与世界来源、无快照拒绝、日志代际与归档、重放先于行为与
seq 断言、镜像声明 guest 程序与中断向量、空闲=严重错误、断电边界、恢复报告。
全部离线：LLM 对话用脚本化执行器，guest 程序用 `plm.python`。
"""

import json
import os

import pytest

from avm.boot import boot
from avm.console import Console, ConsoleConflict
from avm.core import Core, PANEL_CID
from avm.exceptions import KernelPanic
from avm.image import ImageError, load_image
from avm.journal import Journal, JournalError
from avm.snapshot import read_snapshot, write_atomic
from persist_helpers import (
    ScriptedLMU, assert_same_state, load_core, out_path, run_turns, write_image,
)

THREE_TURNS = [
    ("", [{"call_id": "c1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v1"}}]),
    ("", [{"call_id": "c2", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "tail"}}]),
    ("done", []),
]


# ----------------------------------------------------------------------
# 夹具：带 guest 程序声明的镜像
# ----------------------------------------------------------------------

PY_PARA = {
    "kind": "dict", "ctrl": {"type": "para"},
    "value": {"model": {"kind": "str", "value": "plm.python"}},
}


def _py_program(code):
    return {
        "kind": "dict", "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": code},
            "para_ref": {"kind": "str", "value": "$MEM.para.python"},
        },
    }


def _guest_mem(*, idle_code=None, boot_code=None, shutdown_code=None):
    idle_code = idle_code or (
        "def idle():\n"
        "    yield from memory_write('$MEM.markers.idle_ran', '1')\n"
    )
    boot_code = boot_code or (
        "def boot():\n"
        "    yield from memory_write('$MEM.markers.boot_ran', '1')\n"
    )
    shutdown_code = shutdown_code or (
        "def shutdown():\n"
        "    yield from memory_write('$MEM.markers.shutdown_ran', '1')\n"
    )
    mem = {
        "para": {"kind": "dict", "value": {"python": PY_PARA}},
        "model_params": {
            "kind": "dict", "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {"kind": "dict", "value": {
            "init": _py_program(
                "def init():\n"
                "    yield from memory_write('$MEM.markers.init_ran', '1')\n"
            ),
            "boot": _py_program(boot_code),
            "idle": _py_program(idle_code),
            "shutdown": _py_program(shutdown_code),
        }},
        "markers": {"kind": "dict", "value": {
            "init_ran": {"kind": "str", "value": ""},
            "boot_ran": {"kind": "str", "value": ""},
            "idle_ran": {"kind": "str", "value": ""},
            "shutdown_ran": {"kind": "str", "value": ""},
        }},
    }
    return mem


def _guest_image(tmp_path, *, roles=("boot", "idle", "shutdown"), idle_code=None,
                 name="guest.json", **mem_kwargs):
    image = write_image(
        tmp_path, mem=_guest_mem(idle_code=idle_code, **mem_kwargs),
        init={"name": "init", "system_ref": "$MEM.programs.init",
              "user": "yield from init()", "para_ref": "$MEM.para.python"},
        name=name,
    )
    with open(image, encoding="utf-8") as f:
        doc = json.load(f)
    doc["meta"]["guest_programs"] = {
        role: f"$MEM.programs.{role}" for role in roles
    }
    with open(image, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return image


def _advance_panel(core, console):
    """让保留对话把通道里排队的动作跑完（与 Core.run 的连续推进一致）。"""
    if console.panel.cid in core._dormant_cids:
        core._dormant_cids.remove(console.panel.cid)
    core._active_cid = console.panel.cid
    for _ in range(20):
        if core._active_cid != console.panel.cid:
            return
        core.advance_conversation()
    raise AssertionError("保留对话没有让出执行权")


def _power_mem():
    """最小镜像 + OS 约定的断电节点（$MEM.sys.power.request/state）。"""
    return {
        "model_params": {
            "kind": "dict", "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {"kind": "dict", "value": {"main": {
            "kind": "dict", "ctrl": {"type": "settingup"},
            "value": {"content": {"kind": "str", "value": "sys"}},
        }}},
        "src": {"kind": "str", "value": "hello"},
        "sys": {"kind": "dict", "value": {"power": {"kind": "dict", "value": {
            "request": {"kind": "str", "value": ""},
            "state": {"kind": "str", "value": "on"},
        }}}},
    }


# ----------------------------------------------------------------------
# 验收 1：resume = 最近快照 + 日志尾部，续代际
# ----------------------------------------------------------------------

class TestResume:
    def _run_with_listener(self, tmp_path):
        handler = {
            "kind": "dict", "ctrl": {"type": "handler"},
            "value": {
                "content": {"kind": "str",
                            "value": "def handle(event):\n    return {}\n"},
                "condition": {"kind": "str", "value": json.dumps({
                    "events": ["memory_write"], "phase": ["before"],
                    "target": {"kind": "exact", "ref": "$MEM.src"},
                })},
            },
        }
        mem = {
            "model_params": {
                "kind": "dict", "ctrl": {"type": "para"},
                "value": {"model": {"kind": "str", "value": "test-model"}},
            },
            "programs": {"kind": "dict", "value": {"main": {
                "kind": "dict", "ctrl": {"type": "settingup"},
                "value": {"content": {"kind": "str", "value": "sys"}},
            }}},
            "handlers": {"kind": "dict", "value": {"h": handler}},
            "src": {"kind": "str", "value": "hello"},
        }
        responses = [
            ("", [{"call_id": "c1", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.src", "content": "v1"}}]),
            ("", [{"call_id": "c2", "cmd_type": "listen_event",
                   "args": {"action": "listen", "handler_ref": "$MEM.handlers.h"}}]),
            ("", [{"call_id": "c3", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.src", "content": "tail"}}]),
            ("done", []),
        ]
        image = write_image(tmp_path, mem=mem)
        core = load_core(image, responses)
        core.attach_journal(core.journal_path)
        run_turns(core, 2)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        run_turns(core, 2)  # 快照之后提交的世界 + 监听器尾部
        return image, core

    def test_resume_restores_conversations_listeners_schedule_and_seq(self, tmp_path):
        image, mine = self._run_with_listener(tmp_path)
        assert mine.mem["src"] == "tail"
        assert len(mine.events.listeners) == 1

        resumed = boot(image, mode="resume", refresh_models=False)

        assert resumed.mem["src"] == "tail"
        assert [listener.get("listener_id") for listener in resumed.events.listeners] == \
            [listener.get("listener_id") for listener in mine.events.listeners]
        assert resumed._active_cid == mine._active_cid
        assert resumed._ready_cids == mine._ready_cids
        assert resumed._dormant_cids == mine._dormant_cids
        assert resumed._next_cid == mine._next_cid
        assert resumed._journal_seq == mine._journal_seq == 4
        assert resumed.journal.last_seq() == 4
        assert resumed.journal_generation == 1, "resume 续当前代际"
        assert resumed.recovery["mode"] == "resume"
        assert resumed.recovery["source"] == "snapshot"
        assert resumed.recovery["snapshot_seq"] == 2
        assert resumed.recovery["replayed"] == 2
        assert_same_state(mine, resumed)

    def test_resume_without_snapshot_is_refused(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError, match="resume 需要可读的重启快照") as excinfo:
            boot(image, mode="resume", refresh_models=False)
        assert "reset" in str(excinfo.value)
        assert "reboot" in str(excinfo.value)

    def test_resume_without_declared_snapshot_is_refused(self, tmp_path):
        image = write_image(tmp_path, snapshot=False)
        with pytest.raises(ImageError, match="没有声明 snapshot_to"):
            boot(image, mode="resume", refresh_models=False)


# ----------------------------------------------------------------------
# 验收 2/3：reboot 世界来源 B 与无快照拒绝
# ----------------------------------------------------------------------

class TestReboot:
    def _run(self, tmp_path):
        responses = [
            ("", [{"call_id": "c1", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.src", "content": "v1"}}]),
            ("", [{"call_id": "c2", "cmd_type": "memory_make",
                   "args": {"ref": "$MEM", "key": "made", "mem_type": "str"}}]),
            # 快照之后的尾部：世界类变化 + 对话态变化
            ("", [{"call_id": "c3", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.made", "content": "after-snap"}}]),
            ("", [{"call_id": "c4", "cmd_type": "create_cmd",
                   "args": {"system_ref": "$MEM.programs.main",
                            "para_ref": "$MEM.model_params"}}]),
            ("", [{"call_id": "c5", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.made", "content": "tail"}}]),
            ("done", []),
        ]
        image = write_image(tmp_path)
        core = load_core(image, responses)
        core.attach_journal(core.journal_path)
        run_turns(core, 2)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        run_turns(core, 4)
        assert core.mem["made"] == "tail"
        assert set(core._conv_by_cid) == {0, 1}
        assert core._journal_seq == 6
        return image, core

    def test_world_is_snapshot_section_plus_world_effect_tail(self, tmp_path):
        image, mine = self._run(tmp_path)
        rebooted = boot(image, mode="reboot", refresh_models=False)

        # 世界包含快照之后的世界变化；对话态效果（create_cmd 的 conversation_create）
        # 没有被重放：对话只剩 init 重建的一条。
        assert rebooted.mem["src"] == "v1"
        assert rebooted.mem["made"] == "tail"
        assert set(rebooted._conv_by_cid) == {1}, "init 按快照 next_cid 重建"
        assert rebooted._next_cid == 2
        assert rebooted.recovery["mode"] == "reboot"
        assert rebooted.recovery["source"] == "snapshot"
        assert rebooted.recovery["world_source"] == "snapshot+journal"
        assert rebooted.recovery["snapshot_seq"] == 2
        assert rebooted.recovery["replayed"] == 4, "尾部 4 条记录被扫描"
        assert rebooted.recovery["world_effects"] == 2, "只有 2 条世界类效果被应用"
        assert rebooted.recovery["last_exit"] == "forced"
        # 新代际 + 归档旧日志
        assert rebooted.journal_generation == 2
        assert rebooted._journal_seq == 0
        assert rebooted.journal.last_seq() == 0
        assert os.path.isfile(out_path(image, "run.1.journal"))
        old_records = Journal(out_path(image, "run.1.journal")).read()
        assert [record["seq"] for record in old_records] == [1, 2, 3, 4, 5, 6]

    def test_reboot_without_snapshot_is_refused_with_reset_hint(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError, match="reboot 需要可读的重启快照") as excinfo:
            boot(image, mode="reboot", refresh_models=False)
        assert "reset" in str(excinfo.value)


# ----------------------------------------------------------------------
# 验收 4：reset 世界回到原始镜像 + 归档 + 新代际
# ----------------------------------------------------------------------

class TestReset:
    def test_reboot_without_journal_uses_snapshot_world_only(self, tmp_path):
        image = write_image(tmp_path, journal=False)
        core = load_core(image, THREE_TURNS)
        run_turns(core, 3)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        assert core._journal_seq == 0

        rebooted = boot(image, mode="reboot", refresh_models=False)
        assert rebooted.journal is None
        assert rebooted.mem["src"] == "tail"
        assert rebooted.journal_generation == 2
        assert rebooted._journal_seq == 0
        assert rebooted.recovery["world_source"] == "snapshot"

    def test_reset_returns_to_image_and_archives_recovery_inputs(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        assert core.mem["src"] == "tail"

        reset = boot(image, mode="reset", refresh_models=False)

        assert reset.mem["src"] == "hello"
        assert "made" not in reset.mem
        assert sorted(reset._conv_by_cid) == [0]
        assert reset._journal_seq == 0
        assert reset.journal.last_seq() == 0
        assert reset.journal_generation == 2
        assert reset.recovery["mode"] == "reset"
        assert reset.recovery["world_source"] == "image"
        assert reset.recovery["last_exit"] == "normal"
        assert os.path.isfile(out_path(image, "run.1.journal"))
        assert os.path.isfile(out_path(image, "run.snapshot.1.json"))
        assert not os.path.isfile(out_path(image, "run.snapshot.json")), \
            "旧快照被归档，不会被下一次 reboot 误用"

    def test_reset_on_fresh_image_stays_generation_one(self, tmp_path):
        image = write_image(tmp_path)
        reset = boot(image, mode="reset", refresh_models=False)
        assert reset.journal_generation == 1
        assert reset._journal_seq == 0


# ----------------------------------------------------------------------
# 验收 5：代际与 run() 的 seq 断言
# ----------------------------------------------------------------------

class TestGeneration:
    def test_run_after_reboot_does_not_trip_seq_assertion(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 2)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        run_turns(core, 2)

        rebooted = boot(image, mode="reboot", refresh_models=False)
        rebooted.lmu = ScriptedLMU([("done", [])])
        rebooted.run()  # 安全点的 seq 断言不能误报（新代际从 0 开始）

        assert rebooted._journal_seq == 1
        assert rebooted.journal.last_seq() == 1
        assert rebooted.journal_generation == 1 + 1
        assert [record.get("generation") for record in rebooted.journal.read()] == [2]

    def test_snapshot_generation_roundtrip_and_old_snapshot_compat(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        core.journal_generation = 7
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        assert read_snapshot(core.snapshot_path)["generation"] == 7

        # 旧快照没有 generation：按当前代际兼容，不拒绝恢复
        with open(core.snapshot_path, encoding="utf-8") as f:
            doc = json.load(f)
        doc.pop("generation")
        write_atomic(core.snapshot_path, doc)
        restored = load_core(image)
        restored.restore_from(core.snapshot_path)
        assert restored.journal_generation == 1

    def test_old_log_without_generation_fields_is_accepted(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        # 把日志里的 generation 字段剥掉（模拟旧版本写下的日志）
        path = out_path(image, "run.journal")
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        for record in records:
            record.pop("generation", None)
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        resumed = boot(image, mode="resume", refresh_models=False)
        assert resumed.journal_generation == 1


# ----------------------------------------------------------------------
# 验收 9：重放断言（seq / 代际不一致 → JournalError）
# ----------------------------------------------------------------------

class TestReplayAssertions:
    def test_snapshot_seq_ahead_of_journal_raises_journal_error(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        doc = read_snapshot(core.snapshot_path)
        doc["seq"] = doc["seq"] + 5
        write_atomic(core.snapshot_path, doc)

        with pytest.raises(JournalError, match="比日志末"):
            boot(image, mode="resume", refresh_models=False)

    def test_generation_mismatch_raises_journal_error(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 2)
        core.snapshot_to(out_path(image, "run.snapshot.json"))

        rebooted = boot(image, mode="reboot", refresh_models=False)
        rebooted.lmu = ScriptedLMU([("done", [])])
        rebooted.run()  # 新代际日志里有 1 条记录，快照仍是上一代际

        with pytest.raises(JournalError, match="代际"):
            boot(image, mode="resume", refresh_models=False)

    def test_reboot_with_truncated_journal_raises_journal_error(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        path = out_path(image, "run.journal")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])  # 丢掉最后一条：日志比快照短

        with pytest.raises(JournalError, match="日志缺失/被截断"):
            boot(image, mode="reboot", refresh_models=False)


# ----------------------------------------------------------------------
# 验收 6：guest 程序与中断向量
# ----------------------------------------------------------------------

class TestGuestPrograms:
    def test_cold_start_creates_kernel_conversations_and_vectors(self, tmp_path):
        image = _guest_image(tmp_path)
        core = boot(image, mode="reset", refresh_models=False)

        assert core.guest_roles == {
            "boot": {"cid": 1, "entry": "yield from boot()", "kind": "python"},
            "idle": {"cid": 2, "entry": "yield from idle()", "kind": "python"},
            "shutdown": {"cid": 3, "entry": "yield from shutdown()",
                         "kind": "python"},
        }
        assert core.interrupts.vectors == {"boot": 1, "idle": 2, "shutdown": 3}
        for role, cid in (("boot", 1), ("idle", 2), ("shutdown", 3)):
            conv = core.get_conversation(cid)
            assert conv.attrs["guest_role"] == role
            assert conv.is_python
            assert conv.name == f"guest-{role}"
        assert 1 in core._ready_cids, "boot 角色上电后运行一次"
        assert 2 in core._dormant_cids and 3 in core._dormant_cids
        assert core.recovery["guest_programs"] == {"boot": 1, "idle": 2, "shutdown": 3}

    def test_resume_finds_guest_conversations_by_attrs(self, tmp_path):
        image = _guest_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        run_turns(core, 4)  # init 2 回合 + boot 2 回合
        core.snapshot_to(out_path(image, "run.snapshot.json"))

        resumed = boot(image, mode="resume", refresh_models=False)
        assert {role: info["cid"] for role, info in resumed.guest_roles.items()} == {
            "boot": 1, "idle": 2, "shutdown": 3,
        }
        assert resumed.interrupts.vectors == {"boot": 1, "idle": 2, "shutdown": 3}
        assert resumed.get_conversation(2).attrs["guest_role"] == "idle"

    def test_resume_missing_guest_role_is_refused(self, tmp_path):
        image = _guest_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        run_turns(core, 4)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        doc = read_snapshot(core.snapshot_path)
        doc["conversations"] = [
            conv for conv in doc["conversations"]
            if (conv.get("attrs") or {}).get("guest_role") != "idle"
        ]
        write_atomic(core.snapshot_path, doc)

        with pytest.raises(ImageError, match="不允许在 resume 时补建"):
            boot(image, mode="resume", refresh_models=False)

    def test_reboot_recreates_guests_from_snapshot_world(self, tmp_path):
        image = _guest_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        run_turns(core, 4)
        core.snapshot_to(out_path(image, "run.snapshot.json"))
        snapshot_next_cid = core._next_cid

        rebooted = boot(image, mode="reboot", refresh_models=False)
        # init 先按快照 next_cid 重建，guest 角色按固定顺序跟在后面。
        init_cid = snapshot_next_cid
        assert rebooted.guest_roles == {
            "boot": {"cid": init_cid + 1, "entry": "yield from boot()",
                     "kind": "python"},
            "idle": {"cid": init_cid + 2, "entry": "yield from idle()",
                     "kind": "python"},
            "shutdown": {"cid": init_cid + 3,
                         "entry": "yield from shutdown()", "kind": "python"},
        }
        assert rebooted.get_conversation(init_cid + 1).attrs["guest_role"] == "boot"
        assert rebooted.get_conversation(init_cid).name == "init"

    def test_declaration_validation(self, tmp_path):
        image = write_image(tmp_path)
        with open(image, encoding="utf-8") as f:
            doc = json.load(f)
        doc["meta"]["guest_programs"] = {"clock": "$MEM.programs.main"}
        with open(image, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        with pytest.raises(ImageError, match="未知角色"):
            load_image(Core(), image)

        doc["meta"]["guest_programs"] = {"idle": "MEM.programs.main"}
        with open(image, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        with pytest.raises(ImageError, match="必须是 \\$ 开头"):
            load_image(Core(), image)

        doc["meta"]["guest_programs"] = {"idle": "$MEM.src"}
        with open(image, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        with pytest.raises(ImageError, match="必须是程序节点"):
            load_image(Core(), image)


# ----------------------------------------------------------------------
# 验收 7：空闲 = 严重错误
# ----------------------------------------------------------------------

class TestIdleIsFatal:
    def test_idle_interrupt_delivered_then_panic_when_still_idle(self, tmp_path):
        image = _guest_image(tmp_path)
        core = boot(image, mode="reset", refresh_models=False)

        with pytest.raises(KernelPanic, match="没有让系统离开空闲"):
            core.run()

        # idle 角色确实收到了中断并执行了程序
        assert core.mem["markers"]["idle_ran"] == "1"
        assert core.interrupts.last_dispatch[0]["type"] == "idle"
        assert core.interrupts.last_dispatch[0]["target"] == 2
        assert core.monitor.frames(), "严重错误要留监测帧"

    def test_no_idle_role_panics(self, tmp_path):
        image = _guest_image(tmp_path, roles=("boot",))
        core = boot(image, mode="reset", refresh_models=False)
        assert "idle" not in core.interrupts.vectors

        with pytest.raises(KernelPanic, match="没有声明 idle guest 程序"):
            core.run()

    def test_panel_active_is_not_idle(self):
        core = Core()
        core.ensure_operator_conversation()   # 面板等待输入：休眠、由中断唤醒
        core._active_cid = PANEL_CID          # 面板回合进行中
        assert not core._idle_now(), "活跃对话（含面板等待输入）不算空闲"
        core._active_cid = None
        assert core._idle_now(), "没有活跃/就绪/待派发中断才是空闲"

    def test_legacy_image_without_guest_programs_returns_when_idle(self, tmp_path):
        """兼容：没有 guest 声明的旧镜像保持旧语义，队列空了 run() 正常返回。"""
        image = write_image(tmp_path)
        core = load_core(image, [("done", [])])
        assert core.strict_scheduler is False
        core.run()
        assert core._active_cid is None


class TestShutdownSupervisorWake:
    """OS 提交关机请求后，宿主用中断唤醒声明的 shutdown 监督程序（不自行断电）。"""

    def _mem_with_power(self, init_code):
        mem = _guest_mem(idle_code="def idle():\n    pass\n")
        mem["programs"]["value"]["init"] = _py_program(init_code)
        mem["sys"] = {"kind": "dict", "value": {"power": {"kind": "dict", "value": {
            "request": {"kind": "str", "value": ""},
            "state": {"kind": "str", "value": "on"},
        }}}}
        return mem

    def test_power_request_change_wakes_shutdown_role(self, tmp_path):
        init_code = (
            "def init():\n"
            "    yield from memory_write('$MEM.sys.power.request', 'shutdown')\n"
        )
        image = write_image(
            tmp_path, mem=self._mem_with_power(init_code),
            init={"name": "init", "system_ref": "$MEM.programs.init",
                  "user": "yield from init()", "para_ref": "$MEM.para.python"},
        )
        with open(image, encoding="utf-8") as f:
            doc = json.load(f)
        doc["meta"]["guest_programs"] = {"shutdown": "$MEM.programs.shutdown"}
        with open(image, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        core = boot(image, mode="reset", refresh_models=False)
        assert core.recovery["interrupt_vectors"].keys() == {"shutdown"}

        # 没有 idle 角色：监督程序睡回去之后的空闲是严重错误；先验证中断确实送到。
        with pytest.raises(KernelPanic, match="没有声明 idle"):
            core.run()
        assert core.mem["markers"]["shutdown_ran"] == "1"
        assert any(record["type"] == "shutdown"
                   for record in core.interrupts.last_dispatch or [])

    def test_stale_request_at_boot_does_not_wake_supervisor(self, tmp_path):
        init_code = (
            "def init():\n"
            "    yield from memory_write('$MEM.markers.init_ran', '1')\n"
        )
        mem = self._mem_with_power(init_code)
        mem["sys"]["value"]["power"]["value"]["request"]["value"] = "shutdown"
        image = write_image(
            tmp_path, mem=mem,
            init={"name": "init", "system_ref": "$MEM.programs.init",
                  "user": "yield from init()", "para_ref": "$MEM.para.python"},
        )
        with open(image, encoding="utf-8") as f:
            doc = json.load(f)
        doc["meta"]["guest_programs"] = {"shutdown": "$MEM.programs.shutdown"}
        with open(image, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        core = boot(image, mode="reset", refresh_models=False)
        with pytest.raises(KernelPanic, match="没有声明 idle"):
            core.run()
        # 存量请求（resume 会恢复上一次关机写下的值）不重新唤醒监督程序
        assert core.mem["markers"]["shutdown_ran"] == ""
        assert not core.interrupts.last_dispatch


# ----------------------------------------------------------------------
# 验收 8（内核侧）：断电边界与恢复报告
# ----------------------------------------------------------------------

class TestPowerBoundary:
    def _core(self, tmp_path, responses):
        image = write_image(tmp_path, mem=_power_mem())
        core = load_core(image, responses)
        core.attach_journal(core.journal_path)
        return core

    def test_power_state_off_returns_from_run_without_stop_request(self, tmp_path):
        core = self._core(tmp_path, [
            ("", [{"call_id": "c1", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.sys.power.state", "content": "off"}}]),
        ])
        core.run()
        assert core.mem["sys"]["power"]["state"] == "off"
        assert core._stop_requested is False
        assert core._journal_seq == 1, "断电前的回合已提交"
        assert core.journal.last_seq() == 1

    def test_boot_time_power_off_is_not_a_new_request(self, tmp_path):
        core = self._core(tmp_path, [("hello", [])])
        core.mem["sys"]["power"]["state"] = "off"
        core.run()
        # 存量 off 不触发断电：会话照常跑完
        assert core._conv_by_cid[0].messages[-1].content == "hello"
        assert core._stop_requested is False

    def test_recovery_report_fields(self, tmp_path):
        core = self._core(tmp_path, [("done", [])])
        core.run()
        core.snapshot_to(core.snapshot_path)
        resumed = boot(str(tmp_path / "image.json"), mode="resume",
                       refresh_models=False)
        report = resumed.recovery
        for key in ("mode", "world_source", "snapshot_seq", "replayed",
                    "generation", "next_cid", "guest_programs",
                    "interrupt_vectors", "last_exit"):
            assert key in report
        assert report["last_exit"] == "normal"


# ----------------------------------------------------------------------
# 控制台：关机请求走 OS 约定，不再直接 request_stop
# ----------------------------------------------------------------------

class TestConsoleShutdownRequest:
    def _console(self, tmp_path, mem=None):
        image = write_image(tmp_path, mem=mem or _power_mem())
        core = load_core(image)
        core.attach_journal(core.journal_path)
        console = Console(core, mount_io_devices=False)
        return core, console

    def test_shutdown_writes_os_request_through_panel(self, tmp_path):
        core, console = self._console(tmp_path)
        response = console.request_shutdown()

        assert response["status"] == "stopping"
        assert response["entry"] == "$MEM.sys.power.request"
        assert core._stop_requested is False, "关机请求不再直接停内核"
        assert console.lifecycle == "stopping"
        assert console.channel.status()["pending_actions"] == 2

        _advance_panel(core, console)
        assert core.mem["sys"]["power"]["request"] == "shutdown"
        effects = [
            change["effect"]
            for record in core.journal.read()
            for change in record["changes"]
            if change["kind"] == "effect"
        ]
        assert any(effect["kind"] == "memory_set"
                   and effect["ref"] == "$MEM.sys.power.request"
                   and effect["value"] == "shutdown" for effect in effects)
        # 再次提交关机请求被拒
        with pytest.raises(ConsoleConflict):
            console.request_shutdown()

    def test_shutdown_without_os_entry_warns_but_submits(self, tmp_path):
        from persist_helpers import base_image_mem

        core, console = self._console(tmp_path, mem=base_image_mem())
        response = console.request_shutdown()
        assert response["status"] == "stopping"
        assert response["warning"] and "power.request" in response["warning"]
        _advance_panel(core, console)
        # 缺失节点时写入以工具错误结束，不崩内核
        assert core._stop_requested is False

    def test_force_stop_immediately_ends_host_without_snapshot(self, tmp_path):
        import threading

        core, console = self._console(tmp_path)
        exited = threading.Event()
        console.force_stop_hook = exited.set
        console.force_stop_delay = 0.0
        console.request_shutdown()

        response = console.force_stop()

        assert response == {"status": "force-stopped", "snapshot_written": False}
        assert console.lifecycle == "stopped"
        assert exited.wait(timeout=2), "force-stop 回调（宿主退出）被调用"
        assert not os.path.exists(
            out_path(str(tmp_path / "image.json"), "run.snapshot.json")
        ), "强制退出不写快照"


# ----------------------------------------------------------------------
# 模式校验
# ----------------------------------------------------------------------

class TestModeValidation:
    def test_unknown_mode_is_rejected(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError, match="未知启动模式"):
            boot(image, mode="cold", refresh_models=False)

    def test_restore_argument_only_for_resume(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError, match="restore 参数只用于 resume"):
            boot(image, mode="reboot", restore="x.json", refresh_models=False)

    def test_checkpoint_in_rejected_with_snapshot_modes(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError, match="checkpoint_in"):
            boot(image, mode="resume", checkpoint_in="cp.json", refresh_models=False)

    def test_old_call_without_mode_still_works(self, tmp_path):
        image = write_image(tmp_path)
        core = boot(image, refresh_models=False)
        assert core.recovery["mode"] == "auto"
        assert sorted(core._conv_by_cid) == [0]
        assert core.journal_generation == 1
        assert core._journal_seq == 0

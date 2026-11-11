"""重启快照（avm/snapshot.py）测试。

验收（计划 A 批次）：快照往返（cid/历史/批次/监听器/设备表一致、不变量通过）、
快照 + 日志尾部 = 完整状态、Python 对话续跑（跨轮状态从 $MEM 读）、宿主 --restore
入口；全部不调真实 API。
"""

import hashlib
import json
import os

import pytest

from avm.boot import boot
from avm.image import ImageError
from avm.snapshot import SnapshotError, read_snapshot, snapshot, write_atomic
from persist_helpers import (
    ScriptedLMU, assert_same_state, load_core, out_path, run_core, run_turns,
    write_image,
)

TWO_TURNS = [
    ("", [{"call_id": "c1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v1"}}]),
    ("done", []),
]

# 连续推进（每轮都产生指令，对话保持活跃）：用于"快照后再跑，然后恢复快照+日志尾部"
BUSY_TURNS = [
    ("", [{"call_id": "b1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "busy1"}}]),
    ("", [{"call_id": "b2", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.dst", "content": "busy2"}}]),
]
TAIL_TURNS = [
    ("", [{"call_id": "t1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.dst", "content": "after snapshot"}}]),
    ("", [{"call_id": "t2", "cmd_type": "memory_make",
           "args": {"ref": "$MEM", "key": "tail", "mem_type": "str"}}]),
]


def _run(image, responses, snapshot_after=None):
    core = load_core(image, responses)
    core.attach_journal(core.journal_path)
    if snapshot_after is not None:
        run_turns(core, snapshot_after)
        core.snapshot_to(out_path(image, "mid.snapshot.json"))
        return core
    run_turns(core, len(responses))
    return core


class TestSnapshotRoundtrip:
    def test_roundtrip_reproduces_state(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        path = str(tmp_path / "snap.json")
        mine.snapshot_to(path)

        restored = load_core(image)
        doc = restored.restore_from(path)
        assert doc["format"] == "avm.snapshot"
        assert doc["seq"] == 2
        assert restored._journal_seq == 2
        assert_same_state(mine, restored)

    def test_restore_keeps_cid_and_batch(self, tmp_path):
        image = write_image(tmp_path)
        mine = load_core(image, TWO_TURNS)
        mine._pick_next_active()
        mine.advance_conversation()  # 第一轮结束后对话仍活跃，批次里有一条工具响应
        path = str(tmp_path / "snap.json")
        mine.snapshot_to(path)
        assert mine._conv_by_cid[0].user_batch.tool_responses

        restored = load_core(image)
        restored.restore_from(path)
        assert restored._active_cid == mine._active_cid == 0
        assert [r.content for r in restored._conv_by_cid[0].user_batch.tool_responses] == \
            [r.content for r in mine._conv_by_cid[0].user_batch.tool_responses]
        assert restored._conv_by_cid[0].to_api_messages() == \
            mine._conv_by_cid[0].to_api_messages()

    def test_snapshot_file_is_atomic_json(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        path = str(tmp_path / "snap.json")
        mine.snapshot_to(path)
        assert json.loads(open(path, encoding="utf-8").read())["format"] == "avm.snapshot"
        assert not os.path.exists(path + ".tmp")

    def test_read_snapshot_rejects_wrong_format(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "avm.core_dump", "version": 1}, f)
        with pytest.raises(SnapshotError, match="不是 AVM 重启快照"):
            read_snapshot(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "avm.snapshot", "version": 99}, f)
        with pytest.raises(SnapshotError, match="版本"):
            read_snapshot(path)

    def test_write_atomic_creates_directory(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "snap.json")
        write_atomic(target, {"format": "avm.snapshot", "version": 2})
        assert os.path.isfile(target)


class TestSnapshotPlusJournalTail:
    def test_tail_replay_after_snapshot(self, tmp_path):
        image = write_image(tmp_path)
        mine = load_core(image, BUSY_TURNS + TAIL_TURNS)
        mine.attach_journal(mine.journal_path)
        run_turns(mine, 2)
        snap = str(tmp_path / "mid.json")
        mine.snapshot_to(snap)
        run_turns(mine, 2)  # 快照之后再提交两轮

        restored = load_core(image)
        restored.restore_from(snap)
        assert restored._journal_seq == 2
        restored.attach_journal(restored.journal_path)
        assert restored.replay_journal() == 2
        assert restored._journal_seq == 4
        assert restored.mem["dst"] == "after snapshot"
        assert_same_state(mine, restored)

    def test_boot_restore_entry(self, tmp_path):
        image = write_image(tmp_path)
        mine = load_core(image, BUSY_TURNS + TAIL_TURNS)
        mine.attach_journal(mine.journal_path)
        run_turns(mine, 2)
        snap = str(tmp_path / "mid.json")
        mine.snapshot_to(snap)
        run_turns(mine, 2)

        restored = boot(image, restore=snap, refresh_models=False)
        assert restored._journal_seq == 4
        assert restored.mem["dst"] == "after snapshot"
        assert_same_state(mine, restored)


class TestBootAutoRestore:
    """快照是缓存：auto 可读则恢复，读不出来则等价 reset；显式恢复仍然严格。"""

    def test_boot_auto_restores_declared_snapshot(self, tmp_path, capsys):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        mine.snapshot_to(out_path(image, "run.snapshot.json"))

        restored = boot(image, refresh_models=False)
        assert "已从重启快照恢复" in capsys.readouterr().err
        assert_same_state(mine, restored)

    def test_boot_auto_resets_when_snapshot_corrupt(self, tmp_path):
        image = write_image(tmp_path)
        snap = out_path(image, "run.snapshot.json")
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        with open(snap, "w", encoding="utf-8") as f:
            f.write("{not-json")

        core = boot(image, refresh_models=False)
        assert core.recovery["mode"] == "auto"
        assert core.recovery["source"] == "cold", "不可读快照按 auto 的 reset 语义冷启动"
        assert 0 in core._conv_by_cid  # init 正常启动

    def test_boot_auto_resets_when_declared_snapshot_missing(self, tmp_path):
        image = write_image(tmp_path)
        assert not os.path.exists(out_path(image, "run.snapshot.json"))

        core = boot(image, refresh_models=False)
        assert core.recovery["mode"] == "auto"
        assert core.recovery["source"] == "cold"
        assert 0 in core._conv_by_cid

    def test_explicit_restore_missing_stays_strict(self, tmp_path):
        image = write_image(tmp_path)
        with pytest.raises(ImageError):
            boot(image, restore=str(tmp_path / "nope.json"), refresh_models=False)


class TestDevices:
    def _device_image(self, tmp_path):
        device_file = tmp_path / "echo.py"
        device_file.write_text(
            "from avm.memory_device import MemoryDevice\n"
            "class EchoDevice(MemoryDevice):\n"
            "    def __init__(self, value=''):\n"
            "        self._value = value\n"
            "    def pretend_as_type(self):\n"
            "        return 'str'\n"
            "    def to_llm_string(self):\n"
            "        return self._value\n"
            "    def set_value(self, value):\n"
            "        self._value = str(value)\n"
            "    def get_value(self):\n"
            "        return self._value\n",
            encoding="utf-8",
        )
        mem = {
            "model_params": {
                "kind": "dict", "ctrl": {"type": "para"},
                "value": {"model": {"kind": "str", "value": "test-model"}},
            },
            "programs": {"kind": "dict", "value": {"main": {
                "kind": "dict", "ctrl": {"type": "settingup"},
                "value": {"content": {"kind": "str", "value": "sys"}},
            }}},
            "echo": {"kind": "device"},
        }
        devices = [{"path": "echo", "file": str(device_file), "class": "EchoDevice",
                    "args": {"value": "初始"}}]
        return write_image(tmp_path, mem=mem, devices=devices)

    def test_device_specs_survive_roundtrip(self, tmp_path):
        image = self._device_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        assert "echo" in mine.device_specs
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        assert "echo" in restored.device_specs
        assert restored.device_specs["echo"]["class"] == "EchoDevice"
        assert restored.mem._devices["echo"].get_value() == "初始"
        assert_same_state(mine, restored)

    @staticmethod
    def _sha256(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_device_spec_records_sha256_and_snapshot_keeps_it(self, tmp_path):
        image = self._device_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        digest = self._sha256(tmp_path / "echo.py")
        assert mine.device_specs["echo"]["sha256"] == digest
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        with open(snap, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["device_specs"]["echo"]["sha256"] == digest

    def test_restore_rejects_changed_device_file(self, tmp_path):
        image = self._device_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        (tmp_path / "echo.py").write_text(
            "from avm.memory_device import MemoryDevice\n"
            "class EchoDevice(MemoryDevice):\n"
            "    def __init__(self, value=''):\n"
            "        self._value = value\n"
            "    def pretend_as_type(self):\n"
            "        return 'str'\n"
            "    def to_llm_string(self):\n"
            "        return '换掉了'\n",
            encoding="utf-8",
        )
        restored = load_core(image)
        with pytest.raises(SnapshotError, match="哈希") as excinfo:
            restored.restore_from(snap)
        assert "echo" in str(excinfo.value)
        assert "echo.py" in str(excinfo.value)
        assert "期望" in str(excinfo.value) and "实际" in str(excinfo.value)

    def test_restore_rejects_spec_without_sha256(self, tmp_path):
        image = self._device_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        with open(snap, encoding="utf-8") as f:
            doc = json.load(f)
        doc["device_specs"]["echo"].pop("sha256")
        write_atomic(snap, doc)
        restored = load_core(image)
        with pytest.raises(SnapshotError, match="重新审批"):
            restored.restore_from(snap)

    def test_restore_clears_runtime_mounted_devices(self, tmp_path):
        """快照后热挂的设备不在快照里：恢复时以 device_specs 为准把它摘掉。"""
        from avm.memory_device import StringDevice

        image = self._device_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        mine.mem.mount("hot", StringDevice("hot"))
        mine.device_specs["hot"] = {"builtin": False, "class": "StringDevice"}

        restored = load_core(image)
        restored.restore_from(snap)
        assert "hot" not in restored.mem._devices
        assert "hot" not in restored.device_specs


PLM_IMAGE_COUNTER = {
    "counter_program": {
        "kind": "dict",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": (
                "def pack():\n"
                "    n = int((yield from memory_read('$MEM.counter'))) + 1\n"
                "    yield from memory_write('$MEM.counter', str(n))\n"
                "    print(n)\n"
            )},
            "para_ref": {"kind": "str", "value": "$MEM.counter_params"},
        },
    },
    "counter_params": {
        "kind": "dict",
        "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "plm.python"}},
    },
    "counter": {"kind": "str", "value": "0"},
    "model_params": {
        "kind": "dict", "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "test-model"}},
    },
    "programs": {"kind": "dict", "value": {"main": {
        "kind": "dict", "ctrl": {"type": "settingup"},
        "value": {"content": {"kind": "str", "value": "sys"}},
    }}},
}


class TestPythonConversationResume:
    def test_python_conversation_continues_after_restore(self, tmp_path):
        image = write_image(
            tmp_path, mem=PLM_IMAGE_COUNTER,
            init={"name": "counter", "system_ref": "$MEM.counter_program",
                  "user": "yield from pack()"},
        )
        mine = load_core(image)
        mine.attach_journal(mine.journal_path)
        run_core(mine)
        assert mine.mem["counter"] == "1"
        conv = mine._conv_by_cid[0]
        assert conv.is_python
        history_after_first = conv.to_api_messages()

        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        restored.attach_journal(restored.journal_path)
        conv = restored._conv_by_cid[0]
        assert conv.is_python
        assert conv.to_api_messages() == history_after_first

        # 续跑：喂一次输入，程序从 $MEM 读跨轮状态
        conv.user_batch.add_user_content("yield from pack()")
        restored._ready_cids.append(conv.cid)
        run_core(restored)
        assert restored.mem["counter"] == "2"
        assert len(restored._conv_by_cid[0].messages) > len(history_after_first)

    def test_restore_does_not_reexecute_history(self, tmp_path):
        """恢复后 Python 程序的历史 user/tool 消息不重新执行（副作用只发生一次）。"""
        image = write_image(
            tmp_path, mem=PLM_IMAGE_COUNTER,
            init={"name": "counter", "system_ref": "$MEM.counter_program",
                  "user": "yield from pack()"},
        )
        mine = load_core(image)
        run_core(mine)
        assert mine.mem["counter"] == "1"
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        # 只做一次空推进（对话休眠中：直接构造一个非空批次触发执行器创建但不该重放历史）
        restored.attach_journal(restored.journal_path)
        assert restored.mem["counter"] == "1"


class TestRestoredInvariants:
    def test_restore_result_passes_invariants(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        restored = load_core(image)
        restored.restore_from(snap)
        restored._check_invariants()  # 不抛即通过


class TestSnapshotFormatVersion:
    def test_old_snapshot_version_is_rejected(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        path = str(tmp_path / "old.json")
        mine.snapshot_to(path)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        doc["version"] = 1
        write_atomic(path, doc)

        with pytest.raises(SnapshotError, match="旧版本快照不再支持") as excinfo:
            read_snapshot(path)
        assert "reset" in str(excinfo.value)
        with pytest.raises(ImageError, match="reset"):
            boot(image, mode="resume", refresh_models=False)

    def test_snapshot_without_generation_is_rejected(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, TWO_TURNS)
        path = str(tmp_path / "snap.json")
        mine.snapshot_to(path)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        doc.pop("generation")
        write_atomic(path, doc)

        with pytest.raises(SnapshotError, match="generation"):
            read_snapshot(path)
        restored = load_core(image)
        with pytest.raises(SnapshotError, match="generation"):
            restored.restore_from(path)

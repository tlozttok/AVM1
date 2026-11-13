"""设备装载入口（images/devices/device_loader.py）与设备审批哈希绑定测试。

验收（设备审批计划，包机制已删）：正确哈希装载成功且挂载路径正确；错误/缺失哈希
Error 且无挂载；坏 JSON 与未知 action 是 Illegal；password 字段拒绝；恢复时改设备
文件失败；热装设备经日志重放后仍在；快照记录缺 sha256 拒绝恢复；内置设备恢复不受
影响。全部不调真实 API。
"""

import hashlib
import json
import os

import pytest

from avm.boot import boot
from avm.image import ImageError
from avm.snapshot import SnapshotError, write_atomic
from persist_helpers import (
    assert_same_state, load_core, run_core, run_turns, write_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE_LOADER_FILE = os.path.join(
    REPO_ROOT, "images", "devices", "device_loader.py")


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_echo_device(tmp_path, name="echo.py", default_value=""):
    path = tmp_path / name
    path.write_text(
        "from avm.memory_device import MemoryDevice\n"
        "class EchoDevice(MemoryDevice):\n"
        f"    def __init__(self, value={default_value!r}):\n"
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
    return path


def device_loader_image(tmp_path, *, devices=None):
    mem = {
        "model_params": {
            "kind": "dict", "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {"kind": "dict", "value": {"main": {
            "kind": "dict", "ctrl": {"type": "settingup"},
            "value": {"content": {"kind": "str", "value": "sys"}},
        }}},
        "device_loader": {"kind": "device"},
    }
    all_devices = [{
        "path": "device_loader",
        "file": DEVICE_LOADER_FILE,
        "class": "DeviceLoader",
        "args": {},
    }] + list(devices or [])
    return write_image(tmp_path, mem=mem, devices=all_devices)


def command_turn(command, call_id="w1"):
    return ("", [{
        "call_id": call_id,
        "cmd_type": "memory_write",
        "args": {"ref": "$MEM.device_loader",
                 "content": json.dumps(command, ensure_ascii=False)},
    }])


def run_commands(tmp_path, commands):
    image = device_loader_image(tmp_path)
    core = load_core(image, [command_turn(cmd, f"w{i}") for i, cmd in enumerate(commands)]
                     + [("done", [])])
    core.attach_journal(core.journal_path)
    run_turns(core, len(commands) + 1)
    return core, image


def tool_responses(core):
    conv = core._conv_by_cid[0]
    from avm.types import ToolMessage

    return [m.content for m in conv.messages if isinstance(m, ToolMessage)]


def load_device_command(device_file, *, mount="extra", approved=None, **extra):
    command = {
        "action": "load_device", "path": mount,
        "file": str(device_file), "class": "EchoDevice",
        "args": {"value": "热装设备"},
    }
    if approved is not None:
        command["approved_sha256"] = approved
    command.update(extra)
    return command


class TestDeviceLoaderCommands:
    def test_load_device_with_correct_hash(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        digest = sha256_file(device_file)
        core, _ = run_commands(tmp_path, [load_device_command(device_file, approved=digest)])
        assert "extra" in core.mem._devices
        assert core.mem._devices["extra"].get_value() == "热装设备"
        spec = core.device_specs["extra"]
        assert spec["class"] == "EchoDevice"
        assert spec["path"] == str(device_file)
        assert spec["sha256"] == digest
        assert spec["approved_sha256"] == digest
        assert spec["approved_at"]
        assert spec["approved_via"] == "device_loader"
        assert tool_responses(core)[0] == "Success set: $MEM.device_loader"

    def test_load_device_wrong_hash_fails_without_mount(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [
            load_device_command(device_file, approved="0" * 64),
        ])
        response = tool_responses(core)[0]
        assert response.startswith("Error:")
        assert "哈希" in response and "拒绝装载" in response
        assert "extra" not in core.mem._devices
        assert "extra" not in core.device_specs

    def test_load_device_missing_approval_is_error(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [load_device_command(device_file)])
        assert tool_responses(core)[0].startswith("Error:")
        assert "approved_sha256" in tool_responses(core)[0]
        assert "extra" not in core.mem._devices

    def test_load_device_requires_path_file_and_class(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        digest = sha256_file(device_file)
        command = {"action": "load_device", "approved_sha256": digest,
                   "file": str(device_file), "class": "EchoDevice"}
        core, _ = run_commands(tmp_path, [command])
        assert tool_responses(core)[0] == "Illegal: load_device 需要非空挂载 path"
        assert "extra" not in core.mem._devices

    def test_password_is_rejected(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [
            load_device_command(device_file, password="operator-secret"),
        ])
        response = tool_responses(core)[0]
        assert response.startswith("Illegal:")
        assert "password" in response and "approved_sha256" in response
        assert "extra" not in core.mem._devices

    def test_bad_json_is_illegal(self, tmp_path):
        image = device_loader_image(tmp_path)
        core = load_core(image, [(
            "", [{"call_id": "w1", "cmd_type": "memory_write",
                  "args": {"ref": "$MEM.device_loader", "content": "{not json"}}]),
            ("done", []),
        ])
        run_turns(core, 2)
        assert tool_responses(core)[0] == "Illegal: 装载命令不是合法 JSON"

    def test_unknown_action_is_illegal(self, tmp_path):
        # 包动作已删除：老命令落到未知 action 分支，一律拒绝，也不会挂载任何设备。
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [{
            "action": "install_package", "path": str(device_file),
            "approved_sha256": sha256_file(device_file),
        }])
        response = tool_responses(core)[0]
        assert response.startswith("Illegal:")
        assert "action 不支持" in response and "install_package" in response
        assert "extra" not in core.mem._devices


class TestDeviceLoaderPersistence:
    def test_hot_mounted_device_replays_from_journal(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = load_device_command(device_file, approved=sha256_file(device_file))
        mine, image = run_commands(tmp_path, [command])
        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        replayed.replay_journal()
        assert "extra" in replayed.mem._devices
        assert replayed.device_specs["extra"]["class"] == "EchoDevice"
        assert replayed.device_specs["extra"]["sha256"] == sha256_file(device_file)
        assert_same_state(mine, replayed)

    def test_journal_replay_rejects_changed_device_file(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = load_device_command(device_file, approved=sha256_file(device_file))
        _, image = run_commands(tmp_path, [command])
        device_file.write_text(
            "from avm.memory_device import MemoryDevice\n"
            "class EchoDevice(MemoryDevice):\n"
            "    def pretend_as_type(self):\n"
            "        return 'str'\n"
            "    def to_llm_string(self):\n"
            "        return '换掉了'\n",
            encoding="utf-8",
        )
        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        with pytest.raises(Exception, match="哈希"):
            replayed.replay_journal()
        assert "extra" not in replayed.mem._devices

    def test_snapshot_restore_accepts_unchanged_device(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = load_device_command(device_file, approved=sha256_file(device_file))
        mine, image = run_commands(tmp_path, [command])
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        assert restored.mem._devices["extra"].get_value() == "热装设备"
        recorded = restored.device_specs["extra"]
        assert recorded["sha256"] == sha256_file(device_file)
        assert recorded["approved_sha256"] == sha256_file(device_file)
        assert recorded["approved_at"]
        assert recorded["approved_via"] == "device_loader"

    def test_snapshot_restore_rejects_changed_device_file(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = load_device_command(device_file, approved=sha256_file(device_file))
        mine, image = run_commands(tmp_path, [command])
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        device_file.write_text(
            "from avm.memory_device import MemoryDevice\n"
            "class EchoDevice(MemoryDevice):\n"
            "    def pretend_as_type(self):\n"
            "        return 'str'\n"
            "    def to_llm_string(self):\n"
            "        return '换掉了'\n",
            encoding="utf-8",
        )
        with pytest.raises((ImageError, SnapshotError), match="哈希") as excinfo:
            boot(image, restore=snap, refresh_models=False)
        message = str(excinfo.value)
        assert "extra" in message
        assert str(device_file) in message
        assert "期望" in message and "实际" in message

    def test_old_snapshot_spec_without_sha256_is_rejected(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = load_device_command(device_file, approved=sha256_file(device_file))
        mine, image = run_commands(tmp_path, [command])
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)
        with open(snap, encoding="utf-8") as f:
            doc = json.load(f)
        doc["device_specs"]["extra"].pop("sha256")
        write_atomic(snap, doc)

        restored = load_core(image)
        with pytest.raises(SnapshotError, match="重新审批"):
            restored.restore_from(snap)
        assert "extra" not in restored.mem._devices


class TestBuiltinDeviceRestore:
    def test_builtin_specs_without_sha256_still_restore(self, tmp_path):
        image = write_image(tmp_path)
        mine = load_core(image, [("done", [])])
        mine.attach_journal(mine.journal_path)
        run_core(mine)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        for mount in ("conversations", "scheduler", "memory", "monitor",
                      "listeners", "roots", "caps"):
            assert restored.device_specs[mount]["builtin"] is True
            assert mount in restored.mem._devices

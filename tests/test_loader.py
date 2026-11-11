"""装载设备（images/packages/loader.py）与设备审批哈希绑定测试。

验收（设备审批计划）：正确哈希装载成功；错误/缺失哈希 Error 且无挂载/无记录；
程序包装载同时核对设备文件哈希并写安装记录；恢复时改文件失败；去密码；
升级需重新审批；快照记录缺 sha256 拒绝恢复；内置设备恢复不受影响。
全部不调真实 API。
"""

import hashlib
import json
import os

import pytest

from avm import packages
from avm.boot import boot
from avm.image import ImageError
from avm.snapshot import SnapshotError, write_atomic
from persist_helpers import (
    ScriptedLMU, assert_same_state, load_core, out_path, run_core, run_turns,
    write_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER_FILE = os.path.join(REPO_ROOT, "images", "packages", "loader.py")


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


def write_package(tmp_path, *, version="1_0_0", filename="demo-1_0_0.pkg.json",
                  device_file="echo.py", device_sha256=None):
    device_path = tmp_path / device_file
    if not device_path.exists():
        write_echo_device(tmp_path, name=device_file)
    device = {"path": "io", "file": device_file, "class": "EchoDevice",
              "args": {"value": "包内设备"}}
    if device_sha256 is not None:
        device["sha256"] = device_sha256
    package = {
        "meta": {"format": "avm.package", "version": 1, "name": "demo",
                 "package_version": version, "description": "演示程序包"},
        "mem": {
            "data": {"kind": "dict", "value": {"msg": {"kind": "str", "value": "包数据"}}},
            "program": {
                "kind": "dict",
                "ctrl": {"type": "settingup"},
                "value": {"content": {"kind": "str", "value": "包内程序系统消息"}},
            },
            "io": {"kind": "device"},
        },
        "devices": [device],
    }
    path = tmp_path / filename
    path.write_text(json.dumps(package, ensure_ascii=False), encoding="utf-8")
    return filename


def loader_image(tmp_path, *, devices=None):
    mem = {
        "model_params": {
            "kind": "dict", "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {"kind": "dict", "value": {"main": {
            "kind": "dict", "ctrl": {"type": "settingup"},
            "value": {"content": {"kind": "str", "value": "sys"}},
        }}},
        "loader": {"kind": "device"},
    }
    all_devices = [{
        "path": "loader",
        "file": LOADER_FILE,
        "class": "LoaderDevice",
        "args": {},
    }] + list(devices or [])
    return write_image(tmp_path, mem=mem, devices=all_devices)


def command_turn(command, call_id="w1"):
    return ("", [{
        "call_id": call_id,
        "cmd_type": "memory_write",
        "args": {"ref": "$MEM.loader",
                 "content": json.dumps(command, ensure_ascii=False)},
    }])


def run_commands(tmp_path, commands):
    image = loader_image(tmp_path)
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


def install_command(package_file, *, approved=None, **extra):
    command = {"action": "install_package", "path": package_file}
    if approved is not None:
        command["approved_sha256"] = approved
    command.update(extra)
    return command


class TestLoaderCommands:
    def test_load_device_with_correct_hash(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        digest = sha256_file(device_file)
        core, _ = run_commands(tmp_path, [load_device_command(device_file, approved=digest)])
        assert "extra" in core.mem._devices
        assert core.mem._devices["extra"].get_value() == "热装设备"
        spec = core.device_specs["extra"]
        assert spec["class"] == "EchoDevice"
        assert spec["sha256"] == digest
        assert spec["approved_sha256"] == digest
        assert spec["approved_at"]
        assert spec["approved_via"] == "loader"

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

    def test_password_is_rejected(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [
            load_device_command(device_file, password="operator-secret"),
        ])
        response = tool_responses(core)[0]
        assert response.startswith("Illegal:")
        assert "password" in response and "approved_sha256" in response
        assert "extra" not in core.mem._devices

    def test_install_with_correct_package_hash(self, tmp_path):
        package_file = write_package(tmp_path)
        digest = sha256_file(tmp_path / package_file)
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved=digest),
        ])
        assert tool_responses(core)[0] == "Success set: $MEM.loader"
        assert [item["package_version"] for item in packages.list_installed(core)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in core.mem._devices
        assert core.mem._devices[mount].get_value() == "包内设备"
        assert core.device_specs[mount]["sha256"] == sha256_file(tmp_path / "echo.py")

    def test_install_wrong_package_hash_fails_without_record(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved="f" * 64),
        ])
        response = tool_responses(core)[0]
        assert response.startswith("Error:")
        assert "程序包哈希" in response and "拒绝安装" in response
        assert packages.list_installed(core) == []
        assert "packages.demo.1_0_0.io" not in core.mem._devices
        assert "packages" not in core.mem._data

    def test_install_missing_approval_is_error(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [install_command(package_file)])
        assert tool_responses(core)[0].startswith("Error:")
        assert packages.list_installed(core) == []
        assert "packages.demo.1_0_0.io" not in core.mem._devices

    def test_bad_json_is_illegal(self, tmp_path):
        image = loader_image(tmp_path)
        core = load_core(image, [(
            "", [{"call_id": "w1", "cmd_type": "memory_write",
                  "args": {"ref": "$MEM.loader", "content": "{not json"}}]),
            ("done", []),
        ])
        run_turns(core, 2)
        assert tool_responses(core)[0] == "Illegal: 装载命令不是合法 JSON"

    def test_install_twice_second_is_error(self, tmp_path):
        package_file = write_package(tmp_path)
        command = install_command(package_file, approved=sha256_file(tmp_path / package_file))
        core, _ = run_commands(tmp_path, [command, command])
        responses = tool_responses(core)
        assert responses[0] == "Success set: $MEM.loader"
        assert responses[1].startswith("Error: 程序包已安装")

    def test_uninstall_command(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved=sha256_file(tmp_path / package_file)),
            {"action": "uninstall_package", "name": "demo", "package_version": "1_0_0"},
        ])
        assert packages.list_installed(core) == []
        assert "packages.demo.1_0_0.io" not in core.mem._devices

    def test_uninstall_does_not_need_approval(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved=sha256_file(tmp_path / package_file)),
            {"action": "uninstall_package", "name": "demo", "package_version": "1_0_0",
             "approved_sha256": "0" * 64},
        ])
        # 卸载不加载代码，approved 只是被忽略；结果仍然是卸掉
        assert packages.list_installed(core) == []

    def test_package_records_device_hashes_and_approval(self, tmp_path):
        package_file = write_package(tmp_path)
        package_digest = sha256_file(tmp_path / package_file)
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved=package_digest,
                            approved_at="2026-11-11T00:00:00+0800",
                            approved_via="panel-session-1"),
        ])
        record = core.unwrap("$MEM.package_records.demo.1_0_0", for_llm=False)
        assert record["sha256"] == package_digest
        assert record["approved_sha256"] == package_digest
        assert record["approved_at"] == "2026-11-11T00:00:00+0800"
        assert record["approved_via"] == "panel-session-1"
        (device_entry,) = list(record["devices"])
        assert device_entry["mount"] == "packages.demo.1_0_0.io"
        assert device_entry["class"] == "EchoDevice"
        assert device_entry["sha256"] == sha256_file(tmp_path / "echo.py")
        assert device_entry["approved_sha256"] == package_digest
        assert device_entry["approved_at"] == "2026-11-11T00:00:00+0800"
        assert device_entry["approved_via"] == "panel-session-1"

    def test_package_device_pinned_hash_is_checked(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        package_file = write_package(
            tmp_path, device_sha256=sha256_file(device_file),
        )
        # 审批之后、安装之前把设备文件换掉：包哈希没变，但设备条目里的哈希对不上。
        device_file.write_text(
            "from avm.memory_device import MemoryDevice\n"
            "class EchoDevice(MemoryDevice):\n"
            "    def pretend_as_type(self):\n"
            "        return 'str'\n"
            "    def to_llm_string(self):\n"
            "        return '换掉了'\n",
            encoding="utf-8",
        )
        core, _ = run_commands(tmp_path, [
            install_command(package_file, approved=sha256_file(tmp_path / package_file)),
        ])
        response = tool_responses(core)[0]
        assert response.startswith("Error:")
        assert "文件哈希不一致" in response
        assert packages.list_installed(core) == []
        assert "packages.demo.1_0_0.io" not in core.mem._devices

    def test_upgrade_requires_new_package_approval(self, tmp_path):
        write_echo_device(tmp_path)
        package_v1 = write_package(tmp_path, version="1_0_0",
                                   filename="demo-1_0_0.pkg.json")
        old_approval = sha256_file(tmp_path / package_v1)
        # 新版本：同包名、改设备代码；旧批准是旧包文件的哈希，必须重新审批。
        write_echo_device(tmp_path, default_value="新版本设备")
        package_v2 = write_package(tmp_path, version="1_1_0",
                                   filename="demo-1_1_0.pkg.json")
        new_approval = sha256_file(tmp_path / package_v2)

        image = loader_image(tmp_path)
        core = load_core(image, [
            command_turn(install_command(package_v1, approved=old_approval), "w1"),
            command_turn(install_command(package_v2, approved=old_approval), "w2"),
            ("done", []),
        ])
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        responses = tool_responses(core)
        assert responses[0] == "Success set: $MEM.loader"
        assert responses[1].startswith("Error:")
        assert "程序包哈希" in responses[1]
        # 旧批准作废：新版本没装上，current 仍指旧版本。
        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_0_0"
        assert "1_1_0" not in core.mem._data["packages"]["demo"]

        # 重新审批后装载成功，current 才前移。
        core = load_core(image, [
            command_turn(install_command(package_v1, approved=old_approval), "w1"),
            command_turn(install_command(package_v2, approved=new_approval), "w2"),
            ("done", []),
        ])
        core.attach_journal(core.journal_path)
        run_turns(core, 3)
        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_1_0"


class TestLoaderPersistence:
    def test_install_replays_from_journal(self, tmp_path):
        package_file = write_package(tmp_path)
        command = install_command(package_file, approved=sha256_file(tmp_path / package_file))
        mine, image = run_commands(tmp_path, [command])

        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        assert replayed.replay_journal() == 2
        assert [item["package_version"] for item in packages.list_installed(replayed)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in replayed.mem._devices
        assert replayed.mem._devices[mount].get_value() == "包内设备"
        assert replayed.device_specs[mount]["sha256"] == sha256_file(tmp_path / "echo.py")
        assert_same_state(mine, replayed)

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
        assert recorded["approved_via"] == "loader"

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

    def test_end_to_end_restore_and_reuse_package(self, tmp_path):
        """操作员写 loader → 装包 → 用包内程序 → 快照 + 日志恢复 → 仍可用。"""
        package_file = write_package(tmp_path)
        install = install_command(package_file, approved=sha256_file(tmp_path / package_file))
        image = loader_image(tmp_path)
        responses = [
            command_turn(install, "w1"),
            ("", [{"call_id": "c1", "cmd_type": "create_cmd",
                   "args": {"system_ref": "$MEM.packages.demo.1_0_0.program",
                            "para_ref": "$MEM.model_params"}},
                  {"call_id": "c2", "cmd_type": "send_instruction",
                   "args": {"cid": 1, "content": "用包做点事"}}]),
            ("parent done", []),
            ("child done", []),
        ]
        mine = load_core(image, responses)
        mine.attach_journal(mine.journal_path)
        run_core(mine)
        child = mine._conv_by_cid[1]
        assert child.messages[0].content == "包内程序系统消息"
        assert any(m.role == "user" and m.content == "用包做点事" for m in child.messages)

        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = boot(image, restore=snap, refresh_models=False)
        assert [item["package_version"] for item in packages.list_installed(restored)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in restored.mem._devices
        assert restored.mem._devices[mount].get_value() == "包内设备"

        # 恢复后再次使用包内程序：新开一条对话走 create_cmd
        restored.lmu = ScriptedLMU([
            ("", [{"call_id": "r1", "cmd_type": "create_cmd",
                   "args": {"system_ref": "$MEM.packages.demo.1_0_0.program",
                            "para_ref": "$MEM.model_params"}},
                  {"call_id": "r2", "cmd_type": "send_instruction",
                   "args": {"cid": 3, "content": "恢复后再用一次"}}]),
            ("restored parent done", []),
            ("restored child done", []),
        ])
        conv = restored.start("sys", "重新使用程序包", para_ref="$MEM.model_params")
        run_core(restored)
        child = restored._conv_by_cid[3]
        assert child.messages[0].content == "包内程序系统消息"
        assert any(m.role == "user" and m.content == "恢复后再用一次" for m in child.messages)


# ----------------------------------------------------------------------
# 演示镜像（计划 B4）：loader + 示例包 + 内核态安全监听器
# ----------------------------------------------------------------------

DEMO_DIR = os.path.join(REPO_ROOT, "images", "packages")
DEMO_IMAGE = os.path.join(DEMO_DIR, "packages_demo.json")
DEMO_PACKAGE = os.path.join(DEMO_DIR, "example.pkg.json")


class _ActionChannel:
    """操作员通道：按顺序吐出动作，没有动作时返回 end。"""

    def __init__(self, actions):
        self.actions = list(actions)
        self.published = []

    def wait_action(self):
        return self.actions.pop(0) if self.actions else {"end": True}

    def publish(self, kind, data):
        self.published.append((kind, data))


def copy_demo_image(tmp_path):
    with open(DEMO_IMAGE, encoding="utf-8") as f:
        image = json.load(f)
    for device in image.get("devices") or []:
        if isinstance(device.get("file"), str) and not os.path.isabs(device["file"]):
            device["file"] = os.path.join(DEMO_DIR, device["file"])
    image["journal_to"] = "out/run.journal"
    image["snapshot_to"] = "out/run.snapshot.json"
    image["persist_to"] = "out/mem.json"
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def install_command_demo():
    return {"action": "install_package", "path": DEMO_PACKAGE,
            "approved_sha256": sha256_file(DEMO_PACKAGE)}


class TestDemoImage:
    def test_guard_registered_at_startup(self, tmp_path):
        image = copy_demo_image(tmp_path)
        core = load_core(image)
        run_core(core)
        (listener,) = core.events.listeners
        assert listener["owner_cid"] == 0
        assert listener["tier"] == "kernel"
        assert listener["priority"] == 100
        assert listener["condition"]["target"]["refs"]["normal"][0] == "device"

    def test_operator_can_install_package(self, tmp_path):
        image = copy_demo_image(tmp_path)
        core = load_core(image)
        run_core(core)  # init 程序注册守卫后休眠
        channel = _ActionChannel([
            {"tool": "memory_write",
             "arguments": {"ref": "$MEM.loader",
                           "content": json.dumps(install_command_demo(), ensure_ascii=False)}},
            {"end": True},
        ])
        panel = core.bind_console_conversation(channel)
        core._ready_cids.append(panel.cid)
        run_core(core)
        assert [item["package_version"] for item in packages.list_installed(core)] == ["1_0_0"]
        assert "1_0_0" in core.mem._data["packages"]["demo"]

    def test_non_operator_write_is_interrupted(self, tmp_path):
        image = copy_demo_image(tmp_path)
        core = load_core(image)
        run_core(core)  # init 程序注册守卫后休眠
        conv = core.start("sys", "hello", para_ref="$MEM.model_params")
        core.lmu = ScriptedLMU([(
            "", [{"call_id": "w1", "cmd_type": "memory_write",
                  "args": {"ref": "$MEM.loader",
                           "content": json.dumps(install_command_demo(), ensure_ascii=False)}}]),
            ("done", []),
        ])
        run_core(core)
        responses = [m.content for m in conv.messages if m.role == "tool"]
        assert responses and responses[0].startswith("Interrupted:")
        assert packages.list_installed(core) == []


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

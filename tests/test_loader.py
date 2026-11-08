"""装载设备（images/packages/loader.py）测试。

验收（计划 B3/B4）：正确密码成功、错误密码 Error 且不内核恐慌、坏 JSON Illegal、
运行期装载设备；安装的包与设备能随快照 + 日志恢复后继续使用。全部不调真实 API。
"""

import hashlib
import json
import os

import pytest

from avm import packages
from avm.boot import boot
from persist_helpers import (
    ScriptedLMU, assert_same_state, load_core, out_path, run_core, run_turns,
    write_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOADER_FILE = os.path.join(REPO_ROOT, "images", "packages", "loader.py")
PASSWORD = "operator-secret"
PASSWORD_SHA = hashlib.sha256(PASSWORD.encode("utf-8")).hexdigest()


def write_echo_device(tmp_path, name="echo.py"):
    path = tmp_path / name
    path.write_text(
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
    return path


def write_package(tmp_path, *, version="1_0_0", filename="demo-1_0_0.pkg.json"):
    write_echo_device(tmp_path)
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
        "devices": [{"path": "io", "file": "echo.py", "class": "EchoDevice",
                     "args": {"value": "包内设备"}}],
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
        "args": {"password_sha256": PASSWORD_SHA},
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


class TestLoaderCommands:
    def test_install_with_correct_password(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [{
            "action": "install_package", "password": PASSWORD, "path": package_file,
        }])
        assert tool_responses(core)[0] == "Success set: $MEM.loader"
        assert [item["package_version"] for item in packages.list_installed(core)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in core.mem._devices
        assert core.mem._devices[mount].get_value() == "包内设备"

    def test_wrong_password_is_error_not_panic(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [{
            "action": "install_package", "password": "wrong", "path": package_file,
        }])
        assert tool_responses(core)[0] == "Error: 密码不匹配"
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
        command = {"action": "install_package", "password": PASSWORD, "path": package_file}
        core, _ = run_commands(tmp_path, [command, command])
        responses = tool_responses(core)
        assert responses[0] == "Success set: $MEM.loader"
        assert responses[1].startswith("Error: 程序包已安装")

    def test_uninstall_command(self, tmp_path):
        package_file = write_package(tmp_path)
        core, _ = run_commands(tmp_path, [
            {"action": "install_package", "password": PASSWORD, "path": package_file},
            {"action": "uninstall_package", "password": PASSWORD,
             "name": "demo", "package_version": "1_0_0"},
        ])
        assert packages.list_installed(core) == []
        assert "packages.demo.1_0_0.io" not in core.mem._devices

    def test_load_device_command(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _ = run_commands(tmp_path, [{
            "action": "load_device", "password": PASSWORD, "path": "extra",
            "file": str(device_file), "class": "EchoDevice",
            "args": {"value": "热装设备"},
        }])
        assert "extra" in core.mem._devices
        assert core.mem._devices["extra"].get_value() == "热装设备"
        assert core.device_specs["extra"]["class"] == "EchoDevice"


class TestLoaderPersistence:
    def test_install_replays_from_journal(self, tmp_path):
        package_file = write_package(tmp_path)
        command = {"action": "install_package", "password": PASSWORD, "path": package_file}
        mine, image = run_commands(tmp_path, [command])

        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        assert replayed.replay_journal() == 2
        assert [item["package_version"] for item in packages.list_installed(replayed)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in replayed.mem._devices
        assert replayed.mem._devices[mount].get_value() == "包内设备"
        assert_same_state(mine, replayed)

    def test_hot_mounted_device_replays_from_journal(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        command = {"action": "load_device", "password": PASSWORD, "path": "extra",
                   "file": str(device_file), "class": "EchoDevice",
                   "args": {"value": "热装设备"}}
        mine, image = run_commands(tmp_path, [command])
        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        replayed.replay_journal()
        assert "extra" in replayed.mem._devices
        assert replayed.device_specs["extra"]["class"] == "EchoDevice"
        assert_same_state(mine, replayed)

    def test_end_to_end_restore_and_reuse_package(self, tmp_path):
        """操作员写 loader → 装包 → 用包内程序 → 快照 + 日志恢复 → 仍可用。"""
        package_file = write_package(tmp_path)
        image = loader_image(tmp_path)
        responses = [
            command_turn({"action": "install_package", "password": PASSWORD,
                          "path": package_file}, "w1"),
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
DEMO_PASSWORD = "demo-operator"  # 演示镜像 devices.args 里存的是它的 sha256（见 build_image.py）
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


def install_command():
    return {"action": "install_package", "password": DEMO_PASSWORD,
            "path": DEMO_PACKAGE}


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
                           "content": json.dumps(install_command(), ensure_ascii=False)}},
            {"end": True},
        ])
        panel = core.start_operator_conversation(channel)
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
                           "content": json.dumps(install_command(), ensure_ascii=False)}}]),
            ("done", []),
        ])
        run_core(core)
        responses = [m.content for m in conv.messages if m.role == "tool"]
        assert responses and responses[0].startswith("Interrupted:")
        assert packages.list_installed(core) == []

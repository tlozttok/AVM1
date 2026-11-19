"""设备装载入口（images/devices/device_loader.py）的申请/批准/挂载协议测试。

验收（设备装载审批计划，2026-11-16）：根写命令废弃；写 `.request.<名字>` 只登记申请；
面板设备动作批准/拒绝只改设备状态；批准后写 `.resume.<名字>` 在指令里完成挂载并记
`device_mount`；文件在批准前后变化都被拦截；多申请按名字共存；日志重放与快照恢复沿用
既有哈希核对。全部不调真实 API。
"""

import hashlib
import json
import os

import pytest

from avm.boot import boot
from avm.console import OperatorChannel
from avm.image import ImageError
from avm.snapshot import SnapshotError, write_atomic
from persist_helpers import (
    load_core, run_turns, write_image,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE_LOADER_FILE = os.path.join(
    REPO_ROOT, "images", "devices", "device_loader.py")

PY_PARA = {
    "kind": "dict",
    "ctrl": {"type": "para"},
    "value": {"model": {"kind": "str", "value": "plm.python"}},
}


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_echo_device(tmp_path, name="echo.py", value="热装设备"):
    path = tmp_path / name
    path.write_text(
        "from avm.memory_device import MemoryDevice\n"
        "class EchoDevice(MemoryDevice):\n"
        f"    def __init__(self, value={value!r}):\n"
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
        "para": PY_PARA,
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


def make_core(tmp_path, *, devices=None):
    """装载镜像、跑完 init，再把面板对话绑上操作员通道。

    init 用脚本化执行器跑一个空回合；之后的指令都由面板对话（plm.operator）执行，
    这样测试可以在两条指令之间调用设备的批准动作（面板动作本来就在 Core 之外）。
    """
    image = device_loader_image(tmp_path, devices=devices)
    core = load_core(image, [("done", [])])
    core.attach_journal(core.journal_path)
    run_turns(core, 1)
    channel = OperatorChannel()
    panel = core.bind_console_conversation(channel, para_ref="$MEM.para.python")
    return core, image, channel, panel


def run_instructions(core, channel, panel, pairs):
    """让面板对话执行一批 `memory_write`：(ref, content) 序列。"""
    actions = [
        {"tool": "memory_write", "arguments": {"ref": ref, "content": content}}
        for ref, content in pairs
    ] + [{"end": True}]
    channel.submit_actions(actions)
    if panel.cid in core._dormant_cids:
        core._dormant_cids.remove(panel.cid)
    core._active_cid = panel.cid
    for _ in range(20):
        if core._active_cid != panel.cid:
            return
        core.advance_conversation()
    raise AssertionError("面板对话没有让出执行权")


def responses(core, cid):
    from avm.types import ToolMessage

    conv = core._conv_by_cid[cid]
    return [m.content for m in conv.messages if isinstance(m, ToolMessage)]


def last_response(core, panel):
    return responses(core, panel.cid)[-1]


def device(core):
    return core.mem.mounted_devices()["device_loader"]


def action_named(core, name):
    actions = device(core).debug_actions()
    for action in actions:
        if action.get("name") == name:
            return action
    raise AssertionError(
        f"没有名为 {name!r} 的设备动作；现有：{[a.get('name') for a in actions]}")


def action_names(core):
    return [action.get("name") for action in device(core).debug_actions()]


def request_ref(name):
    return f"$MEM.device_loader.request.{name}"


def resume_ref(name):
    return f"$MEM.device_loader.resume.{name}"


def request_value(device_file, **overrides):
    payload = {
        "file": str(device_file),
        "class": "EchoDevice",
        "args": {"value": "热装设备"},
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def submit_request(core, channel, panel, name, device_file, **overrides):
    run_instructions(core, channel, panel, [
        (request_ref(name), request_value(device_file, **overrides)),
    ])
    return last_response(core, panel)


def approve(core, name):
    return action_named(core, f"批准装载 {name}")["run"]({})


def reject(core, name, reason=None):
    args = {} if reason is None else {"reason": reason}
    return action_named(core, f"拒绝装载 {name}")["run"](args)


def resume(core, channel, panel, name):
    run_instructions(core, channel, panel, [(resume_ref(name), "")])
    return last_response(core, panel)


def mounted_hot_device(tmp_path, *, name="extra"):
    """走完申请、批准、resume，返回 (core, image, channel, panel, device_file)。"""
    device_file = write_echo_device(tmp_path)
    core, image, channel, panel = make_core(tmp_path)
    assert submit_request(core, channel, panel, name, device_file) == (
        f"Success set: {request_ref(name)}")
    approve(core, name)
    assert resume(core, channel, panel, name) == f"Success set: {resume_ref(name)}"
    return core, image, channel, panel, device_file


class TestRequestProtocol:
    def test_request_records_pending_without_mount(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        response = submit_request(core, channel, panel, "extra", device_file)
        assert response == "Success set: $MEM.device_loader.request.extra"
        assert "extra" not in core.mem.mounted_devices()
        root = core.unwrap("$MEM.device_loader", for_llm=True)
        assert "待批准 1 台：extra" in root
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：pending" in detail
        assert "sha256：" + sha256_file(device_file) in detail
        listing = core.unwrap("$MEM.device_loader.request", for_llm=True)
        assert "extra  status=pending" in listing

    def test_root_write_is_illegal(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        legacy = json.dumps({
            "action": "load_device", "path": "extra",
            "file": str(device_file), "class": "EchoDevice",
            "approved_sha256": sha256_file(device_file),
        }, ensure_ascii=False)
        run_instructions(core, channel, panel, [("$MEM.device_loader", legacy)])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert ".request.<名字>" in response
        assert "extra" not in core.mem.mounted_devices()

    def test_bare_request_write_is_illegal(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [
            ("$MEM.device_loader.request", "{}"),
        ])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "名字" in response

    def test_bare_resume_write_is_illegal(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [
            ("$MEM.device_loader.resume", ""),
        ])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "名字" in response

    def test_legacy_approval_fields_are_rejected(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        payload = json.dumps({
            "file": str(device_file), "class": "EchoDevice",
            "approved_sha256": sha256_file(device_file),
        }, ensure_ascii=False)
        run_instructions(core, channel, panel, [
            (request_ref("extra"), payload),
        ])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "approved_sha256" in response
        assert "extra" not in core.mem.mounted_devices()

    def test_bad_json_is_illegal(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [(request_ref("extra"), "{not json")])
        response = last_response(core, panel)
        assert response == "Illegal: 装载申请不是合法 JSON"

    def test_unknown_field_is_illegal(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        payload = json.dumps({
            "file": str(device_file), "class": "EchoDevice", "oops": 1,
        }, ensure_ascii=False)
        run_instructions(core, channel, panel, [(request_ref("extra"), payload)])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "oops" in response

    def test_requires_file_and_class(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [
            (request_ref("extra"), json.dumps({"class": "EchoDevice"})),
        ])
        assert last_response(core, panel).startswith("Illegal:")
        run_instructions(core, channel, panel, [
            (request_ref("other"), json.dumps({"file": str(device_file)})),
        ])
        assert last_response(core, panel).startswith("Illegal:")

    def test_args_must_be_object(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        payload = json.dumps({
            "file": str(device_file), "class": "EchoDevice", "args": [1],
        }, ensure_ascii=False)
        run_instructions(core, channel, panel, [(request_ref("extra"), payload)])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "args" in response

    def test_missing_file_is_illegal(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [
            (request_ref("extra"), json.dumps({
                "file": str(tmp_path / "nope.py"), "class": "EchoDevice",
            })),
        ])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "不可读" in response

    def test_class_must_exist_in_file(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        payload = json.dumps({
            "file": str(device_file), "class": "NopeDevice",
        }, ensure_ascii=False)
        run_instructions(core, channel, panel, [(request_ref("extra"), payload)])
        response = last_response(core, panel)
        assert response.startswith("Illegal:")
        assert "NopeDevice" in response

    def test_mount_name_must_be_free(self, tmp_path):
        device_file = write_echo_device(tmp_path, "second.py")
        second = {
            "path": "extra", "file": str(device_file),
            "class": "EchoDevice", "args": {},
        }
        core, _image, channel, panel = make_core(tmp_path, devices=[second])
        assert "extra" in core.mem.mounted_devices()
        req_file = write_echo_device(tmp_path, "echo.py")
        response = submit_request(core, channel, panel, "extra", req_file)
        assert response.startswith("Illegal:")
        assert "占用" in response
        assert device(core) is not None

    def test_same_request_is_idempotent_and_different_conflicts(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        first = submit_request(core, channel, panel, "extra", device_file)
        assert first == "Success set: $MEM.device_loader.request.extra"
        second = submit_request(core, channel, panel, "extra", device_file)
        assert second == "Success set: $MEM.device_loader.request.extra"
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：pending" in detail
        changed = submit_request(core, channel, panel, "extra", device_file,
                                 args={"value": "换参数"})
        assert changed.startswith("Illegal:")
        assert "内容不同" in changed

    def test_multiple_requests_coexist(self, tmp_path):
        file_a = write_echo_device(tmp_path, "a.py")
        file_b = write_echo_device(tmp_path, "b.py")
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "dev_a", file_a)
        submit_request(core, channel, panel, "dev_b", file_b)
        root = core.unwrap("$MEM.device_loader", for_llm=True)
        assert "待批准 2 台" in root
        names = action_names(core)
        assert "批准装载 dev_a" in names and "批准装载 dev_b" in names
        approve(core, "dev_a")
        assert "已批准待挂载 1 台：dev_a" in core.unwrap(
            "$MEM.device_loader", for_llm=True)

    def test_read_resume_is_error(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        run_instructions(core, channel, panel, [])
        with pytest.raises(Exception):
            core.unwrap(resume_ref("extra"), for_llm=True)

    def test_root_mentions_addresses(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        root = core.unwrap("$MEM.device_loader", for_llm=True)
        assert "request.<名字>" in root
        assert "resume.<名字>" in root


class TestPanelApproval:
    def test_no_actions_without_pending(self, tmp_path):
        core, _image, channel, panel = make_core(tmp_path)
        assert device(core).debug_actions() == []

    def test_approve_marks_without_mount(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        result = approve(core, "extra")
        assert result["approved"] == "extra"
        assert result["sha256"] == sha256_file(device_file)
        assert "extra" not in core.mem.mounted_devices()
        assert "extra" not in core.device_specs
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：approved" in detail
        assert "panel-device-action" in detail
        assert "resume.extra" in detail
        assert device(core).debug_actions() == []

    def test_approve_refuses_changed_file(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        device_file.write_text(device_file.read_text() + "\n# 换掉了\n",
                               encoding="utf-8")
        with pytest.raises(ValueError, match="已变化"):
            approve(core, "extra")
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：pending" in detail
        assert "extra" not in core.mem.mounted_devices()

    def test_reject_records_reason(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        result = reject(core, "extra", "代码没看过")
        assert result["rejected"] == "extra"
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：rejected" in detail
        assert "代码没看过" in detail

    def test_rejected_request_can_be_replaced(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        reject(core, "extra", "先拒")
        response = submit_request(core, channel, panel, "extra", device_file,
                                  args={"value": "新版"})
        assert response == "Success set: $MEM.device_loader.request.extra"
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：pending" in detail


class TestResumeMount:
    def test_resume_requires_approval(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        response = resume(core, channel, panel, "extra")
        assert response.startswith("Illegal:")
        assert "未获批准" in response or "pending" in response
        assert "extra" not in core.mem.mounted_devices()

    def test_resume_mounts_with_panel_approval(self, tmp_path):
        core, _image, channel, panel, device_file = mounted_hot_device(tmp_path)
        assert "extra" in core.mem.mounted_devices()
        assert core.mem._devices["extra"].get_value() == "热装设备"
        spec = core.device_specs["extra"]
        digest = sha256_file(device_file)
        assert spec["class"] == "EchoDevice"
        assert spec["sha256"] == digest
        assert spec["approved_sha256"] == digest
        assert spec["approved_at"]
        assert spec["approved_via"] == "panel-device-action"
        root = core.unwrap("$MEM.device_loader", for_llm=True)
        assert "extra" not in root

    def test_resume_after_file_change_marks_rejected(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        approve(core, "extra")
        device_file.write_text(device_file.read_text() + "\n# 批准后换掉\n",
                               encoding="utf-8")
        response = resume(core, channel, panel, "extra")
        assert response.startswith("Error:")
        assert "已变化" in response
        assert "extra" not in core.mem.mounted_devices()
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：rejected" in detail
        assert "已变化" in detail

    def test_mount_failure_keeps_approval_and_last_error(self, tmp_path):
        device_file = write_echo_device(tmp_path)
        core, _image, channel, panel = make_core(tmp_path)
        submit_request(core, channel, panel, "extra", device_file)
        approve(core, "extra")
        # 批准后占住挂载名：resume 的挂载会失败，批准保留、错误可见。
        core.mem.make("$MEM", "extra", "str")
        response = resume(core, channel, panel, "extra")
        assert response.startswith("Error:")
        assert "extra" not in core.mem.mounted_devices()
        detail = core.unwrap(request_ref("extra"), for_llm=True)
        assert "状态：approved" in detail
        assert "上次挂载失败" in detail
        core.mem.detach("$MEM.extra")
        retry = resume(core, channel, panel, "extra")
        assert retry == "Success set: $MEM.device_loader.resume.extra"
        assert "extra" in core.mem.mounted_devices()


class TestPersistence:
    def test_hot_mounted_device_replays_from_journal(self, tmp_path):
        mine, image, _channel, _panel, device_file = mounted_hot_device(tmp_path)
        replayed = load_core(image)
        replayed.attach_journal(replayed.journal_path)
        replayed.replay_journal()
        assert "extra" in replayed.mem._devices
        assert replayed.device_specs["extra"]["class"] == "EchoDevice"
        assert replayed.device_specs["extra"]["sha256"] == sha256_file(device_file)
        assert replayed.device_specs["extra"]["approved_via"] == "panel-device-action"
        from persist_helpers import assert_same_state

        assert_same_state(mine, replayed)

    def test_journal_replay_rejects_changed_device_file(self, tmp_path):
        mine, image, _channel, _panel, device_file = mounted_hot_device(tmp_path)
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
        mine, image, _channel, _panel, device_file = mounted_hot_device(tmp_path)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        assert restored.mem._devices["extra"].get_value() == "热装设备"
        recorded = restored.device_specs["extra"]
        assert recorded["sha256"] == sha256_file(device_file)
        assert recorded["approved_sha256"] == sha256_file(device_file)
        assert recorded["approved_at"]
        assert recorded["approved_via"] == "panel-device-action"

    def test_snapshot_restore_rejects_changed_device_file(self, tmp_path):
        mine, image, _channel, _panel, device_file = mounted_hot_device(tmp_path)
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
        mine, image, _channel, _panel, _device_file = mounted_hot_device(tmp_path)
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
        run_turns(mine, 1)
        snap = str(tmp_path / "snap.json")
        mine.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        for mount in ("conversations", "scheduler", "memory", "monitor",
                      "listeners", "roots", "caps"):
            assert restored.device_specs[mount]["builtin"] is True
            assert mount in restored.mem._devices

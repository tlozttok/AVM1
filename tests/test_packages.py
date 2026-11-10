"""程序包（avm/packages.py）测试：装/列/卸、重复安装、升级指针、设备、回滚、
快照与日志恢复后的包状态。全部本地文件与脚本化执行器，不调真实 API。
"""

import json
import os

import pytest

from avm import packages
from avm.core import Core
from avm.exceptions import VMMemoryError
from persist_helpers import (
    assert_same_state, load_core, out_path, run_core, write_image,
)


def write_device_file(tmp_path, name="echo.py"):
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


def write_package(tmp_path, *, name="demo", version="1_0_0", mem=None, devices=None,
                  filename=None, meta_extra=None, device_file="echo.py"):
    if devices is None:
        devices = [{"path": "io", "file": device_file, "class": "EchoDevice",
                    "args": {"value": "包内设备"}}]
        if mem is None:
            mem = {
                "data": {"kind": "dict", "value": {"msg": {"kind": "str", "value": "hi"}}},
                "io": {"kind": "device"},
            }
    package = {
        "meta": dict({
            "format": "avm.package",
            "version": 1,
            "name": name,
            "package_version": version,
            "description": "示例包",
        }, **(meta_extra or {})),
        "mem": mem or {},
        "devices": devices,
    }
    path = tmp_path / (filename or f"{name}-{version}.pkg.json")
    path.write_text(json.dumps(package, ensure_ascii=False), encoding="utf-8")
    return str(path)


def package_sha256(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestInstall:
    def test_install_creates_subtree_record_and_pointer(self, tmp_path):
        write_device_file(tmp_path)
        path = write_package(tmp_path)
        core = Core()
        record = packages.install(core, path)
        assert record["sha256"]
        assert core.unwrap("$MEM.packages.demo.1_0_0.data.msg", for_llm=False) == "hi"
        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_0_0"
        installed = packages.list_installed(core)
        assert [item["name"] for item in installed] == ["demo"]
        assert installed[0]["source"] == os.path.abspath(path)

    def test_wrong_approval_hash_fails_before_any_effect(self, tmp_path):
        write_device_file(tmp_path)
        path = write_package(tmp_path)
        core = Core()
        before = dict(core.mem._devices)
        with pytest.raises(VMMemoryError, match="程序包哈希"):
            packages.install(core, path, approved_sha256="0" * 64)
        assert core.mem._devices == before
        assert packages.list_installed(core) == []
        assert "packages" not in core.mem._data
        assert "package_records" not in core.mem._data

    def test_install_records_device_hashes_and_approval(self, tmp_path):
        write_device_file(tmp_path)
        path = write_package(tmp_path)
        digest = package_sha256(path)
        core = Core()
        packages.install(
            core, path, approved_sha256=digest,
            approved_at="2026-11-11T00:00:00+0800", approved_via="panel-1",
        )
        record = core.unwrap("$MEM.package_records.demo.1_0_0", for_llm=False)
        assert record["sha256"] == digest
        assert record["approved_sha256"] == digest
        assert record["approved_at"] == "2026-11-11T00:00:00+0800"
        assert record["approved_via"] == "panel-1"
        (entry,) = list(record["devices"])
        assert entry["sha256"] == package_sha256(str(tmp_path / "echo.py"))
        assert core.device_specs["packages.demo.1_0_0.io"]["sha256"] == entry["sha256"]
        assert core.device_specs["packages.demo.1_0_0.io"]["approved_sha256"] == digest
        assert core.device_specs["packages.demo.1_0_0.io"]["approved_via"] == "panel-1"

    def test_install_twice_is_rejected(self, tmp_path):
        write_device_file(tmp_path)
        path = write_package(tmp_path)
        core = Core()
        packages.install(core, path)
        with pytest.raises(VMMemoryError, match="已安装"):
            packages.install(core, path)

    def test_upgrade_moves_current_pointer(self, tmp_path):
        write_device_file(tmp_path)
        core = Core()
        packages.install(core, write_package(tmp_path, version="1_0_0"))
        packages.install(core, write_package(tmp_path, version="1_1_0"))
        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_1_0"
        assert len(packages.list_installed(core)) == 2

    def test_bad_format_is_rejected(self, tmp_path):
        bad = tmp_path / "bad.pkg.json"
        bad.write_text(json.dumps({"meta": {"format": "nope", "version": 1,
                                            "name": "x", "package_version": "1"}}),
                       encoding="utf-8")
        with pytest.raises(VMMemoryError, match="不是 AVM 程序包"):
            packages.install(Core(), str(bad))

    def test_version_with_dot_is_rejected(self, tmp_path):
        """版本号是内存键：点号是路径分隔符，带点版本装得进去却寻不到（作者 2026-11-07 决定先禁用）。"""
        bad = tmp_path / "dot.pkg.json"
        bad.write_text(json.dumps({"meta": {"format": "avm.package", "version": 1,
                                            "name": "demo", "package_version": "1.2.0"},
                                   "mem": {}, "devices": []}), encoding="utf-8")
        with pytest.raises(VMMemoryError, match="点号"):
            packages.install(Core(), str(bad))


class TestDeviceAndRollback:
    def test_package_device_is_mounted_under_subtree(self, tmp_path):
        write_device_file(tmp_path)
        core = Core()
        packages.install(core, write_package(tmp_path))
        mount = "packages.demo.1_0_0.io"
        assert mount in core.mem._devices
        assert core.mem._devices[mount].get_value() == "包内设备"
        assert core.device_specs[mount]["class"] == "EchoDevice"

    def test_missing_device_marker_is_rejected(self, tmp_path):
        write_device_file(tmp_path)
        package = write_package(
            tmp_path,
            mem={"data": {"kind": "dict", "value": {}}},  # 没有 io 的 kind=device 标记
        )
        core = Core()
        with pytest.raises(VMMemoryError, match="kind=device"):
            packages.install(core, package)
        assert "packages.demo.1_0_0.io" not in core.mem._devices

    def test_failure_rolls_back_device_and_nodes(self, tmp_path):
        write_device_file(tmp_path)
        package = write_package(tmp_path)
        core = Core()
        core.mem["packages"] = "占位字符串"  # 装到一半必然失败
        before_devices = dict(core.mem._devices)
        with pytest.raises(VMMemoryError):
            packages.install(core, package)
        assert core.mem._devices == before_devices
        assert "packages.demo.1_0_0.io" not in core.device_specs
        assert "record" not in str(packages.list_installed(core))


class TestUninstall:
    def test_uninstall_removes_subtree_devices_and_record(self, tmp_path):
        write_device_file(tmp_path)
        core = Core()
        packages.install(core, write_package(tmp_path))
        packages.uninstall(core, "demo", "1_0_0")
        assert "1_0_0" not in core.mem._data["packages"]["demo"]
        assert "packages.demo.1_0_0.io" not in core.mem._devices
        assert packages.list_installed(core) == []

    def test_uninstall_missing_is_rejected(self, tmp_path):
        with pytest.raises(VMMemoryError, match="未安装"):
            packages.uninstall(Core(), "ghost", "1_0_0")


class TestPersistence:
    def test_snapshot_roundtrip_keeps_packages_and_devices(self, tmp_path):
        write_device_file(tmp_path)
        image = write_image(tmp_path)
        core = load_core(image, [("done", [])])
        core.attach_journal(core.journal_path)
        run_core(core)
        packages.install(core, write_package(tmp_path))
        snap = str(tmp_path / "snap.json")
        core.snapshot_to(snap)

        restored = load_core(image)
        restored.restore_from(snap)
        assert [item["package_version"] for item in packages.list_installed(restored)] == ["1_0_0"]
        mount = "packages.demo.1_0_0.io"
        assert mount in restored.mem._devices
        assert restored.mem._devices[mount].get_value() == "包内设备"
        assert_same_state(core, restored)


class TestUpgradeRollback:
    def test_failed_upgrade_restores_current_and_removes_partial(self, tmp_path):
        write_device_file(tmp_path)
        core = Core()
        packages.install(core, write_package(tmp_path, version="1_0_0"))
        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_0_0"
        core.mem["package_records"] = "占位字符串"  # 让新版本的安装记录写不进去

        with pytest.raises(VMMemoryError):
            packages.install(core, write_package(tmp_path, version="1_1_0"))

        assert core.unwrap("$MEM.packages.demo.current", for_llm=False) == "1_0_0"
        assert "1_1_0" not in core.mem._data["packages"]["demo"]
        assert "packages.demo.1_1_0.io" not in core.mem._devices
        assert [item["package_version"] for item in packages.list_installed(core)] == []

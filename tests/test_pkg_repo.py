"""PackageRepoDevice（base 的 OS 包仓库设备）测试：装载/读接口、路径安全、包格式与只读。

全离线：临时目录 + 普通内存/设备操作，不需要 MockLMU，也不调真实 API。
"""

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pytest

from avm.core import Core, MemoryReadInstruction, MemoryWriteInstruction
from avm.exceptions import MemoryTypeError, VMMemoryError
from avm.image import load_image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_FILE = REPO_ROOT / "images" / "devices" / "pkg_repo.py"
BASE_IMAGE = REPO_ROOT / "images" / "base" / "base.json"

MANIFEST = {"name": "demo", "package_version": "1.0.0", "description": "示例包"}
DATA = {
    "kind": "dict",
    "meta": "示例数据",
    "value": {
        "greeting": {"kind": "str", "value": "你好"},
        "items": {"kind": "list", "value": [{"kind": "str", "value": "a"}]},
    },
}
INSTALL = "def install():\n    return 'ok'\n"
ECHO_DEVICE = "print('device code must never run')\n"
_ABSENT = object()


def _device_class():
    spec = importlib.util.spec_from_file_location("_test_pkg_repo", DEVICE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PackageRepoDevice


def _write(path: Path, value) -> str:
    """按值写一个包文件；dict/list 写 JSON，str 写原文，_ABSENT 表示不创建。"""
    if value is _ABSENT:
        return ""
    if isinstance(value, bytes):
        path.write_bytes(value)
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    # newline="" 关掉平台换行翻译：测试要断言"读回来与写进去逐字节相同"，
    # 而 write_text 默认在 Windows 上把 \n 翻成 \r\n，断言必然对不上。
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return text


def _make_package(root: Path, name: str = "demo", *, manifest=MANIFEST, data=DATA,
                  install=INSTALL, devices=None):
    """在 root 下造一个包；返回（包目录，manifest 原文，data 原文）。"""
    pkg = root / name
    pkg.mkdir(parents=True)
    manifest_text = _write(pkg / "manifest.json", manifest)
    data_text = _write(pkg / "data.json", data)
    if install is not _ABSENT:
        _write(pkg / "install.py", install)
    if devices is not None:
        (pkg / "devices").mkdir()
        for filename, content in devices.items():
            _write(pkg / "devices" / filename, content)
    return pkg, manifest_text, data_text


def _device(root: Path, **kwargs):
    return _device_class()(root=str(root), **kwargs)


def _core(device):
    core = Core()
    core.mem.mount("pkgrepo", device)
    device.mount_path = "pkgrepo"
    return core


def _assert_old_package(core):
    manifest = json.loads(core.unwrap("$MEM.pkgrepo.manifest", for_llm=True))
    assert manifest["name"] == "old"


def _fs_supports(build) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            build(Path(tmp))
        except OSError:
            return False
        return True


def _make_symlink(base: Path) -> None:
    target = base / "target"
    target.mkdir()
    (base / "link").symlink_to(target, target_is_directory=True)


HAS_SYMLINK = _fs_supports(_make_symlink)
requires_symlink = pytest.mark.skipif(
    not HAS_SYMLINK,
    reason="当前环境不允许创建符号链接（Windows 需管理员或开发者模式）：越界用例只能在类 POSIX 上跑",
)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "pkgs"
    root.mkdir()
    return root


@pytest.fixture
def loaded(root):
    """先装好 old 包；验证失败类用例结束后用它断言旧包仍可用。"""
    _make_package(root, "old", manifest={"name": "old", "package_version": "0.0.1"})
    core = _core(_device(root))
    core.mem.set("$MEM.pkgrepo", "old")
    return root, core


class TestLoadAndRead:
    def test_manual_before_load(self, root):
        core = _core(_device(root))
        text = core.unwrap("$MEM.pkgrepo", for_llm=True)
        assert "pkgrepo[" in text
        assert "未装载包" in text
        assert "memory_write $MEM.pkgrepo" in text
        assert "$MEM.pkgrepo.manifest" in text
        assert "$MEM.pkgrepo.read" in text
        assert "OS 约定" in text
        assert "不 import" in text

    def test_load_and_read_interfaces(self, root):
        pkg, manifest_text, data_text = _make_package(
            root, devices={"echo.py": ECHO_DEVICE}
        )
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")

        manual = core.unwrap("$MEM.pkgrepo", for_llm=True)
        assert "已装载包" in manual
        assert "path=demo" in manual
        assert "name=demo" in manual
        assert "package_version=1.0.0" in manual
        assert "文件=4" in manual

        assert core.unwrap("$MEM.pkgrepo.manifest", for_llm=True) == manifest_text
        assert core.unwrap("$MEM.pkgrepo.data", for_llm=True) == data_text
        assert core.unwrap("$MEM.pkgrepo.installer", for_llm=True) == INSTALL
        assert core.unwrap('$MEM.pkgrepo.read."devices/echo.py"', for_llm=True) == ECHO_DEVICE
        assert core.unwrap('$MEM.pkgrepo.read."manifest.json"', for_llm=True) == manifest_text

        lines = core.unwrap("$MEM.pkgrepo.files", for_llm=True).splitlines()
        assert [line.split()[0] for line in lines] == [
            "manifest.json", "data.json", "install.py", "devices/echo.py",
        ]
        for line in lines:
            rel, size, digest = line.split()
            raw = (pkg / rel).read_bytes()
            assert int(size) == len(raw)
            assert digest == hashlib.sha256(raw).hexdigest()

    def test_read_dotted_filename_uses_quote_syntax(self, root):
        source = "print('dotted')\n"
        _make_package(root, devices={"echo.extra.py": source})
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        assert core.unwrap(
            '$MEM.pkgrepo.read."devices/echo.extra.py"', for_llm=True
        ) == source
        assert "devices/echo.extra.py" in core.unwrap("$MEM.pkgrepo.files", for_llm=True)

    def test_missing_installer_errors(self, root):
        _make_package(root, install=_ABSENT)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        assert "install.py" not in core.unwrap("$MEM.pkgrepo.files", for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap("$MEM.pkgrepo.installer", for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.pkgrepo.read."install.py"', for_llm=True)

    def test_clear_loaded_package(self, root):
        _make_package(root)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        assert "已装载包" in core.unwrap("$MEM.pkgrepo", for_llm=True)

        core.mem.set("$MEM.pkgrepo", "")
        assert "未装载包" in core.unwrap("$MEM.pkgrepo", for_llm=True)
        for ref in ("$MEM.pkgrepo.manifest", "$MEM.pkgrepo.data",
                    "$MEM.pkgrepo.installer", "$MEM.pkgrepo.files",
                    '$MEM.pkgrepo.read."manifest.json"'):
            with pytest.raises(VMMemoryError):
                core.unwrap(ref, for_llm=True)

    def test_whitespace_write_clears(self, root):
        _make_package(root)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        core.mem.set("$MEM.pkgrepo", "  \t \n ")
        assert "未装载包" in core.unwrap("$MEM.pkgrepo", for_llm=True)

    def test_replace_package(self, root):
        _make_package(root, "first", manifest={"name": "first", "package_version": "1"})
        _, second_text, _ = _make_package(
            root, "second", manifest={"name": "second", "package_version": "2"}
        )
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "first")
        assert json.loads(core.unwrap("$MEM.pkgrepo.manifest", for_llm=True))["name"] == "first"
        core.mem.set("$MEM.pkgrepo", "second")
        assert core.unwrap("$MEM.pkgrepo.manifest", for_llm=True) == second_text
        assert "name=second" in core.unwrap("$MEM.pkgrepo", for_llm=True)

    def test_extra_files_are_not_exposed(self, root):
        pkg, _manifest_text, _data_text = _make_package(root, devices={"echo.py": ECHO_DEVICE})
        (pkg / "notes.txt").write_text("not part of the package\n", encoding="utf-8")
        (pkg / "devices" / "readme.md").write_text("ignored\n", encoding="utf-8")
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        files = core.unwrap("$MEM.pkgrepo.files", for_llm=True)
        assert "notes.txt" not in files and "readme.md" not in files
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.pkgrepo.read."notes.txt"', for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.pkgrepo.read."devices/readme.md"', for_llm=True)

    def test_devices_subdirectories_are_not_listed(self, root):
        pkg, _m, _d = _make_package(root, devices={"echo.py": ECHO_DEVICE})
        (pkg / "devices" / "sub").mkdir()
        (pkg / "devices" / "sub" / "deep.py").write_text("x\n", encoding="utf-8")
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        assert "deep.py" not in core.unwrap("$MEM.pkgrepo.files", for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.pkgrepo.read."devices/sub/deep.py"', for_llm=True)

    def test_package_python_is_never_imported_or_executed(self, root, tmp_path):
        install_marker = tmp_path / "install-executed"
        device_marker = tmp_path / "device-executed"
        install_source = f"open({str(install_marker)!r}, 'w').write('x')\n"
        device_source = f"open({str(device_marker)!r}, 'w').write('x')\n"
        _make_package(root, install=install_source, devices={"boom.py": device_source})
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        assert not install_marker.exists()
        assert not device_marker.exists()
        assert core.unwrap(
            '$MEM.pkgrepo.read."devices/boom.py"', for_llm=True
        ) == device_source

    def test_unknown_verb(self, root):
        _make_package(root)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        with pytest.raises(VMMemoryError):
            core.unwrap("$MEM.pkgrepo.remove", for_llm=True)

    def test_unquoted_read_path_rejected(self, root):
        _make_package(root)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        with pytest.raises(VMMemoryError):
            core.unwrap("$MEM.pkgrepo.read.manifest.json", for_llm=True)


class TestPathSafety:
    def test_absolute_path_rejected(self, root):
        _make_package(root)
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", str(root / "demo"))

    def test_parent_segment_rejected(self, root):
        _make_package(root)
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "../outside")
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "demo/../../outside")

    def test_missing_directory_rejected(self, root):
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "nope")

    def test_file_instead_of_directory_rejected(self, root):
        (root / "plain.txt").write_text("x\n", encoding="utf-8")
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "plain.txt")

    def test_empty_non_string_write_rejected(self, root):
        device = _device(root)
        with pytest.raises(VMMemoryError):
            device.set_value(123)

    def test_multiline_path_rejected(self, root):
        _make_package(root)
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "demo\nother")

    @requires_symlink
    def test_symlink_escaping_root_rejected(self, root, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        _make_package(outside, "pkg")
        (root / "link").symlink_to(outside / "pkg", target_is_directory=True)
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "link")

    @requires_symlink
    def test_package_file_symlink_escape_rejected(self, root, tmp_path):
        pkg, _m, _d = _make_package(root)
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps(DATA, ensure_ascii=False), encoding="utf-8")
        (pkg / "data.json").unlink()
        (pkg / "data.json").symlink_to(outside)
        core = _core(_device(root))
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "demo")

    def test_path_failure_keeps_loaded_package(self, loaded):
        root, core = loaded
        for bad in (str(root / "old"), "../outside", "nope"):
            with pytest.raises(VMMemoryError):
                core.mem.set("$MEM.pkgrepo", bad)
            _assert_old_package(core)


class TestPackageValidation:
    @pytest.mark.parametrize(
        "manifest",
        [
            _ABSENT,
            "{not json",
            [1, 2],
            {"package_version": "1.0"},
            {"name": "  ", "package_version": "1.0"},
            {"name": 1, "package_version": "1.0"},
            {"name": "x"},
            {"name": "x", "package_version": ""},
        ],
        ids=[
            "missing", "invalid-json", "not-object", "missing-name",
            "empty-name", "non-string-name", "missing-version", "empty-version",
        ],
    )
    def test_bad_manifest_rejected(self, loaded, manifest):
        root, core = loaded
        _make_package(root, "bad", manifest=manifest)
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    @pytest.mark.parametrize(
        "data",
        [
            _ABSENT,
            "{not json",
            [1],
            {"kind": "str", "value": "x"},
            {"value": {}},
            {"kind": "dict", "value": {"x": "plain"}},
            {"kind": "dict", "value": {"x": {"kind": "nope"}}},
            {"kind": "dict", "value": {"x": {"kind": "dict", "value": []}}},
            {"kind": "dict", "value": {"x": {"kind": "list", "value": {}}}},
        ],
        ids=[
            "missing", "invalid-json", "root-not-object", "root-not-dict",
            "root-without-kind", "nested-raw-value", "nested-bad-kind",
            "dict-value-not-object", "list-value-not-array",
        ],
    )
    def test_bad_data_rejected(self, loaded, data):
        root, core = loaded
        _make_package(root, "bad", data=data)
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    def test_non_utf8_manifest_rejected(self, loaded):
        root, core = loaded
        pkg = root / "bad"
        pkg.mkdir()
        (pkg / "manifest.json").write_bytes(b"\xff\xfe\x00")
        (pkg / "data.json").write_text(json.dumps(DATA), encoding="utf-8")
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    def test_non_utf8_device_file_rejected(self, loaded):
        root, core = loaded
        pkg, _m, _d = _make_package(root, "bad", devices={"boom.py": "x\n"})
        (pkg / "devices" / "boom.py").write_bytes(b"\xff\xfe")
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    def test_devices_path_must_be_directory(self, loaded):
        root, core = loaded
        pkg, _m, _d = _make_package(root, "bad")
        (pkg / "devices").write_text("not a directory\n", encoding="utf-8")
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    def test_required_file_must_be_regular_file(self, loaded):
        root, core = loaded
        pkg = root / "bad"
        (pkg / "manifest.json").mkdir(parents=True)
        (pkg / "data.json").write_text(json.dumps(DATA), encoding="utf-8")
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(core)

    @pytest.mark.parametrize(
        "target",
        ["manifest.json", "data.json", "install.py", "devices/echo.py"],
        ids=["manifest", "data", "install", "device"],
    )
    def test_file_over_max_file_bytes_rejected(self, loaded, target):
        root, _loaded_core = loaded
        overrides = {}
        if target == "manifest.json":
            overrides["manifest"] = dict(MANIFEST, padding="x" * 400)
        elif target == "data.json":
            overrides["data"] = {
                "kind": "dict",
                "value": {"padding": {"kind": "str", "value": "x" * 400}},
            }
        elif target == "install.py":
            overrides["install"] = "x" * 400
        else:
            overrides["devices"] = {"echo.py": "x" * 400}
        _make_package(root, "bad", **overrides)
        small = _core(_device(root, max_file_bytes=350))
        small.mem.set("$MEM.pkgrepo", "old")
        with pytest.raises(VMMemoryError):
            small.mem.set("$MEM.pkgrepo", "bad")
        _assert_old_package(small)

    def test_error_does_not_echo_file_content(self, loaded):
        root, core = loaded
        _make_package(root, "bad", data='{"secret": "TOPSECRET-CONTENT"')
        with pytest.raises(VMMemoryError) as exc:
            core.mem.set("$MEM.pkgrepo", "bad")
        assert "TOPSECRET-CONTENT" not in str(exc.value)
        assert "data.json" in str(exc.value)
        _assert_old_package(core)


class TestReadOnly:
    def test_pretend_as_type(self, root):
        assert _device(root).pretend_as_type() == "str"

    def test_subpath_writes_are_typed_errors(self, root):
        _pkg, _manifest_text, data_text = _make_package(root)
        core = _core(_device(root))
        core.mem.set("$MEM.pkgrepo", "demo")
        for ref in (
            "$MEM.pkgrepo.manifest",
            "$MEM.pkgrepo.data",
            "$MEM.pkgrepo.files",
            '$MEM.pkgrepo.read."devices/echo.py"',
        ):
            with pytest.raises(MemoryTypeError):
                core.mem.set(ref, "x")
        # 子路径写失败不影响读
        assert core.unwrap("$MEM.pkgrepo.data", for_llm=True) == data_text

    def test_memory_write_instruction_returns_error(self, root):
        _pkg, _manifest_text, data_text = _make_package(root)
        core = _core(_device(root))
        conv = core.start("sys", "user")
        MemoryWriteInstruction(
            "tc1", conv.cid, {"ref": "$MEM.pkgrepo", "content": "demo"}
        ).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content.startswith("Success")
        MemoryWriteInstruction(
            "tc2", conv.cid, {"ref": "$MEM.pkgrepo.manifest", "content": "x"}
        ).execute(core, conv)
        response = conv.user_batch.tool_responses[-1].content
        assert response.startswith("Error:") and "只读" in response
        MemoryReadInstruction(
            "tc3", conv.cid, {"ref": "$MEM.pkgrepo.data"}
        ).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content == data_text


class TestConstruction:
    def test_default_root_is_repo_root(self):
        device = _device_class()()
        assert device._root == REPO_ROOT.resolve()
        assert "未装载包" in device.to_llm_string()
        assert device._max_file_bytes == 1000000

    def test_missing_root_rejected(self, tmp_path):
        with pytest.raises(VMMemoryError):
            _device_class()(root=str(tmp_path / "nope"))

    def test_file_root_rejected(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text("x\n", encoding="utf-8")
        with pytest.raises(VMMemoryError):
            _device_class()(root=str(path))

    @pytest.mark.parametrize("limit", [0, -1, "abc"])
    def test_bad_max_file_bytes_rejected(self, root, limit):
        with pytest.raises(VMMemoryError):
            _device_class()(root=str(root), max_file_bytes=limit)


def _prepared_image(tmp_path, package_root):
    """复制 base 镜像到 tmp：设备文件改绝对路径，pkgrepo 的 root 指向临时包根。"""
    image = json.loads(BASE_IMAGE.read_text(encoding="utf-8"))
    base_dir = BASE_IMAGE.parent
    for dev in image.get("devices") or []:
        file_path = dev.get("file")
        if isinstance(file_path, str) and not os.path.isabs(file_path):
            dev["file"] = os.path.normpath(os.path.join(base_dir, file_path))
        if dev.get("path") == "pkgrepo":
            dev["args"] = {"root": str(package_root)}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return path


class TestBaseImage:
    def test_base_image_mounts_pkgrepo(self):
        core = load_image(Core(), str(BASE_IMAGE))
        assert "pkgrepo" in core.mem._devices
        device = core.mem._devices["pkgrepo"]
        assert type(device).__name__ == "PackageRepoDevice"
        text = core.unwrap("$MEM.pkgrepo", for_llm=True)
        assert "pkgrepo[" in text
        assert "未装载包" in text

    def test_end_to_end_with_temporary_root(self, tmp_path):
        package_root = tmp_path / "pkgs"
        package_root.mkdir()
        _pkg, _manifest_text, data_text = _make_package(package_root)
        image_path = _prepared_image(tmp_path, package_root)
        core = load_image(Core(), str(image_path))
        conv = core.get_conversation(0)

        MemoryWriteInstruction(
            "w1", conv.cid, {"ref": "$MEM.pkgrepo", "content": "demo"}
        ).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content == "Success set: $MEM.pkgrepo"

        MemoryReadInstruction(
            "r1", conv.cid, {"ref": "$MEM.pkgrepo.data"}
        ).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content == data_text

        MemoryReadInstruction(
            "r2", conv.cid, {"ref": "$MEM.pkgrepo.files"}
        ).execute(core, conv)
        listing = conv.user_batch.tool_responses[-1].content
        assert "manifest.json" in listing and "devices" not in listing

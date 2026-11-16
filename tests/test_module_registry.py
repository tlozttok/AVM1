"""运行期模块注册：目录 live 化、$MEM.module_loader 设备、-1 面板刷新与持久化。

对应计划 `proposed/plan-module-registry-2026-11-16.md`。
"""

import json

import pytest

from avm.capabilities import default_kernel_caps
from avm.core import Core, _build_conversation, _build_imports, _derive_child_caps
from avm.exceptions import VMMemoryError
from avm.modules import (
    BASE_MODULES,
    MODULE_CATALOG,
    grantable_module_names,
    register_module,
    registered_module_descriptors,
)
from avm.sandbox import SecurityError
from avm.snapshot import SnapshotError, restore, snapshot
from avm.types import MetaDict, MetaList, StrNode


def _lib_file(tmp_path, name="mylib"):
    path = tmp_path / f"{name}.py"
    path.write_text(
        "def hello():\n    return 'hello-from-lib'\n\nVALUE = 7\n", encoding="utf-8")
    return path


def _forget_runtime_modules():
    """模拟新进程启动：测试进程里清掉运行期注册（真实启动只有静态目录）。"""
    from avm import modules as module_registry
    module_registry.MODULE_CATALOG.pop("mylib", None)
    module_registry.REGISTERED_MODULE_DESCRIPTORS.pop("mylib", None)
    module_registry._invalidate_catalog_caches()


def _core(tmp_path):
    core = Core()
    core.image_dir = str(tmp_path)
    core.mem["plm_params"] = MetaDict(
        data={"model": "plm.python"}, ctrl={"type": "para"})
    return core


def _register_command(name, file_path=None, module=None, exports=("hello", "VALUE")):
    command = {"action": "register", "name": name, "exports": list(exports)}
    if file_path is not None:
        command["file"] = str(file_path)
    else:
        command["module"] = module
    return json.dumps(command)


def _python_conv(core, imports, parent):
    node = MetaDict(data={"content": "x = 1"}, ctrl={"type": "python"})
    node["caps"] = MetaDict(data={"imports": MetaList([StrNode(n) for n in imports])})
    core.mem["prog"] = node
    conv = _build_conversation(core, "$MEM.prog", "$MEM.plm_params", parent, is_sub=False)
    return conv


def _exec(core, conv, code):
    para = core.unwrap("$MEM.plm_params", for_llm=False)
    executor = core._get_executor(conv, para)
    conv.append_user_message(code)
    return executor.exec(conv, para)[0]


class TestRegistryLive:
    def test_register_updates_live_getters_and_catalog_device(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        assert "mylib" not in grantable_module_names()
        core.mem.set("$MEM.module_loader", _register_command("mylib", path))
        assert "mylib" in grantable_module_names()
        assert "mylib" in MODULE_CATALOG
        entry = json.loads(core.unwrap("$MEM.module_catalog.mylib", for_llm=False))
        assert entry["kind"] == "grantable"
        assert sorted(entry["exports"]) == ["VALUE", "hello"]
        assert "mylib" in core.unwrap("$MEM.module_catalog", for_llm=True)

    def test_kernel_default_is_live(self, tmp_path):
        core = _core(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", _lib_file(tmp_path)))
        assert "mylib" in default_kernel_caps().imports
        host_conv = core.start("sys", "user")
        assert "mylib" in _build_imports(host_conv)

    def test_registration_descriptor_recorded(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", path))
        descriptors = registered_module_descriptors()
        assert descriptors["mylib"]["source_kind"] == "file"
        assert descriptors["mylib"]["exports"] == ["hello", "VALUE"]
        assert len(descriptors["mylib"]["sha256"]) == 64

    def test_builtin_registration_rejects_bad_exports(self):
        with pytest.raises(SecurityError, match="基线"):
            register_module("math", exports={"x": 1})
        with pytest.raises(SecurityError, match="空"):
            register_module("empty", exports={})
        with pytest.raises(SecurityError, match="模块对象"):
            register_module("bad", exports={"m": __import__("json")})
        with pytest.raises(SecurityError, match="重复"):
            register_module("dummy", exports={"x": 1})
            register_module("dummy", exports={"x": 1})


class TestDeviceRegistration:
    def test_register_from_file_and_use_in_new_conversation(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", path))

        panel = core.ensure_operator_conversation()
        assert "mylib" in panel.caps.imports
        conv = _python_conv(core, ["mylib"], panel)
        assert _exec(core, conv, "import mylib\nprint(mylib.hello() + ':' + str(mylib.VALUE))") == (
            "hello-from-lib:7\n")

    def test_register_from_installed_module(self, tmp_path):
        core = _core(tmp_path)
        core.mem.set("$MEM.module_loader",
                     _register_command("stats", module="statistics", exports=["mean"]))
        panel = core.ensure_operator_conversation()
        conv = _python_conv(core, ["stats"], panel)
        assert _exec(core, conv, "import stats\nprint(stats.mean([1, 2, 3]))") == "2\n"

    def test_existing_conversation_not_widened(self, tmp_path):
        core = _core(tmp_path)
        old_conv = core.start("old", "user")
        core.mem.set("$MEM.module_loader", _register_command("mylib", _lib_file(tmp_path)))
        assert "mylib" not in _build_imports(old_conv)
        node = MetaDict(data={"content": "x = 1"}, ctrl={"type": "python"})
        node["caps"] = MetaDict(data={"imports": MetaList([StrNode("mylib")])})
        with pytest.raises(VMMemoryError, match="不可超出父对话"):
            _derive_child_caps(core, old_conv, node, is_sub=False)

    def test_duplicate_same_descriptor_is_idempotent(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        command = _register_command("mylib", path)
        core.mem.set("$MEM.module_loader", command)
        core.mem.set("$MEM.module_loader", command)  # init 重跑/重放幂等
        assert "mylib" in MODULE_CATALOG

    def test_duplicate_different_descriptor_rejected(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", path))
        with pytest.raises(VMMemoryError, match="不同描述"):
            core.mem.set("$MEM.module_loader",
                         _register_command("mylib", path, exports=["hello"]))

    def test_missing_export_rejected(self, tmp_path):
        core = _core(tmp_path)
        with pytest.raises(VMMemoryError, match="缺少导出名"):
            core.mem.set("$MEM.module_loader", _register_command(
                "mylib", _lib_file(tmp_path), exports=["nope"]))

    def test_bad_command_rejected(self, tmp_path):
        core = _core(tmp_path)
        with pytest.raises(SecurityError):
            core.mem.set("$MEM.module_loader", "not-json")
        with pytest.raises(SecurityError, match="二选一"):
            core.mem.set("$MEM.module_loader", json.dumps(
                {"action": "register", "name": "x", "file": "a.py",
                 "module": "json", "exports": ["x"]}))


class TestInstructionPath:
    def test_memory_write_instruction_registers(self, tmp_path):
        """注册入口就是设备：从对话发一条 memory_write 即注册。"""
        core = _core(tmp_path)
        conv = core.start("sys", "user")
        core._active_cid = conv.cid
        conv.user_batch.tool_responses.clear()
        core._process_return_calls([{
            "cmd_type": "memory_write",
            "call_id": "c1",
            "args": {"ref": "$MEM.module_loader",
                     "content": _register_command("mylib", _lib_file(tmp_path))},
        }], conv)
        assert "mylib" in MODULE_CATALOG
        assert "mylib" in grantable_module_names()


class TestPanelConversation:
    def test_panel_created_after_registration_gets_module(self, tmp_path):
        core = _core(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", _lib_file(tmp_path)))
        panel = core.ensure_operator_conversation()
        assert "mylib" in panel.caps.imports

    def test_panel_refreshed_when_reused(self, tmp_path):
        core = _core(tmp_path)
        panel = core.ensure_operator_conversation()
        before = panel.caps.imports
        core.mem.set("$MEM.module_loader", _register_command("mylib", _lib_file(tmp_path)))
        assert "mylib" not in before
        assert "mylib" in core.ensure_operator_conversation().caps.imports


class TestPersistence:
    def test_snapshot_includes_module_specs_and_restores(self, tmp_path):
        core = _core(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", _lib_file(tmp_path)))
        doc = snapshot(core)
        assert "mylib" in doc["module_specs"]
        _forget_runtime_modules()

        restored = Core()
        restore(restored, doc)
        assert "mylib" in grantable_module_names()
        host_conv = restored.start("sys", "user")
        assert "mylib" in _build_imports(host_conv)
        conv = _python_conv(restored, ["mylib"], host_conv)
        assert _exec(restored, conv, "import mylib\nprint(mylib.VALUE)") == "7\n"

    def test_old_snapshot_without_module_specs_restores(self, tmp_path):
        core = _core(tmp_path)
        doc = snapshot(core)
        doc.pop("module_specs", None)
        restored = Core()
        restore(restored, doc)
        assert "mylib" not in MODULE_CATALOG

    def test_missing_module_file_fails_restore(self, tmp_path):
        core = _core(tmp_path)
        path = _lib_file(tmp_path)
        core.mem.set("$MEM.module_loader", _register_command("mylib", path))
        doc = snapshot(core)
        path.unlink()
        _forget_runtime_modules()
        with pytest.raises(SnapshotError, match="恢复失败"):
            restore(Core(), doc)

    def test_journal_effect_idempotent(self, tmp_path):
        from avm.journal import _effect_module_register, is_world_effect
        core = _core(tmp_path)
        spec = {
            "name": "mylib",
            "source_kind": "file",
            "source": str(_lib_file(tmp_path)),
            "exports": ["hello", "VALUE"],
            "note": "",
        }
        assert is_world_effect("module_register")
        _effect_module_register(core, {"spec": spec})
        assert "mylib" in MODULE_CATALOG
        _effect_module_register(core, {"spec": spec})  # 幂等
        assert "mylib" in MODULE_CATALOG

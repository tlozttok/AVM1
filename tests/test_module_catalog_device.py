"""$MEM.module_catalog：宿主挂载的只读模块目录设备。

对应计划 `docs/private/Agents/proposed/plan-sandbox-imports-2026-11-15.md` 的 2.7。
"""

import json

import pytest

from avm.capabilities import AddressSpace
from avm.core import Core
from avm.exceptions import VMMemoryError
from avm.image import load_image
from avm.modules import BASE_MODULES, GRANTABLE_MODULE_NAMES, MODULE_CATALOG
from avm.snapshot import restore, snapshot


def _core():
    return Core()


class TestCatalogRead:
    def test_root_summary(self):
        core = _core()
        text = core.unwrap("$MEM.module_catalog", for_llm=True)
        for name in BASE_MODULES:
            assert name in text
        for name in GRANTABLE_MODULE_NAMES:
            assert name in text

    def test_single_module_entry(self):
        core = _core()
        entry = json.loads(core.unwrap("$MEM.module_catalog.itertools", for_llm=False))
        assert entry["name"] == "itertools"
        assert entry["kind"] == "grantable"
        assert "islice" in entry["exports"]
        math_entry = json.loads(core.unwrap("$MEM.module_catalog.math", for_llm=False))
        assert math_entry["kind"] == "base"

    def test_exports_subpath(self):
        core = _core()
        exports = json.loads(core.unwrap("$MEM.module_catalog.re.exports", for_llm=False))
        assert "search" in exports
        assert "enum" not in exports

    def test_unknown_module_rejected(self):
        core = _core()
        with pytest.raises(VMMemoryError, match="不在宿主模块目录"):
            core.unwrap("$MEM.module_catalog.os", for_llm=False)

    def test_catalog_matches_import_catalog(self):
        core = _core()
        for name in MODULE_CATALOG:
            entry = json.loads(core.unwrap(f"$MEM.module_catalog.{name}", for_llm=False))
            assert entry["exports"] == sorted(MODULE_CATALOG[name].exports)


class TestReadOnly:
    def test_root_write_rejected(self):
        core = _core()
        with pytest.raises(VMMemoryError, match="不支持"):
            core.mem.set("$MEM.module_catalog", "x")

    def test_subpath_write_rejected(self):
        core = _core()
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.module_catalog.itertools", "x")

    def test_device_registered_and_in_specs(self):
        core = _core()
        assert "module_catalog" in core.mem._devices
        assert core.device_specs["module_catalog"]["builtin"] is True
        assert core.device_specs["module_catalog"]["class"] == "ModuleCatalogDevice"


class TestVisibility:
    def test_user_space_needs_granted_root(self):
        core = _core()
        user_space = AddressSpace({"work": ("w",)})
        with pytest.raises(VMMemoryError, match="引用根不可用"):
            core.unwrap("$mods.itertools", for_llm=False, space=user_space)
        space = AddressSpace({"mods": ("module_catalog",)})
        entry = json.loads(core.unwrap("$mods.itertools", for_llm=False, space=space))
        assert entry["name"] == "itertools"

    def test_device_survives_image_mem_replace(self, tmp_path):
        image = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {
                    "kind": "dict",
                    "ctrl": {"type": "para"},
                    "value": {"model": {"kind": "str", "value": "test-model"}},
                },
            },
            "para_ref": "$MEM.model_params",
            "init": {"name": "init", "system": "你是入口。", "user": "开始"},
        }
        path = tmp_path / "image.json"
        path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
        core = load_image(Core(), str(path))
        assert "module_catalog" in core.mem._devices
        assert "module_catalog" in core.unwrap("$MEM", for_llm=True)


class TestSnapshotBackfill:
    def test_old_snapshot_without_catalog_entry_gets_backfilled(self):
        core = _core()
        doc = snapshot(core)
        # 模拟旧快照：生成时还没有 module_catalog 这条宿主设备
        del doc["device_specs"]["module_catalog"]
        restored = Core()
        restore(restored, doc)
        assert "module_catalog" in restored.mem._devices
        assert restored.device_specs["module_catalog"]["builtin"] is True
        assert "itertools" in restored.unwrap("$MEM.module_catalog", for_llm=True)
        with pytest.raises(VMMemoryError, match="不在宿主模块目录"):
            restored.unwrap("$MEM.module_catalog.os", for_llm=False)

"""caps.imports：模块导入白名单的数据层、委托规则与 Core 传播。

对应计划 `docs/private/Agents/proposed/plan-sandbox-imports-2026-11-15.md` 的阶段 C。
"""

import json

import pytest

from avm.capabilities import (
    Capability,
    MODE_USER,
    caps_from_dict,
    caps_to_dict,
    default_kernel_caps,
)
from avm.core import Core, PLMExecutor, _build_conversation, _build_imports, _derive_child_caps
from avm.exceptions import IllegalOperationError, VMMemoryError
from avm.modules import BASE_MODULES, GRANTABLE_MODULE_NAMES
from avm.python_server import create_plm
from avm.types import (
    Conversation, MetaDict, MetaList, StrNode, SystemMessage, UserMessage,
)


def _caps_node(imports):
    return MetaDict(data={"imports": MetaList([StrNode(name) for name in imports])})


def _program_node(content="x = 1", imports=None, node_type="python"):
    data = {"content": content}
    node = MetaDict(data=data, ctrl={"type": node_type})
    if imports is not None:
        node["caps"] = _caps_node(imports)
    return node


def _core():
    core = Core()
    core.mem["plm_params"] = MetaDict(
        data={"model": "plm.python"}, ctrl={"type": "para"})
    return core


def _parent_conv(core):
    conv = core.start("sys", "user", para_ref="$MEM.plm_params")
    return conv


class TestCapabilityImportsData:
    def test_default_kernel_has_all_grantable_modules(self):
        caps = default_kernel_caps()
        assert caps.imports == GRANTABLE_MODULE_NAMES

    def test_bare_capability_has_no_grants(self):
        assert Capability().imports == frozenset()

    @pytest.mark.parametrize("bad", [("a.b",), (3,), ("",), ("math", None)])
    def test_bad_import_names_rejected(self, bad):
        with pytest.raises(IllegalOperationError):
            Capability(imports=list(bad))

    def test_serialization_roundtrip(self):
        caps = Capability(mode=MODE_USER, roots={"work": ("w",)},
                          tools=["memory_read"], imports=["itertools"])
        data = caps_to_dict(caps)
        assert data["imports"] == ["itertools"]
        assert caps_from_dict(data) == caps

    def test_old_caps_dict_without_imports_restores_empty(self):
        data = caps_to_dict(Capability(imports=["itertools"]))
        del data["imports"]
        assert caps_from_dict(data).imports == frozenset()

    def test_conversation_checkpoint_roundtrip(self):
        caps = Capability(mode=MODE_USER, roots={"work": ("w",)}, tools=[],
                          imports=["itertools", "collections"])
        conv = Conversation(
            messages=[SystemMessage(content="s"), UserMessage(content="u")],
            caps=caps,
        )
        restored = Conversation.from_checkpoint(
            json.loads(json.dumps(conv.to_checkpoint())))
        assert restored.caps.imports == {"itertools", "collections"}


class TestDeriveChildImports:
    def test_granted_subset(self):
        core = _core()
        parent = _parent_conv(core)
        node = _program_node(imports=["itertools"])
        caps = _derive_child_caps(core, parent, node, is_sub=False)
        assert caps.imports == {"itertools"}

    def test_absent_spec_inherits_parent(self):
        core = _core()
        parent = _parent_conv(core)
        parent.caps = parent.caps.copy(imports=["itertools"])
        node = _program_node(imports=None)
        assert _derive_child_caps(core, parent, node, is_sub=False).imports == {"itertools"}

    def test_unknown_module_rejected(self):
        core = _core()
        parent = _parent_conv(core)
        node = _program_node(imports=["numpy"])
        with pytest.raises(VMMemoryError, match="目录之外"):
            _derive_child_caps(core, parent, node, is_sub=False)

    def test_baseline_module_cannot_be_declared(self):
        core = _core()
        parent = _parent_conv(core)
        node = _program_node(imports=["math"])
        with pytest.raises(IllegalOperationError, match="基线"):
            _derive_child_caps(core, parent, node, is_sub=False)

    def test_cannot_exceed_parent(self):
        core = _core()
        parent = _parent_conv(core)
        parent.caps = parent.caps.copy(imports=["itertools"])
        node = _program_node(imports=["collections"])
        with pytest.raises(VMMemoryError, match="不可超出父对话"):
            _derive_child_caps(core, parent, node, is_sub=False)


class TestBuildImports:
    def test_effective_set_is_baseline_plus_grants(self):
        core = _core()
        conv = core.start("sys", "user")
        conv.caps = conv.caps.copy(imports=["itertools"])
        assert _build_imports(conv) == sorted(set(BASE_MODULES) | {"itertools"})

    def test_kernel_default_effective_set_is_full_catalog(self):
        core = _core()
        conv = core.start("sys", "user")
        assert _build_imports(conv) == sorted(set(BASE_MODULES) | set(GRANTABLE_MODULE_NAMES))


class TestExecutorPropagation:
    def _python_conv(self, core, imports):
        parent = _parent_conv(core)
        core.mem["prog"] = _program_node(content="x = 1", imports=sorted(imports))
        conv = _build_conversation(core, "$MEM.prog", "$MEM.plm_params", parent, is_sub=False)
        conv.append_user_message("pass")
        return conv

    def test_executor_receives_imports(self):
        core = _core()
        conv = self._python_conv(core, {"itertools"})
        para = core.unwrap("$MEM.plm_params", for_llm=False)
        executor = core._get_executor(conv, para)
        executor.exec(conv, para)
        assert executor.plm._imports == frozenset(set(BASE_MODULES) | {"itertools"})

    def test_caps_self_description_exposes_imports(self):
        core = _core()
        conv = self._python_conv(core, {"itertools"})
        summary = json.loads(core.unwrap(f"$MEM.caps.{conv.cid}", for_llm=False))
        assert summary["imports"]["base"] == list(BASE_MODULES)
        assert summary["imports"]["granted"] == ["itertools"]

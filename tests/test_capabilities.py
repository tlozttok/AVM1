"""对话能力（caps）数据模型：默认值、校验、地址上下文与检查点往返。

对应权限体系批次一 S1（plan-permission-phase1-2026-10-29.md）。
"""

import json

import pytest

from avm.capabilities import (
    ALL_TOOL_NAMES,
    MEM_ROOT_NAME,
    MODE_KERNEL,
    MODE_USER,
    AddressSpace,
    Capability,
    caps_from_dict,
    caps_to_dict,
    check_root_name,
    default_kernel_caps,
)
from avm.core import _instruction_registry
from avm.exceptions import IllegalOperationError
from avm.types import Conversation, SystemMessage, UserMessage


class TestInstructionNames:
    def test_tool_names_match_instruction_registry(self):
        """ALL_TOOL_NAMES 与指令注册表是同一份清单，防止两处漂移。"""
        assert set(_instruction_registry()) == set(ALL_TOOL_NAMES)


class TestCapabilityDefaults:
    def test_default_kernel_caps(self):
        caps = default_kernel_caps()
        assert caps.mode == MODE_KERNEL
        assert caps.roots == {MEM_ROOT_NAME: ()}
        assert caps.has_mem()
        assert caps.tools == ALL_TOOL_NAMES
        assert caps.groups == ()

    def test_user_caps_with_workspace_and_subset_tools(self):
        caps = Capability(
            mode=MODE_USER,
            groups=("worker",),
            roots={"work": ("workspaces", "w1")},
            tools=["memory_read", "memory_write"],
        )
        assert not caps.has_mem()
        assert caps.root_segments("work") == ("workspaces", "w1")
        assert caps.root_segments("nope") is None
        assert caps.tool_allowed("memory_read")
        assert not caps.tool_allowed("create_cmd")


class TestValidation:
    def test_user_mode_rejects_mem_root(self):
        with pytest.raises(IllegalOperationError):
            Capability(mode=MODE_USER, roots={MEM_ROOT_NAME: ()})

    @pytest.mark.parametrize("name", ["", "a.b", "$x", "conv", "batch", None, 3])
    def test_bad_root_names(self, name):
        with pytest.raises(IllegalOperationError):
            check_root_name(name)

    def test_mem_allowed_only_for_internal_tables(self):
        check_root_name(MEM_ROOT_NAME, allow_reserved=frozenset({MEM_ROOT_NAME}))
        caps = Capability(roots={MEM_ROOT_NAME: ()})
        assert caps.has_mem()

    def test_bad_mode(self):
        with pytest.raises(IllegalOperationError):
            Capability(mode="root")

    def test_unknown_tool(self):
        with pytest.raises(IllegalOperationError):
            Capability(tools=["memory_read", "format_disk"])

    def test_root_segments_must_be_sequence(self):
        with pytest.raises(IllegalOperationError):
            Capability(roots={"data": "MEM.a"})

    def test_root_segments_no_empty(self):
        with pytest.raises(IllegalOperationError):
            Capability(roots={"data": ("a", "", "b")})


class TestAddressSpace:
    def test_default_space_has_mem(self):
        assert AddressSpace().has_mem()
        assert AddressSpace().names() == (MEM_ROOT_NAME,)

    def test_space_from_caps(self):
        caps = Capability(mode=MODE_USER, roots={"work": ("w", "1")}, tools=[])
        space = caps.space()
        assert not space.has_mem()
        assert space.root("work") == ("w", "1")
        assert space == AddressSpace({"work": ("w", "1")})


class TestCapsSerialization:
    def test_dict_roundtrip(self):
        caps = Capability(
            mode=MODE_USER,
            groups=("worker", "auditor"),
            roots={"work": ("workspaces", "w2"), "lib": ("lib",)},
            tools=["memory_read", "send_instruction"],
        )
        data = caps_to_dict(caps)
        assert json.loads(json.dumps(data)) == data  # JSON 安全
        assert caps_from_dict(data) == caps

    def test_conversation_checkpoint_roundtrip(self):
        caps = Capability(
            mode=MODE_USER,
            groups=("worker",),
            roots={"work": ("workspaces", "w3")},
            tools=["memory_read"],
        )
        conv = Conversation(
            messages=[SystemMessage(content="s"), UserMessage(content="u")],
            caps=caps,
            attrs={"role": "greeter", "approved": "1"},
        )
        restored = Conversation.from_checkpoint(json.loads(json.dumps(conv.to_checkpoint())))
        assert restored.caps == caps
        assert restored.attrs == {"role": "greeter", "approved": "1"}
        assert restored.to_api_messages() == conv.to_api_messages()

    def test_old_checkpoint_without_caps_defaults_to_kernel(self):
        cp = {
            "version": 1,
            "name": "old",
            "para_ref": "$MEM.model_params",
            "is_python": False,
            "is_sub": False,
            "is_root": True,
            "metadata": {},
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
        }
        conv = Conversation.from_checkpoint(cp)
        assert conv.caps == default_kernel_caps()
        assert conv.attrs == {}

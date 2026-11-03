"""创建委托与模式：程序 ctrl.caps 解析、子能力计算、模式不可提权。

对应权限体系批次一 S3（plan-permission-phase1-2026-10-29.md）。
"""

import json

import pytest

from avm.capabilities import (
    ALL_TOOL_NAMES,
    MODE_KERNEL,
    MODE_USER,
    Capability,
    default_kernel_caps,
)
from avm.core import Core
from avm.image import ImageError
from avm.types import MetaDict, MetaList, StrNode


def _caps_node(caps):
    data = {}
    for key in ("mode", "workspace"):
        if key in caps:
            data[key] = StrNode(str(caps[key]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            data[key] = MetaList([StrNode(str(x)) for x in caps[key]])
    for key, value in caps.items():
        if key not in data and key not in ("mode", "workspace"):
            data[key] = MetaList([StrNode(str(x)) for x in value]) if isinstance(value, list) else StrNode(str(value))
    return MetaDict(data=data)


def _core():
    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    core.mem["para"] = MetaDict(data={"model": "test"}, ctrl={"type": "para"})
    conv = core.start("sys", "user", para_ref="$MEM.model_params")
    core._active_cid = conv.cid
    return core, conv


def _program(core, key="prog", caps=None, ctype="settingup", content="sys"):
    node = MetaDict(data={"content": content}, ctrl={"type": ctype})
    if caps is not None:
        node["caps"] = _caps_node(caps)
    core.mem[key] = node
    return f"$MEM.{key}"


def _call(core, conv, cmd, args, call_id="c1"):
    conv.user_batch.tool_responses.clear()
    core._process_return_calls(
        [{"cmd_type": cmd, "call_id": call_id, "args": args}], conv
    )
    return conv.user_batch.tool_responses[-1].content


def _call_maybe(core, conv, cmd, args, call_id="c1"):
    """create_sub 成功时不写工具响应（父进入等待），返回空串。"""
    conv.user_batch.tool_responses.clear()
    core._process_return_calls(
        [{"cmd_type": cmd, "call_id": call_id, "args": args}], conv
    )
    if not conv.user_batch.tool_responses:
        return ""
    return conv.user_batch.tool_responses[-1].content


def _created_cid(text):
    assert text.startswith("Success created: cid="), text
    return int(text.split("=")[-1])


def _user_conv(core, roots=None, tools=None, groups=()):
    """受限父对话：默认手里有工作空间与常见程序/参数节点，便于测试创建指令。"""
    conv = core.start("sys", "user", para_ref="$MEM.model_params")
    conv.caps = Capability(
        mode=MODE_USER,
        groups=groups,
        roots=roots or {
            "work": ("ws",),
            "prog": ("prog",),
            "para": ("para",),
            "utext": ("user_text",),
        },
        tools=tools if tools is not None else ALL_TOOL_NAMES,
    )
    return conv


class TestProgramCapsParsing:
    def test_unknown_field_rejected(self):
        core, conv = _core()
        _program(core, caps={"mode": "user", "wat": 1})
        text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        assert text.startswith("Error:") or text.startswith("Illegal:")
        assert "未知字段" in text

    def test_ctrl_caps_rejected(self):
        core, conv = _core()
        core.mem["prog"] = MetaDict(
            data={"content": "sys"}, ctrl={"type": "settingup", "caps": "{}"}
        )
        text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        assert "ctrl.caps" in text and "value.caps" in text


class TestChildCaps:
    def test_legacy_program_without_caps_inherits_full_kernel(self):
        core, conv = _core()
        _program(core, caps=None)
        cid = _created_cid(_call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"}))
        child = core.get_conversation(cid)
        assert child.caps == default_kernel_caps()

    def test_user_mode_grants_groups_roots_tools(self):
        core, conv = _core()
        core.mem["lib"] = MetaDict(data={"x": "v"})
        core.root_registry["lib"] = ("lib",)
        _program(core, caps={
            "mode": "user",
            "groups": ["worker"],
            "roots": ["lib"],
            "tools": ["memory_read"],
        })
        text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.prog", "para_ref": "$MEM.para"})
        child = core.get_conversation(_created_cid(text))
        assert child.caps.mode == MODE_USER
        assert child.caps.groups == ("worker",)
        assert child.caps.roots["lib"] == ("lib",)
        assert child.caps.roots["work"][0] == "workspaces"
        assert child.caps.tools == frozenset({"memory_read"})
        assert child.workspace_root_name == "work"

    def test_tools_cannot_exceed_parent(self):
        from avm.core import _derive_child_caps
        from avm.exceptions import VMMemoryError

        core, conv = _core()
        # 父对话工具收窄到 memory_read
        conv.caps = Capability(mode=MODE_KERNEL, tools=["memory_read"])
        _program(core, caps={"tools": ["memory_read", "create_cmd"]})
        with pytest.raises(VMMemoryError) as ei:
            _derive_child_caps(core, conv, core.mem["prog"], is_sub=False)
        assert "工具集不可超出父对话" in str(ei.value)

    def test_user_parent_cannot_create_kernel_child(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        parent = _user_conv(core, tools=ALL_TOOL_NAMES)
        _program(core, caps={"mode": "kernel"})
        text = _call(core, parent, "create_cmd", {"system_ref": "$prog", "para_ref": "$para"})
        assert "模式不可提权" in text

    def test_sub_cannot_cross_mode(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        parent = _user_conv(core)
        core.mem["user_text"] = "hi"
        _program(core, caps={"mode": "user"})
        text = _call_maybe(core, parent, "create_sub", {
            "system_ref": "$prog", "user_ref": "$utext", "para_ref": "$para",
        })
        # 用户态父 → 程序声明 user：模式相同，但根表沿用父根；创建应成功（无工具响应）
        assert text == ""

        kernel_text = _call_maybe(core, conv, "create_sub", {
            "system_ref": "$MEM.prog", "user_ref": "$MEM.user_text", "para_ref": "$MEM.para",
        })
        assert "亚对话不能跨模式创建" in kernel_text

    def test_same_domain_sub_inherits_parent_roots(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        parent = _user_conv(core)
        core.mem["user_text"] = "hi"
        _program(core, caps={"tools": ["memory_read"]})
        text = _call_maybe(core, parent, "create_sub", {
            "system_ref": "$prog", "user_ref": "$utext", "para_ref": "$para",
        })
        # create_sub 成功没有成功响应（父进入休眠）；亚对话应存在且继承根
        subs = [c for c in core._conv_by_cid.values() if c.is_sub]
        assert len(subs) == 1
        assert subs[0].caps.mode == MODE_USER
        assert subs[0].caps.roots["work"] == ("ws",)
        assert subs[0].caps.tools == frozenset({"memory_read"})

    def test_same_domain_sub_cannot_add_roots(self):
        core, conv = _core()
        core.mem["ws"] = MetaDict(data={})
        core.mem["lib"] = MetaDict(data={})
        core.root_registry["lib"] = ("lib",)
        parent = _user_conv(core)
        core.mem["user_text"] = "hi"
        _program(core, caps={"roots": ["lib"]})
        text = _call(core, parent, "create_sub", {
            "system_ref": "$prog", "user_ref": "$utext", "para_ref": "$para",
        })
        assert "不能向子对话授予父对话没有的根" in text


def _caps_image_node(caps):
    value = {}
    for key in ("mode", "workspace"):
        if key in caps:
            value[key] = {"kind": "str", "value": str(caps[key])}
    for key in ("groups", "roots", "tools"):
        if key in caps:
            value[key] = {"kind": "list", "value": [
                {"kind": "str", "value": str(x)} for x in caps[key]
            ]}
    return {"kind": "dict", "value": value}


class TestCtrlCapsRejectedByImage:
    def test_image_ctrl_caps_rejected(self, tmp_path):
        from avm.image import ImageError, load_image
        image = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                                 "value": {"model": {"kind": "str", "value": "test"}}},
                "prog": {"kind": "dict", "ctrl": {"type": "settingup", "caps": "{}"},
                         "value": {"content": {"kind": "str", "value": "sys"}}},
                "u": {"kind": "str", "value": "go"},
            },
            "para_ref": "$MEM.model_params",
            "init": {"system_ref": "$MEM.prog", "user_ref": "$MEM.u"},
        }
        path = tmp_path / "i.json"
        path.write_text(json.dumps(image), encoding="utf-8")
        with pytest.raises(ImageError) as ei:
            load_image(Core(), str(path))
        assert "value.caps" in str(ei.value)


class TestInitCaps:
    def _image(self, tmp_path, caps):
        image = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                                 "value": {"model": {"kind": "str", "value": "test"}}},
                "programs": {"kind": "dict", "value": {
                    "init": {"kind": "dict", "ctrl": {"type": "settingup"},
                             "value": {"content": {"kind": "str", "value": "sys"},
                                       "caps": _caps_image_node(caps)}},
                }},
                "u": {"kind": "str", "value": "go"},
            },
            "para_ref": "$MEM.model_params",
            "init": {"system_ref": "$MEM.programs.init", "user_ref": "$MEM.u"},
        }
        path = tmp_path / "image.json"
        path.write_text(json.dumps(image), encoding="utf-8")
        return str(path)

    def test_init_user_mode_rejected(self, tmp_path):
        from avm.image import load_image

        path = self._image(tmp_path, {"mode": "user"})
        with pytest.raises(ImageError) as ei:
            load_image(Core(), path)
        assert "内核态" in str(ei.value)

    def test_init_caps_narrow_tools(self, tmp_path):
        from avm.image import load_image

        path = self._image(tmp_path, {"mode": "kernel", "tools": ["memory_read"]})
        core = load_image(Core(), path)
        root = core.get_conversation(0)
        assert root.caps.mode == MODE_KERNEL
        assert root.caps.tools == frozenset({"memory_read"})

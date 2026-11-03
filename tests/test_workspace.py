"""工作空间分配、security 段与私有性不变式。

对应权限体系批次一 S4（plan-permission-phase1-2026-10-29.md）：
- 镜像 security.roots 注册表与 security.workspace_region；
- 用户态对话创建时分配私有工作空间（前缀无关、保留区）；
- 区域子树禁止链接/设备挂载；
- 根目标前缀无关；私有工作空间互不可见。
"""

import json
import os

import pytest

from avm.capabilities import ALL_TOOL_NAMES, Capability
from avm.core import Core
from avm.exceptions import IllegalOperationError, VMMemoryError
from avm.image import ImageError, load_image
from avm.memory import Memory
from avm.memory_device import StringDevice
from avm.types import LinkNode, MetaDict, MetaList, StrNode

IMAGE = os.path.join(os.path.dirname(__file__), "..", "images", "security_demo.json")


def _caps_node(caps):
    data = {}
    for key in ("mode", "workspace"):
        if key in caps:
            data[key] = StrNode(str(caps[key]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            data[key] = MetaList([StrNode(str(x)) for x in caps[key]])
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


def _create_user(core, conv, caps):
    core.mem["user_text"] = "任务"
    _program(core, "worker", caps)
    text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.worker", "para_ref": "$MEM.para"})
    assert text.startswith("Success created: cid="), text
    return int(text.split("=")[-1])


class TestSecurityConfig:
    def test_install_creates_region_and_registry(self):
        core = Core()
        core.install_security_config({"lib": ("lib",)}, ("workspaces",))
        assert core.root_registry == {"lib": ("lib",)}
        assert core.workspace_region == ("workspaces",)
        assert "workspaces" in core.mem._data

    def test_registry_targets_prefix_free(self):
        core = Core()
        with pytest.raises(VMMemoryError) as ei:
            core.install_security_config({"a": ("x",), "b": ("x", "y")}, ("workspaces",))
        assert "前缀冲突" in str(ei.value)

    def test_registry_target_cannot_overlap_region(self):
        core = Core()
        with pytest.raises(VMMemoryError):
            core.install_security_config({"a": ("workspaces",)}, ("workspaces",))
        with pytest.raises(VMMemoryError):
            core.install_security_config({"a": ("workspaces", "lib")}, ("workspaces",))

    def test_region_subtree_cannot_contain_links(self):
        core = Core()
        core.mem["workspaces"] = MetaDict(data={"lk": LinkNode(target="$MEM.x")})
        core.mem["x"] = "v"
        with pytest.raises(VMMemoryError) as ei:
            core.install_security_config({}, ("workspaces",))
        assert "链接" in str(ei.value)

    def test_region_cannot_contain_device_mount(self):
        core = Core()
        core.mem.mount("workspaces.dev", StringDevice("v"))
        with pytest.raises(VMMemoryError):
            core.install_security_config({}, ("workspaces",))

    def test_region_must_be_dict(self):
        core = Core()
        core.mem["workspaces"] = "not-a-dict"
        with pytest.raises(VMMemoryError):
            core.install_security_config({}, ("workspaces",))


class TestWorkspaceAllocation:
    def test_user_conversation_gets_private_workspace(self):
        core, conv = _core()
        core.mem["lib"] = MetaDict(data={"greeting": "hi"})
        core.install_security_config({"lib": ("lib",)}, ("workspaces",))
        cid = _create_user(core, conv, {
            "mode": "user", "roots": ["lib"],
            "tools": ["memory_read", "memory_write"],
        })
        child = core.get_conversation(cid)
        target = child.caps.roots["work"]
        assert target[:1] == ("workspaces",)
        workspace = core.mem.resolve_path(["MEM"] + list(target), for_llm=False)
        assert isinstance(workspace, MetaDict)
        assert workspace.get_ctrl()["owner"] == str(conv.cid)
        # 授予根随 caps 进根表
        assert child.caps.roots["lib"] == ("lib",)
        # 用户态对话看不到 $MEM 与其他根
        assert child.caps.mode == "user"
        assert not child.caps.has_mem()

    def test_two_workspaces_are_distinct(self):
        core, conv = _core()
        core.install_security_config({}, ("workspaces",))
        cid1 = _create_user(core, conv, {"mode": "user"})
        cid2 = _create_user(core, conv, {"mode": "user"})
        t1 = core.get_conversation(cid1).caps.roots["work"]
        t2 = core.get_conversation(cid2).caps.roots["work"]
        assert t1 != t2
        assert t1[0] == t2[0] == "workspaces"

    def test_allocation_rejects_overlapping_root(self):
        core, conv = _core()
        core.install_security_config({}, ("workspaces",))
        # 手动塞一个覆盖区域的根（模拟外部得来的能力数据）
        conv.caps = conv.caps.copy(roots={"MEM": (), "cover": ("workspaces",)})
        with pytest.raises(VMMemoryError) as ei:
            core._allocate_workspace(conv.cid)
        assert "前缀冲突" in str(ei.value)

    def test_workspace_root_name_from_caps(self):
        core, conv = _core()
        core.install_security_config({}, ("workspaces",))
        core.mem["user_text"] = "t"
        _program(core, "worker", {"mode": "user", "workspace": "sandbox"})
        text = _call(core, conv, "create_cmd", {"system_ref": "$MEM.worker", "para_ref": "$MEM.para"})
        child = core.get_conversation(int(text.split("=")[-1]))
        assert "sandbox" in child.caps.roots and "work" not in child.caps.roots

    def test_workspace_name_collision_with_grant_rejected(self):
        core, conv = _core()
        core.mem["lib"] = MetaDict(data={})
        core.install_security_config({"work": ("lib",)}, ("workspaces",))
        text = _call(core, conv, "create_cmd", {
            "system_ref": (_program(core, "worker", {
                "mode": "user", "roots": ["work"], "workspace": "work",
            })),
            "para_ref": "$MEM.para",
        })
        assert "重名" in text


class TestSecurityDemoImage:
    def test_load_demo_and_create_user(self):
        core = load_image(Core(), IMAGE)
        root = core.get_conversation(0)
        assert root.caps.mode == "kernel"
        assert core.root_registry == {"lib": ("lib",)}
        assert core.workspace_region == ("workspaces",)
        text = _call(core, root, "create_cmd", {
            "system_ref": "$MEM.programs.worker", "para_ref": "$MEM.model_params",
        })
        child = core.get_conversation(int(text.split("=")[-1]))
        assert child.caps.mode == "user"
        assert child.caps.roots["lib"] == ("lib",)
        assert child.caps.roots["work"][0] == "workspaces"

    def test_demo_user_cannot_read_mem(self):
        core = load_image(Core(), IMAGE)
        root = core.get_conversation(0)
        text = _call(core, root, "create_cmd", {
            "system_ref": "$MEM.programs.worker", "para_ref": "$MEM.model_params",
        })
        child = core.get_conversation(int(text.split("=")[-1]))
        core._active_cid = child.cid
        assert _call(core, child, "memory_read", {"ref": "$lib.greeting"}) == "hello from lib"
        assert _call(core, child, "memory_read", {"ref": "$MEM"}).startswith("Error:")
        assert _call(core, child, "memory_read", {"ref": "$work"}).startswith("dict[keys=[]")


class TestImageSecurityValidation:
    def _image(self, tmp_path, security, extra_mem=None, devices=None):
        mem = {
            "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                             "value": {"model": {"kind": "str", "value": "test"}}},
            "programs": {"kind": "dict", "value": {
                "init": {"kind": "dict", "ctrl": {"type": "settingup"},
                         "value": {"content": {"kind": "str", "value": "sys"}}},
            }},
            "u": {"kind": "str", "value": "go"},
        }
        if extra_mem:
            mem.update(extra_mem)
        image = {
            "meta": {"version": 1},
            "mem": mem,
            "para_ref": "$MEM.model_params",
            "init": {"system_ref": "$MEM.programs.init", "user_ref": "$MEM.u"},
        }
        if security is not None:
            image["security"] = security
        if devices is not None:
            image["devices"] = devices
        path = tmp_path / "img.json"
        path.write_text(json.dumps(image), encoding="utf-8")
        return str(path)

    def test_unknown_security_key(self, tmp_path):
        path = self._image(tmp_path, {"roots": {}, "wat": 1})
        with pytest.raises(ImageError) as ei:
            load_image(Core(), path)
        assert "未知字段" in str(ei.value)

    def test_prefix_conflict_at_load(self, tmp_path):
        path = self._image(tmp_path, {"roots": {"a": "$MEM.x", "b": "$MEM.x.y"}})
        with pytest.raises(ImageError):
            load_image(Core(), path)

    def test_device_mount_inside_region_rejected(self, tmp_path):
        dev_file = tmp_path / "dev.py"
        dev_file.write_text(
            "from avm.memory_device import StringDevice\nclass Dev(StringDevice):\n    pass\n",
            encoding="utf-8",
        )
        path = self._image(
            tmp_path,
            {"roots": {}, "workspace_region": "$MEM.workspaces"},
            devices=[{"path": "workspaces.dev", "file": "dev.py", "class": "Dev", "args": {}}],
        )
        # 设备文件写在镜像同目录，file 用相对路径
        with pytest.raises(ImageError) as ei:
            load_image(Core(), path)
        assert "链接" in str(ei.value)

    def test_default_region_without_security_section(self, tmp_path):
        path = self._image(tmp_path, None)
        core = load_image(Core(), path)
        assert core.root_registry == {}
        assert core.workspace_region == ("workspaces",)
        assert "workspaces" not in core.mem._data  # 没有用户态对话时不物化

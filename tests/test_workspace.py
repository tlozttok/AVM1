"""动态授予根、security 段与根表私有性。

对应 plan-dynamic-roots-2026-11-10（删除内核工作空间）：
- 内核不再分配工作空间：用户态对话创建后不自动出现 $work；
- 软件用 memory_make 建目录、写 $MEM.conversations.<cid>.roots 动态加根/撤根；
- 镜像 security.roots 注册表保留；security.workspace_region 与 caps.workspace 被拒绝；
- 根校验（保留区、设备路径、前缀无关、与其他根冲突）保留在 Core。
"""

import json

import pytest

from avm.core import Core
from avm.exceptions import IllegalOperationError, VMMemoryError
from avm.image import ImageError, load_image
from avm.memory_device import StringDevice
from avm.types import MetaDict, MetaList, StrNode


def _security_image(tmp_path) -> str:
    """在测试内构造的最小 security 段镜像：内核 init + 只持有 $lib 根的用户态程序。

    原示例镜像已删除；这里只保留本文件用到的节点（$lib.greeting、programs.worker
    的 caps、para 与 init 入口），覆盖根注册表与用户态根可见性。
    """
    image = {
        "meta": {"version": 1},
        "mem": {
            "model_params": {
                "kind": "dict", "ctrl": {"type": "para"},
                "value": {"model": {"kind": "str", "value": "test"}},
            },
            "lib": {
                "kind": "dict",
                "value": {"greeting": {"kind": "str", "value": "hello from lib"}},
            },
            "programs": {
                "kind": "dict",
                "value": {
                    "kernel_init": {
                        "kind": "dict", "ctrl": {"type": "settingup"},
                        "value": {"content": {"kind": "str", "value": "kernel init"}},
                    },
                    "worker": {
                        "kind": "dict", "ctrl": {"type": "settingup"},
                        "value": {
                            "content": {"kind": "str", "value": "worker"},
                            "caps": {
                                "kind": "dict",
                                "value": {
                                    "mode": {"kind": "str", "value": "user"},
                                    "roots": {"kind": "list", "value": [
                                        {"kind": "str", "value": "lib"},
                                    ]},
                                },
                            },
                        },
                    },
                },
            },
            "init_prompt": {"kind": "str", "value": "开始"},
        },
        "security": {"roots": {"lib": "$MEM.lib"}},
        "para_ref": "$MEM.model_params",
        "init": {"name": "init", "system_ref": "$MEM.programs.kernel_init",
                 "user_ref": "$MEM.init_prompt"},
    }
    path = tmp_path / "security.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _caps_node(caps):
    data = {}
    if "mode" in caps:
        data["mode"] = StrNode(str(caps["mode"]))
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


def _create_user(core, conv, caps=None):
    _program(core, "worker", caps if caps is not None else {"mode": "user"})
    text = _call(core, conv, "create_cmd", {
        "system_ref": "$MEM.worker", "para_ref": "$MEM.para",
    })
    assert text.startswith("Success created: cid="), text
    return core.get_conversation(int(text.split("=")[-1]))


def _grant_work(core, conv, extra=None, name="work"):
    """测试辅助：软件建目录 + 动态加根（内核不再自动分配 $work）。"""
    assert "workdirs" not in core.mem or isinstance(core.mem["workdirs"], MetaDict)
    if "workdirs" not in core.mem:
        core.mem.make("$MEM", "workdirs", "dict")
    dir_key = f"w{conv.cid}"
    core.mem.make("$MEM.workdirs", dir_key, "dict")
    roots = {name: ["workdirs", dir_key]}
    roots.update(extra or {})
    core.set_conversation_roots(conv.cid, roots)
    return ("workdirs", dir_key)


class TestSecurityConfig:
    def test_install_root_registry(self):
        core = Core()
        core.install_security_config({"lib": ("lib",)})
        assert core.root_registry == {"lib": ("lib",)}
        assert not hasattr(core, "workspace_region")
        assert "workspaces" not in core.mem._data  # 内核不物化任何工作空间区域

    def test_registry_targets_prefix_free(self):
        core = Core()
        with pytest.raises(VMMemoryError) as ei:
            core.install_security_config({"a": ("x",), "b": ("x", "y")})
        assert "前缀冲突" in str(ei.value)

    def test_registry_rejects_control_device_targets(self):
        core = Core()
        with pytest.raises(VMMemoryError):
            core.install_security_config({"admin": ("conversations",)})


class TestDynamicRoots:
    def test_user_conversation_has_no_auto_work_root(self):
        core, root = _core()
        child = _create_user(core, root)
        assert child.caps.mode == "user"
        assert sorted(child.caps.roots) == ["caps"]
        assert _call(core, child, "memory_read", {"ref": "$work"}).startswith("Error:")

    def test_grant_directory_then_read(self):
        core, root = _core()
        child = _create_user(core, root)
        target = _grant_work(core, child)
        assert child.caps.roots["work"] == target
        work_dir = "$MEM." + ".".join(target)
        core.mem.make(work_dir, "greeting", "str")
        core.mem.set(work_dir + ".greeting", "hi")
        assert _call(core, child, "memory_read", {"ref": "$work.greeting"}) == "hi"

    def test_grant_keeps_caps_self_root(self):
        core, root = _core()
        child = _create_user(core, root)
        caps_target = child.caps.roots["caps"]
        _grant_work(core, child)
        assert child.caps.roots["caps"] == caps_target

    def test_write_is_replacement_not_merge(self):
        core, root = _core()
        child = _create_user(core, root)
        _grant_work(core, child)
        core.mem.make("$MEM", "other", "dict")
        core.set_conversation_roots(child.cid, {"other": ["other"]})
        assert "work" not in child.caps.roots
        assert child.caps.roots["other"] == ("other",)
        assert _call(core, child, "memory_read", {"ref": "$work"}).startswith("Error:")
        assert _call(core, child, "memory_read", {"ref": "$other"}).startswith("dict[keys=")

    def test_remove_root(self):
        core, root = _core()
        child = _create_user(core, root)
        _grant_work(core, child)
        core.set_conversation_roots(child.cid, {})
        assert "work" not in child.caps.roots
        assert _call(core, child, "memory_read", {"ref": "$work"}).startswith("Error:")

    def test_kernel_conversation_roots_rejected(self):
        core, root = _core()
        with pytest.raises(IllegalOperationError) as ei:
            core.set_conversation_roots(root.cid, {"lib": ["lib"]})
        assert "内核态" in str(ei.value)
        assert root.caps.roots == {"MEM": ()}

    def test_caps_root_cannot_be_overwritten(self):
        core, root = _core()
        child = _create_user(core, root)
        with pytest.raises(IllegalOperationError) as ei:
            core.set_conversation_roots(child.cid, {"caps": ["lib"]})
        assert "Core 分配" in str(ei.value)
        assert child.caps.roots["caps"][0] == "caps"

    def test_grant_target_validation(self):
        core, root = _core()
        child = _create_user(core, root)
        core.root_registry["lib"] = ("lib",)
        cases = [
            {"MEM": ["x"]},                 # 保留根名
            {"x": []},                      # $MEM 根本身
            {"x": "lib"},                   # 不是段序列
            {"x": ["conversations"]},       # 控制面设备
            {"x": ["listeners"]},
            {"x": ["roots"]},
            {"x": ["caps"]},
            {"x": ["lib", "sub"]},          # 与根注册表前缀冲突
        ]
        for roots_map in cases:
            with pytest.raises((IllegalOperationError, VMMemoryError)):
                core.set_conversation_roots(child.cid, roots_map)
        assert "work" not in child.caps.roots

    def test_grant_conflicts_with_other_conversation_roots(self):
        core, root = _core()
        first = _create_user(core, root)
        second = _create_user(core, root)
        target = _grant_work(core, first)
        with pytest.raises(IllegalOperationError) as ei:
            core.set_conversation_roots(second.cid, {"parent": ["workdirs"]})
        assert "前缀冲突" in str(ei.value)
        # 相同目标允许共享（旧校验语义：前缀冲突才算冲突）
        core.set_conversation_roots(second.cid, {"copy": list(target)})
        assert second.caps.roots["copy"] == target

    def test_grant_rejects_internal_device_paths(self):
        core, root = _core()
        child = _create_user(core, root)
        core.mem.mount("box.dev", StringDevice("v"))
        with pytest.raises(IllegalOperationError) as ei:
            core.set_conversation_roots(child.cid, {"box": ["box"]})
        assert "设备" in str(ei.value)
        # 显式授予设备本身允许
        core.set_conversation_roots(child.cid, {"dev": ["box", "dev"]})
        assert child.caps.roots["dev"] == ("box", "dev")


class TestSecurityImage:
    def test_load_image_and_create_user(self, tmp_path):
        core = load_image(Core(), _security_image(tmp_path))
        root = core.get_conversation(0)
        assert root.caps.mode == "kernel"
        assert core.root_registry == {"lib": ("lib",)}
        text = _call(core, root, "create_cmd", {
            "system_ref": "$MEM.programs.worker", "para_ref": "$MEM.model_params",
        })
        child = core.get_conversation(int(text.split("=")[-1]))
        assert child.caps.mode == "user"
        assert child.caps.roots["lib"] == ("lib",)
        assert "work" not in child.caps.roots

    def test_user_cannot_read_mem_or_work(self, tmp_path):
        core = load_image(Core(), _security_image(tmp_path))
        root = core.get_conversation(0)
        text = _call(core, root, "create_cmd", {
            "system_ref": "$MEM.programs.worker", "para_ref": "$MEM.model_params",
        })
        child = core.get_conversation(int(text.split("=")[-1]))
        core._active_cid = child.cid
        assert _call(core, child, "memory_read", {"ref": "$lib.greeting"}) == "hello from lib"
        assert _call(core, child, "memory_read", {"ref": "$MEM"}).startswith("Error:")
        assert _call(core, child, "memory_read", {"ref": "$work"}).startswith("Error:")


class TestImageSecurityValidation:
    def _image(self, tmp_path, security, extra_mem=None):
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
        path = tmp_path / "img.json"
        path.write_text(json.dumps(image), encoding="utf-8")
        return str(path)

    def test_unknown_security_key(self, tmp_path):
        path = self._image(tmp_path, {"roots": {}, "wat": 1})
        with pytest.raises(ImageError) as ei:
            load_image(Core(), path)
        assert "未知字段" in str(ei.value)

    def test_workspace_region_rejected_with_hint(self, tmp_path):
        path = self._image(tmp_path, {"roots": {},
                                      "workspace_region": "$MEM.workspaces"})
        with pytest.raises(ImageError) as ei:
            load_image(Core(), path)
        assert "workspace_region" in str(ei.value) and "已移除" in str(ei.value)

    def test_prefix_conflict_at_load(self, tmp_path):
        path = self._image(tmp_path, {"roots": {"a": "$MEM.x", "b": "$MEM.x.y"}})
        with pytest.raises(ImageError):
            load_image(Core(), path)

    def test_default_without_security_section(self, tmp_path):
        path = self._image(tmp_path, None)
        core = load_image(Core(), path)
        assert core.root_registry == {}
        assert "workspaces" not in core.mem._data

"""程序包：运行期安装/卸载的"内存子树 + 设备"单元。

包格式（一个 JSON 文件）：

```json
{"meta": {"format": "avm.package", "version": 1,
          "name": "demo", "package_version": "1.0.0", "description": "..."},
 "mem": { ... 与镜像 mem 段同格式 ... },
 "devices": [{ "path": "io", "file": "io.py", "class": "EchoDevice", "args": {},
                "sha256": "<可选：设备文件内容哈希>"}]}
```

安装落点：`$MEM.packages.<name>.<package_version>` 是包内存子树的根节点；
`$MEM.package_records.<name>.<package_version>` 是安装记录（来源路径、包文件 sha256、
安装时间、设备清单）。设备清单每项带设备文件 sha256；loader 安装时还带
approved_sha256 / approved_at / approved_via（整包批准：approved_sha256 是包文件哈希）。
`$MEM.packages.<name>.current` 指向最近安装的版本字符串
（升级 = 装新版本后指针改指新版本，一条 `memory_write`，原子）。

设备条目的 `path` 是**相对包子树的挂载路径**（如 `io` → `$MEM.packages.<name>.<version>.io`）：
这样卸载 detach 整棵子树时挂载点链接一并回收，不会留下悬空链接。设备文件 `file`
相对包文件所在目录解析；设备条目可选的 `sha256` 会在安装时与文件实际哈希核对。

安装顺序：校验 → 挂设备 → 建子树 → 写记录 → 更新 current。任一步失败回滚已挂设备
与已建节点；回滚的操作不进日志——只有成功的安装才把效果提交给回合日志。卸载只做
"卸设备 + detach 子树 + 删记录"三件事；悬空引用不主动清理，访问得到既有的
`MemoryLinkBrokenError`（计划 B1）。

这条模块只做机制，不做包依赖解析、版本约束、签名校验（计划"不做什么"）。
"""

import hashlib
import json
import os
import sys
import time
from typing import Optional

from .exceptions import VMMemoryError
from .memory import Memory, check_new_key
from .types import MetaDict, MetaList, StrNode

PACKAGE_FORMAT = "avm.package"
PACKAGE_VERSION = 1

PACKAGES_ROOT = "packages"
RECORDS_ROOT = "package_records"
_PACKAGES_META = "已安装程序包"
_RECORDS_META = "程序包安装记录"


# ----------------------------------------------------------------------
# 读取与校验
# ----------------------------------------------------------------------

def read_package(path: str) -> dict:
    """读程序包并校验形状；格式问题抛 VMMemoryError（面向对话是 Error 级）。"""
    if not isinstance(path, str) or not path:
        raise VMMemoryError("程序包路径必须是非空字符串")
    if not os.path.isfile(path):
        raise VMMemoryError(f"程序包不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            package = json.load(f)
    except (OSError, ValueError) as e:
        raise VMMemoryError(f"程序包不是合法 JSON：{path}") from e
    if not isinstance(package, dict):
        raise VMMemoryError(f"程序包根节点必须是对象：{path}")
    meta = package.get("meta")
    if not isinstance(meta, dict):
        raise VMMemoryError(f"程序包缺少 meta 段：{path}")
    if meta.get("format") != PACKAGE_FORMAT:
        raise VMMemoryError(
            f"不是 AVM 程序包（format={meta.get('format')!r}）：{path}"
        )
    if meta.get("version") != PACKAGE_VERSION:
        raise VMMemoryError(
            f"不支持的程序包格式版本 {meta.get('version')!r}（当前支持 {PACKAGE_VERSION}）"
        )
    name = meta.get("name")
    package_version = meta.get("package_version")
    for label, value in (("name", name), ("package_version", package_version)):
        if not isinstance(value, str) or not value:
            raise VMMemoryError(f"程序包 meta.{label} 必须是非空字符串")
        if "." in value:
            raise VMMemoryError(f"程序包 meta.{label} 不能包含点号（点号是路径分隔符）：{value!r}")
    mem = package.get("mem") or {}
    if not isinstance(mem, dict):
        raise VMMemoryError("程序包 mem 段必须是对象")
    devices = package.get("devices") or []
    if not isinstance(devices, list):
        raise VMMemoryError("程序包 devices 段必须是数组")
    return package


def package_path(name: str, package_version: str) -> list:
    """包内存在 $MEM 下的真实段序列（不含 MEM 根）。"""
    return [PACKAGES_ROOT, name, package_version]


def record_path(name: str, package_version: str) -> list:
    return [RECORDS_ROOT, name, package_version]


def _node_at(core, segments: list):
    node = core.mem._data
    for segment in segments:
        if not isinstance(node, (dict, MetaDict)):
            return None
        node = node.get(segment)
        if node is None:
            return None
    return node


def list_installed(core) -> list:
    """列出已安装程序包（读安装记录；调试/测试与装载设备回执用）。"""
    records = core.mem._data.get(RECORDS_ROOT)
    if not isinstance(records, MetaDict):
        return []
    out = []
    for name, versions in records.items():
        if not isinstance(versions, MetaDict):
            continue
        for package_version, record in versions.items():
            entry = {"name": name, "package_version": package_version}
            if isinstance(record, MetaDict):
                for key in ("source", "sha256", "installed_at"):
                    value = record.get(key)
                    if isinstance(value, str):
                        entry[key] = value
            out.append(entry)
    return sorted(out, key=lambda item: (item["name"], item["package_version"]))


# ----------------------------------------------------------------------
# 安装
# ----------------------------------------------------------------------

def install(core, path: str, *, base_dir: Optional[str] = None,
            approved_sha256: Optional[str] = None,
            approved_at: Optional[str] = None,
            approved_via: Optional[str] = None) -> dict:
    """安装程序包（挂设备 → 建子树 → 写记录 → 更新 current），返回安装记录。

    `approved_sha256` 是 loader 转来的程序包文件批准哈希；给了就必须与包文件
    实际 sha256 相等。设备文件的内容哈希在装载时重算并写进设备 spec / 安装记录；
    spec 里的 approved_sha256 是这次整包批准的程序包哈希，恢复/重放只用 spec 的
    sha256 核对设备文件。`approved_at` / `approved_via` 只写记录，不参与校验。

    失败回滚已发生的设备挂载与内存节点，异常抛 VMMemoryError（对话得到 Error）。
    只有成功安装才把状态变化提交给回合日志（见 `_commit_effects`）。
    """
    from .image import (
        ImageError, _device_entry_fields, _parse_node, _sha256_file,
        mount_device_entry,
    )

    package_sha256 = None
    if os.path.isfile(path):
        try:
            package_sha256 = _sha256_file(os.path.abspath(path))
        except OSError as e:
            raise VMMemoryError(f"读取程序包失败：{path}: {e}") from e
    if approved_sha256 is not None:
        if not isinstance(approved_sha256, str) or not approved_sha256.strip():
            raise VMMemoryError("程序包批准哈希必须是非空字符串")
        if package_sha256 is None:
            raise VMMemoryError(f"程序包不存在：{path}")
        if approved_sha256.strip().lower() != package_sha256:
            raise VMMemoryError(
                f"程序包哈希与批准哈希不一致（{path}）："
                f"期望 {approved_sha256}，实际 {package_sha256}；拒绝安装"
            )
    package = read_package(path)
    meta = package["meta"]
    name = meta["name"]
    package_version = meta["package_version"]
    mem_section = package.get("mem") or {}
    devices_section = package.get("devices") or []
    check_new_key(name)
    check_new_key(package_version)

    target = package_path(name, package_version)
    if _node_at(core, target) is not None:
        raise VMMemoryError(
            f"程序包已安装：{name} {package_version}"
            f"（地址已存在——只安装一次，不覆盖；升级请装新版本）"
        )

    # 1) 解析包内存子树与设备标记（纯解析，无副作用）
    markers = []
    parsed = {}
    for key, node in mem_section.items():
        value = _parse_node(node, target + [key], markers)
        if value is not None:
            parsed[key] = value
    base = base_dir
    if base is None:
        base = os.path.dirname(os.path.abspath(path))
    device_mounts = {}
    for dev in devices_section:
        if not isinstance(dev, dict):
            raise VMMemoryError("程序包 devices 条目必须是对象")
        rel = dev.get("path")
        if not isinstance(rel, str) or not rel:
            raise VMMemoryError("程序包 devices 条目必须包含非空 path")
        if rel.startswith("$"):
            raise VMMemoryError(f"程序包 devices.path 必须是相对包子树的路径，得到 {rel!r}")
        mount = ".".join(target + rel.split("."))
        if mount not in markers:
            raise VMMemoryError(
                f"程序包设备 {rel} 没有在 mem 段标记 kind=device"
            )
        # 包安装同时核对每台设备的文件哈希：装载前先算一次，交给挂载入口再算，
        # 两边不一致说明文件在安装过程中被换掉，拒绝安装（窗口收敛到一次安装内）。
        try:
            fields = _device_entry_fields(dict(dev, path=mount), base)
        except ImageError as e:
            raise VMMemoryError(f"程序包设备装载失败：{e}") from e
        if fields["file_path"] is not None:
            try:
                actual = _sha256_file(fields["file_path"])
            except OSError as e:
                raise VMMemoryError(
                    f"程序包设备 {mount} 的文件不可读：{fields['file_path']}: {e}"
                ) from e
            expected = dev.get("sha256")
            if expected is not None:
                if not isinstance(expected, str) or not expected:
                    raise VMMemoryError(
                        f"程序包设备 {mount} 的 sha256 必须是非空字符串"
                    )
                if expected.strip().lower() != actual:
                    raise VMMemoryError(
                        f"程序包设备 {mount} 的文件哈希不一致（{fields['file_path']}）："
                        f"期望 {expected}，实际 {actual}；拒绝安装"
                    )
            # 把这次安装看到的内容哈希交给挂载入口核对并记录（包内设备也进来源表）。
            dev = dict(dev, sha256=actual)
        device_mounts[mount] = dict(dev, path=mount)

    effects = []
    created = []      # (真实段, 是否由本函数创建)
    mounted = []
    current_rollback = None  # (包名节点, 旧 current 值)：失败时把版本指针还原
    try:
        # 2) 建子树（先于挂设备：设备挂载点链接要物化在已建好的版本节点里）
        packages_node, created_packages = _ensure_root(core, PACKAGES_ROOT, _PACKAGES_META)
        created.append(([PACKAGES_ROOT], created_packages))
        name_node, created_name = _ensure_child(core, packages_node, name,
                                                f"程序包 {name}")
        created.append(([PACKAGES_ROOT, name], created_name))
        version_node = MetaDict(
            data=parsed,
            metadata=meta.get("description") or f"程序包 {name} {package_version}",
        )
        name_node[package_version] = version_node
        created.append((target, True))
        if created_name:
            # 第一次装这个包：current 指针随包名节点一起建出来
            name_node["current"] = StrNode(package_version)
        effects.append({
            "kind": "memory_attach", "path": target,
            "node": Memory.serialize_node(version_node),
        })
        effects.append({
            "kind": "memory_attach", "path": [PACKAGES_ROOT, name],
            "node": Memory.serialize_node(name_node),
        })
        if created_packages:
            effects.append({
                "kind": "memory_attach", "path": [PACKAGES_ROOT],
                "node": Memory.serialize_node(packages_node),
            })
        if not created_name:
            current_rollback = (name_node, name_node.get("current"))
            _set_current(core, name, package_version, effects)

        # 3) 挂设备（此时版本节点已在树里，挂载点链接物化在包子树内）
        for mount, dev in device_mounts.items():
            mount_dev = dict(dev)
            if approved_at is not None:
                mount_dev["approved_at"] = approved_at
            if approved_via is not None:
                mount_dev["approved_via"] = approved_via
            try:
                spec = mount_device_entry(core, mount_dev, base)
            except ImageError as e:
                raise VMMemoryError(f"程序包设备装载失败：{e}") from e
            # 整包批准：approved_sha256 是程序包文件的哈希（批准整包 = 批准其中的设备）；
            # 设备文件自己的内容哈希在 spec["sha256"]。恢复/重放只用 sha256 核对。
            if approved_sha256 is not None:
                spec["approved_sha256"] = approved_sha256
            if approved_at is not None:
                spec["approved_at"] = approved_at
            if approved_via is not None:
                spec["approved_via"] = approved_via
            if (approved_sha256 is not None or approved_at is not None
                    or approved_via is not None):
                core.device_specs[mount] = spec
            mounted.append(mount)
            effects.append({"kind": "device_mount", "mount": mount, "spec": spec})

        # 4) 写安装记录
        record = _install_record(
            path, devices=effects, package_version=package_version,
            package_sha256=package_sha256, approved_sha256=approved_sha256,
            approved_at=approved_at, approved_via=approved_via,
        )
        record_node = _record_node(record)
        records_root, created_records = _ensure_root(core, RECORDS_ROOT, _RECORDS_META)
        created.append(([RECORDS_ROOT], created_records))
        records_name, created_records_name = _ensure_child(
            core, records_root, name, f"程序包 {name} 的安装记录")
        created.append(([RECORDS_ROOT, name], created_records_name))
        records_name[package_version] = record_node
        created.append((record_path(name, package_version), True))
        effects.append({
            "kind": "memory_attach",
            "path": record_path(name, package_version),
            "node": Memory.serialize_node(record_node),
        })
        if created_records:
            effects.append({
                "kind": "memory_attach", "path": [RECORDS_ROOT],
                "node": Memory.serialize_node(records_root),
            })
        record["package_path"] = "$MEM." + ".".join(target)
    except BaseException:
        _rollback(core, mounted, created, current_rollback)
        raise

    _commit_effects(core, effects)
    record["devices"] = sorted(mounted)
    return record


def _install_record(path: str, *, devices: list, package_version: str,
                    package_sha256: Optional[str] = None,
                    approved_sha256: Optional[str] = None,
                    approved_at: Optional[str] = None,
                    approved_via: Optional[str] = None) -> dict:
    source = os.path.abspath(path)
    if package_sha256 is None:
        with open(source, "rb") as f:
            package_sha256 = hashlib.sha256(f.read()).hexdigest()
    device_entries = []
    for effect in devices:
        if effect.get("kind") != "device_mount":
            continue
        spec = effect.get("spec") or {}
        entry = MetaDict(data={})
        entry["mount"] = StrNode(effect["mount"])
        entry["class"] = StrNode(str(spec.get("class") or ""))
        if spec.get("path"):
            entry["path"] = StrNode(str(spec["path"]))
        if spec.get("id"):
            entry["id"] = StrNode(str(spec["id"]))
        # 包内每个设备的文件内容哈希：重启恢复按它核对，不再信任"装载时看到的文件"。
        entry["sha256"] = StrNode(str(spec.get("sha256") or ""))
        for key in ("approved_sha256", "approved_at", "approved_via"):
            value = spec.get(key)
            if value:
                entry[key] = StrNode(str(value))
        if spec.get("args"):
            entry["args"] = StrNode(json.dumps(spec["args"], ensure_ascii=False))
        device_entries.append(entry)
    record = {
        "source": source,
        "sha256": package_sha256,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_version": package_version,
        "devices": device_entries,
    }
    # 整包批准：approved_sha256 绑定的是程序包文件；设备文件的哈希在各自条目里。
    if approved_sha256 is not None:
        record["approved_sha256"] = approved_sha256
    if approved_at is not None:
        record["approved_at"] = approved_at
    if approved_via is not None:
        record["approved_via"] = approved_via
    return record


def _record_node(record: dict) -> MetaDict:
    node = MetaDict(data={}, metadata="程序包安装记录")
    for key in ("source", "sha256", "installed_at", "package_version",
                "approved_sha256", "approved_at", "approved_via"):
        if key in record:
            node[key] = StrNode(str(record.get(key, "")))
    devices = MetaList(data=[], metadata="本包挂载的设备")
    for entry in record.get("devices") or []:
        devices.append(entry)
    node["devices"] = devices
    return node


def _ensure_root(core, key: str, metadata: str):
    """确保 $MEM.<key> 是 dict 节点；返回 (节点, 是否本次创建)。"""
    existing = core.mem._data.get(key)
    if existing is None:
        node = MetaDict(data={}, metadata=metadata)
        core.mem._data[key] = node
        return node, True
    if not isinstance(existing, MetaDict):
        raise VMMemoryError(f"$MEM.{key} 已存在但不是 dict 节点，无法安装程序包")
    return existing, False


def _ensure_child(core, parent: MetaDict, key: str, metadata: str):
    """确保 parent[key] 是 dict 节点；返回 (节点, 是否本次创建)。"""
    existing = parent.get(key)
    if existing is None:
        node = MetaDict(data={}, metadata=metadata)
        parent[key] = node
        return node, True
    if not isinstance(existing, MetaDict):
        raise VMMemoryError(f"程序包路径 $MEM...{key} 已存在但不是 dict 节点")
    return existing, False


def _set_current(core, name: str, package_version: str, effects: list) -> None:
    """把 $MEM.packages.<name>.current 写成版本字符串（一条 memory_set 效果，原子）。"""
    node = _node_at(core, [PACKAGES_ROOT, name])
    current = node.get("current")
    if current is not None and not isinstance(current, str):
        raise VMMemoryError(f"$MEM.packages.{name}.current 不是字符串节点，无法更新版本指针")
    node["current"] = StrNode(package_version)
    effects.append({
        "kind": "memory_set",
        "ref": f"$MEM.{PACKAGES_ROOT}.{name}.current",
        "value": package_version,
        "space": None,
    })


def _commit_effects(core, effects: list) -> None:
    """把安装/卸载效果提交给回合日志（无日志/非回合执行期是空操作）。"""
    core.note_effects(effects)


def _rollback(core, mounted: list, created: list, current_rollback=None) -> None:
    """安装失败回滚：先卸设备、再按创建逆序 detach 已建节点、还原版本指针。回滚不进日志。"""
    if current_rollback is not None:
        name_node, old_current = current_rollback
        try:
            if old_current is None:
                if "current" in name_node:
                    del name_node["current"]
            else:
                name_node["current"] = old_current
        except Exception:
            pass
    for mount in reversed(mounted):
        try:
            if mount in core.mem._devices:
                core.unmount_device(mount)
        except Exception as e:  # 回滚必须继续
            print(f"[AVM] 程序包回滚：卸载设备 {mount} 失败（{e}）", file=sys.stderr)
    for segments, was_created in reversed(created):
        if not was_created:
            continue
        try:
            ref = "$MEM." + ".".join(segments)
            core.mem.detach(ref)
        except Exception:
            pass


# ----------------------------------------------------------------------
# 卸载
# ----------------------------------------------------------------------

def uninstall(core, name: str, package_version: str) -> dict:
    """卸载程序包：卸设备 → detach 子树 → 删安装记录。失败抛 VMMemoryError。

    悬空引用（其他内存节点指向包内路径）不主动清理：访问得到既有的链接无效错误，
    清理留给以后的内存管理（计划 B1"不做什么"）。
    """
    check_new_key(name)
    check_new_key(package_version)
    target = package_path(name, package_version)
    if _node_at(core, target) is None:
        raise VMMemoryError(f"程序包未安装：{name} {package_version}")
    effects = []
    prefix = ".".join(target)
    for mount in sorted(core.device_specs):
        if mount == prefix or mount.startswith(prefix + "."):
            core.unmount_device(mount)
            effects.append({"kind": "device_unmount", "mount": mount})
    version_ref = "$MEM." + ".".join(target)
    core.mem.detach(version_ref)
    effects.append({"kind": "memory_detach", "path": list(target)})
    record_segments = record_path(name, package_version)
    if _node_at(core, record_segments) is not None:
        core.mem.detach("$MEM." + ".".join(record_segments))
        effects.append({"kind": "memory_detach", "path": list(record_segments)})
    _commit_effects(core, effects)
    return {
        "name": name,
        "package_version": package_version,
        "package_path": "$MEM." + prefix,
    }

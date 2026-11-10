"""程序包：运行期安装/卸载的"内存子树 + 设备"单元。

包格式（一个 JSON 文件）：

```json
{"meta": {"format": "avm.package", "version": 1,
          "name": "demo", "package_version": "1.0.0", "description": "..."},
 "mem": { ... 与镜像 mem 段同格式 ... },
 "devices": [{ "path": "io", "file": "io.py", "class": "EchoDevice", "args": {} }]}
```

安装落点：`$MEM.packages.<name>.<package_version>` 是包内存子树的根节点；
`$MEM.package_records.<name>.<package_version>` 是安装记录（来源路径、sha256、
安装时间、设备清单）。`$MEM.packages.<name>.current` 指向最近安装的版本字符串
（升级 = 装新版本后指针改指新版本，一条 `memory_write`，原子）。

设备条目的 `path` 是**相对包子树的挂载路径**（如 `io` → `$MEM.packages.<name>.<version>.io`）：
这样卸载 detach 整棵子树时挂载点链接一并回收，不会留下悬空链接。设备文件 `file`
相对包文件所在目录解析。

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

def install(core, path: str, *, base_dir: Optional[str] = None) -> dict:
    """安装程序包（挂设备 → 建子树 → 写记录 → 更新 current），返回安装记录。

    失败回滚已发生的设备挂载与内存节点，异常抛 VMMemoryError（对话得到 Error）。
    只有成功安装才把状态变化提交给回合日志（见 `_commit_effects`）。
    """
    from .image import ImageError, _parse_node, mount_device_entry

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
            try:
                spec = mount_device_entry(core, dev, base)
            except ImageError as e:
                raise VMMemoryError(f"程序包设备装载失败：{e}") from e
            mounted.append(mount)
            effects.append({"kind": "device_mount", "mount": mount, "spec": spec})

        # 4) 写安装记录
        record = _install_record(path, devices=effects, package_version=package_version)
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


def _install_record(path: str, *, devices: list, package_version: str) -> dict:
    source = os.path.abspath(path)
    with open(source, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
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
        if spec.get("args"):
            entry["args"] = StrNode(json.dumps(spec["args"], ensure_ascii=False))
        device_entries.append(entry)
    return {
        "source": source,
        "sha256": digest,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_version": package_version,
        "devices": device_entries,
    }


def _record_node(record: dict) -> MetaDict:
    node = MetaDict(data={}, metadata="程序包安装记录")
    for key in ("source", "sha256", "installed_at", "package_version"):
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

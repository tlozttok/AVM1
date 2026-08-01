"""AVM 系统镜像：一个 JSON + 若干设备 Python 文件 = 标准 AVM 系统镜像

用法: python -m avm.image <image.json>

镜像格式见 docs/system-image.md。镜像加载过程：
1. 解析 mem 段为显式节点树（kind/meta/ctrl/value），kind=device 为可读性标记（可出现在任意层级）；
2. 按 para_ref 校验 para 节点（ctrl.type == "para"）并配置 Core；
3. 按 devices 段加载设备插件文件（必须继承 MemoryDevice）并挂载；
4. 启动 init 对话（0 号，默认名 "init"）；
5. run() 执行；若镜像声明 persist_to，结束后写回内存。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

from .core import Core
from .exceptions import VMMemoryError
from .memory_device import MemoryDevice
from .types import MetaDict, MetaList

IMAGE_VERSION = 1
KINDS = ("str", "dict", "list", "device")


class ImageError(ValueError):
    """镜像格式错误：面向运行者，应修复镜像而不是吞掉"""


def _parse_node(value, path, markers):
    """解析单个显式节点。path 是从 mem 根到当前节点的点号路径（列表）。
    kind=device 为可读性标记：收集其点号路径到 markers，不写入数据树。"""
    if not isinstance(value, dict) or "kind" not in value:
        raise ImageError(f"{path}: mem 节点必须是显式对象（含 kind 字段），实际为 {type(value).__name__}")
    kind = value.get("kind")
    if kind not in KINDS:
        raise ImageError(f"{path}: 未知 kind {kind!r}，允许 {KINDS}")
    meta = value.get("meta")
    ctrl = value.get("ctrl")
    if ctrl is not None and not isinstance(ctrl, dict):
        raise ImageError(f"{path}: ctrl 必须是对象（dict[str,str]）")
    node_value = value.get("value")

    if kind == "str":
        if not isinstance(node_value, str):
            raise ImageError(f"{path}: str 节点的 value 必须是字符串")
        return node_value
    if kind == "dict":
        if not isinstance(node_value, dict):
            raise ImageError(f"{path}: dict 节点的 value 必须是对象")
        data = {}
        for k, v in node_value.items():
            parsed = _parse_node(v, path + [k], markers)
            if parsed is not None:
                data[k] = parsed
        return MetaDict(data=data, metadata=meta, ctrl=ctrl)
    if kind == "list":
        if not isinstance(node_value, list):
            raise ImageError(f"{path}: list 节点的 value 必须是数组")
        return MetaList(
            data=[_parse_node(v, path + [str(i)], markers) for i, v in enumerate(node_value)],
            metadata=meta, ctrl=ctrl,
        )
    # kind == "device"
    markers.append(".".join(path))
    return None


def _load_device_module(file_path: str, dev_path: str):
    name = "_avm_image_dev_" + hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImageError(f"设备 {dev_path}: 无法加载文件 {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mount_devices(core: Core, image_path: str, devices: list, mem_device_paths: list):
    base = os.path.dirname(os.path.abspath(image_path))
    mounted = set()
    for dev in devices:
        path = dev.get("path")
        if not isinstance(path, str) or not path:
            raise ImageError("devices 条目必须包含非空 path")
        file_rel = dev.get("file")
        class_name = dev.get("class")
        if not isinstance(file_rel, str) or not isinstance(class_name, str):
            raise ImageError(f"设备 {path}: 必须包含 file 和 class")
        file_path = file_rel if os.path.isabs(file_rel) else os.path.join(base, file_rel)
        if not os.path.isfile(file_path):
            raise ImageError(f"设备 {path}: 设备文件不存在 {file_path}")
        module = _load_device_module(file_path, path)
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type) or not issubclass(cls, MemoryDevice):
            raise ImageError(f"设备 {path}: {class_name} 必须是继承 MemoryDevice 的类")
        args = dev.get("args") or {}
        if not isinstance(args, dict):
            raise ImageError(f"设备 {path}: args 必须是对象")
        try:
            device = cls(**args)
        except TypeError as e:
            raise ImageError(f"设备 {path}: 实例化失败（args 与构造函数不匹配）: {e}")
        core.mem.mount(path, device)
        mounted.add(path)

    for mp in mem_device_paths:
        if mp not in mounted:
            raise ImageError(f"mem 中标记 kind=device 的路径 {mp} 未在 devices 段声明")


def _resolve_para(core: Core, para_ref: str) -> str:
    try:
        node = core.unwrap(para_ref, for_llm=False)
    except VMMemoryError as e:
        raise ImageError(f"para_ref {para_ref} 不存在: {e}")
    if not isinstance(node, MetaDict) or (node.get_ctrl() or {}).get("type") != "para":
        raise ImageError(f"para_ref {para_ref} 指向的节点必须是 ctrl.type='para' 的 dict 节点")
    return para_ref


def _start_init(core: Core, init: dict):
    name = init.get("name", "init")
    system_ref, user_ref = init.get("system_ref"), init.get("user_ref")
    system, user = init.get("system"), init.get("user")
    if (system is None) == (system_ref is None):
        raise ImageError("init 必须且只能提供 system 与 system_ref 之一")
    if (user is None) == (user_ref is None):
        raise ImageError("init 必须且只能提供 user 与 user_ref 之一")
    if system_ref is not None:
        if not isinstance(system_ref, str) or not system_ref.startswith("$"):
            raise ImageError("system_ref 必须是 $ 开头的内存引用（字面量请用 system 字段）")
        system = core.unwrap(system_ref)
    if user_ref is not None:
        if not isinstance(user_ref, str) or not user_ref.startswith("$"):
            raise ImageError("user_ref 必须是 $ 开头的内存引用（字面量请用 user 字段）")
        user = core.unwrap(user_ref)
    if not isinstance(system, str) or not isinstance(user, str):
        raise ImageError("init 的 system/user 最终必须是字符串（引用需指向 str 节点）")
    conv = core.start(system, user, core.para_ref)
    conv.name = name


def load_image(image_path: str) -> Core:
    """加载系统镜像并完成全部启动配置（不运行）。"""
    if not os.path.isfile(image_path):
        raise ImageError(f"镜像文件不存在: {image_path}")
    with open(image_path, "r", encoding="utf-8") as f:
        image = json.load(f)
    if not isinstance(image, dict):
        raise ImageError("镜像根节点必须是对象")
    meta = image.get("meta") or {}
    version = meta.get("version", 1)
    if version != IMAGE_VERSION:
        raise ImageError(f"不支持的镜像版本 {version}（当前支持 {IMAGE_VERSION}）")
    if not isinstance(image.get("mem"), dict):
        raise ImageError("镜像必须包含 mem 段（对象）")

    core = Core()
    device_markers = []
    root = MetaDict(data={})
    for key, node in image["mem"].items():
        parsed = _parse_node(node, [key], device_markers)
        if parsed is None:
            continue
        root[key] = parsed
    core.mem._data = root

    core.para_ref = _resolve_para(core, image.get("para_ref", "$MEM.model_params"))
    _mount_devices(core, image_path, image.get("devices") or [], device_markers)

    init = image.get("init")
    if init:
        if not isinstance(init, dict):
            raise ImageError("init 段必须是对象")
        _start_init(core, init)

    persist_to = image.get("persist_to")
    if persist_to is not None:
        if not isinstance(persist_to, str):
            raise ImageError("persist_to 必须是字符串路径")
        core.persist_to = persist_to if os.path.isabs(persist_to) else os.path.join(
            os.path.dirname(os.path.abspath(image_path)), persist_to)
    return core


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 系统镜像启动器")
    parser.add_argument("image", help="系统镜像 JSON 文件路径")
    args = parser.parse_args(argv)
    try:
        core = load_image(args.image)
    except (ImageError, VMMemoryError, OSError, json.JSONDecodeError) as e:
        print(f"[AVM] 镜像错误: {e}", file=sys.stderr)
        return 1
    try:
        core.run()
    finally:
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[AVM] 内存已写回 {persist_to}")
            except OSError as e:
                print(f"[AVM] 写回失败: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

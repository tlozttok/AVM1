"""AVM 系统镜像：一个 JSON + 若干设备 Python 文件 = 标准 AVM 系统镜像

用法: python -m avm.image <image.json>

镜像格式见 docs/system-image.md。镜像加载过程：
1. 解析 mem 段为显式节点树（kind/meta/ctrl/value），kind=device 为可读性标记（可出现在任意层级）；
2. 按 para_ref 校验 para 节点（ctrl.type == "para"）并配置 Core；
3. 按 devices 段加载设备插件文件（必须继承 MemoryDevice）并挂载；
4. 启动 init 对话（0 号，默认名 "init"；入口可以是 settingup LLM 程序，也可以是 python PLM 程序）；
5. run() 执行；若镜像声明 persist_to，结束后写回内存。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from typing import Optional

from .core import Core, _build_conversation
from .exceptions import VMMemoryError, RunnerStop
from .memory_device import MemoryDevice, BUILTIN_DEVICE_CLASSES
from .runner_prompt import prompt_api_error
from .types import MetaDict, MetaList, LinkNode

IMAGE_VERSION = 1
KINDS = ("str", "dict", "list", "device", "link")


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
    if kind == "link":
        # 链接节点：target 以 $ 开头是内存树地址，否则是设备标识符（内核侧声明，不开放指令创建）
        target = value.get("target")
        if not isinstance(target, str) or not target:
            raise ImageError(f"{path}: link 节点的 target 必须是非空字符串")
        return LinkNode(target=target, metadata=meta, ctrl=ctrl)
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
        if not isinstance(dev, dict):
            # devices 段是数组但条目不是对象（如 ["a"]）时，dev.get 抛 AttributeError
            raise ImageError(f"devices 条目必须是对象，得到 {type(dev).__name__}")
        path = dev.get("path")
        if not isinstance(path, str) or not path:
            raise ImageError("devices 条目必须包含非空 path")
        class_name = dev.get("class")
        args = dev.get("args") or {}
        if not isinstance(args, dict):
            raise ImageError(f"设备 {path}: args 必须是对象")
        builtin = bool(dev.get("builtin", False))
        if builtin:
            if not isinstance(class_name, str) or class_name not in BUILTIN_DEVICE_CLASSES:
                raise ImageError(
                    f"设备 {path}: builtin=true 时 class 必须是内建注册表中的类（{list(BUILTIN_DEVICE_CLASSES)}）"
                )
            if dev.get("file") is not None:
                raise ImageError(f"设备 {path}: builtin=true 不允许提供 file")
            cls = BUILTIN_DEVICE_CLASSES[class_name]
        else:
            if not isinstance(dev.get("file"), str) or not isinstance(class_name, str):
                raise ImageError(f"设备 {path}: 必须包含 file 和 class")
            file_rel = dev["file"]
            file_path = file_rel if os.path.isabs(file_rel) else os.path.join(base, file_rel)
            if not os.path.isfile(file_path):
                raise ImageError(f"设备 {path}: 设备文件不存在 {file_path}")
            module = _load_device_module(file_path, path)
            cls = getattr(module, class_name, None)
            if not isinstance(cls, type) or not issubclass(cls, MemoryDevice):
                raise ImageError(f"设备 {path}: {class_name} 必须是继承 MemoryDevice 的类")
        try:
            device = cls(**args)
        except TypeError as e:
            raise ImageError(f"设备 {path}: 实例化失败（args 与构造函数不匹配）: {e}")
        device_id = dev.get("id")
        if device_id is not None:
            if not isinstance(device_id, str) or not device_id:
                raise ImageError(f"设备 {path}: id 必须是非空字符串")
            device.device_id = device_id
        core.mem.mount(path, device)
        device.mount_path = path
        # 调试用：把 Core 引用交给设备（检查点 /save 等调试命令需要访问运行时状态）
        device.attach_core(core)
        mounted.add(path)

    for mp in mem_device_paths:
        if mp not in mounted:
            raise ImageError(f"mem 中标记 kind=device 的路径 {mp} 未在 devices 段声明")

    # 挂载完成后校验所有 LinkDevice 目标可达（面向运行者的诊断错误可含真实路径）。
    for device in core.mem._devices.values():
        if getattr(device, "is_link", False):
            try:
                device.validate()
            except Exception as e:
                raise ImageError(f"链接 {device.mount_path} 目标不可达: {e}")


def _resolve_para(core: Core, para_ref: str) -> str:
    try:
        node = core.unwrap(para_ref, for_llm=False)
    except VMMemoryError as e:
        raise ImageError(f"para_ref {para_ref} 不存在: {e}")
    if not isinstance(node, MetaDict) or (node.get_ctrl() or {}).get("type") != "para":
        raise ImageError(f"para_ref {para_ref} 指向的节点必须是 ctrl.type='para' 的 dict 节点")
    return para_ref


def _init_para_ref(core: Core, init: dict, node: Optional[MetaDict], node_type: Optional[str]) -> str:
    """init 的 para 引用：init.para_ref → 程序节点的 para_ref 字段 → 镜像顶层 para_ref。

    python 程序（PLM 入口）在三个来源都缺时报错：顶层 para 按约定是 LLM para，
    拿它建 Python 对话只会在模型命名空间校验上撞墙，这里早报更清楚。
    """
    for where, value in (
        ("init.para_ref", init.get("para_ref")),
        ("程序节点的 para_ref", node.get("para_ref") if isinstance(node, MetaDict) else None),
    ):
        if value is None:
            continue
        if not isinstance(value, str) or not value.startswith("$"):
            raise ImageError(f"{where} 必须是 $ 开头的内存引用，得到 {value!r}")
        return value
    if node_type == "python":
        raise ImageError(
            "init 指向 python 程序节点时必须有 para 引用：在 init 段写 para_ref，"
            "或在程序节点里写 para_ref（顶层 para_ref 是 LLM 参数，不能建 Python 对话）"
        )
    return core.para_ref


def _start_init(core: Core, init: dict):
    name = init.get("name", "init")
    system_ref, user_ref = init.get("system_ref"), init.get("user_ref")
    system, user = init.get("system"), init.get("user")
    if (system is None) == (system_ref is None):
        raise ImageError("init 必须且只能提供 system 与 system_ref 之一")
    if (user is None) == (user_ref is None):
        raise ImageError("init 必须且只能提供 user 与 user_ref 之一")
    node: Optional[MetaDict] = None
    node_type: Optional[str] = None
    if system_ref is not None:
        if not isinstance(system_ref, str) or not system_ref.startswith("$"):
            raise ImageError("system_ref 必须是 $ 开头的内存引用（字面量请用 system 字段）")
        node = core.unwrap(system_ref, for_llm=False)
        node_type = (node.get_ctrl() or {}).get("type") if isinstance(node, MetaDict) else None
        if node_type not in ("settingup", "python"):
            raise ImageError(
                f"system_ref {system_ref} 必须指向 ctrl.type='settingup'（LLM 程序）或 "
                "'python'（Python 程序）的 dict 节点（str 节点、无类型 dict 节点不接受）"
            )
        content = node.get("content")
        if not isinstance(content, str):
            raise ImageError(f"程序节点 {system_ref} 必须有字符串 content 字段")
        system = content
    if user_ref is not None:
        if not isinstance(user_ref, str) or not user_ref.startswith("$"):
            raise ImageError("user_ref 必须是 $ 开头的内存引用（字面量请用 user 字段）")
        user = core.unwrap(user_ref, for_llm=False)
    if not isinstance(system, str) or not isinstance(user, str):
        raise ImageError(
            "init 的 system/user 必须是字符串（system 字面量或程序节点的 content；"
            "user_ref 需指向 str 节点）"
        )
    para_ref = _init_para_ref(core, init, node, node_type)
    if node_type == "python":
        # PLM 入口：0 号对话是 Python 对话。分派与模型命名空间校验复用 _build_conversation，
        # 它建好 SystemMessage（程序代码）、注册对话、设 para_ref 与 is_python；
        # 这里补用户消息与"0 号 + 就绪"语义（与 Core.start 对齐，区别只在执行器是 PLM）。
        conv = _build_conversation(core, system_ref, para_ref, parent=None, is_sub=False)
        conv.append_user_message(user)
        conv.is_root = True
        conv.name = name
        core._ready_cids.append(conv.cid)
        return
    conv = core.start(system, user, para_ref)
    conv.name = name


def load_image(core: Core, image_path: str, checkpoint_in: Optional[str] = None,
               host_flags: Optional[dict] = None) -> Core:
    """把系统镜像加载进一个空核心并完成全部装载配置（不运行）。

    生命周期三段里的第二段：创建空核心（Core()）→ 加载镜像（本函数）→ 启动核心
    （avm/boot.py 的 start_core；宿主通常直接用加载区的 boot() 走完三段）。

    checkpoint_in：调试用——启动时加载单对话检查点（跳过 init，恢复为就绪对话）。
    host_flags：启动标志——本次运行由哪个宿主托管（如 {"console": True}）。宿主事实，
    不属于镜像内容，不进 $MEM。在挂载设备之前写入 Core，设备可在 attach_core 里读；
    核心启动后不再变化。不传则空，设备行为与从前一致。
    """
    if core._conv_by_cid:
        raise ImageError(
            f"load_image 需要一个空核心（当前已有 {len(core._conv_by_cid)} 个对话）："
            f"先用 Core() 建空核心，再往里装载"
        )
    if not os.path.isfile(image_path):
        raise ImageError(f"镜像文件不存在: {image_path}")
    with open(image_path, "r", encoding="utf-8") as f:
        image = json.load(f)
    if not isinstance(image, dict):
        raise ImageError("镜像根节点必须是对象")
    meta = image.get("meta")
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        # `or {}` 会放过真值非对象（如 1、"x"、[1]），随后 meta.get 抛 AttributeError
        raise ImageError(f"meta 必须是对象，得到 {type(meta).__name__}")
    version = meta.get("version", 1)
    if version != IMAGE_VERSION:
        raise ImageError(f"不支持的镜像版本 {version}（当前支持 {IMAGE_VERSION}）")
    if not isinstance(image.get("mem"), dict):
        raise ImageError("镜像必须包含 mem 段（对象）")

    if host_flags is not None and not isinstance(host_flags, dict):
        raise ImageError("host_flags 必须是对象")
    for key in (host_flags or {}):
        if not isinstance(key, str) or not key:
            raise ImageError("host_flags 的键必须是非空字符串")
    # 启动标志：在挂载设备之前就位，这样每个设备的 attach_core 都读得到。
    core.host_flags = dict(host_flags or {})
    core.image_dir = os.path.dirname(os.path.abspath(image_path))
    core.image_name = os.path.splitext(os.path.basename(image_path))[0]
    budget = meta.get("instruction_budget")
    if isinstance(budget, int) and budget > 0:
        core._instruction_budget = budget
    device_markers = []
    root = MetaDict(data={})
    for key, node in image["mem"].items():
        parsed = _parse_node(node, [key], device_markers)
        if parsed is None:
            continue
        root[key] = parsed
    core.mem._data = root
    # Core 的内建设备在 __init__ 里挂载，早于这次整树替换；按设备表把挂载点链接重新物化
    core.mem.materialize_mount_links()
    # 启动自带的模型目录同样会被整树替换冲掉，这里重装（幂等，镜像已有的字段不覆盖）
    core.install_model_catalog()

    para_ref = image.get("para_ref", "$MEM.model_params")
    if not isinstance(para_ref, str) or not para_ref:
        # 显式 null / 数字 / 空串都会让 core.unwrap 抛 AttributeError（不是 ImageError）
        raise ImageError(f"para_ref 必须是非空字符串，得到 {para_ref!r}")
    core.para_ref = _resolve_para(core, para_ref)
    _mount_devices(core, image_path, image.get("devices") or [], device_markers)

    init = image.get("init")
    if checkpoint_in:
        # 调试用：后续启动加载已保存的对话，而不是重新从 init 开始
        if not isinstance(checkpoint_in, str) or not checkpoint_in:
            raise ImageError("checkpoint_in 必须是文件路径")
        try:
            core.restore_conversation(checkpoint_in, schedule="ready")
        except OSError as e:
            raise ImageError(f"检查点不存在或不可读: {checkpoint_in}: {e}")
    elif init:
        if not isinstance(init, dict):
            raise ImageError("init 段必须是对象")
        _start_init(core, init)
    else:
        # 静默的"空镜像"：没有对话会被启动，run() 立即返回、退出码 0、persist_to 照写
        print(
            "[AVM] 警告：镜像没有 init 段（或为空），不会有对话启动，run() 将立即返回",
            file=sys.stderr,
        )

    persist_to = image.get("persist_to")
    if persist_to is not None:
        if not isinstance(persist_to, str):
            raise ImageError("persist_to 必须是字符串路径")
        core.persist_to = persist_to if os.path.isabs(persist_to) else os.path.join(
            os.path.dirname(os.path.abspath(image_path)), persist_to)
    # 内核转储（诊断快照，不是恢复输入）：core_dump 开关、core_dump_to 落点，与 persist_to 同为顶层字段
    core_dump = image.get("core_dump")
    if core_dump is not None and not isinstance(core_dump, bool):
        raise ImageError("core_dump 必须是布尔值")
    core.core_dump_enabled = True if core_dump is None else core_dump
    core_dump_to = image.get("core_dump_to")
    if core_dump_to is not None:
        if not isinstance(core_dump_to, str) or not core_dump_to:
            raise ImageError("core_dump_to 必须是非空字符串路径")
        core.core_dump_to = core_dump_to if os.path.isabs(core_dump_to) else os.path.join(
            os.path.dirname(os.path.abspath(image_path)), core_dump_to)
    return core


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 系统镜像启动器")
    parser.add_argument("image", help="系统镜像 JSON 文件路径")
    parser.add_argument("--checkpoint-in", metavar="PATH",
                        help="调试用：启动时加载单对话检查点（跳过 init，恢复为就绪对话）")
    from .boot import add_core_dump_args
    add_core_dump_args(parser)
    args = parser.parse_args(argv)
    # 局部导入：boot 依赖本模块的 load_image，模块级导入会成环
    from .boot import boot, dump_on_death
    try:
        core = boot(args.image, checkpoint_in=args.checkpoint_in)
    except (ImageError, VMMemoryError, OSError, json.JSONDecodeError) as e:
        print(f"[AVM] 镜像错误: {e}", file=sys.stderr)
        return 1
    if args.no_core_dump:
        core.core_dump_enabled = False
    status = 0
    try:
        core.run()
    except RunnerStop as e:
        print(f"[AVM] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[AVM] 收到中断，写回后退出", file=sys.stderr)
        status = 130
    except BaseException as e:
        # 内核死亡：先落现场快照，再保留原始 traceback 往上抛
        dump_on_death(core, e, args.core_dump)
        raise
    finally:
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[AVM] 内存已写回 {persist_to}")
            except OSError as e:
                print(f"[AVM] 写回失败: {e}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())

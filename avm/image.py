"""AVM 系统镜像：一个 JSON + 若干设备/PLM 的 Python 文件 = 标准 AVM 系统镜像

用法: python -m avm.image <image.json>

镜像格式见 docs/system-image.md。镜像加载过程：
1. 解析 mem 段为显式节点树（kind/meta/ctrl/value），kind=device 为可读性标记（可出现在任意层级）；
2. 装载 plms 段的自定义 PLM（必须继承 PLMServer）进本核的 Core.plms；
3. 按 para_ref 校验 para 节点（ctrl.type == "para"）并配置 Core；
4. 按 devices 段加载设备插件文件（必须继承 MemoryDevice）并挂载；
5. 启动 init 对话（0 号，默认名 "init"；入口可以是 settingup LLM 程序，也可以是 python PLM 程序）；
   重启恢复路径（restore=True）跳过这一步，对话与内存随后由快照/日志接管；
6. run() 执行；若镜像声明 persist_to 写回内存、journal_to 逐回合记日志、
   snapshot_to 在正常退出时写重启快照。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from typing import Optional

from .core import Core, _build_conversation, _derive_child_caps
from .capabilities import grant_target_segments
from .exceptions import IllegalOperationError, VMMemoryError, RunnerStop
from .memory_device import MemoryDevice, BUILTIN_DEVICE_CLASSES
from .python_server import PLM_REGISTRY, PLMServer
from .runner_prompt import prompt_api_error
from .types import MetaDict, MetaList, LinkNode, StrNode

IMAGE_VERSION = 1
KINDS = ("str", "dict", "list", "device", "link")
# 镜像 meta.guest_programs 允许的角色：内核按这个固定顺序在冷启动时创建对话。
# 宿主用中断唤醒这些对话（向量名 = 角色名），不代替它们直接 send_instruction。
GUEST_ROLES = ("boot", "idle", "shutdown")


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
    if ctrl is not None and "caps" in ctrl:
        raise ImageError(
            f"{path}: 能力声明必须放在 value.caps 数据节点里，ctrl.caps 不再支持"
        )
    node_value = value.get("value")

    if kind == "str":
        if not isinstance(node_value, str):
            raise ImageError(f"{path}: str 节点的 value 必须是字符串")
        if meta is not None:
            # 字符串只有控制数据，没有描述数据；描述放到父容器或删除
            raise ImageError(f"{path}: str 节点没有描述数据（meta）；请改用 ctrl 或把描述放到父容器")
        return StrNode(node_value, ctrl=ctrl)
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


def _load_image_module(file_path: str, kind: str, subject: str):
    """按文件路径加载镜像自带的 Python 模块（设备与 PLM 共用）。

    kind 进模块名前缀（dev / plm），避免同一文件扮演两种角色时模块名相撞；
    subject 是报错时的对象名（如 "设备 inputs"、"PLM plm.community"）。
    """
    name = f"_avm_image_{kind}_" + hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImageError(f"{subject}: 无法加载文件 {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_device_module(file_path: str, dev_path: str):
    return _load_image_module(file_path, "dev", f"设备 {dev_path}")


def _load_plms(core: Core, image_path: str, plms: list):
    """装载镜像的 plms 段：把 {model, file, class} 声明的类装进本核的 core.plms。

    形状与 devices 段对称（file 相对镜像目录，class 必须是 PLMServer 的子类）。
    区别：PLM 是按名构造的类，装载期不实例化，也不进内存树。
    必须在 install_model_catalog 之前调用——模型目录是 PLM 注册表的可见面。
    """
    if not isinstance(plms, list):
        raise ImageError(f"plms 段必须是数组，得到 {type(plms).__name__}")
    base = os.path.dirname(os.path.abspath(image_path))
    modules = {}  # 同一个文件被引用多次时只加载一次
    seen = set()
    for item in plms:
        if not isinstance(item, dict):
            raise ImageError(f"plms 条目必须是对象，得到 {type(item).__name__}")
        model = item.get("model")
        file_rel = item.get("file")
        class_name = item.get("class")
        if not isinstance(model, str) or not model:
            raise ImageError("plms 条目必须包含非空 model")
        if not isinstance(file_rel, str) or not file_rel:
            raise ImageError(f"PLM {model}: 必须包含非空 file")
        if not isinstance(class_name, str) or not class_name:
            raise ImageError(f"PLM {model}: 必须包含非空 class")
        if model in PLM_REGISTRY:
            raise ImageError(f"PLM {model}: 与内建 PLM 重名，镜像不能覆盖宿主注册表")
        if model in seen:
            raise ImageError(f"PLM {model}: 同一个 plms 段里重复声明")
        seen.add(model)
        file_path = file_rel if os.path.isabs(file_rel) else os.path.join(base, file_rel)
        if not os.path.isfile(file_path):
            raise ImageError(f"PLM {model}: 文件不存在 {file_path}")
        module = modules.get(file_path)
        if module is None:
            module = _load_image_module(file_path, "plm", f"PLM {model}")
            modules[file_path] = module
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type) or not issubclass(cls, PLMServer):
            raise ImageError(f"PLM {model}: {class_name} 必须是继承 PLMServer 的类")
        core.plms[model] = cls


def _sha256_file(path: str) -> str:
    """流式计算文件内容哈希（设备文件可能不小，不整块读进内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_entry_fields(dev: dict, base: Optional[str], subject: str = "设备"):
    """解析设备条目的形状并解析文件路径，不 import、不读文件。

    返回字段字典：path/subject/class/args/builtin/file_path/device_id。
    镜像 devices 段、程序包 devices 段、日志重放与 loader 热装共用。
    """
    if not isinstance(dev, dict):
        raise ImageError(f"{subject}条目必须是对象，得到 {type(dev).__name__}")
    path = dev.get("path")
    if not isinstance(path, str) or not path:
        raise ImageError(f"{subject}条目必须包含非空 path")
    subject = f"设备 {path}"
    class_name = dev.get("class")
    args = dev.get("args") or {}
    if not isinstance(args, dict):
        raise ImageError(f"{subject}: args 必须是对象")
    builtin = bool(dev.get("builtin", False))
    file_path = None
    if builtin:
        if not isinstance(class_name, str) or class_name not in BUILTIN_DEVICE_CLASSES:
            raise ImageError(
                f"{subject}: builtin=true 时 class 必须是内建注册表中的类（{list(BUILTIN_DEVICE_CLASSES)}）"
            )
        if dev.get("file") is not None:
            raise ImageError(f"{subject}: builtin=true 不允许提供 file")
    else:
        if not isinstance(dev.get("file"), str) or not isinstance(class_name, str):
            raise ImageError(f"{subject}: 必须包含 file 和 class")
        file_rel = dev["file"]
        file_path = file_rel if os.path.isabs(file_rel) else os.path.join(base or os.getcwd(), file_rel)
        if not os.path.isfile(file_path):
            raise ImageError(f"{subject}: 设备文件不存在 {file_path}")
    return {
        "path": path,
        "subject": subject,
        "class": class_name,
        "args": args,
        "builtin": builtin,
        "file_path": file_path,
        "device_id": dev.get("id"),
    }


def _verify_device_sha256(fields: dict, dev: dict) -> Optional[str]:
    """装载前核对设备文件哈希；返回实际哈希（内建返回 None）。

    这是哈希校验的唯一落点（设备审批计划的 B 段），四条装载路径都经过
    `mount_device_entry`。规则：

    - 带 `approved_sha256`（loader 热装请求）时，文件哈希必须与批准值相等；
    - 不带批准值但 spec 里有 `sha256`（快照恢复 / 日志重放 / 镜像重载）时，
      必须与记录值相等；
    - 两者都没有（镜像首次冷装载）时，计算并记录实际哈希。

    不相等抛 ImageError（面向运行者），调用方不会 import、不会挂载。
    """
    file_path = fields["file_path"]
    if file_path is None:
        return None
    actual = _sha256_file(file_path)
    subject = fields["subject"]
    for label, key in (("批准哈希", "approved_sha256"), ("记录哈希", "sha256")):
        expected = dev.get(key)
        if expected is None:
            continue
        if not isinstance(expected, str) or not expected:
            raise ImageError(f"{subject}: {label}必须是非空字符串，得到 {expected!r}")
        if expected.strip().lower() != actual:
            raise ImageError(
                f"{subject}: 文件哈希与{label}不一致（{file_path}）："
                f"期望 {expected}，实际 {actual}；拒绝装载"
            )
    return actual


def resolve_device_entry(dev: dict, base: Optional[str], subject: str = "设备",
                         *, file_sha256: Optional[str] = None):
    """解析一条 devices 条目（镜像 devices 段或程序包 devices 段）→ (cls, args, id, spec)。

    纯解析 + 加载设备模块，不挂载。spec 是设备来源表条目（快照/日志重放用）：
    绝对路径 + 类名 + 构造参数 + 标识符 + 是否内建；文件型设备带内容哈希 sha256。
    """
    fields = _device_entry_fields(dev, base, subject)
    subject = fields["subject"]
    path = fields["path"]
    class_name = fields["class"]
    args = fields["args"]
    file_path = fields["file_path"]
    if fields["builtin"]:
        cls = BUILTIN_DEVICE_CLASSES[class_name]
    else:
        module = _load_device_module(file_path, path)
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type) or not issubclass(cls, MemoryDevice):
            raise ImageError(f"{subject}: {class_name} 必须是继承 MemoryDevice 的类")
        if file_sha256 is None:
            file_sha256 = _sha256_file(file_path)
    spec = {
        "path": file_path,
        "file": dev.get("file"),
        "class": class_name,
        "args": args,
        "id": dev.get("id"),
        "builtin": fields["builtin"],
    }
    if file_path is not None:
        spec["sha256"] = file_sha256
    # 审批记录只在 loader 装载时随请求带上；镜像冷装载与日志重放都不带这些字段。
    for key in ("approved_sha256", "approved_at", "approved_via"):
        value = dev.get(key)
        if value is not None:
            spec[key] = value
    return cls, args, dev.get("id"), spec


def instantiate_device(cls, args: dict, device_id, subject: str = "设备"):
    """实例化一台设备并设置标识符（args 与构造函数不匹配是镜像/包格式错误）。"""
    try:
        device = cls(**args)
    except TypeError as e:
        raise ImageError(f"{subject}: 实例化失败（args 与构造函数不匹配）: {e}")
    if device_id is not None:
        if not isinstance(device_id, str) or not device_id:
            raise ImageError(f"{subject}: id 必须是非空字符串")
        device.device_id = device_id
    return device


def mount_device_entry(core: Core, dev: dict, base: Optional[str] = None) -> dict:
    """装载一台设备：校验文件哈希 → 解析 → 实例化 → 挂载 → 登记设备来源表。

    哈希校验只在这里发生（设备审批计划的唯一落点）：文件型设备在 import 之前
    重算 sha256，与请求的 approved_sha256 或 spec 的 sha256 比对；不相等拒绝装载。
    挂载是纯内存操作（Memory.mount 只改设备表与挂载点链接节点，不发事件），
    所以在指令执行中途调用是安全的；程序包装载、日志重放与镜像装载共用这一条路径。
    """
    fields = _device_entry_fields(dev, base)
    actual_sha256 = _verify_device_sha256(fields, dev)
    cls, args, device_id, spec = resolve_device_entry(
        dev, base, file_sha256=actual_sha256,
    )
    path = fields["path"]
    device = instantiate_device(cls, args, device_id, subject=fields["subject"])
    core.mem.mount(path, device)
    device.mount_path = path
    # 调试用：把 Core 引用交给设备（检查点 /save 等调试命令需要访问运行时状态）
    device.attach_core(core)
    core.device_specs[path] = spec
    return spec


def _mount_devices(core: Core, image_path: str, devices: list, mem_device_paths: list):
    base = os.path.dirname(os.path.abspath(image_path))
    mounted = set()
    for dev in devices:
        mount_device_entry(core, dev, base)
        if isinstance(dev, dict) and isinstance(dev.get("path"), str):
            mounted.add(dev["path"])

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


_SECURITY_KEYS = frozenset({"roots"})


def _parse_security(core: Core, image: dict) -> None:
    """解析并安装镜像顶层 security 段；缺省时保持空注册表。

    工作目录不再由内核分配：security.workspace_region 已移除，目录由软件用
    memory_make 建，根由 `$MEM.conversations.<cid>.roots` 动态授予。
    """
    security = image.get("security")
    if security is None:
        return
    if not isinstance(security, dict):
        raise ImageError("security 必须是对象")
    if "workspace_region" in security:
        raise ImageError(
            "security.workspace_region 已移除：内核不再分配工作空间。"
            "改用软件建目录（memory_make）后写 $MEM.conversations.<cid>.roots 动态加根"
        )
    unknown = sorted(set(security) - _SECURITY_KEYS)
    if unknown:
        raise ImageError(f"security 含未知字段：{unknown}（允许：{sorted(_SECURITY_KEYS)}）")
    roots_raw = security.get("roots") or {}
    if not isinstance(roots_raw, dict):
        raise ImageError("security.roots 必须是对象")
    roots_spec = {}
    for name, ref in roots_raw.items():
        try:
            if not isinstance(name, str) or not name:
                raise IllegalOperationError("根名必须是非空字符串")
            roots_spec[name] = grant_target_segments(ref)
        except (IllegalOperationError, VMMemoryError) as e:
            raise ImageError(f"security.roots.{name}: {e}")
    try:
        core.install_security_config(roots_spec)
    except (IllegalOperationError, VMMemoryError) as e:
        raise ImageError(f"security 段非法：{e}")


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
        if node is not None:
            try:
                _derive_child_caps(core, None, node, is_sub=False, is_init=True)
            except (VMMemoryError, IllegalOperationError) as e:
                raise ImageError(f"init 程序的能力声明非法：{e}")
        conv = _build_conversation(core, system_ref, para_ref, parent=None, is_sub=False, is_init=True)
        conv.append_user_message(user)
        conv.is_root = True
        conv.name = name
        core._ready_cids.append(conv.cid)
        return
    conv = core.start(system, user, para_ref)
    if node is not None:
        # LLM 入口同样按程序 ctrl.caps 计算 init 能力；init 强制内核态
        try:
            caps = _derive_child_caps(core, None, node, is_sub=False, is_init=True)
        except (VMMemoryError, IllegalOperationError) as e:
            raise ImageError(f"init 程序的能力声明非法：{e}")
        conv.caps = caps
    conv.name = name


def _parse_guest_programs(raw) -> dict:
    """解析镜像 meta.guest_programs（角色 → 程序节点引用）；形状错误面向镜像作者。

    角色名限定 `boot`/`idle`/`shutdown`（宿主的中断向量名就是角色名）；
    引用必须是以 `$` 开头的内存引用，指向 ctrl.type 为 python/settingup 的程序节点。
    节点内容与 para 在真正创建对话时才校验（此时内存树/恢复世界已经就位）。
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ImageError(
            f"meta.guest_programs 必须是对象（角色 → 程序节点引用），得到 {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - set(GUEST_ROLES))
    if unknown:
        raise ImageError(
            f"meta.guest_programs 含未知角色 {unknown}（允许：{list(GUEST_ROLES)}）"
        )
    decl = {}
    for role in GUEST_ROLES:
        ref = raw.get(role)
        if ref is None:
            continue
        if not isinstance(ref, str) or not ref.startswith("$"):
            raise ImageError(
                f"meta.guest_programs.{role} 必须是 $ 开头的内存引用，得到 {ref!r}"
            )
        decl[role] = ref
    return decl


def _guest_entry(role: str, node_type: str) -> str:
    """guest 角色的入口正文：中断 payload 投进对话时程序据此开始本轮。

    - python 程序：按约定定义与角色同名的生成器函数，入口是 `yield from <角色>()`
      （与镜像 init 段写 user="yield from boot()" 是同一写法）；
    - settingup（LLM）程序：角色名就是通知正文，由提示词解释该做什么。
    """
    if node_type == "python":
        return f"yield from {role}()"
    return role


def _guest_para_ref(core: Core, ref: str, node: MetaDict, node_type: str, role: str) -> str:
    """guest 程序的 para 引用：程序节点的 para_ref → 镜像顶层 para_ref（仅 LLM 程序）。

    python 程序必须显式给 para_ref（顶层 para 按约定是 LLM para，拿它建 Python
    对话只会在模型命名空间校验上撞墙）；与 init 的解析规则保持一致。
    """
    value = node.get("para_ref")
    if value is not None:
        if not isinstance(value, str) or not value.startswith("$"):
            raise ImageError(
                f"guest 程序 {ref}（角色 {role}）的 para_ref 必须是 $ 开头的内存引用"
            )
        return value
    if node_type == "python":
        raise ImageError(
            f"guest 程序 {ref}（角色 {role}）是 python 程序时必须在程序节点写 para_ref"
            "（para 的 model 必须是 PLM 注册名）"
        )
    return core.para_ref


def _guest_node(core: Core, ref: str, role: str):
    """解析并按程序节点校验 guest 引用，返回 (node, node_type)。"""
    try:
        node = core.unwrap(ref, for_llm=False)
    except VMMemoryError as e:
        raise ImageError(f"meta.guest_programs.{role} 指向的节点不存在：{ref}: {e}")
    node_type = (node.get_ctrl() or {}).get("type") if isinstance(node, MetaDict) else None
    if node_type not in ("settingup", "python"):
        raise ImageError(
            f"meta.guest_programs.{role} = {ref} 必须是程序节点"
            "（ctrl.type='settingup' 或 'python' 的 dict 节点）"
        )
    if not isinstance(node.get("content"), str):
        raise ImageError(f"guest 程序节点 {ref} 必须有字符串 content 字段")
    return node, node_type


def start_guest_programs(core: Core, decl: Optional[dict] = None) -> dict:
    """冷启动：按 meta.guest_programs 为每个角色创建内核态对话并注册中断向量。

    - 按 GUEST_ROLES 的固定顺序创建（cid 按创建顺序分配）；
    - `boot` 立刻进就绪队列并预置入口正文（上电后运行一次）；
      `idle`/`shutdown` 进休眠队列，只由宿主中断唤醒；
    - 对话 attrs["guest_role"] 记录角色，resume 靠它找回（镜像层权限策略不写内存树）。
    返回角色 → {"cid", "entry", "kind"}，同时写入 `core.guest_roles`。
    """
    roles = dict(decl if decl is not None else getattr(core, "guest_programs_decl", {}) or {})
    created = {}
    for role in GUEST_ROLES:
        ref = roles.get(role)
        if ref is None:
            continue
        node, node_type = _guest_node(core, ref, role)
        para_ref = _guest_para_ref(core, ref, node, node_type, role)
        try:
            conv = _build_conversation(
                core, ref, para_ref, parent=None, is_sub=False, is_init=True,
            )
        except (VMMemoryError, IllegalOperationError) as e:
            raise ImageError(f"guest 程序 {ref}（角色 {role}）创建失败：{e}")
        conv.name = f"guest-{role}"
        conv.is_root = True
        conv.attrs["guest_role"] = role
        entry = _guest_entry(role, node_type)
        if role == "boot":
            conv.append_user_message(entry)
            core._ready_cids.append(conv.cid)
        else:
            core._dormant_cids.append(conv.cid)
        core.interrupts.register(role, conv.cid)
        created[role] = {"cid": conv.cid, "entry": entry, "kind": node_type}
    core.guest_roles = created
    return created


def restore_guest_roles(core: Core, decl: Optional[dict] = None) -> dict:
    """resume：按 attrs["guest_role"] 找回快照里的 guest 对话并注册中断向量。

    缺失声明过的角色时拒绝启动（不做补建）：补建会分配新 cid，与已提交日志里的
    旧 cid 错位，后续重放必然对不上。快照里有、但镜像已不再声明的角色只告警，
    对话按普通对话保留（不再注册向量）。
    """
    roles = dict(decl if decl is not None else getattr(core, "guest_programs_decl", {}) or {})
    found = {}
    for conv in core._conv_by_cid.values():
        role = (conv.attrs or {}).get("guest_role")
        if not isinstance(role, str) or not role:
            continue
        if role not in GUEST_ROLES:
            raise ImageError(
                f"快照里对话 cid={conv.cid} 的 attrs['guest_role']={role!r} "
                f"不是已声明的角色（允许：{list(GUEST_ROLES)}）"
            )
        if role in found:
            raise ImageError(
                f"快照里 guest 角色 {role!r} 对应多个对话"
                f"（cid={found[role].cid} 与 cid={conv.cid}），拒绝启动"
            )
        found[role] = conv
    created = {}
    for role in GUEST_ROLES:
        ref = roles.get(role)
        conv = found.get(role)
        if ref is None:
            if conv is not None:
                print(
                    f"[AVM] 镜像不再声明 guest 角色 {role!r}：快照里的对话 cid={conv.cid} "
                    "按普通对话保留，不注册中断向量",
                    file=sys.stderr,
                )
            continue
        if conv is None:
            raise ImageError(
                f"resume 需要 guest 程序对话 {role!r}，但快照里没有 attrs['guest_role']="
                f"{role!r} 的对话；不允许在 resume 时补建（新 cid 会让后续日志错位）："
                "请用 boot(mode='reboot') 或 boot(mode='reset') 重建（会重跑 init 与 guest 程序）"
            )
        _node, node_type = _guest_node(core, ref, role)
        if conv.is_python != (node_type == "python"):
            raise ImageError(
                f"guest 角色 {role!r} 的快照对话 cid={conv.cid} 程序类型与镜像声明不一致："
                f"对话 is_python={conv.is_python}，声明节点是 {node_type}"
            )
        entry = _guest_entry(role, node_type)
        core.interrupts.register(role, conv.cid)
        created[role] = {"cid": conv.cid, "entry": entry, "kind": node_type}
    core.guest_roles = created
    return created


def start_cold(core: Core) -> dict:
    """冷启动（load_image(restore=True) 之后）：补跑 init，再按声明创建 guest 程序。

    reboot/reset 用：世界先由快照/原始镜像就位，再让 init 对话与 guest 程序对话
    出现在新世界里（cid 从 Core._next_cid 起按创建顺序分配）。
    """
    init = getattr(core, "_image_init", None)
    if init is not None:
        if not isinstance(init, dict):
            raise ImageError("init 段必须是对象")
        _start_init(core, init)
    else:
        print(
            "[AVM] 警告：镜像没有 init 段（或为空），冷启动不会有 init 对话",
            file=sys.stderr,
        )
    return start_guest_programs(core)


def load_image(core: Core, image_path: str, checkpoint_in: Optional[str] = None,
               host_flags: Optional[dict] = None, restore: bool = False) -> Core:
    """把系统镜像加载进一个空核心并完成全部装载配置（不运行）。

    生命周期三段里的第二段：创建空核心（Core()）→ 加载镜像（本函数）→ 启动核心
    （avm/boot.py 的 start_core；宿主通常直接用加载区的 boot() 走完三段）。

    checkpoint_in：调试用——启动时加载单对话检查点（跳过 init，恢复为就绪对话）。
    restore：重启恢复的内部细节——跳过 init（对话与内存稍后由重启快照/日志恢复），
    并把 init 段与 guest 程序声明暂存在 Core 上，供冷启动模式稍后补跑（`start_cold`）。
    镜像其余段（设备、安全配置、para）照常装载，因为恢复路径也要按当前镜像/设备文件重建。
    host_flags：启动标志——本次运行由哪个宿主托管（如 {"console": True}）。宿主事实，
    不属于镜像内容，不进 $MEM。在挂载设备之前写入 Core，设备可在 attach_core 里读；
    核心启动后不再变化。不传则空，设备行为与从前一致。

    镜像的 plms 段（顶层数组，条目 {model, file, class}）在这里装载进 core.plms，
    与内建 PLM_REGISTRY 并列；没有该段时行为与从前一致。
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
    snapshot_every = meta.get("snapshot_every")
    if snapshot_every is not None:
        if (not isinstance(snapshot_every, int) or isinstance(snapshot_every, bool)
                or snapshot_every < 0):
            raise ImageError("meta.snapshot_every 必须是非负整数")
        core.snapshot_every = snapshot_every or None
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
    # 镜像声明的 PLM 必须在安装模型目录之前装载：$MEM.model_catalog.plm 是它的可见面，
    # 装完目录再注册就不会出现在对话看得见的清单里。
    _load_plms(core, image_path, image.get("plms") or [])
    # 启动自带的模型目录同样会被整树替换冲掉，这里重装（幂等，镜像已有的字段不覆盖）
    core.install_model_catalog()

    para_ref = image.get("para_ref", "$MEM.model_params")
    if not isinstance(para_ref, str) or not para_ref:
        # 显式 null / 数字 / 空串都会让 core.unwrap 抛 AttributeError（不是 ImageError）
        raise ImageError(f"para_ref 必须是非空字符串，得到 {para_ref!r}")
    core.para_ref = _resolve_para(core, para_ref)
    _mount_devices(core, image_path, image.get("devices") or [], device_markers)
    # security 段在设备挂载之后安装：区域子树里的设备挂载链接此时已物化，能被拒绝
    _parse_security(core, image)

    init = image.get("init")
    # guest 程序声明是镜像事实：装载期校验形状，冷启动在 init 之后创建对话，
    # 恢复路径先暂存，resume 按 attrs 找回（见 avm.boot 的三种启动模式）。
    guest_decl = _parse_guest_programs(meta.get("guest_programs"))
    core.guest_programs_decl = guest_decl
    core.strict_scheduler = bool(guest_decl)
    if restore:
        if checkpoint_in:
            raise ImageError("restore 与 checkpoint_in 不能同时使用（重启恢复与单对话恢复是两条路径）")
        # 恢复路径：对话与内存由快照/日志接管；init 段暂存给冷启动模式补跑。
        core._image_init = init
    elif checkpoint_in:
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
    if not restore and guest_decl:
        # 冷启动：init 之后按固定顺序创建 guest 程序对话并注册中断向量。
        start_guest_programs(core)

    journal_to = image.get("journal_to")
    if journal_to is not None:
        if not isinstance(journal_to, str) or not journal_to:
            raise ImageError("journal_to 必须是非空字符串路径")
        core.journal_path = journal_to if os.path.isabs(journal_to) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(image_path)), journal_to))
    snapshot_to = image.get("snapshot_to")
    if snapshot_to is not None:
        if not isinstance(snapshot_to, str) or not snapshot_to:
            raise ImageError("snapshot_to 必须是非空字符串路径")
        core.snapshot_path = snapshot_to if os.path.isabs(snapshot_to) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(image_path)), snapshot_to))
    persist_to = image.get("persist_to")
    if persist_to is not None:
        if not isinstance(persist_to, str):
            raise ImageError("persist_to 必须是字符串路径")
        core.persist_to = persist_to if os.path.isabs(persist_to) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(image_path)), persist_to))
    # 内核转储（诊断快照，不是恢复输入）：core_dump 开关、core_dump_to 落点，与 persist_to 同为顶层字段
    core_dump = image.get("core_dump")
    if core_dump is not None and not isinstance(core_dump, bool):
        raise ImageError("core_dump 必须是布尔值")
    core.core_dump_enabled = True if core_dump is None else core_dump
    core_dump_to = image.get("core_dump_to")
    if core_dump_to is not None:
        if not isinstance(core_dump_to, str) or not core_dump_to:
            raise ImageError("core_dump_to 必须是非空字符串路径")
        # normpath：镜像里写的相对落点常用 "/"，拼进目录后归一成宿主分隔符，
        # 否则 Windows 上会产出 "dir\out/x.json" 这种混合形式
        core.core_dump_to = core_dump_to if os.path.isabs(core_dump_to) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(image_path)), core_dump_to))
    return core


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 系统镜像启动器")
    parser.add_argument("image", help="系统镜像 JSON 文件路径")
    parser.add_argument("--checkpoint-in", metavar="PATH",
                        help="调试用：启动时加载单对话检查点（跳过 init，恢复为就绪对话）")
    from .boot import add_core_dump_args, add_mode_arg, add_restore_arg
    add_core_dump_args(parser)
    add_restore_arg(parser)
    add_mode_arg(parser)
    args = parser.parse_args(argv)
    # 局部导入：boot 依赖本模块的 load_image，模块级导入会成环
    from .boot import boot, dump_on_death, write_snapshot
    from .journal import JournalError
    try:
        core = boot(
            args.image, mode=args.mode,
            checkpoint_in=args.checkpoint_in, restore=args.restore,
        )
    except (ImageError, JournalError, VMMemoryError, OSError, json.JSONDecodeError) as e:
        print(f"[AVM] 镜像错误: {e}", file=sys.stderr)
        return 1
    if args.no_core_dump:
        core.core_dump_enabled = False
    status = 0
    snapshot_ok = True  # 只有安全点退出才写重启快照
    try:
        core.run()
    except RunnerStop as e:
        print(f"[AVM] {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("[AVM] 收到中断，写回后退出", file=sys.stderr)
        status = 130
        snapshot_ok = False
    except BaseException as e:
        # 内核死亡：先落现场快照，再保留原始 traceback 往上抛
        dump_on_death(core, e, args.core_dump)
        snapshot_ok = False
        raise
    finally:
        if snapshot_ok:
            write_snapshot(core)
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

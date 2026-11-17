import dataclasses
import traceback
from typing import Optional, List, Dict, Type, Any

import json
import os
import sys
import time

from .types import MetaList, MetaDict
from .capabilities import (
    CONTROL_DEVICE_ROOTS, MEM_ROOT_NAME, MODE_KERNEL, MODE_USER, Capability, caps_to_dict,
    check_root_name, default_kernel_caps, grant_target_segments, path_is_prefix,
    targets_prefix_free,
)
from .exceptions import IllegalOperationError, VMSyntaxError, VMMemoryError, APIError, RunnerStop, KernelPanic
from .api_errors import CONTEXT_LIMIT, INSUFFICIENT_BALANCE, CATEGORIES, classify_api_error
from .types import SystemMessage, UserMessage, Conversation, UserMessageBatch, message_to_api_dict, LinkNode
from .types import message_to_dict, batch_to_dict
from .memory import Memory, check_new_key
from .memory_device import MemoryDevice
from .monitor import Monitor
from .modules import (
    BASE_MODULES, MODULE_CATALOG, ModuleCatalogDevice, ModuleLoaderDevice,
    grantable_module_names, module_descriptor, register_module_from_source,
    register_module as register_catalog_module, registered_module_descriptors,
)
from .events import EventSystem, build_event, EVENT_TYPES
from .journal import (
    JOURNAL_RECORD_VERSION, Journal, JournalError, apply as apply_journal_record,
    conversation_create_payload, scheduler_to_snapshot,
)
from .info_devices import (
    CapsDevice, ConversationsDevice, RootsDevice, SchedulerDevice,
    ListenersDevice, MemorySummaryDevice, MonitorDevice,
)
from .interrupts import InterruptTable

# 指令集与工具面（计划 S1：从 core.py 抽到 instructions.py；旧名保留兼容）
from .executor import Executor, assemble_messages, commit_round, parse_json_args
from .services import Services, default_services
from .instructions import (
    Instruction, MemoryReadInstruction, MemoryWriteInstruction, MemoryMakeInstruction,
    MemoryDeleteInstruction, EditMetadataInstruction, CreateInstruction, CreateSubInstruction,
    SendInstruction, CloseInstruction, ListenEventInstruction,
    _instruction_registry, _build_tools, _build_imports,
    _effect_space, _note_memory_write, _report_tool_error, _canonical_ref,
    _condition_before_target, _dispatch_before_event,
    _parameters_of, _value_matches, _validate_instruction_args, _respond_error,
    _publish_after_safely, _publish_core_after_safely,
    _INSTRUCTION_ERRORS, _UNEXECUTED_TOOL_RESPONSE,
    instruction_registry, build_tools, build_imports,
)






# 操作员对话的系统消息：说明这条对话的执行器是操作员而不是模型。
_OPERATOR_SYSTEM_PROMPT = (
    "这是操作员对话。执行器是控制台的操作员，不是模型：本对话不产生推理结果，"
    "每轮由操作员提交一次工具调用；操作员选择继续时本对话结束本轮交互。"
)












# 指令执行体约定的错误类：可判定、可返回给对话的一类。九条指令共用同一套，
# 不在这套之内的异常是内核缺陷，由 advance_conversation 记帧后向上抛（不在这里吞）。

# 保留对话 cid（作者规定）：宿主侧操作员对话的身份，不参与正常分配，也不会撞上
# `_next_cid` 的正数。它对普通对话隐藏：不出现在 $MEM.conversations / $MEM.scheduler
# 里，也不可用 send_instruction / close_conversation 或 $conv.<cid> / $batch.<cid>
# 寻址。隐藏只是显示/寻址策略：对话本身随快照/日志正常恢复，日志里它的回合按普通
# 回合重放。
PANEL_CID = -1


# 启动自带的内存节点：模型目录。普通内存节点，不是设备——不产生事件、不参与调度。
# 名字里的点号是禁的（$MEM 寻址按点切段），所以 plm.* 与带点的模型 id 一律装进列表项的字段。
MODEL_CATALOG_KEY = "model_catalog"
_CATALOG_META = "模型目录：可用的 LLM 模型与本地 PLM；两侧都写明各自接受的参数"


def _publish_memory_after(
    core: 'Core',
    event_type: str,
    conv: Conversation,
    target: str,
    payload: dict,
    result: Optional[dict] = None,
    follow_target: Optional[str] = None,
    space=None,
) -> None:
    """读取/写入成功后发布 memory/device after 事实事件。

    事件 target 用调用者给的地址（append 时保持原 ref，带 -1）；追设备能力时要
    一个可解析的地址，追加写入由调用方通过 follow_target 传真实下标地址。
    `space` 是发起对话的地址上下文（根表）。
    """
    follow = core.mem.follow_link_target(follow_target or target, space=space)
    if follow[0] == "memory":
        core.events.publish_after_event(
            event_type, conv, target, payload, source_kind="memory",
            target_ref=("memory", tuple(follow[1])),
        )
        return
    _, mount_key, remainder = follow
    device = core.mem._devices[mount_key]
    extra = device.after_event(event_type, remainder, result or {})
    if extra is None:
        return
    merged = dict(extra)
    merged["ref"] = target
    merged["actor_cid"] = conv.cid
    core.events.publish_after_event(
        event_type, conv, target, merged, device_path=mount_key,
        target_ref=("device", mount_key, tuple(remainder)),
    )




# 协议层拒绝（JSON 解析失败、未知工具名、缺少可配对的 call_id）时回给对话的占位文本。
# 内容不携带错误细节：错误面向运行者（_report_tool_error），这里只为保住
# 「assistant 的 tool_calls 与 tool 消息一一配对」这条协议不变量。












































def _check_model_namespace(core: 'Core', program_type: str, para: MetaDict, para_ref: str) -> None:
    """输入边界的命名空间交叉校验：程序类型与 model 必须配套。

    python 程序 → model 必须是 PLM 注册名；settingup 程序 → model 必须是上游 LLM 模型名。
    上游清单没取到时只警告不拦：`/models` 只是参考，取不到不该让镜像起不来（讨论案第五节）。
    """
    catalog = core.mem._data.get(MODEL_CATALOG_KEY)
    if not isinstance(catalog, MetaDict):
        return
    model = para.get("model")
    if program_type == "python":
        if core.services.plm.is_host_only(model):
            # 宿主独占 PLM：拒绝理由涉及宿主/调试实现，不返回给对话
            raise IllegalOperationError("操作被外部原因拒绝")
        names = sorted(
            e.get("name") for e in (catalog.get("plm") or []) if isinstance(e, MetaDict)
        )
        if not isinstance(model, str) or not model:
            raise IllegalOperationError(
                f"{para_ref} 缺少 model：python 程序的 model 必须是 PLM 注册名（可用：{names}）"
            )
        if model not in names:
            raise IllegalOperationError(
                f"{para_ref} 的 model {model!r} 不是 PLM 注册名（可用：{names}）；"
                f"LLM 模型名只能配 settingup 程序"
            )
        return
    if not isinstance(model, str) or not model:
        # 缺 model 与上游清单无关，输入边界就能判定
        raise IllegalOperationError(
            f"{para_ref} 缺少 model：settingup 程序的 model 必须是 LLM 模型名"
        )
    llm = catalog.get("llm")
    if not isinstance(llm, MetaDict) or llm.get("source") != "api":
        print(
            f"[AVM] 模型目录未取到上游清单，跳过 LLM 模型名校验：{para_ref} 的 model {model!r}",
            file=sys.stderr,
        )
        return
    listed = sorted(
        e.get("id") for e in (llm.get("models") or []) if isinstance(e, MetaDict)
    )
    if model not in listed:
        raise IllegalOperationError(
            f"{para_ref} 的 model {model!r} 不在上游模型清单里（可用：{listed}）"
        )


_CAPS_SPEC_KEYS = frozenset({"mode", "groups", "roots", "tools", "imports"})


def _read_caps_value(caps_node, key: str):
    """把 value.caps 下的一个数据节点读成 Python 值。"""
    value = caps_node.get(key)
    if key in ("groups", "roots", "tools", "imports"):
        if not isinstance(value, MetaList):
            raise IllegalOperationError(
                f"程序节点的 value.caps.{key} 必须是 list 数据节点"
            )
        items = []
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise IllegalOperationError(
                    f"程序节点的 value.caps.{key}[{i}] 必须是 str 数据节点"
                )
            items.append(str(item))
        return items
    if not isinstance(value, str):
        raise IllegalOperationError(
            f"程序节点的 value.caps.{key} 必须是 str 数据节点"
        )
    return str(value)


def _program_caps_spec(node) -> dict:
    """读取程序节点 value.caps 数据；缺省空规格（继承父能力）。

    能力声明是节点数据，不是控制数据：ctrl 只描述类型与数据组织格式。
    ctrl.caps 一律拒绝（镜像装载与运行期创建两处入口）。
    """
    if not isinstance(node, MetaDict):
        return {}
    ctrl = node.get_ctrl() or {}
    if "caps" in ctrl:
        raise IllegalOperationError(
            "程序节点的能力声明必须放在 value.caps 数据节点里；ctrl.caps 不再支持"
        )
    caps_node = node.get("caps")
    if caps_node is None:
        return {}
    if not isinstance(caps_node, MetaDict):
        raise IllegalOperationError("程序节点的 value.caps 必须是 dict 数据节点")
    unknown = sorted(set(caps_node.keys()) - _CAPS_SPEC_KEYS)
    if "workspace" in unknown:
        raise IllegalOperationError(
            "程序节点的 value.caps.workspace 已移除：工作目录不再由内核分配。"
            "改用软件建目录（memory_make）后写 $MEM.conversations.<cid>.roots 动态加根"
        )
    if unknown:
        raise IllegalOperationError(
            f"程序节点的 value.caps 含未知字段：{unknown}（允许：{sorted(_CAPS_SPEC_KEYS)}）"
        )
    return {key: _read_caps_value(caps_node, key) for key in caps_node.keys()}


def _derive_child_caps(core: 'Core', parent: Optional[Conversation], node, is_sub: bool,
                       is_init: bool = False) -> Capability:
    """按程序 ctrl.caps 与父能力计算子对话能力。

    规则（批次一）：
    - 模式：缺省继承；用户态不能创建内核态；亚对话不能跨模式；init 强制内核态。
    - 工具集：缺省继承；显式声明时必须 ⊆ 父工具集。
    - 组：缺省继承；用户态父对话只能给子对话父已有的组（内核态父可指定）。
    - 根：内核态子对话固定 `{MEM: ()}`；用户态子对话的授予根来自 caps.roots 的
      `根名 -> $MEM 路径`（同域创建时只能沿用父根，不能新增）。用户态对话的工作
      目录不再由内核分配：目录由软件建，根由 `$MEM.conversations.<cid>.roots` 动态授予。
    """
    spec = _program_caps_spec(node)
    parent_caps = parent.caps if parent is not None else default_kernel_caps()
    parent_mode = parent_caps.mode

    mode = spec.get("mode", parent_mode)
    if mode not in (MODE_KERNEL, MODE_USER):
        raise IllegalOperationError(f"程序 caps.mode 必须是 kernel/user，得到 {mode!r}")
    if is_init and mode != MODE_KERNEL:
        raise VMMemoryError("init 程序必须是内核态（没有内核态就无法管理权限）")
    if parent is not None:
        if parent_mode == MODE_USER and mode != MODE_USER:
            raise VMMemoryError("模式不可提权：用户态对话只能创建用户态对话")
        if is_sub and mode != parent_mode:
            raise VMMemoryError("亚对话不能跨模式创建")

    tools_spec = spec.get("tools")
    if tools_spec is None:
        tools = parent_caps.tools
    else:
        if not isinstance(tools_spec, list) or not all(isinstance(x, str) for x in tools_spec):
            raise IllegalOperationError("程序 caps.tools 必须是字符串列表")
        tools = frozenset(tools_spec)
        extra = tools - parent_caps.tools
        if extra:
            raise VMMemoryError(f"工具集不可超出父对话：{sorted(extra)}")

    imports_spec = spec.get("imports")
    if imports_spec is None:
        imports = parent_caps.imports
    else:
        if not isinstance(imports_spec, list) or not all(isinstance(x, str) for x in imports_spec):
            raise IllegalOperationError("程序 caps.imports 必须是字符串列表")
        imports = frozenset(imports_spec)
        declared_base = sorted(imports & set(BASE_MODULES))
        if declared_base:
            raise IllegalOperationError(
                f"程序 caps.imports 不能声明基线模块：{declared_base}"
                f"（基线 {list(BASE_MODULES)} 不可裁剪，也不需要声明）"
            )
        unknown = imports - grantable_module_names()
        if unknown:
            raise VMMemoryError(
                f"程序 caps.imports 引用了宿主模块目录之外的模块：{sorted(unknown)}"
                f"（可授予：{sorted(grantable_module_names())}）"
            )
        extra = imports - parent_caps.imports
        if extra:
            raise VMMemoryError(f"模块导入不可超出父对话：{sorted(extra)}")

    groups_spec = spec.get("groups")
    if groups_spec is None:
        groups = parent_caps.groups
    else:
        if not isinstance(groups_spec, list) or not all(isinstance(x, str) and x for x in groups_spec):
            raise IllegalOperationError("程序 caps.groups 必须是非空字符串列表")
        groups = tuple(dict.fromkeys(groups_spec))
        if parent is not None and parent_mode == MODE_USER and not set(groups) <= set(parent_caps.groups):
            raise VMMemoryError("组不可提权：用户态对话只能创建同组或更少组的子对话")

    if mode == MODE_KERNEL:
        roots = {MEM_ROOT_NAME: ()}
    else:
        grants = {}
        roots_spec = spec.get("roots") or []
        if not isinstance(roots_spec, list) or not all(isinstance(x, str) for x in roots_spec):
            raise IllegalOperationError(
                "程序 caps.roots 必须是根名列表（名字来自镜像 security.roots 注册表）"
            )
        for name in roots_spec:
            check_root_name(name)
            if name == MEM_ROOT_NAME:
                raise IllegalOperationError("程序 caps.roots 不能声明 MEM 根")
            target = core.root_registry.get(name)
            if target is None:
                raise VMMemoryError(f"程序 caps.roots 引用了未注册的根：{name!r}")
            grants[name] = tuple(target)
        same_domain = is_sub or (parent is not None and parent_mode == MODE_USER)
        if same_domain:
            inherited = {n: t for n, t in parent_caps.roots.items() if n != MEM_ROOT_NAME}
            for name, target in grants.items():
                if inherited.get(name) != target:
                    raise VMMemoryError(f"不能向子对话授予父对话没有的根：{name!r}")
            roots = inherited
        else:
            roots = grants

    return Capability(mode=mode, groups=groups, roots=roots, tools=tools, imports=imports)


def _build_conversation(core: 'Core', system_ref: str, para_ref: str, parent: Conversation, is_sub: bool,
                        space=None, is_init: bool = False) -> Conversation:
    node = core.unwrap(system_ref, for_llm=False, space=space)
    # 程序节点校验：与 Python 对话要求 ctrl.type='python' 对等，LLM 对话要求 ctrl.type='settingup'。
    # str 节点、无类型 dict 节点（如参考手册）都不是程序，create_cmd/create_sub 不接受。
    node_type = (node.get_ctrl() or {}).get("type") if isinstance(node, MetaDict) else None
    if node_type not in ("settingup", "python"):
        raise VMMemoryError(
            f"{system_ref} 不是程序节点：system_ref 必须指向 ctrl.type='settingup'（LLM 程序）"
            f"或 'python'（Python 程序）的 dict 节点"
        )
    content = node.get("content")
    if not isinstance(content, str):
        raise VMMemoryError(
            f"{system_ref} 缺 content：ctrl.type='{node_type}' 节点的 value 必须含字符串 content"
        )
    system = content
    is_python = (node_type == "python")
    messages = [SystemMessage(content=system)]
    para_node = core.unwrap(para_ref, for_llm=False, space=space)
    para_type = (para_node.get_ctrl() or {}).get("type") if isinstance(para_node, MetaDict) else None
    if para_type != "para":
        raise VMMemoryError(
            f"{para_ref} 不是参数节点：para_ref 必须指向 ctrl.type='para' 的 dict 节点"
        )
    _check_model_namespace(core, node_type, para_node, para_ref)
    caps = _derive_child_caps(core, parent, node, is_sub, is_init=is_init)
    conv = Conversation(
        messages=messages,
        cid=0,
        is_sub=is_sub,
        parent=parent,
        is_root=False,
        caps=caps,
    )
    core._register(conv)
    if conv.caps.mode == MODE_USER:
        # 能力自述：保留根 caps 绑定到 $MEM.caps.<cid>，由 Core 拥有，不参与授予
        roots = dict(conv.caps.roots)
        roots["caps"] = ("caps", str(conv.cid))
        conv.caps = conv.caps.copy(roots=roots)
    conv.para_ref = para_ref
    conv.is_python = is_python
    return conv






class Core:
    def __init__(self, services: Optional[Services] = None):
        # 服务层（计划 S5）：LLM 与 PLM 服务由组合根注入；默认装配见 services.py。
        self.services: Services = services if services is not None else default_services()
        self._conv_by_cid: Dict[int, Conversation] = {}
        self._ready_cids: List[int] = []
        self._dormant_cids: List[int] = []
        self._active_cid: Optional[int] = None
        self._next_cid: int = 0
        self.mem: Memory = Memory()
        self._executors: Dict[int, Executor] = {}  # cid → PLM 执行器（Python 对话；PLM 状态跨轮次保留）
        self._finished: set = set()  # 已关闭（个体结束）的对话 cid；记录供监测/信息设备展示
        self._instruction_budget: int = 50  # 每次激活的指令预算（核心中断阈值）
        self._activation_instructions: int = 0
        self.image_dir: Optional[str] = None  # 调试用：镜像目录（检查点默认路径）
        self.image_name: Optional[str] = None  # 调试用：镜像名（不含扩展名，检查点默认路径）
        # 启动标志：本次运行由哪个宿主托管（如 {"console": True}）。宿主事实，不属于镜像内容；
        # 由 load_image 在挂载设备之前写入，核心启动后不再变化。设备在 attach_core 里读，
        # 核心不读、不校验、不监督。
        self.host_flags: Dict[str, Any] = {}
        # 宿主侧 IO 总线：由 console 宿主安装。与 host_flags 一样是宿主事实，
        # 不属于镜像内容、不进 $MEM；旧 $MEM.inputs / $MEM.outputs 设备不读它，
        # 操作面板宿主的 IO 设备由 Console 单独挂载（见 avm/io_devices.py）。
        self.io_bus = None
        # 操作面板的生命周期：宿主请求停止（关机的安全点标志）。
        self._stop_requested: bool = False
        # 本次启动的恢复摘要（boot 填写；/api/state 暴露给面板首屏）。
        self.recovery: Dict[str, Any] = {
            "source": "cold", "snapshot_seq": None, "replayed": 0,
            "restored": {}, "warnings": [],
        }
        # 镜像声明的 PLM 实现：model 名 → 类，由 load_image 按镜像的 plms 段装载。
        # 与宿主内建的 PLM_REGISTRY 并列，内建优先（镜像不得同名覆盖，装载期拒绝）。
        # 放在这里是因为本函数末尾的 install_model_catalog 要枚举它。
        self.plms: Dict[str, type] = {}
        # 安全配置（批次一 S4）：根注册表（名字 -> 真实段）。
        # 用户态对话的根表由软件动态授予（$MEM.conversations.<cid>.roots），内核不分配目录。
        self.root_registry: Dict[str, tuple] = {}
        self.debug: bool = False
        self.monitor: Monitor = Monitor()
        self.para_ref: str = "$MEM.model_params"
        # API 失败时的运行者策略：callable(error, attempt) -> "retry" | "reload" | "exit"。
        # 这是宿主覆盖：装了它，内核不再套用内置的三类处置；没装时由
        # `advance_conversation` 按分类决定（余额不足暂停 / 上下文超限跳过 / 其余重试）。
        # 终端提示见 avm/runner_prompt.py，由宿主显式安装。
        self.api_error_policy = None
        # 内置 API 失败策略：余额不足暂停、上下文超限跳过对话、其余重试。
        # 连续失败达到这个次数就暂停执行；镜像 meta.api_error.max_retries 覆盖，
        # 运行期可直接改 Core.api_error_max_retries。
        self.api_error_max_retries: int = 5
        self.paused: bool = False
        self.pause_reason: Optional[str] = None
        self.pause_category: Optional[str] = None
        # 本次暂停由哪个对话的连续失败触发；clear_api_pause 只重置它的计数，
        # 不遍历对话表（宿主线程恢复时主线程可能还在动表）。
        self._paused_cid: Optional[int] = None
        # run() 只做一次启动初始化（基线帧、断电边沿基线、resume 的 boot 唤醒）；
        # API 暂停后宿主再次进入 run() 时不能重复做这些事。
        self._run_started: bool = False
        # 内核转储：镜像可用 core_dump_to / core_dump 覆盖（见 load_image）
        self.core_dump_to: Optional[str] = None
        self.core_dump_enabled: bool = True
        self.events: EventSystem = EventSystem(self)
        self.interrupts: InterruptTable = InterruptTable(self)  # 中断向量表：外部（控制台）发起的控制权转移
        self._turn_events: List[dict] = []
        # 持久运行：回合级效果日志（journal）与重启快照（snapshot）。
        # journal 为 None 时所有记录点是空操作——无日志镜像行为不变。
        self.journal: Optional[Journal] = None
        self.journal_path: Optional[str] = None   # 镜像 journal_to（load_image 解析后写入）
        self.snapshot_path: Optional[str] = None  # 镜像 snapshot_to（load_image 解析后写入）
        self._journal_seq: int = 0                # 已提交/已重放到的日志位置
        # 日志代际：resume 续当前代际；reboot/reset 由宿主归档当前日志并开新代际（seq 归零）。
        self.journal_generation: int = 1
        self._journal_batch: Dict[int, dict] = {}  # cid → 上次提交的待发批次（增量记录用）
        self._scheduler_committed: Optional[dict] = None  # 上次提交后的调度状态（增量基线）
        self.snapshot_every: Optional[int] = None  # 每 N 个已提交回合写一次定时快照；None 关闭
        self._last_snapshot_seq: int = 0           # 上次写快照对应的日志 seq
        self._turn_effects: List[dict] = []       # 本轮指令产生的语义效果（回合结束提交）
        self._replaying: bool = False             # 重放期间禁止再记录
        self._recording: bool = False             # 只在回合执行体内记录
        # 调度严格模式：镜像声明了 meta.guest_programs 时打开。打开后"活跃为空、
        # 就绪为空且无待派发中断"不再视为跑完：run() 向已注册的 `idle` 中断向量
        # 派发；派发后仍然是空闲（没有可运行目标、或 guest idle 处理程序没离开
        # 空闲）就 KernelPanic（空闲 = 严重错误）。宿主 console 把保留对话注册为
        # `idle` 向量，面板阻塞等操作员就是空闲态。嵌入式宿主未接入生命周期管理
        # （镜像没有声明 guest 程序、宿主也没注册 idle 向量）时，run() 在队列空了
        # 时正常返回。
        self.strict_scheduler: bool = False
        self.guest_roles: Dict[str, dict] = {}      # 角色 → {"cid", "entry", "kind"}（宿主中断向量来源）
        self.guest_programs_decl: Dict[str, str] = {}  # 镜像 meta.guest_programs（角色 → 程序节点引用）
        self._image_init = None                     # load_image(restore=True) 暂存的 init 段（冷启动补跑）
        self._power_state_seen = None               # 断电判定的边沿基线（见 power_off_requested）
        # 恢复出来的对话（快照/日志）：Python 执行器首次创建时要预热一次
        self._warmup_needed: set = set()
        # 设备来源表：挂载地址 → 设备 spec（文件/类/参数/标识符；内建设备 builtin）。
        # 快照要覆盖热装设备，而设备表里只有实例，没有"这是从哪来的"。
        self.device_specs: Dict[str, dict] = {}
        self.install_host_devices()
        # 启动自带的内存节点：模型目录（静态部分；LLM 清单由宿主 refresh_model_catalog 填）
        self.install_model_catalog()

    def install_host_devices(self) -> None:
        """挂载宿主内建设备（幂等）。

        Core 启动时调用一次；镜像整树替换后由 `materialize_mount_links` 重物化挂载点；
        旧快照/旧日志恢复后要再调用一次，补齐快照生成时还不存在的宿主设备
        （例如 `module_catalog` 是后加的）。
        """
        factories = {
            "conversations": lambda: ConversationsDevice(self),
            "scheduler": lambda: SchedulerDevice(self),
            "memory": lambda: MemorySummaryDevice(self),
            "monitor": lambda: MonitorDevice(self),
            "listeners": lambda: ListenersDevice(self),
            "roots": lambda: RootsDevice(self),
            "caps": lambda: CapsDevice(self),
            "module_catalog": lambda: ModuleCatalogDevice(),
            "module_loader": lambda: ModuleLoaderDevice(self),
        }
        for mount, factory in factories.items():
            if mount in self.mem._devices:
                continue
            device = factory()
            self.mem.mount(mount, device)
            self.device_specs[mount] = {
                "builtin": True,
                "class": type(device).__name__,
            }

    # ------------------------------------------------------------------
    # 模块目录（运行期注册）
    # ------------------------------------------------------------------

    def refresh_panel_caps(self) -> None:
        """-1 面板对话是完整权限、不是创建时固定的：目录变化后刷新它的导入白名单。

        只刷新这一个保留对话；已有普通对话不扩权（作者 2026-11-16 定夺：新库只给
        之后创建的新对话用）。
        """
        conv = self._conv_by_cid.get(PANEL_CID)
        if conv is None:
            return
        current = grantable_module_names()
        if conv.caps.imports != current:
            conv.caps = conv.caps.copy(imports=current)

    def register_module(self, name, *, source_kind="builtin", source=None,
                        exports=None, export_names=None, note="", journal=True):
        """运行期注册一个可授予模块（模块目录的唯一写入口）。

        - `source_kind="builtin"`：直接给 exports 映射（宿主代码/测试；不持久化，
          因为模块对象由宿主代码拥有）；
        - `source_kind="file"` / `"module"`：宿主文件/已安装模块按导出名单裁剪；
          注册描述进 journal（世界效果）与快照，重启恢复。
        - 注册后刷新 -1 面板对话的完整目录；已有普通对话不扩权。
        """
        if source_kind == "builtin":
            spec = register_catalog_module(name, exports=exports, note=note or "宿主注册")
        else:
            spec = register_module_from_source(
                name, source_kind, source, tuple(export_names or ()), note or "")
            if journal:
                self.note_effect("module_register", spec=module_descriptor(name))
        self.refresh_panel_caps()
        return spec

    def is_hidden_cid(self, cid) -> bool:
        """对话是否对普通对话隐藏、不可寻址（显示/寻址策略，不影响恢复）。

        判定只读对话自己的 `attrs["hidden"]`（宿主在保留对话创建时写 "1"）；
        cid 是身份，不参与显示策略。
        """
        conv = self._conv_by_cid.get(cid)
        return conv is not None and (conv.attrs or {}).get("hidden") == "1"

    # ------------------------------------------------------------------
    # 安全配置：根注册表与动态授予根（批次一 S4）
    # ------------------------------------------------------------------

    def _validate_root_specs(self, roots_spec: dict, check_live: bool = True) -> dict:
        """校验根注册表：名字、目标语法、前缀无关、保留区、存活对话根表。

        镜像装载（check_live=False）与运行期替换（check_live=True）共用。
        """
        checked = {}
        for name, target in roots_spec.items():
            check_root_name(name)
            target = tuple(target)
            if name == MEM_ROOT_NAME:
                raise VMMemoryError("security.roots 不能注册 MEM 根")
            if not target:
                raise VMMemoryError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if target and target[0] in CONTROL_DEVICE_ROOTS:
                raise VMMemoryError(
                    f"根 {name!r} 的目标落在内核控制面设备路径 $MEM.{target[0]}"
                )
            if not targets_prefix_free(list(checked.values()), target):
                raise VMMemoryError(f"根 {name!r} 的目标与已注册根前缀冲突")
            checked[name] = target
        if check_live:
            others = []
            for conv in self._conv_by_cid.values():
                for root_name, root_target in conv.caps.roots.items():
                    if root_name == MEM_ROOT_NAME:
                        continue
                    others.append((conv.cid, tuple(root_target)))
            for name, target in checked.items():
                for cid, other_target in others:
                    if target == other_target:
                        continue
                    if path_is_prefix(target, other_target) or path_is_prefix(other_target, target):
                        raise VMMemoryError(
                            f"根 {name!r} 的目标与存活对话 {cid} 的根表前缀冲突"
                        )
        return checked

    def install_security_config(self, roots_spec: dict) -> None:
        """装载镜像 security 段：校验并安装根注册表。

        校验项：根名规则、目标前缀无关、保留区。失败抛 VMMemoryError
        （镜像装载器翻成 ImageError）。工作目录不再由内核分配：目录由软件用
        memory_make 建，根由 `$MEM.conversations.<cid>.roots` 动态授予。
        """
        checked = self._validate_root_specs(roots_spec, check_live=False)
        self.root_registry = checked

    def replace_root_registry(self, roots_spec: dict) -> None:
        """运行期整体替换根注册表（$MEM.roots 控制面），先校验后提交。"""
        checked = self._validate_root_specs(roots_spec, check_live=True)
        self.root_registry = checked

    def _core_roots(self, conv) -> dict:
        """Core 拥有的根（目前只有能力自述根 caps）：控制面不可覆盖、重放时保留。"""
        target = conv.caps.roots.get("caps")
        return {"caps": tuple(target)} if target is not None else {}

    def _validate_grants(self, conv, grants: dict) -> None:
        """授予根的全量校验：保留区、设备路径、新根之间与其他存活根/根注册表的前缀关系。

        `grants` 是"根名 -> 目标段序列"，不含 Core 自述根；先全量校验后提交。
        """
        for name, target in grants.items():
            if not target:
                raise IllegalOperationError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if target[0] in CONTROL_DEVICE_ROOTS:
                raise IllegalOperationError(
                    f"根 {name!r} 的目标落在内核控制面设备路径 $MEM.{target[0]}"
                )
            for mount in self.mem._devices:
                mount_segments = tuple(mount.split("."))
                if target == mount_segments:
                    continue  # 显式授予设备本身是允许的
                if path_is_prefix(target, mount_segments) or path_is_prefix(mount_segments, target):
                    raise IllegalOperationError(
                        f"根 {name!r} 的目标落在保留的内核设备路径 $MEM.{mount}"
                    )
        # 新根之间前缀无关（相等目标允许共享）
        items = list(grants.items())
        for i, (n1, t1) in enumerate(items):
            for n2, t2 in items[i + 1:]:
                if t1 == t2:
                    continue
                if path_is_prefix(t1, t2) or path_is_prefix(t2, t1):
                    raise IllegalOperationError(f"根 {n1!r} 与 {n2!r} 的目标前缀冲突")
        # 与其他存活对话的根表、根注册表比较
        others = []
        for cid, other in self._conv_by_cid.items():
            if cid == conv.cid:
                continue
            for other_name, other_target in other.caps.roots.items():
                if other_name == MEM_ROOT_NAME:
                    continue
                others.append((cid, other_name, tuple(other_target)))
        for name, target in self.root_registry.items():
            others.append((None, name, tuple(target)))
        for name, target in items:
            for cid, other_name, other_target in others:
                if target == other_target:
                    continue  # 相同目标允许共享
                if path_is_prefix(target, other_target) or path_is_prefix(other_target, target):
                    raise IllegalOperationError(
                        f"根 {name!r} 与已有根 {other_name!r} 的目标前缀冲突"
                    )

    def set_conversation_roots(self, cid: int, roots_map: dict) -> None:
        """整体替换用户态对话的非 core 根表（控制面 `$MEM.conversations.<cid>.roots`）。

        `roots_map` 是"根名 -> 目标段序列（list/tuple，不含 MEM 根）"的完整映射；
        控制面语义是替换，不是合并。Core 自述根（caps）不可覆盖。
        校验通过后先记 conversation_roots 效果、再写 caps（日志与内存一致）。
        """
        conv = self._conv_by_cid.get(cid)
        if conv is None:
            raise VMMemoryError(f"对话 {cid} 不存在")
        if cid in self._finished:
            raise VMMemoryError(f"对话 {cid} 已关闭")
        if conv.caps.mode == MODE_KERNEL:
            raise IllegalOperationError(
                "内核态对话的根表固定含 MEM，不能通过控制面改写（用户态对话才可授予根）"
            )
        if not isinstance(roots_map, dict):
            raise IllegalOperationError("roots 必须是映射（根名 -> 目标段序列）")
        core_roots = self._core_roots(conv)
        grants: Dict[str, tuple] = {}
        for name, segments in roots_map.items():
            if not isinstance(name, str) or not name:
                raise IllegalOperationError(f"根名必须是非空字符串，得到 {name!r}")
            if name in core_roots:
                raise IllegalOperationError(f"根名 {name!r} 由 Core 分配，不能通过控制面覆盖")
            check_root_name(name)
            if isinstance(segments, str) or not isinstance(segments, (list, tuple)):
                raise IllegalOperationError(
                    f"根 {name!r} 的目标必须是段序列（list/tuple），得到 {type(segments).__name__}"
                )
            target = tuple(segments)
            if not target:
                raise IllegalOperationError(f"根 {name!r} 的目标不能是 $MEM 根本身")
            if any(not isinstance(seg, str) or not seg for seg in target):
                raise IllegalOperationError(f"根 {name!r} 的目标段必须是非空字符串：{target!r}")
            grants[name] = target
        self._validate_grants(conv, grants)
        new_roots = dict(core_roots)
        new_roots.update(grants)
        self.note_effect(
            "conversation_roots", cid=cid,
            roots={name: list(target) for name, target in grants.items()},
        )
        conv.caps = conv.caps.copy(roots=new_roots)

    def _register(self, conv: Conversation, cid: Optional[int] = None) -> Conversation:
        conv.cid = self._next_cid if cid is None else cid
        assert conv.cid not in self._conv_by_cid, f"cid {conv.cid} 已存在"
        self._conv_by_cid[conv.cid] = conv
        if cid is None:
            self._next_cid += 1
        return conv

    def get_conversation(self, cid: int) -> Optional[Conversation]:
        return self._conv_by_cid.get(cid)

    def checkpoint_path(self, cid: int) -> str:
        """调试用：单对话检查点的默认路径（镜像目录/out/<镜像名>.conv.json；无镜像上下文时用 checkpoints/conv-<cid>.json）"""
        if self.image_dir and self.image_name:
            return os.path.join(self.image_dir, "out", f"{self.image_name}.conv.json")
        return os.path.join("checkpoints", f"conv-{cid}.json")

    def save_conversation(self, cid: int, path: str) -> None:
        """调试用：保存单对话检查点（独立 JSON 文件，无损、确定性）。

        若历史末尾是未配对的 assistant 工具调用（如等待输入响应时），截掉该条，
        保证恢复后的历史是"截至最后一条完整交换"的干净前缀，to_api_messages()
        与保存时的请求前缀逐字节一致（前缀缓存命中）。v1 不支持 Python 对话。
        """
        conv = self._conv_by_cid.get(cid)
        if conv is None:
            raise VMMemoryError(f"对话 {cid} 不存在")
        if conv.is_python:
            raise ValueError("v1 检查点不支持 Python 对话")
        cp = conv.to_checkpoint()
        msgs = cp["messages"]
        last_parts = (msgs[-1].get("parts") or []) if msgs else []
        if (msgs and msgs[-1].get("role") == "assistant"
                and any(part.get("type") == "tool_call" for part in last_parts)):
            msgs = msgs[:-1]  # 末尾未配对工具调用：截断到最后一个完整交换
            cp["messages"] = msgs
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cp, f, ensure_ascii=False, indent=2)

    def restore_conversation(self, path: str, schedule: str = "dormant") -> Conversation:
        """调试用：从检查点恢复对话（注册新 cid；v1 不支持 Python 对话）。

        schedule: "dormant"（默认，等待被调用）或 "ready"（排队等待调度）。
        """
        with open(path, encoding="utf-8") as f:
            cp = json.load(f)
        conv = Conversation.from_checkpoint(cp)
        if conv.is_python:
            raise ValueError("v1 检查点不支持 Python 对话")
        self._register(conv)
        if schedule == "ready":
            self._ready_cids.append(conv.cid)
        else:
            self._dormant_cids.append(conv.cid)
        return conv

    # ------------------------------------------------------------------
    # 回合级效果日志（journal，批次 A）
    # ------------------------------------------------------------------

    def attach_journal(self, path: str) -> "Journal":
        """挂上回合日志（后续回合从这里追加）。重放用 `replay_journal`。"""
        self.detach_journal()
        self.journal = Journal(path)
        # 调度增量的基线：从装载后的状态开始记，第一条记录覆盖 run 前的选择与回合内迁移。
        self._scheduler_committed = self._scheduler_state()
        self._refresh_journal_batch()
        return self.journal

    def detach_journal(self) -> None:
        """摘掉日志（之后的回合不再记录；已提交的记录留在文件里）。"""
        self.journal = None

    def _journaling(self) -> bool:
        """当前是否正在为回合日志记录（无日志/重放/非回合执行期都是假）。

        记录点先问这一句再构造载荷：无日志镜像不该为日志做序列化工作。
        """
        return self.journal is not None and not self._replaying and self._recording

    def note_effect(self, kind: str, **payload) -> None:
        """记一条语义效果：无日志、重放期间、非回合执行期都是空操作。"""
        if not self._journaling():
            return
        self._turn_effects.append({"kind": kind, **payload})

    def note_effects(self, effects: List[dict]) -> None:
        """批量记效果（一步产生多条状态变化的操作用）。"""
        if not self._journaling():
            return
        self._turn_effects.extend(effects)

    def _scheduler_state(self) -> dict:
        """当前调度状态的只读快照（回合开始/提交时各取一次，做增量用）。"""
        return {
            "active": self._active_cid,
            "ready": list(self._ready_cids),
            "dormant": list(self._dormant_cids),
            "finished": set(self._finished),
            "activation_instructions": int(self._activation_instructions),
            "next_cid": int(self._next_cid),
        }

    @staticmethod
    def _scheduler_delta(before: dict, after: dict):
        """把回合前后的调度状态压成一条增量变化；没有变化返回 None。"""
        delta = {}
        if before["active"] != after["active"]:
            delta["active"] = after["active"]
        for key in ("ready", "dormant"):
            before_list = before[key]
            after_list = after[key]
            before_set = set(before_list)
            after_set = set(after_list)
            added = [cid for cid in after_list if cid not in before_set]
            removed = [cid for cid in before_list if cid not in after_set]
            if added:
                delta[key + "_add"] = added
            if removed:
                delta[key + "_remove"] = removed
        added_finished = sorted(after["finished"] - before["finished"])
        removed_finished = sorted(before["finished"] - after["finished"])
        if added_finished:
            delta["finished_add"] = added_finished
        if removed_finished:
            delta["finished_remove"] = removed_finished
        if before["activation_instructions"] != after["activation_instructions"]:
            delta["activation_instructions"] = after["activation_instructions"]
        if before["next_cid"] != after["next_cid"]:
            delta["next_cid"] = after["next_cid"]
        if not delta:
            return None
        return {"kind": "scheduler", **delta}

    def _refresh_journal_batch(self) -> None:
        """重放/恢复完成后，把"上次提交的批次"对齐到当前对话状态。"""
        self._journal_batch = {
            cid: batch_to_dict(conv.user_batch) for cid, conv in self._conv_by_cid.items()
        }

    def _refresh_journal_scheduler(self) -> None:
        """重放/恢复完成后，把调度增量基线对齐到当前调度状态。"""
        self._scheduler_committed = self._scheduler_state()

    def _commit_journal_record(self, conv: Conversation, history_delta: List[dict]) -> None:
        """回合结束提交一条记录（安全点调用；异常回合不提交）。

        记录是**有序变化序列**：本轮追加的消息、指令产生的效果、待发批次的变化、
        调度队列的变化。顺序即执行顺序；重放只按顺序应用，不再按字段分阶段重组。
        """
        effects = self._turn_effects
        self._turn_effects = []
        if self.journal is None:
            return
        changes: List[dict] = []
        for item in history_delta:
            changes.append({"kind": "message", "message": item})
        for effect in effects:
            changes.append({"kind": "effect", "effect": effect})
        cid = conv.cid
        empty_batch = {"tool_responses": [], "user_contents": []}
        # 批次变化可能发生在别的对话上（delivery 效果给目标加输入），所以对所有对话
        # 取差量；数量与对话数同级，单条改动通常为空。
        for other_cid, other_conv in self._conv_by_cid.items():
            current_batch = batch_to_dict(other_conv.user_batch)
            previous = self._journal_batch.get(other_cid, empty_batch)
            if current_batch != previous:
                changes.append(
                    {"kind": "batch", "cid": other_cid, "batch": current_batch}
                )
            self._journal_batch[other_cid] = current_batch
        scheduler_before = self._scheduler_committed
        if scheduler_before is None:
            scheduler_before = self._scheduler_state()
        delta = self._scheduler_delta(scheduler_before, self._scheduler_state())
        if delta is not None:
            changes.append(delta)
        self._scheduler_committed = self._scheduler_state()
        record = {
            "v": JOURNAL_RECORD_VERSION,
            "generation": int(self.journal_generation),
            "cid": cid,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "changes": changes,
        }
        self._journal_seq = self.journal.append(record)

    def replay_journal(self, records: Optional[List[dict]] = None) -> int:
        """按 seq 顺序重放日志（默认从日志文件续在已应用位置之后）。

        幂等：应用位置由 `_journal_seq` 推进，同一段日志重放两次第二次是空操作。
        返回本次实际应用的记录数。日志不连续（缺记录）直接报错。
        """
        if self.journal is None:
            raise JournalError("没有挂载日志：先 attach_journal")
        pending = (
            self.journal.iter_records(after_seq=self._journal_seq)
            if records is None else list(records)
        )
        applied = 0
        for record in pending:
            seq = record.get("seq")
            if not isinstance(seq, int):
                raise JournalError(f"日志记录缺少合法 seq：{record!r}")
            if seq <= self._journal_seq:
                continue
            if seq != self._journal_seq + 1:
                raise JournalError(
                    f"日志不连续：当前已应用到 seq={self._journal_seq}，"
                    f"下一条是 seq={seq}（缺记录，拒绝重放）"
                )
            apply_journal_record(self, record)
            self._journal_seq = seq
            applied += 1
        if applied:
            self._refresh_journal_batch()
            self._refresh_journal_scheduler()
        return applied

    # ------------------------------------------------------------------
    # 重启快照（snapshot，批次 A）
    # ------------------------------------------------------------------

    def snapshot_to(self, path: Optional[str] = None) -> str:
        """写重启快照（默认镜像声明的 snapshot_to）；返回落点。"""
        from .snapshot import snapshot, write_atomic

        target = path or self.snapshot_path
        if not target:
            raise ValueError("没有快照路径：传 path 或让镜像声明 snapshot_to")
        write_atomic(target, snapshot(self))
        self._last_snapshot_seq = self._journal_seq
        return target

    def restore_from(self, path: str) -> dict:
        """从重启快照恢复（替换调度/对话/内存/监听器/设备表）；返回快照文档。"""
        from .snapshot import read_snapshot, restore

        doc = read_snapshot(path)
        restore(self, doc)
        return doc

    def unmount_device(self, mount: str) -> None:
        """卸载设备：设备表与设备来源表一起删（包卸载、日志重放、宿主共用）。"""
        self.device_specs.pop(mount, None)
        self.mem.unmount(mount)

    @staticmethod
    def _respond_unexecuted(conv: Conversation, call_id):
        """协议层拒绝：写一条中性工具响应保住配对；call_id 不可用时无法配对，只记 stderr。"""
        if isinstance(call_id, str) and call_id:
            conv.user_batch.add_tool_response(_UNEXECUTED_TOOL_RESPONSE, call_id)

    def _process_return_calls(self, return_calls: list, conv: Conversation):
        registry = _instruction_registry()
        for rc in return_calls:
            cmd_type = rc.get("cmd_type", "")
            call_id = rc.get("call_id", "")
            args = rc.get("args", {})
            if cmd_type == "json_error":
                detail = args if isinstance(args, dict) else {}
                _report_tool_error(
                    conv,
                    f"工具调用 {detail.get('name', '')} 参数不是合法 JSON: {detail.get('error', '')}，原始参数: {detail.get('raw', '')}",
                )
                self._respond_unexecuted(conv, call_id)
                continue
            if cmd_type not in registry:
                # 未知工具名的唯一出口。两条解析路径（LMU 的 _return_calls_from_message、
                # PLM 的 _parse_tool_calls）都只把真名放进 cmd_type，查表在这里做一次。
                # 处置与 json_error 一致：报给运行者，回一条中性占位保住配对。
                _report_tool_error(
                    conv,
                    f"未知工具 {cmd_type}，参数: {json.dumps(args, ensure_ascii=False)}",
                )
                self._respond_unexecuted(conv, call_id)
                continue
            if not isinstance(call_id, str) or not call_id:
                # 没有 call_id 就无法写配对响应，也不能伪造一个（assistant 那侧的 id 不是它）
                _report_tool_error(conv, f"工具 {cmd_type} 缺少可配对的 call_id，跳过")
                continue
            if not conv.caps.tool_allowed(cmd_type):
                # 能力边界：schema 与 PLM 注入已裁剪，这里再兜一道实际调度
                _report_tool_error(
                    conv,
                    f"工具 {cmd_type} 未授权给本对话（mode={conv.caps.mode}），参数: "
                    f"{json.dumps(args, ensure_ascii=False)}",
                )
                conv.user_batch.add_tool_response(
                    f"Error: 工具 {cmd_type} 未授权给本对话", call_id
                )
                continue
            instr_cls = registry[cmd_type]
            error = _validate_instruction_args(instr_cls, args)
            if error is not None:
                _report_tool_error(conv, f"工具 {cmd_type} 参数不合法：{error}")
                conv.user_batch.add_tool_response(f"Illegal: {error}", call_id)
                continue
            instr = instr_cls(call_id, conv.cid, args)
            instr.execute(self, conv)

    def _on_conversation_finished(self, conv: Conversation, result: str):
        if conv.is_sub:
            parent = conv.parent
            if parent is not None and parent.cid not in self._finished:
                if conv.metadata.get("returned"):
                    _report_tool_error(conv, "亚对话已完成过一次，本次完成不再写回父对话")
                else:
                    # 亚对话紧耦合：完成即自动写回并唤醒父
                    call_id = conv.metadata.get("call_id", "direct_output")
                    parent.user_batch.add_tool_response(result, call_id)
                    self.note_effect(
                        "delivery", to_cid=parent.cid,
                        tool_responses=[{"content": result, "tool_call_id": call_id}],
                    )
                    conv.metadata["returned"] = True
                # 用完即删：返回通道只用一次，个体不再保留
                self._delete_conversation(conv.cid)
                if not self._is_waiting_for_return(parent):
                    # 同一轮可能声明多个 create_sub：全部返回后才唤醒父
                    self._resume(parent.cid)
                else:
                    # 父还在等别的亚对话：让出活跃权，继续调度（活跃指针不能留在已删个体上）
                    self._active_cid = None
                    self._pick_next_active()
                return
            # 父已关闭或不存在：没有返回通道，亚对话用完即删
            self._delete_conversation(conv.cid)
            self._active_cid = None
            self._pick_next_active()
            return
        # 交互结束：对话进入休眠（个体保留，等待事件/再次被调用）。
        # "finished" 是概念态（被内核级关闭指令关闭的对话），Core 调度状态只有活跃/休眠/就绪。
        if conv.cid not in self._dormant_cids:
            self._dormant_cids.append(conv.cid)
        self._active_cid = None
        self._pick_next_active()

    def _resume(self, cid: int):
        """将等待中的对话唤醒为活跃对话（亚对话完成唤醒父：同一调用链，延续预算不重置）。"""
        if cid in self._finished:
            return  # 已关闭的对话不可再被激活
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        self._active_cid = cid

    def _close_conversation(self, cid: int) -> None:
        """内核级关闭：对话个体结束，不需要被关闭对话的回应。

        效果：标记 finished、移出调度队列、释放 PLM 执行器（变量空间）。
        关闭后对话不可再调度、不可再被调用（send_instruction 会以 Error 拒绝）。
        关闭一个还没返回的亚对话时，给等待中的父对话补一条响应——否则父的那次
        `create_sub` 永远没有 tool 消息，父下次运行时历史配对不变量会被破坏。
        """
        if cid in self._finished:
            return
        self.note_effect("conversation_close", cid=cid)
        conv = self._conv_by_cid.get(cid)
        if conv is not None and conv.is_sub:
            # 亚对话用完即删：先给等待中的父补响应，再移除个体
            self._answer_parent_on_sub_close(conv)
            self._delete_conversation(cid)
            if self._active_cid == cid:
                self._active_cid = None
            return
        self._finished.add(cid)
        if cid in self._ready_cids:
            self._ready_cids.remove(cid)
        if cid in self._dormant_cids:
            self._dormant_cids.remove(cid)
        self.events.cancel_all_for_owner(cid)
        self._executors.pop(cid, None)  # 释放 PLM 状态（命名空间等）

    def _delete_conversation(self, cid: int) -> None:
        """亚对话用完即删：移除个体与全部内核引用（调度队列、监听器、执行器、finished 标记）。

        级联：以它为父的亚对话一并删除（它们已经没有返回通道）。create_cmd 子对话
        是独立个体（is_sub=False），不在此列。
        """
        self.note_effect("conversation_delete", cid=cid)
        for child in list(self._conv_by_cid.values()):
            if child.is_sub and child.parent is not None and child.parent.cid == cid:
                self._delete_conversation(child.cid)
        conv = self._conv_by_cid.pop(cid, None)
        if conv is None:
            return
        for queue in (self._ready_cids, self._dormant_cids):
            while cid in queue:
                queue.remove(cid)
        self._finished.discard(cid)
        self.events.cancel_all_for_owner(cid)
        self._executors.pop(cid, None)

    def _answer_parent_on_sub_close(self, sub: Conversation,
                                    reason: Optional[str] = None) -> None:
        """亚对话在返回结果前被关闭/跳过：给还活着的父对话补一条 Error 响应并唤醒它。"""
        if not sub.is_sub or sub.metadata.get("returned"):
            return
        parent = sub.parent
        if parent is None or parent.cid in self._finished:
            return
        call_id = sub.metadata.get("call_id", "direct_output")
        text = (f"Error: 亚对话（cid={sub.cid}）在返回结果前被关闭"
                if reason is None
                else f"Error: 亚对话（cid={sub.cid}）{reason}")
        parent.user_batch.add_tool_response(text, call_id)
        self.note_effect(
            "delivery", to_cid=parent.cid,
            tool_responses=[{
                "content": text,
                "tool_call_id": call_id,
            }],
        )
        sub.metadata["returned"] = True
        # 同一轮多个 create_sub：父还有别的待答调用时只补响应，等全部返回后再唤醒
        if parent.cid != self._active_cid and not self._is_waiting_for_return(parent):
            if parent.cid in self._dormant_cids:
                self._dormant_cids.remove(parent.cid)
            if parent.cid not in self._ready_cids:
                self._ready_cids.insert(0, parent.cid)  # 亚对话结束后父对话先跑
        _report_tool_error(sub, f"亚对话被跳过，已给父对话 cid={parent.cid} 补一条响应")

    def _pick_next_active(self):
        while self._ready_cids:
            picked = self._ready_cids.pop(0)
            conv = self._conv_by_cid.get(picked)
            if conv is not None and self._is_waiting_for_return(conv):
                # 等亚对话返回的对话不能先跑：退回休眠，响应到了再由 _resume 唤醒。
                # 它批次里的内容保留，等被唤醒时一起处理。
                if picked not in self._dormant_cids:
                    self._dormant_cids.append(picked)
                continue
            if picked in self._dormant_cids:
                self._dormant_cids.remove(picked)
            if not self._same_chain(self._active_cid, picked):
                self._activation_instructions = 0  # 调用链断开：新链重新给满预算
            self._active_cid = picked
            return
        self._active_cid = None

    def _same_chain(self, prev_cid, new_cid) -> bool:
        """亚对话与父对话是同一调用链（预算共享）：
        亚对话激活时续用父的剩余预算；亚对话完成、父恢复时也续用。
        create_cmd 子对话（is_sub=False）独立预算，不在此列。"""
        prev = self._conv_by_cid.get(prev_cid)
        new = self._conv_by_cid.get(new_cid)
        if prev is None or new is None:
            return False
        if new.is_sub and new.parent is not None and new.parent.cid == prev.cid:
            return True
        if prev.is_sub and prev.parent is not None and prev.parent.cid == new.cid:
            return True
        return False

    def advance_conversation(self):
        self._turn_events.clear()
        self._turn_effects.clear()  # 只记录本轮；异常回合的效果随下一轮一起丢弃
        conv = self._conv_by_cid[self._active_cid]
        executor = None
        attempt = 0
        history_delta: List[dict] = []
        while True:
            try:
                # 取 para 与选执行器也放进 try：它们会因为配置错配抛错，
                # 原先在 try 之外，逃出去连监测帧都没有（问题 1）。
                para = self._get_para(conv)
                executor = self._get_executor(conv, para)
                before_count = len(conv.messages)
                result, return_calls, _ = executor.exec(conv, para)
                # 外部调用成功：该对话的连续失败计数与最近错误一并清零。
                conv.api_error_streak = 0
                conv.last_api_error = None
                # 本轮写进历史的"实际消息"（消费进历史的批次 + 新的 assistant 消息）。
                # 日志按实际消息记而不是按生成它的效果推导：before 事件可以改写 payload。
                history_delta = [
                    message_to_dict(m) for m in conv.messages[before_count:]
                ]
                conv.user_batch.clear()

                self._recording = True
                try:
                    if return_calls:
                        self._process_return_calls(return_calls, conv)
                        self._activation_instructions += len(return_calls)
                        if conv.cid in self._finished or conv.cid not in self._conv_by_cid:
                            # 内核级关闭或亚对话用完即删：本轮结束后选择下一个活跃对话
                            self._active_cid = None
                            self._pick_next_active()
                        elif conv.cid in self._dormant_cids:
                            # 对话自己让出（等待返回/子完成）
                            self._pick_next_active()
                        elif self._activation_instructions >= self._instruction_budget:
                            # 核心中断：工具返回已加入 batch、状态完整；让出执行权，进就绪队列末尾
                            self._activation_instructions = 0  # 链被中断，重新计数
                            if conv.cid not in self._ready_cids:
                                self._ready_cids.append(conv.cid)
                            self._pick_next_active()
                    else:
                        self._on_conversation_finished(conv, result)
                finally:
                    self._recording = False
            except APIError as e:
                # 外部失败：先分类并记帧。失败时历史与输入批次都没动过，所以重试
                # 同一轮是干净的。宿主装了 api_error_policy 时走宿主策略（旧语义）；
                # 没装时走内置策略：余额不足暂停、上下文超限跳过对话、其余重试。
                attempt += 1
                category, detail = self._classify_api_error(e)
                self._remember_api_error(conv, detail)
                self.monitor.record(self, error=e, last_call=self._last_call_for(executor))
                if self.api_error_policy is not None:
                    if self._resolve_api_error(e, conv, attempt, executor) == "exit":
                        raise RunnerStop(f"API 调用失败，运行者选择退出：{e}") from e
                    continue
                if category == INSUFFICIENT_BALANCE:
                    self._pause_for_api_error(
                        category,
                        f"API 调用失败：余额不足，执行已暂停（{e}）",
                        cid=conv.cid,
                    )
                    return
                if category == CONTEXT_LIMIT:
                    self._skip_conversation_for_context_limit(conv)
                    return
                conv.api_error_streak += 1
                limit = self._api_error_retry_limit()
                if conv.api_error_streak > limit:
                    self._pause_for_api_error(
                        category,
                        f"API 调用连续失败 {conv.api_error_streak} 次"
                        f"（重试上限 {limit}，超过后暂停），执行已暂停（{e}）",
                        cid=conv.cid,
                    )
                    return
                continue
            except Exception as e:
                self.monitor.record(self, error=e, last_call=self._last_call_for(executor))
                raise
            else:
                self.monitor.record(self, last_call=self._last_call_for(executor))
                self._commit_journal_record(conv, history_delta)
                self._maybe_periodic_snapshot()
                return

    def _last_call_for(self, executor):
        """最近一次 API 调用的全文；本轮事件挂在它上面给监测帧。"""
        last_call = getattr(executor, "last_call", None)
        if last_call is not None and self._turn_events:
            last_call = {**last_call, "events": list(self._turn_events)}
        return last_call

    def _resolve_api_error(self, error: APIError, conv: Conversation, attempt: int, executor) -> str:
        """问运行者 API 失败怎么办，返回 retry / reload / exit。

        没装回调、回调自己出错、返回未知动作，一律按退出处理：停下等一个不会来的
        选择比退出更糟。策略的落点是宿主（终端提示见 avm/runner_prompt.py）。
        """
        policy = self.api_error_policy
        if policy is None:
            print(f"[AVM] API 调用失败且没有安装 api_error_policy，按退出处理：{error}", file=sys.stderr)
            return "exit"
        try:
            action = policy(error, attempt)
        except Exception as e:
            print(
                f"[AVM] api_error_policy 自身异常（{type(e).__name__}: {e}），按退出处理",
                file=sys.stderr,
            )
            return "exit"
        if action not in ("retry", "reload", "exit"):
            print(f"[AVM] api_error_policy 返回未知动作 {action!r}，按退出处理", file=sys.stderr)
            return "exit"
        if action == "reload":
            reset = getattr(executor, "reset_client", None)
            if callable(reset):
                reset()
        return action

    # ------------------------------------------------------------------
    # 内置 API 失败策略
    # ------------------------------------------------------------------

    def _classify_api_error(self, error: APIError):
        """取 (category, detail)：外部执行器已经分类的直接用，否则在这里补分类。

        LMU 在抛错前把分类放进 APIError；其它执行器（镜像自定义 PLM、测试替身）
        可能只抛一个普通 APIError，这里按同样的纯函数再判一次。
        """
        category = getattr(error, "category", None)
        detail = getattr(error, "detail", None)
        if category in CATEGORIES:
            detail = dict(detail or {})
            detail.setdefault("category", category)
            detail.setdefault("message", str(error))
            detail.setdefault("status_code", getattr(error, "status_code", None))
            detail.setdefault("context_limit", None)
            return category, detail
        return classify_api_error(
            error, status_code=getattr(error, "status_code", None)
        )

    def _remember_api_error(self, conv: Conversation, detail: dict) -> None:
        """把分类信息挂到对话运行态（不写内存树）；成功一轮后由调用方清掉。

        只保留设备契约里的五个字段：外部执行器给的 detail 可能带额外键，设备面
        不因此漂移。
        """
        conv.last_api_error = {
            "category": detail.get("category"),
            "message": detail.get("message", ""),
            "status_code": detail.get("status_code"),
            "context_limit": detail.get("context_limit"),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def _api_error_retry_limit(self) -> int:
        """连续失败达到这个次数就暂停；运行期改 Core.api_error_max_retries 立即生效。"""
        limit = getattr(self, "api_error_max_retries", 5)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            return 5
        return limit

    def _pause_for_api_error(self, category: str, reason: str,
                             cid: Optional[int] = None) -> None:
        """暂停执行：run() 在下一个安全点返回，宿主持有恢复权；不关对话、不抛异常。"""
        self.paused = True
        self.pause_category = category
        self.pause_reason = reason
        self._paused_cid = cid

    def clear_api_pause(self) -> None:
        """清除 API 暂停并重置本次重试计数（宿主 resume 时调用）。"""
        self.paused = False
        self.pause_category = None
        self.pause_reason = None
        cid, self._paused_cid = self._paused_cid, None
        if cid is None:
            return
        conv = self._conv_by_cid.get(cid)
        if conv is not None:
            conv.api_error_streak = 0

    def _skip_conversation_for_context_limit(self, conv: Conversation) -> None:
        """上下文超限：本轮跳过该对话（不关闭、不重试、不消费输入），继续调度其他对话。

        跳过不是结束：错误信息挂在对话运行态（`$MEM.conversations.<cid>.context`），
        回收/总结是软件任务，由 OS 软件读设备信息后决定（裁剪上下文、总结、或关闭）。
        普通对话回休眠等待再次被唤醒；亚对话不能留在等待链外——给父补一条
        Error 响应后按"用完即删"删除个体。
        """
        cid = conv.cid
        conv.api_error_streak = 0
        if conv.is_sub:
            self._answer_parent_on_sub_close(conv, reason="上下文超限，已跳过该亚对话")
            self._delete_conversation(cid)
            if self._active_cid == cid:
                self._active_cid = None
            self._pick_next_active()
            return
        while cid in self._ready_cids:
            self._ready_cids.remove(cid)
        if (cid in self._conv_by_cid and cid not in self._finished
                and cid not in self._dormant_cids):
            self._dormant_cids.append(cid)
        if self._active_cid == cid:
            self._active_cid = None
        self._pick_next_active()

    def _get_executor(self, conv: Conversation, para: MetaDict):
        """按对话的程序类型选择执行器（由 Core 管理，不挂在对话上）。
        Python 对话：model 必须在内建 PLM_REGISTRY 或本核的 self.plms 中，
        否则报错（不静默回退）；LLM 对话：用 core.lmu（model 交给 API 校验）。"""
        if conv.cid in self._executors:
            return self._executors[conv.cid]
        model = para.get("model")
        plm_service = self.services.plm
        if conv.is_python:
            if not isinstance(model, str) or model not in plm_service.available(self.plms):
                raise ValueError(
                    f"Python 对话 {conv.cid} 的 model {model!r} 不在 PLM_REGISTRY 中"
                    f"（可用: {plm_service.available(self.plms)}）"
                )
            executor = plm_service.create_executor(model, extra=self.plms)
            self._executors[conv.cid] = executor
            self._warm_restored_python_executor(executor, conv)
            return executor
        return self.lmu

    def _warm_restored_python_executor(self, executor, conv: Conversation) -> None:
        """恢复出来的 Python 对话：重建执行器后重放 system 消息、把已有历史标记为已处理。

        PLM 的命名空间不序列化（计划 A5 待定夺第 1 项的默认取舍）：跨轮状态必须由
        程序自己落在 `$MEM`；执行器重建后 setup 代码要再执行一遍，而历史里的 user/tool
        消息不能再执行（否则副作用重复发生）。新对话（历史只有 system）不走这里。
        """
        if conv.cid not in self._warmup_needed:
            return
        self._warmup_needed.discard(conv.cid)
        messages = conv.to_api_messages()
        executor.warm_restore(messages, {"tools": _build_tools(conv), "imports": _build_imports(conv)})

    def _get_para(self, conv: Conversation):
        """按对话自己的 para_ref 读取模型调用参数（缺省用 Core.para_ref）"""
        para_ref = getattr(conv, "para_ref", None) or self.para_ref
        try:
            para = self.unwrap(para_ref, for_llm=False)
            if not isinstance(para, MetaDict) or (para.get_ctrl() or {}).get("type") != "para":
                # 类型检查补全：指向的节点不是 para 节点时不再静默使用；回退默认参数并提示运行者
                _report_tool_error(
                    conv,
                    f"para_ref {para_ref} 不是 ctrl.type='para' 的 dict 节点，回退默认参数（model=gpt-4）",
                )
                para = MetaDict(data={"model": "gpt-4"})
        except VMMemoryError:
            para = MetaDict(data={"model": "gpt-4"})
        return para

    def publish_memory_after(self, event_type: str, conv: Conversation, target: str,
                             payload: dict, result: Optional[dict] = None,
                             follow_target: Optional[str] = None, space=None) -> None:
        """供指令集发布 memory/device after 事实事件（计划 S1）。

        instructions.py 不 import Core；指令经 core 对象调用本方法，Core 内的
        `_publish_memory_after` 保持可被测试替换（monkeypatch avm.core 上的名字）。
        """
        return _publish_memory_after(self, event_type, conv, target, payload,
                                     result=result, follow_target=follow_target,
                                     space=space)

    @property
    def lmu(self):
        """兼容名：LLM 服务执行器（计划 S5 保留属性，测试与 monitor 零修改）。"""
        return self.services.llm

    @lmu.setter
    def lmu(self, value):
        self.services.llm = value

    def build_conversation(self, system_ref: str, para_ref: str, parent: Conversation,
                           is_sub: bool, space=None) -> Conversation:
        """供指令集构造子对话（创建与能力派生留在 Core）。

        instructions.py 不 import Core；CreateInstruction/CreateSubInstruction 经
        core 对象调用这个方法（计划 S1）。
        """
        return _build_conversation(self, system_ref, para_ref, parent=parent,
                                   is_sub=is_sub, space=space)

    def start(self, system: str, user: str, para_ref: str = "$MEM.model_params") -> Conversation:
        conv = Conversation(
            messages=[SystemMessage(content=system), UserMessage(content=user)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=True,
        )
        self._register(conv)
        conv.para_ref = para_ref
        self._ready_cids.append(conv.cid)
        return conv

    def ensure_operator_conversation(self, para_ref: Optional[str] = None) -> Conversation:
        """确保保留对话（cid = PANEL_CID）存在并返回它；不绑定执行器。

        - 身份固定：cid 是作者规定的保留值，不参与正常分配，也不依赖创建顺序；
        - 初始进休眠队列、不进就绪队列：它不主动抢占，只有中断能激活它；
        - 装载时、日志重放遇到它的记录时、控制台绑定时都走这里；
        - 执行器由 bind_console_conversation 在宿主启动后注入（plm.operator 需要通道，
          注册表无参构造出来的实例不能用）。
        """
        conv = self._conv_by_cid.get(PANEL_CID)
        if conv is not None:
            # -1 是完整权限、非创建时固定的：每次取用时按当前目录刷新。
            self.refresh_panel_caps()
            return conv
        conv = Conversation(
            messages=[SystemMessage(content=_OPERATOR_SYSTEM_PROMPT)],
            cid=0,
            is_sub=False,
            parent=None,
            is_root=False,
            # 宿主标记：控制台持有的保留对话对普通对话隐藏。判定只读 attrs。
            attrs={"host": "console", "hidden": "1"},
        )
        self._register(conv, cid=PANEL_CID)
        conv.para_ref = para_ref or self.para_ref
        conv.is_python = True
        self._dormant_cids.append(conv.cid)
        return conv

    def bind_console_conversation(
        self,
        channel,
        para_ref: Optional[str] = None,
    ) -> Conversation:
        """把控制台通道绑到保留对话；没有就先建出来，返回该对话。"""
        conv = self.ensure_operator_conversation(para_ref)
        conv.para_ref = para_ref or self.para_ref
        self._executors[conv.cid] = self.services.plm.create_executor("plm.operator", channel=channel)
        return conv

    # ------------------------------------------------------------------
    # 安全点自检（六条具名不变量）
    # ------------------------------------------------------------------

    def _check_invariants(self) -> None:
        """六条具名不变量的结构自检。违反即 KernelPanic：出现说明某条边界少了一条判定。

        只在安全点调用（run 的每轮推理之前）。成本是常数级遍历，加上活跃对话的一小段历史。
        """
        # 1 调度互斥：就绪与休眠不相交，就绪队列无重复
        both = set(self._ready_cids) & set(self._dormant_cids)
        if both:
            raise KernelPanic(f"调度互斥被破坏：cid {sorted(both)} 同时在就绪与休眠队列")
        if len(set(self._ready_cids)) != len(self._ready_cids):
            raise KernelPanic(f"就绪队列有重复 cid：{self._ready_cids}")
        # 6 预算计数：非负整数
        if not isinstance(self._activation_instructions, int) or self._activation_instructions < 0:
            raise KernelPanic(f"指令预算计数不合法：{self._activation_instructions!r}")
        # 2 活跃有来源
        if self._active_cid is not None:
            conv = self._conv_by_cid.get(self._active_cid)
            if conv is None:
                raise KernelPanic(f"活跃 cid {self._active_cid} 不在对话表里")
            if self._active_cid in self._finished:
                raise KernelPanic(f"活跃 cid {self._active_cid} 已 finished")
            # 3 历史配对、4 值域：只查正要发出去的那条对话
            self._check_pairing(conv)
            self._check_value_domain(conv)
        # 5 设备表一致
        self._check_device_table()

    @staticmethod
    def _pending_call_ids(conv: Conversation) -> List[str]:
        """还没拿到 tool 响应的调用 id（历史与待发批次合并后比对）。

        目前只有 `create_sub` 会留下待答调用：父对话在等亚对话返回。
        """
        messages = conv.messages
        last = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and getattr(messages[i], "tool_calls", None):
                last = i
                break
        if last is None:
            return []
        wanted = []
        for tc in messages[last].tool_calls or []:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if isinstance(call_id, str) and call_id:
                wanted.append(call_id)
        got = {m.tool_call_id for m in messages[last + 1:] if m.role == "tool"}
        got |= {r.tool_call_id for r in conv.user_batch.tool_responses}
        return [call_id for call_id in wanted if call_id not in got]

    def _is_waiting_for_return(self, conv: Conversation) -> bool:
        """对话是否在等工具返回（亚对话）。等返回的对话不能被外部消息唤醒运行。"""
        return bool(self._pending_call_ids(conv))

    @staticmethod
    def _check_pairing(conv: Conversation) -> None:
        """调用 API 前，带 tool_calls 的 assistant 消息要与 tool 消息一一配对。

        待发批次（user_batch）也算数：工具响应先入批次，下一轮推理才写进历史。
        没有 id 的工具调用两边都配不上，不在这里判定。
        """
        messages = conv.messages
        last = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and getattr(messages[i], "tool_calls", None):
                last = i
                break
        if last is None:
            return
        wanted = []
        for tc in messages[last].tool_calls or []:
            call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if isinstance(call_id, str) and call_id:
                wanted.append(call_id)
        if not wanted:
            return
        got = [m.tool_call_id for m in messages[last + 1:] if m.role == "tool"]
        got += [r.tool_call_id for r in conv.user_batch.tool_responses]
        if sorted(wanted) != sorted(got):
            raise KernelPanic(
                f"对话 {conv.cid} 的历史配对被破坏：assistant 声明 {sorted(wanted)}，"
                f"现存 tool 消息 {sorted(got)}"
            )

    @staticmethod
    def _check_value_domain(conv: Conversation) -> None:
        """进入 LLM 可见通道的值必须是字符串。查待发批次与最后一条消息。"""
        for resp in conv.user_batch.tool_responses:
            if not isinstance(resp.content, str) or not isinstance(resp.tool_call_id, str):
                raise KernelPanic(
                    f"对话 {conv.cid} 的工具响应值域不合法："
                    f"content={type(resp.content).__name__} tool_call_id={type(resp.tool_call_id).__name__}"
                )
        for content in conv.user_batch.user_contents:
            if not isinstance(content, str):
                raise KernelPanic(f"对话 {conv.cid} 的待发用户内容不是字符串：{type(content).__name__}")
        if conv.messages:
            content = conv.messages[-1].content
            if content is not None and not isinstance(content, str):
                raise KernelPanic(
                    f"对话 {conv.cid} 最后一条消息的内容不是字符串：{type(content).__name__}"
                )

    def _check_device_table(self) -> None:
        """设备表与挂载点一致：每个挂载地址有链接节点；每个标识符至少有一个挂载地址。"""
        mounted_ids = set()
        for address, device in list(self.mem._devices.items()):
            mounted_ids.add(getattr(device, "device_id", None) or address)
            node = self.mem._data
            for seg in address.split("."):
                node = node.get(seg) if isinstance(node, (dict, MetaDict)) else None
                if node is None:
                    break
            if not isinstance(node, LinkNode):
                raise KernelPanic(f"设备表不一致：{address} 没有挂载点链接节点")
        for device_id in list(self.mem._device_ids):
            if device_id not in mounted_ids:
                raise KernelPanic(f"设备表不一致：标识符 {device_id!r} 没有挂载地址")

    # ------------------------------------------------------------------
    # 模型目录（$MEM.model_catalog）
    # ------------------------------------------------------------------

    def install_model_catalog(self) -> None:
        """安装/补齐模型目录的静态部分。幂等；镜像或检查点里已有的字段不覆盖。

        整树替换（load_image）之后必须重装一次，否则镜像会把启动节点冲掉。
        静态部分不依赖网络：PLM 注册表、para 能用的 API 参数、必填字段。
        """
        root = self.mem._data
        catalog = root.get(MODEL_CATALOG_KEY)
        if not isinstance(catalog, MetaDict):
            catalog = MetaDict(data={}, metadata=_CATALOG_META)
            root[MODEL_CATALOG_KEY] = catalog
        llm = catalog.get("llm")
        if not isinstance(llm, MetaDict):
            llm = MetaDict(data={}, metadata="可用的 LLM 模型；source 说明这份清单的来源")
            catalog["llm"] = llm
        llm.setdefault("source", "unavailable")
        llm.setdefault("required", "model")
        params = llm.get("params")
        if not isinstance(params, MetaDict):
            params = MetaDict(data={}, metadata="可以写进 para 的 API 参数（model 之外全部可选）")
            llm["params"] = params
        llm_service = self.services.llm
        numeric_params = set(llm_service.numeric_param_names())
        for name in llm_service.api_param_names():
            if name not in params:
                params[name] = "数值型" if name in numeric_params else "字符串/原样"
        if not isinstance(llm.get("models"), MetaList):
            llm["models"] = MetaList(data=[], metadata="上游模型清单；每项一个 id")
        if not isinstance(catalog.get("plm"), MetaList):
            catalog["plm"] = MetaList(
                data=self.services.plm.catalog_entries(self.plms),
                metadata="PLM 注册表：本地实现，不需要网络",
            )


    def refresh_model_catalog(self, timeout: float = 10.0) -> dict:
        """拉一次上游 `/models` 写进目录。任何失败都收敛成 source="unavailable"，不抛。

        防御按讨论案第五节：自己校验信封、除 id 外全部按可选字段读、忽略未知字段。
        清单只当参考——取不到不拦启动，也不拦模型名（后者见 _check_model_namespace）。
        """
        self.install_model_catalog()
        llm = self.mem._data[MODEL_CATALOG_KEY]["llm"]

        def _unavailable(reason: str) -> dict:
            llm["source"] = "unavailable"
            llm["reason"] = reason
            print(
                f"[AVM] 模型目录：没能取到上游清单（{reason}），本轮不校验 LLM 模型名",
                file=sys.stderr,
            )
            return {"source": "unavailable", "reason": reason}

        try:
            # 用原始响应自己解析：SDK 默认模式对非 OpenAI 信封抛的是 AttributeError/TypeError，
            # 分不清"这家不是 OpenAI 形状"与"SDK 有 bug"；自己校验信封才有确定的诊断。
            payload = self.lmu.fetch_models_raw(timeout)
        except Exception as e:
            return _unavailable(f"{type(e).__name__}: {e}")
        if (
            not isinstance(payload, dict)
            or payload.get("object") != "list"
            or not isinstance(payload.get("data"), list)
        ):
            return _unavailable("上游 /models 的信封不是 OpenAI 形状（object=list 且 data 是数组）")
        entries = []
        skipped = 0
        for item in payload["data"]:
            if not isinstance(item, dict):
                skipped += 1
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                skipped += 1
                continue
            entry = MetaDict(data={"id": model_id}, metadata="上游模型 id")
            for key in ("name", "owned_by", "context_window", "max_output_tokens",
                        "input_modalities", "output_modalities"):
                value = item.get(key)
                if value is None:
                    continue
                # 内存约定：para 与目录里的值都以 str 存储
                entry[key] = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            effort = item.get("effort")
            if isinstance(effort, dict):
                levels = effort.get("supported_levels")
                if isinstance(levels, list):
                    entry["reasoning_effort_levels"] = ", ".join(str(v) for v in levels)
                default_level = effort.get("default_level")
                if isinstance(default_level, str) and default_level:
                    entry["default_effort_level"] = default_level
            entries.append(entry)
        llm["models"] = MetaList(data=entries, metadata="上游模型清单；每项一个 id")
        llm["source"] = "api"
        if "reason" in llm:
            del llm["reason"]
        print(
            f"[AVM] 模型目录：从上游取到 {len(entries)} 个模型"
            + (f"（跳过 {skipped} 条形状不合法的条目）" if skipped else ""),
            file=sys.stderr,
        )
        self._warn_unlisted_para_model(llm)
        return {"source": "api", "count": len(entries), "skipped": skipped}

    def _warn_unlisted_para_model(self, llm: MetaDict) -> None:
        """默认 para 的 model 不在清单里时提醒运行者（例如用着已退役的别名）。"""
        try:
            para = self.unwrap(self.para_ref, for_llm=False)
        except Exception:
            return
        model = para.get("model") if isinstance(para, MetaDict) else None
        if not isinstance(model, str) or not model:
            return
        listed = {e.get("id") for e in (llm.get("models") or []) if isinstance(e, MetaDict)}
        if model not in listed:
            print(
                f"[AVM] 模型目录：para {self.para_ref} 的 model {model!r} 不在上游清单里"
                f"（可能是拼写错误或已退役的别名）",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # 内核转储（诊断快照，不是恢复输入）
    # ------------------------------------------------------------------

    def core_dump_path(self) -> str:
        """默认转储落点：<镜像目录>/out/<镜像名>.core.json；无镜像上下文时用 cwd 下的 core.<pid>.json。"""
        if self.image_dir and self.image_name:
            return os.path.join(self.image_dir, "out", f"{self.image_name}.core.json")
        return os.path.join(os.getcwd(), f"core.{os.getpid()}.json")

    def dump_core(
        self,
        path: Optional[str] = None,
        error: Optional[BaseException] = None,
        *,
        messages_per_conversation: int = 200,
        max_frames: int = 200,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> Optional[str]:
        """把死亡现场写成一个 JSON，返回写入路径；失败返回 None。

        诊断快照，**不是恢复输入**（恢复依据是回合日志，快照是它的缓存）。全程不抛：任何一段序列化失败
        都降级为占位——转储绝不能成为新的内核退出路径。
        """
        target = path or self.core_dump_to or self.core_dump_path()
        try:
            doc = self._build_core_dump(error, messages_per_conversation, max_frames)
            text = json.dumps(doc, ensure_ascii=False, indent=2)
            if len(text.encode("utf-8")) > max_bytes:
                # 内存树按作者决定不截断，先砍对话正文，再砍对话本身（旧的先砍）
                for brief in doc["conversations"]:
                    if brief.get("messages"):
                        brief["messages"] = []
                        brief["messages_truncated"] = True
                        doc["truncated"]["conversations"].append(brief["cid"])
                text = json.dumps(doc, ensure_ascii=False, indent=2)
            if len(text.encode("utf-8")) > max_bytes:
                for brief in doc["conversations"][:-1]:
                    brief.pop("messages", None)
                    if brief["cid"] not in doc["truncated"]["conversations"]:
                        doc["truncated"]["conversations"].append(brief["cid"])
                text = json.dumps(doc, ensure_ascii=False, indent=2)
            doc["truncated"]["total_bytes"] = len(text.encode("utf-8"))
            text = json.dumps(doc, ensure_ascii=False, indent=2)
            os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
            return target
        except Exception as e:
            print(
                f"[AVM] 内核转储写入失败（{type(e).__name__}: {e}），原始错误不受影响",
                file=sys.stderr,
            )
            return None

    def _build_core_dump(self, error: Optional[BaseException], limit: int, max_frames: int) -> dict:
        truncated = {"conversations": [], "frames": False, "total_bytes": 0}
        err = None
        if error is not None:
            try:
                tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            except Exception as e:  # traceback 都可能失败：降级
                tb = f"<traceback 不可用：{type(e).__name__}: {e}>"
            err = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": tb,
                "panicked": isinstance(error, KernelPanic),
            }
        frames = []
        try:
            all_frames = self.monitor.frames()
            if len(all_frames) > max_frames:
                truncated["frames"] = True
                all_frames = all_frames[-max_frames:]
            frames = [dataclasses.asdict(f) for f in all_frames]
        except Exception as e:
            frames = [{"error": f"帧序列化失败：{type(e).__name__}: {e}"}]
        try:
            listeners = [dict(x) for x in self.events.listeners]
        except Exception as e:
            listeners = [{"error": f"{type(e).__name__}: {e}"}]
        try:
            memory = self.mem.to_image_dict()
        except Exception as e:
            memory = {"kind": "dict", "value": {}, "meta": f"<内存树序列化失败：{type(e).__name__}: {e}>"}
        return {
            "format": "avm.core_dump",
            "version": 1,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "image": os.path.join(self.image_dir or "", self.image_name or "") or None,
            "host_flags": dict(self.host_flags or {}),
            "para_ref": self.para_ref,
            "note": "诊断快照，不是恢复输入；memory 段与 persist_to 同格式",
            "error": err,
            "scheduler": {
                "active": self._active_cid,
                "ready": list(self._ready_cids),
                "dormant": list(self._dormant_cids),
                "finished": sorted(self._finished),
                "instruction_budget": self._instruction_budget,
                "activation_instructions": self._activation_instructions,
            },
            "memory": memory,
            "security": {
                "roots": {name: list(target) for name, target in self.root_registry.items()},
            },
            "conversations": [
                self._conversation_dump(self._conv_by_cid[cid], limit)
                for cid in sorted(self._conv_by_cid)
            ],
            "listeners": listeners,
            "devices": [
                {"mount": mount, "type": type(dev).__name__, "id": getattr(dev, "device_id", None)}
                for mount, dev in list(self.mem._devices.items())
            ],
            "interrupts": {"last_dispatch": list(self.interrupts.last_dispatch)},
            "frames": frames,
            "truncated": truncated,
        }

    def _conversation_dump(self, conv: Conversation, limit: int) -> dict:
        """单个对话的现场：状态、身份、历史（保最后 limit 条）、待发批次。"""
        state = "?"
        if conv.cid == self._active_cid:
            state = "active"
        elif conv.cid in self._finished:
            state = "finished"
        elif conv.cid in self._ready_cids:
            state = "ready"
        elif conv.cid in self._dormant_cids:
            state = "dormant"
        try:
            messages = conv.to_checkpoint().get("messages", [])
        except Exception as e:
            messages = [{"error": f"历史序列化失败：{type(e).__name__}: {e}"}]
        total = len(messages)
        brief = {
            "cid": conv.cid,
            "identity": conv.identity,
            "name": getattr(conv, "name", None),
            "state": state,
            "is_python": conv.is_python,
            "is_sub": conv.is_sub,
            "parent": conv.parent.cid if conv.parent is not None else None,
            "para_ref": conv.para_ref,
            "metadata": dict(conv.metadata or {}),
            "caps": caps_to_dict(conv.caps),
            "attrs": dict(conv.attrs or {}),
            "message_count": total,
            "messages": messages[-limit:] if limit else [],
            "messages_truncated": total > limit,
            "batch": {
                "tool_responses": [
                    {"content": r.content, "tool_call_id": r.tool_call_id}
                    for r in conv.user_batch.tool_responses
                ],
                "user_contents": list(conv.user_batch.user_contents),
            },
        }
        return brief

    # -- 操作面板生命周期 ------------------------------------------------

    def _maybe_periodic_snapshot(self) -> None:
        """每 snapshot_every 个已提交回合在安全点写一次定时快照。

        快照是缓存：写失败只警告，不改变运行；日志仍是持久依据。
        """
        every = self.snapshot_every
        if not every or not self.snapshot_path or self.journal is None:
            return
        if self._journal_seq - self._last_snapshot_seq < every:
            return
        try:
            self.snapshot_to()
        except Exception as e:
            print(
                f"[AVM] 定时快照写入失败（{type(e).__name__}: {e}）",
                file=sys.stderr,
            )

    def request_stop(self) -> None:
        """宿主请求停止：run() 在下一个安全点返回，交宿主写快照。

        只置标志、不动当前回合——半截回合不能拍快照。若对话正阻塞在设备读里，
        这一轮到不了安全点；那是宿主侧的阻塞态问题，不在这里处理。
        正常断电（OS 写 `$MEM.sys.power.state == "off"`）不走这里：宿主在安全点
        看到 off 后直接退出（见 `power_off_requested`）。本入口留给强制退出路径。
        """
        self._stop_requested = True

    def read_power_state(self):
        """读 OS 约定的断电节点 `$MEM.sys.power.state`；不存在/不是字符串返回 None。"""
        try:
            value = self.unwrap("$MEM.sys.power.state", for_llm=False)
        except VMMemoryError:
            return None
        return value if isinstance(value, str) else None

    def power_off_requested(self) -> bool:
        """安全点断电判定（边沿触发）：非 off → off 的一次跳变才算一次断电。

        快照与日志重放会把上一次正常关机写下的 off 一起恢复回来；把启动时的存量
        off 当成新请求会让 VM 一上电就退出。宿主只响应本次上电之后 OS 新写入的 off。
        VM 不写 clean、不改 `$MEM.sys.boot.*`（脏关闭判定属于 OS 计划的数据面）。
        """
        state = self.read_power_state()
        previous = self._power_state_seen
        self._power_state_seen = state
        return state == "off" and previous != "off"

    def _check_journal_seq(self) -> None:
        """安全点断言：日志末 seq 不得大于内存已提交 seq。

        正常提交时两者相等（append 返回的 seq 立刻写进 `_journal_seq`）；恢复路径
        必须先把快照 seq 之后的日志尾部重放完（boot 负责）。大于说明日志里有未重放
        的记录，继续跑会把内存当成比日志新——拒绝继续。
        """
        if self.journal is None:
            return
        last = self.journal.last_seq()
        if last > self._journal_seq:
            raise JournalError(
                f"安全点 seq 断言失败：日志末 seq={last} 大于内存已提交 seq="
                f"{self._journal_seq}；先把日志重放完再运行"
            )

    def _idle_now(self) -> bool:
        """安全点空闲判定：活跃为空、就绪为空、无待派发中断。

        活跃对话（包括阻塞在操作员通道上等动作的面板对话）不算空闲——它的回合
        还没到安全点。宿主 console 把面板对话注册成 `idle` 中断向量：没有对话可跑
        时执行权交给面板，面板在自己的推理里阻塞等操作员，这就是系统的空闲态。
        严格调度模式下空闲是严重错误（guest 活体错误），见 `run`。
        """
        return (
            self._active_cid is None
            and not self._ready_cids
            and self.interrupts.pending_count == 0
        )

    def _panic_idle(self, message: str):
        """空闲 = 严重错误：记监测帧后抛 KernelPanic（宿主记录并退出）。"""
        error = KernelPanic(message)
        self.monitor.record(self, error=error)
        raise error

    def _wake_resumed_boot(self) -> None:
        """resume 后让声明的 boot 角色再跑一次，并把 last_exit 放进入口 payload。

        快照恢复不跑 init、也不自动运行 boot（冷启动由就绪队列驱动）。但 OS 的
        boot-check 需要知道这次是正常恢复还是脏关闭；VM 只提供 `recovery["last_exit"]`
        数据面，用中断把 boot 唤醒是宿主不代替 guest 发指令的既有规则。payload 的
        形状由 guest 程序类型决定（python：入口正文；LLM：角色名 + 数据行）。
        """
        mode = self.recovery.get("mode")
        resumed = mode == "resume" or (
            mode in (None, "auto") and self.recovery.get("source") == "snapshot"
        )
        if not resumed:
            return
        info = self.guest_roles.get("boot")
        if info is None:
            return
        last_exit = str(self.recovery.get("last_exit") or "unknown")
        entry = info.get("entry") or ""
        if info.get("kind") == "python":
            payload = "yield from boot(" + repr(last_exit) + ")"
        else:
            payload = entry + "\nlast_exit=" + last_exit
        self.interrupts.raise_("boot", payload=payload)

    def run(self):
        if not self._run_started:
            self._run_started = True
            self._pick_next_active()

            self.monitor.record_baseline(self)
            # 启动时的存量断电不是新请求：边沿基线从安全点第一次观察开始。
            self._power_state_seen = self.read_power_state()
            # resume：boot 角色由中断唤醒，宿主的事实 last_exit 随 payload 进入口。
            self._wake_resumed_boot()
        idle_retry = False
        idle_cid = None

        while True:
            # 安全点：上一轮推理（含该轮产生的全部指令）已执行完，是状态完整、可恢复
            # 的边界——预算中断用的是同一个边界。中断在这里夺取执行权。
            # 中断表为空时这一行没有副作用。
            self.interrupts.dispatch()
            # API 暂停：宿主负责恢复。暂停检查放在中断派发之后，让已经挂起的
            # shutdown 等中断仍能在这个安全点落到接收方；对话本身不推进。
            if self.paused:
                return
            # 日志与内存的提交位置必须追平（恢复路径由 boot 保证）。
            self._check_journal_seq()
            try:
                self._check_invariants()
            except KernelPanic as e:
                self.monitor.record(self, error=e)
                raise
            if self._stop_requested:
                # 停在安全点：上一轮已完整提交，宿主可以拍快照后干净退出。
                return
            if self.power_off_requested():
                # 正常断电边界：OS 在回合里写下 off；宿主在安全点看到后写快照/persist
                # 并退出（宿主退出路径负责，不经 request_stop）。这是 VM 唯一读取的
                # OS 节点：断电寄存器，软件通过中断与普通写入维护它。
                return
            if self._active_cid is None and self._ready_cids:
                self._pick_next_active()
            if self._idle_now():
                # 空闲时把执行权交给注册的 idle 向量。宿主 console 把保留对话注册成
                # idle：面板对话的推理阻塞等操作员，交给它就是"系统空闲"；只有镜像
                # 声明的 guest idle 程序才需要入口正文（否则本轮不会开始）。
                vector_cid = self.interrupts.vectors.get("idle")
                guest_idle = self.guest_roles.get("idle")
                guest_vector = (
                    guest_idle is not None and guest_idle.get("cid") == vector_cid
                )
                if vector_cid is None:
                    if not self.strict_scheduler:
                        # 嵌入式宿主未接入生命周期管理（镜像没有声明 guest 程序，
                        # 宿主也没注册 idle 向量）：队列空了就是"跑完了"，run() 返回。
                        return
                    self._panic_idle(
                        "系统空闲，但镜像没有声明 idle guest 程序"
                        "（meta.guest_programs.idle），宿主也没有注册 idle 中断向量："
                        "空闲是 guest 活体错误"
                    )
                if guest_vector and idle_retry:
                    self._panic_idle(
                        "系统空闲：guest idle 处理程序本轮没有让系统离开空闲"
                        "（活跃为空、就绪为空、无待派发中断）"
                    )
                payload = guest_idle.get("entry") if guest_vector else None
                self.interrupts.raise_("idle", payload=payload)
                self.interrupts.dispatch()
                if self._active_cid is None and not self._ready_cids:
                    if self._idle_now():
                        self._panic_idle(
                            "idle 中断已派发，但没有任何对话可运行"
                            "（向量目标缺失或被关闭）"
                        )
                    # 派发期间又有别的挂起中断：回到循环顶部派发它。
                    continue
                idle_cid = vector_cid
                idle_retry = guest_vector
            elif self._active_cid is None and not self._ready_cids:
                continue  # 并发挂起的中断：回到循环顶部派发
            elif self._active_cid is not None and self._active_cid != idle_cid:
                # 空闲之后有别的对话拿到执行权：系统确实离开了空闲。
                idle_retry = False
            self.advance_conversation()

    def _canonical_ref_without_final(self, target, space=None):
        """canonical 化的变体：不跟随最终段的链接（删链接时裁决对象是链接节点）。"""
        try:
            split = self.mem.split_path(target, space=space)
        except Exception:
            return None
        if split[0] == "device":
            return ("device", split[1], tuple(split[2]))
        segments = list(split[1])
        if not segments:
            return ("memory", ())
        parent_ref = "$MEM" if len(segments) == 1 else "$MEM." + ".".join(segments[:-1])
        try:
            follow = self.mem.follow_link_target(parent_ref, space=space)
        except Exception:
            return ("memory", tuple(segments))
        if follow[0] == "memory":
            return ("memory", tuple(follow[1]) + (segments[-1],))
        return ("device", follow[1], tuple(follow[2]) + (segments[-1],))

    def canonical_ref(self, target, space=None, follow_final: bool = True):
        """把请求别名规范化为真实目标引用（跟随链接；不存在时取最长存在前缀）。

        follow_final=False：只跟随父路径上的链接，最终段保持原样（删除链接时用）。
        

        返回 ("memory", 真实段序列) / ("device", 挂载键, 剩余段)；
        无法解析返回 None。只用于事件匹配与权限判定，不展示给普通处理器。
        """
        if not isinstance(target, str) or not target.startswith("$"):
            return None
        if not follow_final:
            return self._canonical_ref_without_final(target, space=space)
        # 先尝试整条路径决议（会跟随链接与 LinkDevice 透明挂载）
        try:
            follow = self.mem.follow_link_target(target, space=space)
        except Exception:
            follow = None
        if follow is not None:
            if follow[0] == "memory":
                return ("memory", tuple(follow[1]))
            return ("device", follow[1], tuple(follow[2]))
        try:
            split = self.mem.split_path(target, space=space)
        except Exception:
            return None
        if split[0] == "device":
            return ("device", split[1], tuple(split[2]))
        segments = list(split[1])
        for i in range(len(segments), -1, -1):
            ref = "$MEM" if i == 0 else "$MEM." + ".".join(segments[:i])
            try:
                follow = self.mem.follow_link_target(ref, space=space)
            except Exception:
                continue
            suffix = tuple(segments[i:])
            if follow[0] == "memory":
                return ("memory", tuple(follow[1]) + suffix)
            return ("device", follow[1], tuple(follow[2]) + suffix)
        return ("memory", tuple(segments))

    def emit_device_event(self, device_path: str, target: str, payload: dict) -> dict:
        """设备自改事件入口（v1）。

        device_path 是内存设备挂载路径（不含 $MEM 前缀），例如 "io"。
        只支持 memory_write before 事件；target 必须位于该设备路径下。
        走正常 before 管道，并返回处理后的 event，设备据此决定是否执行自改。
        """
        if device_path not in self.mem._devices:
            raise VMMemoryError(f"设备 {device_path} 未挂载")
        split = self.mem.split_path(target)
        if len(split) != 3 or split[0] != "device" or split[1] != device_path:
            raise VMMemoryError(
                f"设备自改事件 target 必须属于 $MEM.{device_path}：{target}"
            )
        event = build_event(
            self,
            "memory_write",
            "before",
            target=target,
            payload=payload,
            source={"kind": "device", "path": "$MEM." + device_path},
            target_ref=("device", split[1], tuple(split[2])),
        )
        event = self.events.dispatch(event)
        self.events.deliver_notifies(event)
        return event

    def unwrap(self, value, for_llm=True, space=None):
        if not value.startswith("$"):
            return value
        # TODO(issue): 这里一次把整条引用按点号切分，设备拿到的是散开的段
        # （$MEM.AVMRepo.read."avm/core.py".0 → ["read", '"avm/core', 'py"', "0"]），
        # AVMRepo 因此要用引号把区间还原。是否改成逐步切分——逐段走内存树，命中设备时把
        # 剩余的原字符串整段交给设备、由设备自己定义子路径语法——现在不能确定；
        # 它与 split_path 的最长挂载键匹配、设备授权是同一件事，要一起想。
        value = [value[0], *value[1:].split(".")]
        if len(value) > 1 and value[1] in ("conv", "batch"):
            assert value[0] == "$"
            if space is not None and not space.has_mem():
                raise VMMemoryError(
                    f"引用根不可用：{value[1]!r}（本对话根表不含 MEM）"
                )
            try:
                cid = int(value[2])
            except (IndexError, ValueError):
                raise VMMemoryError(f"${value[1]}.<cid> 的 cid 必须是数字：{value[2:] or '（缺）'}")
            conv = self._conv_by_cid.get(cid)
            if conv is None or self.is_hidden_cid(cid):
                raise VMMemoryError(f"对话 {cid} 不存在：${value[1]}.{cid}")
            return conv if value[1] == "conv" else conv.user_batch
        else:
            return self.mem.unwrap(value, for_llm=for_llm, space=space)


def __getattr__(name):
    """兼容旧导入路径：LMU/PLMExecutor 已迁到服务模块（计划 S5）。"""
    if name == "LMU":
        from .llm import LMU
        return LMU
    if name == "PLMExecutor":
        from .plm import PLMExecutor
        return PLMExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

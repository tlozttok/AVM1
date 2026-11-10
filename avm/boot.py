"""加载区：宿主启动一个 AVM 内核的统一入口。

生命周期在这里显式分三段，而不是藏在一次调用里（作者 2026-10-16）：

1. **创建空核心**：`Core()`；
2. **加载镜像**：`avm.image.load_image(core, image_path)`——解析 mem / devices / init /
   persist_to，并在整树替换之后重装启动自带的模型目录（`$MEM.model_catalog`）；
3. **启动核心**：`start_core(core)`——刷新模型目录（拉一次上游 `/models`，失败只警告）、
   装运行者策略（API 失败时问运行者的终端提示）。

之后把 Core 交给宿主自己 `run()`。宿主通常只调 `boot()`，它把三段串起来；要单独测试或
分步使用某一段时，直接调对应的函数。

**三种启动模式**（计划 `plan-vm-lifecycle-2026-11-10`）：

- `resume`：最近重启快照 + 日志尾部重放；恢复对话/监听器/调度/`_next_cid`；不跑 init；
  续当前日志代际。没有可读快照拒绝启动（日志不能单独重建对话）。
- `reboot`：世界 = 最近快照的世界段（内存/安全配置/para/设备来源表）+ 日志尾部里的
  世界类效果重放（对话态效果跳过）；对话/监听器/调度全部丢弃，init 重跑，`_next_cid`
  来自快照；归档旧日志、开新代际。没有可读快照拒绝启动并提示 `reset`。
- `reset`：原始镜像；归档旧日志（与旧快照）、开新代际。

不传 `mode` 时保持旧行为（`mode="auto"` 的语义）：镜像声明了 `snapshot_to` 且快照可读
就用它恢复，否则冷启动 + 日志重放。镜像声明了 `meta.guest_programs` 时，冷启动在 init
之后创建 guest 程序对话并注册中断向量，resume 按 `attrs["guest_role"]` 找回。

装载期的错误（坏镜像、缺 para、设备加载失败）照旧抛给调用方，由宿主决定怎么报——加载区
只负责顺序与默认值，不吞错。日志与内存提交位置不一致时抛 `JournalError`（面向运行者）。
"""

import json
import os
import re
import sys
from typing import Optional

from .core import Core
from .image import ImageError, load_image, restore_guest_roles, start_cold
from .journal import Journal, JournalError, apply_world_record, archive_file
from .runner_prompt import prompt_api_error
from .snapshot import SnapshotError, read_snapshot, restore_world

BOOT_MODES = ("resume", "reboot", "reset")


# ----------------------------------------------------------------------
# 三段生命周期
# ----------------------------------------------------------------------

def start_core(
    core: Core,
    *,
    refresh_models: bool = True,
    timeout: float = 10.0,
    api_error_policy=None,
) -> Core:
    """启动核心（第三段）：刷新模型目录、装运行者策略。不跑调度循环。

    api_error_policy 缺省装终端三选项；要关掉它，传一个返回 "exit" 的策略，或
    在返回后把 `core.api_error_policy` 置空。
    """
    core.api_error_policy = api_error_policy if api_error_policy is not None else prompt_api_error
    if refresh_models:
        core.refresh_model_catalog(timeout=timeout)
    return core


def dump_on_death(core: Core, error, path: Optional[str] = None) -> Optional[str]:
    """宿主在死亡路径上调一次：把现场写进内核转储。

    诊断快照，不是恢复输入。`core_dump_enabled` 为假（镜像 `core_dump: false` 或
    宿主 `--no-core-dump`）时什么都不做。写入失败只记 stderr，不改变调用方的异常。
    """
    if not getattr(core, "core_dump_enabled", True):
        return None
    target = path or getattr(core, "core_dump_to", None) or core.core_dump_path()
    written = core.dump_core(target, error)
    if written:
        print(f"[AVM] 内核转储已写入 {written}", file=sys.stderr)
    return written


def write_snapshot(core: Core) -> Optional[str]:
    """宿主在正常退出路径上调一次：镜像声明了 snapshot_to 就写重启快照。

    只应在安全点调用（run 返回、运行者选择退出）：快照是重启的加速缓存，半截回合的
    状态不能拍。写失败只记 stderr，不改变退出行为。快照丢失不影响数据：日志仍是依据。
    """
    target = getattr(core, "snapshot_path", None)
    if not target:
        return None
    try:
        written = core.snapshot_to(target)
    except Exception as e:
        print(f"[AVM] 重启快照写入失败（{type(e).__name__}: {e}）", file=sys.stderr)
        return None
    print(f"[AVM] 重启快照已写入 {written}", file=sys.stderr)
    return written


# ----------------------------------------------------------------------
# 命令行开关（三个宿主共用）
# ----------------------------------------------------------------------

def add_core_dump_args(parser) -> None:
    """三个宿主共用的转储开关：--core-dump <path> / --no-core-dump。"""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--core-dump", metavar="PATH",
        help="内核死亡（不变量被破坏或未捕获缺陷）时把现场写入该路径（缺省 镜像目录/out/<镜像名>.core.json）",
    )
    group.add_argument(
        "--no-core-dump", action="store_true", help="关闭内核转储",
    )


def add_restore_arg(parser) -> None:
    """三个宿主共用的重启恢复入口：--restore <snapshot>。"""
    parser.add_argument(
        "--restore", metavar="SNAPSHOT",
        help="从重启快照恢复（等价 resume 模式，快照路径显式给出；随后按镜像 journal_to 重放日志尾部）。"
             "不传时若镜像声明了 snapshot_to 且文件可读，会自动使用它",
    )


def add_mode_arg(parser) -> None:
    """启动模式开关（三个宿主共用）：--mode auto|resume|reboot|reset。"""
    parser.add_argument(
        "--mode", choices=("auto",) + BOOT_MODES, default="auto",
        help="启动模式（默认 auto=旧行为：有可读快照用它，否则冷启动）。"
             "resume=最近快照+日志尾部重放；reboot=快照世界+世界类效果重放并重跑 init"
             "（需要快照）；reset=原始镜像并归档旧日志开新代际",
    )


# ----------------------------------------------------------------------
# 恢复输入的读取与判定
# ----------------------------------------------------------------------

def _read_image_json(image_path: str) -> Optional[dict]:
    try:
        with open(image_path, "r", encoding="utf-8") as f:
            image = json.load(f)
    except (OSError, ValueError):
        return None
    return image if isinstance(image, dict) else None


def _declared_path(image_path: str, key: str) -> Optional[str]:
    """读镜像顶层路径字段并按镜像目录解析；无声明或读不到镜像时返回 None。

    与 load_image 的解析规则一致（相对镜像文件所在目录）。这里只用于启动前判断
    恢复输入是否可用；镜像本身的校验仍在 load_image。
    """
    image = _read_image_json(image_path)
    if image is None:
        return None
    value = image.get(key)
    if not isinstance(value, str) or not value:
        return None
    if os.path.isabs(value):
        return value
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(image_path)), value)
    )


def _declared_snapshot_path(image_path: str) -> Optional[str]:
    return _declared_path(image_path, "snapshot_to")


def _declared_journal_path(image_path: str) -> Optional[str]:
    return _declared_path(image_path, "journal_to")


def _snapshot_generation(doc: dict) -> int:
    """快照记录的日志代际；旧快照缺字段按 1（第一代）兼容。"""
    raw = doc.get("generation")
    return raw if isinstance(raw, int) and raw >= 1 else 1


def _try_read_snapshot(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return read_snapshot(path)
    except (OSError, ValueError):
        return None


def _last_exit_status(snapshot_path: Optional[str],
                      journal_path: Optional[str]) -> str:
    """上次退出状态（数据面，不写 clean）：normal / forced / unknown。

    正常断电由宿主在写完快照后退出，所以"快照 seq == 日志末 seq"是正常退出的证据；
    日志比快照长（或只有日志没有快照）说明上次进程在提交之后、写快照之前没了
    （force-stop / 崩溃）。两者都没有是 unknown（首次启动）。VM 只报告，不代 OS
    写 `$MEM.sys.boot.*`。
    """
    doc = _try_read_snapshot(snapshot_path)
    journal_last = None
    journal_exists = bool(journal_path and os.path.isfile(journal_path))
    if journal_exists:
        journal_last = Journal(journal_path).last_seq()
    if doc is None:
        return "forced" if journal_last else "unknown"
    snapshot_seq = int(doc.get("seq", 0) or 0)
    if not journal_exists:
        # 镜像没有 journal_to：快照存在即正常退出；声明了日志却缺失且快照有内容 → 未知。
        return "normal" if (journal_path is None or snapshot_seq == 0) else "unknown"
    if journal_last == snapshot_seq:
        return "normal"
    if journal_last > snapshot_seq:
        return "forced"
    return "unknown"


def _generation_hint(snapshot_path: Optional[str], journal_path: Optional[str]) -> int:
    """reset 归档旧日志时用的代际：快照记录 → 日志首条记录 → 已有档案的最大代际 + 1。"""
    doc = _try_read_snapshot(snapshot_path)
    if doc is not None:
        return _snapshot_generation(doc)
    if journal_path:
        if os.path.isfile(journal_path):
            generation = Journal(journal_path).generation()
            if isinstance(generation, int) and generation >= 1:
                return generation
        base, ext = os.path.splitext(journal_path)
        directory = os.path.dirname(os.path.abspath(journal_path))
        try:
            names = os.listdir(directory)
        except OSError:
            names = []
        pattern = re.compile(
            r"^" + re.escape(os.path.basename(base)) + r"\.(\d+)" + re.escape(ext) + r"$"
        )
        generations = [int(m.group(1)) for name in names
                       if (m := pattern.match(name))]
        if generations:
            return max(generations) + 1
    return 1


# ----------------------------------------------------------------------
# 恢复报告
# ----------------------------------------------------------------------

def _new_recovery(mode: str, last_exit: str) -> dict:
    return {
        "mode": mode,             # resume / reboot / reset / auto（旧调用方式）
        "source": "cold",         # cold / snapshot / journal（面板首屏兼容字段）
        "world_source": "image",  # image / snapshot / snapshot+journal / image+journal
        "snapshot_seq": None,
        "generation": 1,
        "replayed": 0,
        "world_effects": 0,
        "restored": {},
        "next_cid": 0,
        "guest_programs": {},
        "interrupt_vectors": {},
        "last_exit": last_exit,   # normal / forced / unknown（供 OS/操作员判读）
        "warnings": [],
    }


def _finish_recovery(core: Core, recovery: dict) -> dict:
    """补齐运行期事实（代际/cid/guest 向量）并打印报告；写回 core.recovery。"""
    recovery["generation"] = int(getattr(core, "journal_generation", 1) or 1)
    recovery["next_cid"] = int(core._next_cid)
    recovery["guest_programs"] = {
        role: info["cid"] for role, info in (core.guest_roles or {}).items()
    }
    recovery["interrupt_vectors"] = dict(core.interrupts.vectors)
    core.recovery = recovery
    core._last_snapshot_seq = core._journal_seq
    _print_recovery(recovery)
    return recovery


def _print_recovery(recovery: dict) -> None:
    line = (
        f"[AVM] 启动：模式={recovery['mode']} 世界来源={recovery['world_source']} "
        f"快照seq={recovery['snapshot_seq']} 重放={recovery['replayed']}条"
    )
    if recovery.get("world_effects"):
        line += f"（世界效果 {recovery['world_effects']} 条）"
    line += (
        f" 代际={recovery['generation']} next_cid={recovery['next_cid']} "
        f"上次退出={recovery['last_exit']}"
    )
    print(line, file=sys.stderr)
    guests = recovery.get("guest_programs") or {}
    if guests:
        print(
            f"[AVM] guest 程序：{guests}；中断向量：{recovery['interrupt_vectors']}",
            file=sys.stderr,
        )
    for warning in recovery.get("warnings") or []:
        print(f"[AVM] {warning}", file=sys.stderr)


# ----------------------------------------------------------------------
# 重放
# ----------------------------------------------------------------------

def _restore_snapshot(core: Core, path: str) -> dict:
    """恢复快照并打印一行（旧宿主测试依赖这一行）；格式错误翻成 ImageError。"""
    try:
        doc = core.restore_from(path)
    except SnapshotError as e:
        raise ImageError(f"重启快照恢复失败：{e}") from e
    except ValueError as e:
        raise ImageError(f"重启快照恢复失败：{e}") from e
    print(
        f"[AVM] 已从重启快照恢复（seq={doc.get('seq', 0)}，"
        f"{len(core._conv_by_cid)} 个对话）",
        file=sys.stderr,
    )
    return doc


def _resume_journal_tail(core: Core, doc: dict) -> int:
    """resume：重放快照 seq 之后的日志尾部，并断言重放结束 seq 与日志末一致。

    不一致（日志比快照短、代际不符、重放后仍落后）全部抛 JournalError：日志与快照
    来自不同世界时继续跑只会把状态带得更歪。
    """
    snapshot_seq = int(doc.get("seq", 0) or 0)
    snapshot_generation = _snapshot_generation(doc)
    if not core.journal_path:
        if snapshot_seq:
            raise JournalError(
                f"resume：快照 seq={snapshot_seq}，但镜像没有 journal_to，日志尾部无法重放"
            )
        return 0
    journal = core.attach_journal(core.journal_path)
    last = journal.last_seq()
    log_generation = journal.generation()
    if last > 0 and log_generation is not None and log_generation != snapshot_generation:
        raise JournalError(
            f"resume：快照代际 {snapshot_generation} 与日志代际 {log_generation} 不一致，"
            "拒绝重放；请用 boot(mode='reboot') 重建世界或 boot(mode='reset') 重开"
        )
    if last < snapshot_seq:
        raise JournalError(
            f"resume：快照 seq={snapshot_seq} 比日志末 seq={last} 大（日志缺失/被截断），"
            "无法续当前代际；请用 boot(mode='reset')"
        )
    # 旧日志没有 generation 字段：沿用快照记录的代际（新记录按它继续记）。
    core.journal_generation = (
        log_generation if isinstance(log_generation, int) else snapshot_generation
    )
    applied = core.replay_journal()
    if core._journal_seq != snapshot_seq + applied:
        raise JournalError(
            f"resume：重放结束 seq 不一致：恢复基线={snapshot_seq} + 尾部 {applied} 条 "
            f"!= 内存已提交 seq={core._journal_seq}"
        )
    if core._journal_seq != journal.last_seq():
        raise JournalError(
            f"resume：重放结束 seq 不一致：内存={core._journal_seq}，"
            f"日志末={journal.last_seq()}"
        )
    return applied


def _replay_world_tail(core: Core, journal_path: Optional[str], after_seq: int,
                       generation: int):
    """reboot：重放快照 seq 之后的世界类效果；返回 (记录数, 世界效果数, 末 seq)。

    只应用 `memory_set/memory_make/memory_delete/memory_attach/memory_detach/
    edit_metadata/device_mount/device_unmount`；消息、批次、调度、对话/监听器/
    投递等对话态效果一律跳过（对话由 init 重建）。
    """
    if not journal_path or not os.path.isfile(journal_path):
        if after_seq:
            raise JournalError(
                f"reboot：快照 seq={after_seq} 但没有可读日志"
                f"（journal_to={journal_path!r}），无法重放快照之后的世界变化；"
                "请用 boot(mode='reset')"
            )
        return 0, 0, after_seq
    journal = Journal(journal_path)
    last = journal.last_seq()
    if last < after_seq:
        raise JournalError(
            f"reboot：快照 seq={after_seq} 比日志末 seq={last} 大（日志缺失/被截断），"
            "无法重放快照之后的世界变化；请用 boot(mode='reset')"
        )
    log_generation = journal.generation()
    if log_generation is not None and log_generation != generation:
        raise JournalError(
            f"reboot：快照代际 {generation} 与日志代际 {log_generation} 不一致，"
            "无法重放世界尾部；请用 boot(mode='reset')"
        )
    applied = effects = 0
    expected = after_seq
    for record in journal.iter_records(after_seq):
        seq = record.get("seq")
        if not isinstance(seq, int) or seq != expected + 1:
            raise JournalError(
                f"reboot：日志不连续：期望 seq={expected + 1}，得到 {seq!r}（拒绝重放）"
            )
        effects += apply_world_record(core, record)
        expected = seq
        applied += 1
    if expected != last:
        raise JournalError(
            f"reboot：重放结束 seq 不一致：重放到={expected}，日志末={last}"
        )
    return applied, effects, expected


def _start_new_generation(core: Core, journal_path: Optional[str],
                          generation: int, *, bump_when_absent: bool) -> tuple:
    """归档旧日志（若存在）并从 seq 0 开新代际；返回 (新代际, 归档路径)。

    reboot 总是进入新代际（bump_when_absent=True）；reset 在没有任何旧恢复输入时
    留在当前代际（首次启动的原始镜像没有"旧世界"可归档）。
    """
    archived, archived_generation = archive_file(journal_path, generation)
    if archived is not None or bump_when_absent:
        new_generation = archived_generation + 1
    else:
        new_generation = generation
    _begin_generation(core, journal_path, new_generation)
    return new_generation, archived


def _begin_generation(core: Core, journal_path: Optional[str], generation: int) -> None:
    """把 Core 切到新代际：seq 归零、挂上（可能新建的）空日志文件。"""
    core.journal_generation = generation
    core._journal_seq = 0
    core._last_snapshot_seq = 0
    if not journal_path:
        return
    directory = os.path.dirname(os.path.abspath(journal_path))
    try:
        os.makedirs(directory, exist_ok=True)
        # 空文件立即落盘：新代际"从 seq 0 开始"在文件系统上可见（半截日志语义不变）。
        open(journal_path, "a", encoding="utf-8").close()
    except OSError as e:
        print(f"[AVM] 新代际日志创建失败（{journal_path}）：{e}", file=sys.stderr)
        return
    core.attach_journal(journal_path)
    core._journal_seq = 0
    core._last_snapshot_seq = 0


def _restore_from_journal(core: Core, label: str) -> int:
    """旧式启动（mode=auto）的冷回退：挂日志并从已应用位置重放，返回重放条数。

    只用于"快照不可用"的兼容路径：日志是持久化依据，但单独靠日志重建对话是
    尽力而为（快照才是可靠恢复输入）。异常翻成镜像错误。
    """
    if not core.journal_path:
        return 0
    journal = core.attach_journal(core.journal_path)
    generation = journal.generation()
    if isinstance(generation, int) and generation >= 1:
        core.journal_generation = generation
    try:
        applied = core.replay_journal()
    except JournalError as e:
        raise ImageError(f"日志重放失败：{e}") from e
    if applied:
        print(f"[AVM] {label}：重放日志 {applied} 条（seq={core._journal_seq}）", file=sys.stderr)
    return applied


# ----------------------------------------------------------------------
# boot：三种模式 + 旧调用方式
# ----------------------------------------------------------------------

def _check_mode(mode) -> None:
    if mode is None:
        return
    if not isinstance(mode, str) or mode not in BOOT_MODES:
        raise ImageError(
            f"未知启动模式 {mode!r}；允许：{', '.join(BOOT_MODES)}"
            "（resume=最近快照+日志尾部重放；reboot=快照世界+世界类效果重放；"
            "reset=原始镜像）"
        )


def _require_snapshot(path: Optional[str], mode: str) -> dict:
    """模式启动必须有的快照；拒绝时给明确文本（不静默退化 reset）。"""
    if not path:
        raise ImageError(
            f"{mode} 需要重启快照：镜像没有声明 snapshot_to，没有可恢复的快照缓存"
            + ("；改用 boot(mode='reset') 回到原始镜像"
               if mode == "reboot" else "；改用 boot(mode='reset')（reboot 同样需要快照）")
        )
    try:
        return read_snapshot(path)
    except (OSError, ValueError) as e:
        if mode == "reboot":
            raise ImageError(
                f"{mode} 需要可读的重启快照：{e}；没有快照时世界无法重建，"
                "请用 boot(mode='reset') 回到原始镜像"
            ) from e
        raise ImageError(
            f"{mode} 需要可读的重启快照：{e}；日志不能单独重建对话，"
            "请用 boot(mode='reboot') 重建世界（reboot 同样需要快照）"
            "或 boot(mode='reset') 回到原始镜像"
        ) from e


def _next_cid_from(doc: dict) -> int:
    value = doc.get("next_cid")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _restored_counts(core: Core) -> dict:
    """快照恢复后重建了什么：数量只做展示，取不到就给 0，不影响启动。"""
    listeners = getattr(core.events, "_listeners", None)
    try:
        listener_count = len(listeners) if listeners is not None else 0
    except TypeError:
        listener_count = 0
    return {
        "conversations": len(core._conv_by_cid),
        "listeners": listener_count,
        "devices": len(core.mem.mounted_devices()),
    }


def _boot_resume(image_path: str, snapshot_path: str, last_exit: str,
                 host_flags: Optional[dict], checkpoint_in: Optional[str]) -> Core:
    if checkpoint_in:
        raise ImageError(
            "resume 从快照恢复，不能同时用 checkpoint_in（单对话检查点属于冷启动调试路径）"
        )
    doc = _require_snapshot(snapshot_path, "resume")
    core = load_image(Core(), image_path, host_flags=host_flags, restore=True)
    doc = _restore_snapshot(core, snapshot_path)
    applied = _resume_journal_tail(core, doc)
    restore_guest_roles(core)
    recovery = _new_recovery("resume", last_exit)
    recovery["source"] = "snapshot"
    recovery["world_source"] = "snapshot+journal" if applied else "snapshot"
    recovery["snapshot_seq"] = int(doc.get("seq", 0) or 0)
    recovery["replayed"] = applied
    recovery["restored"] = _restored_counts(core)
    _finish_recovery(core, recovery)
    return core


def _boot_reboot(image_path: str, snapshot_path: str, journal_path: Optional[str],
                 last_exit: str, host_flags: Optional[dict],
                 checkpoint_in: Optional[str]) -> Core:
    if checkpoint_in:
        raise ImageError(
            "reboot 会丢弃全部对话，不能同时用 checkpoint_in（单对话检查点属于冷启动调试路径）"
        )
    doc = _require_snapshot(snapshot_path, "reboot")
    generation = _snapshot_generation(doc)
    snapshot_seq = int(doc.get("seq", 0) or 0)
    core = Core()
    # 快照的 cid 计数先就位：init 与 guest 程序按它继续分配 cid（世界重建不重置编号）。
    core._next_cid = _next_cid_from(doc)
    load_image(core, image_path, host_flags=host_flags, restore=True)
    try:
        restore_world(core, doc)
    except (SnapshotError, ValueError) as e:
        raise ImageError(f"reboot 世界恢复失败：{e}") from e
    replayed, world_effects, _end = _replay_world_tail(
        core, journal_path, snapshot_seq, generation,
    )
    start_cold(core)
    new_generation, _archived = _start_new_generation(
        core, journal_path, generation, bump_when_absent=True,
    )
    core._check_invariants()
    recovery = _new_recovery("reboot", last_exit)
    recovery["source"] = "snapshot"
    recovery["world_source"] = "snapshot+journal" if replayed else "snapshot"
    recovery["snapshot_seq"] = snapshot_seq
    recovery["replayed"] = replayed
    recovery["world_effects"] = world_effects
    recovery["restored"] = _restored_counts(core)
    _finish_recovery(core, recovery)
    return core


def _boot_reset(image_path: str, snapshot_path: Optional[str],
                journal_path: Optional[str], last_exit: str,
                host_flags: Optional[dict], checkpoint_in: Optional[str]) -> Core:
    generation = _generation_hint(snapshot_path, journal_path)
    core = load_image(Core(), image_path, checkpoint_in=checkpoint_in,
                      host_flags=host_flags)
    archived_journal, archived_generation = archive_file(journal_path, generation)
    # 旧快照是旧世界的缓存：reset 后它没有意义，归档而不是留着让下一次 reboot 误用。
    archived_snapshot, _snap_generation = archive_file(snapshot_path, archived_generation)
    if archived_journal or archived_snapshot:
        new_generation = archived_generation + 1
    else:
        new_generation = generation
    _begin_generation(core, journal_path, new_generation)
    recovery = _new_recovery("reset", last_exit)
    recovery["source"] = "cold"
    recovery["world_source"] = "image"
    recovery["restored"] = _restored_counts(core)
    _finish_recovery(core, recovery)
    return core


def _boot_legacy(image_path: str, snapshot_path: Optional[str],
                 last_exit: str, host_flags: Optional[dict],
                 checkpoint_in: Optional[str]) -> Core:
    """旧调用方式（不传 mode）：保持 plan-persistent-runtime 的行为。

    快照是缓存：声明了且可读就自动恢复（等价 resume）；缺失或损坏只警告，退回
    "镜像 + 日志重放"（等价旧行为）。新的模式语义只在显式 mode 下生效。
    """
    recovery = _new_recovery("auto", last_exit)
    candidate = snapshot_path if snapshot_path and os.path.isfile(snapshot_path) else None
    if candidate is not None:
        try:
            read_snapshot(candidate)
        except (OSError, ValueError) as e:
            recovery["warnings"].append(f"重启快照不可用（{e}），改为从镜像 + 日志重建")
            candidate = None
    core = load_image(Core(), image_path, checkpoint_in=checkpoint_in,
                      host_flags=host_flags, restore=bool(candidate))
    if candidate is not None:
        doc = _restore_snapshot(core, candidate)
        recovery["source"] = "snapshot"
        recovery["snapshot_seq"] = int(doc.get("seq", 0) or 0)
        try:
            applied = _resume_journal_tail(core, doc)
        except JournalError as e:
            raise ImageError(f"日志重放失败：{e}") from e
        recovery["replayed"] = applied
        recovery["world_source"] = "snapshot+journal" if applied else "snapshot"
        restore_guest_roles(core)
        recovery["restored"] = _restored_counts(core)
    else:
        if core.journal_path:
            applied = _restore_from_journal(core, "从日志恢复")
            if applied:
                recovery["source"] = "journal"
                recovery["replayed"] = applied
                recovery["world_source"] = "image+journal"
        recovery["restored"] = _restored_counts(core)
    _finish_recovery(core, recovery)
    return core


def boot(
    image_path: str,
    *,
    mode: Optional[str] = None,
    checkpoint_in: Optional[str] = None,
    restore: Optional[str] = None,
    host_flags: Optional[dict] = None,
    refresh_models: bool = True,
    timeout: float = 10.0,
    api_error_policy=None,
) -> Core:
    """加载区的便捷入口：创建空核心 → 加载镜像 →（按模式恢复/重放）→ 启动核心。

    mode：resume / reboot / reset；不传时保持旧行为（快照可用则恢复，否则冷启动）。
    restore：显式快照路径，等价 resume（严格——快照读不出来直接报错）。
    checkpoint_in：调试用单对话检查点；只用于冷启动（reset / 不传 mode）。
    """
    _check_mode(mode)
    snapshot_path = _declared_snapshot_path(image_path)
    journal_path = _declared_journal_path(image_path)
    if restore is not None:
        if mode not in (None, "resume"):
            raise ImageError(
                "restore 参数只用于 resume：reboot/reset 从镜像声明的 snapshot_to 取快照"
            )
        snapshot_path = restore
        mode = "resume"
    last_exit = _last_exit_status(snapshot_path, journal_path)
    if checkpoint_in is not None and mode in ("resume", "reboot"):
        raise ImageError(
            f"{mode} 从快照恢复，不能同时用 checkpoint_in"
            "（单对话检查点属于冷启动调试路径）"
        )
    if mode is None:
        core = _boot_legacy(image_path, snapshot_path, last_exit, host_flags, checkpoint_in)
    elif mode == "resume":
        core = _boot_resume(image_path, snapshot_path, last_exit, host_flags, checkpoint_in)
    elif mode == "reboot":
        core = _boot_reboot(image_path, snapshot_path, journal_path, last_exit,
                            host_flags, checkpoint_in)
    else:
        core = _boot_reset(image_path, snapshot_path, journal_path, last_exit,
                           host_flags, checkpoint_in)
    return start_core(
        core,
        refresh_models=refresh_models,
        timeout=timeout,
        api_error_policy=api_error_policy,
    )

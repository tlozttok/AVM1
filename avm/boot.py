"""加载区：宿主启动一个 AVM 内核的统一入口。

生命周期在这里显式分三段，而不是藏在一次调用里（作者 2026-10-16）：

1. **创建空核心**：`Core()`；
2. **加载镜像**：`avm.image.load_image(core, image_path)`——解析 mem / devices / init /
   persist_to，并在整树替换之后重装启动自带的模型目录（`$MEM.model_catalog`）；
3. **启动核心**：`start_core(core)`——刷新模型目录（拉一次上游 `/models`，失败只警告）、
   装运行者策略（API 失败时问运行者的终端提示）。

之后把 Core 交给宿主自己 `run()`。宿主通常只调 `boot()`，它把三段串起来；要单独测试或
分步使用某一段时，直接调对应的函数。

装载期的错误（坏镜像、缺 para、设备加载失败）照旧抛给调用方，由宿主决定怎么报——加载区
只负责顺序与默认值，不吞错。
"""

import json
import os
import sys
from typing import Optional

from .core import Core
from .image import ImageError, load_image
from .runner_prompt import prompt_api_error
from .snapshot import read_snapshot


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
        help="从重启快照恢复（跳过镜像 init；随后按镜像 journal_to 重放日志尾部）。"
             "不传时若镜像声明了 snapshot_to 且文件可读，会自动使用它",
    )


def _declared_snapshot_path(image_path: str) -> Optional[str]:
    """读镜像顶层 snapshot_to 并解析成路径；无声明或读不到镜像时返回 None。

    与 load_image 的解析规则一致（相对镜像文件所在目录）。这里只用于启动前判断
    快照是否可用；镜像本身的校验仍在 load_image。
    """
    try:
        with open(image_path, "r", encoding="utf-8") as f:
            image = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(image, dict):
        return None
    snapshot_to = image.get("snapshot_to")
    if not isinstance(snapshot_to, str) or not snapshot_to:
        return None
    if os.path.isabs(snapshot_to):
        return snapshot_to
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(image_path)), snapshot_to)
    )


def _restore_from_journal(core: Core, label: str) -> int:
    """挂日志并从已应用位置重放，返回重放条数。

    幂等；重放失败是恢复输入损坏，翻成镜像错误。返回值供 boot() 写恢复报告。
    """
    from .journal import JournalError

    if not core.journal_path:
        return 0
    core.attach_journal(core.journal_path)
    try:
        applied = core.replay_journal()
    except JournalError as e:
        raise ImageError(f"日志重放失败：{e}") from e
    if applied:
        print(f"[AVM] {label}：重放日志 {applied} 条（seq={core._journal_seq}）", file=sys.stderr)
    return applied


def boot(
    image_path: str,
    *,
    checkpoint_in: Optional[str] = None,
    restore: Optional[str] = None,
    host_flags: Optional[dict] = None,
    refresh_models: bool = True,
    timeout: float = 10.0,
    api_error_policy=None,
) -> Core:
    """加载区的便捷入口：创建空核心 → 加载镜像 → （恢复）→ 启动核心，返回尚未运行的 Core。

    restore：显式恢复入口。给了就严格——快照读不出来直接报错。
    不传时按"快照是缓存"处理：镜像声明了 `snapshot_to` 且文件可读就自动用它，缺失或
    损坏只警告，退回"镜像 + 日志"重放。日志始终是持久化依据；快照只省重放时间。
    """
    # 恢复报告（F）：来源、快照 seq、重放条数、重建数量与警告。
    # 面板首屏从 /api/state 读它；装载期的终端输出照旧，不吞错。
    recovery = {
        "source": "cold", "snapshot_seq": None, "replayed": 0,
        "restored": {}, "warnings": [],
    }
    explicit = restore is not None
    candidate = restore
    if not explicit:
        candidate = _declared_snapshot_path(image_path)
        if candidate is not None:
            try:
                read_snapshot(candidate)
            except (OSError, ValueError) as e:
                warning = f"重启快照不可用（{e}），改为从镜像 + 日志重建"
                recovery["warnings"].append(warning)
                print(f"[AVM] {warning}", file=sys.stderr)
                candidate = None
    core = load_image(
        Core(), image_path,
        checkpoint_in=checkpoint_in, host_flags=host_flags, restore=bool(candidate),
    )
    if candidate:
        try:
            doc = core.restore_from(candidate)
        except ValueError as e:  # SnapshotError 是 ValueError
            raise ImageError(f"重启快照恢复失败：{e}") from e
        recovery["source"] = "snapshot"
        recovery["snapshot_seq"] = doc.get("seq", 0)
        recovery["restored"] = _restored_counts(core)
        print(
            f"[AVM] 已从重启快照恢复（seq={doc.get('seq', 0)}，"
            f"{len(core._conv_by_cid)} 个对话）",
            file=sys.stderr,
        )
        if core.journal_path:
            recovery["replayed"] = _restore_from_journal(core, "重启恢复")
        else:
            recovery["warnings"].append(
                "镜像没有 journal_to，快照之后的日志尾部无法重放"
            )
            print(
                "[AVM] 警告：镜像没有 journal_to，快照之后的日志尾部无法重放",
                file=sys.stderr,
            )
    else:
        if core.journal_path:
            core.attach_journal(core.journal_path)
            if core.journal.last_seq() > 0:
                recovery["source"] = "journal"
                recovery["replayed"] = _restore_from_journal(core, "从日志恢复")
    core.recovery = recovery
    return start_core(
        core,
        refresh_models=refresh_models,
        timeout=timeout,
        api_error_policy=api_error_policy,
    )


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

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

import sys
from typing import Optional

from .core import Core
from .image import load_image
from .runner_prompt import prompt_api_error


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


def boot(
    image_path: str,
    *,
    checkpoint_in: Optional[str] = None,
    host_flags: Optional[dict] = None,
    refresh_models: bool = True,
    timeout: float = 10.0,
    api_error_policy=None,
) -> Core:
    """加载区的便捷入口：创建空核心 → 加载镜像 → 启动核心，返回尚未运行的 Core。"""
    core = load_image(Core(), image_path, checkpoint_in=checkpoint_in, host_flags=host_flags)
    return start_core(
        core,
        refresh_models=refresh_models,
        timeout=timeout,
        api_error_policy=api_error_policy,
    )

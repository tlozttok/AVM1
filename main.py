#!/usr/bin/env python3
"""AVM 调试入口：加载系统镜像并运行。

用法:
    python main.py <image.json>             # 加载镜像并运行
    python main.py <image.json> --frames    # 运行结束后打印监测帧摘要（stderr）
    python main.py <image.json> --transcript api.txt  # API 调用全文落盘

文件内配置调试（VSCode 里固定运行 python main.py，参数改 debug.json 即可）:
    python main.py                          # 从 debug.json 读取 image/frames/transcript
    python main.py --debug-config other.json
"""
import argparse
import json
import os
import sys

from avm.exceptions import VMMemoryError
from avm.image import ImageError, load_image
from avm.monitor import Monitor


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"调试配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"调试配置必须是对象: {path}")
    return cfg


def _print_frames(core) -> None:
    frames = core.monitor.frames()
    if not frames:
        print("[AVM] 无监测帧", file=sys.stderr)
        return
    print(f"[AVM] 监测帧：基线 + {len(frames) - 1} 轮推进", file=sys.stderr)
    for f in frames:
        line = (
            f"  seq={f.seq} active={f.sched['active_cid']} "
            f"ready={f.sched['ready_cids']} dormant={f.sched['dormant_cids']}"
        )
        tools = [t.get("cmd_type") for t in (f.lmu.get("tool_calls") or [])]
        if tools:
            line += f" tools={tools}"
        result = f.lmu.get("result")
        if result:
            line += f" result={result[:80]!r}"
        error = f.lmu.get("error")
        if error:
            line += f" ERROR={error}"
        print(line, file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AVM 调试入口")
    parser.add_argument("image", nargs="?", help="系统镜像 JSON 文件路径（缺省时从 --debug-config 读取）")
    parser.add_argument("--frames", action="store_true", help="运行结束后打印监测帧摘要（stderr）")
    parser.add_argument("--transcript", metavar="PATH", help="API 调用全文写入该文件（监测器）")
    parser.add_argument("--debug-config", metavar="PATH", default="debug.json",
                        help="无 image 参数时的配置文件（默认 debug.json）")
    args = parser.parse_args(argv)

    frames = args.frames
    transcript = args.transcript
    image = args.image
    config_dir = os.getcwd()
    if image is None:
        cfg_path = args.debug_config
        try:
            cfg = _load_config(cfg_path)
        except (OSError, ValueError) as e:
            print(f"[AVM] 调试配置错误: {e}", file=sys.stderr)
            return 1
        image = cfg.get("image")
        if not isinstance(image, str) or not image:
            print(f"[AVM] {cfg_path} 中缺少 image 字段，且未提供命令行参数", file=sys.stderr)
            return 1
        config_dir = os.path.dirname(os.path.abspath(cfg_path))
        image = image if os.path.isabs(image) else os.path.join(config_dir, image)
        frames = frames or bool(cfg.get("frames"))
        transcript = transcript or cfg.get("transcript")
    if transcript and not os.path.isabs(transcript):
        transcript = os.path.join(config_dir, transcript)

    try:
        core = load_image(image)
    except (ImageError, VMMemoryError, OSError, ValueError) as e:
        print(f"[AVM] 镜像错误: {e}", file=sys.stderr)
        return 1

    if transcript:
        core.monitor = Monitor(max_frames=2000, transcript_path=transcript)

    try:
        core.run()
    finally:
        persist_to = getattr(core, "persist_to", None)
        if persist_to:
            try:
                os.makedirs(os.path.dirname(persist_to) or ".", exist_ok=True)
                core.mem.save(persist_to)
                print(f"[AVM] 内存已写回 {persist_to}", file=sys.stderr)
            except OSError as e:
                print(f"[AVM] 写回失败: {e}", file=sys.stderr)
        if frames:
            _print_frames(core)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Markdown 链接检查（固定动作）。

检查两件事：

1. **断链**：git 跟踪的 Markdown 文件里，每一个相对链接/图片目标都必须存在。
2. **悬空**：被跟踪的文档不得引用未被跟踪的文件——上传层的文档不能依赖本机留档。

用法：

    python scripts/check_doc_links.py          # 检查，有问题时退出码 1
    python scripts/check_doc_links.py -v       # 同时打印检查过的链接数
    python scripts/check_doc_links.py --local-records
                                               # 断链检查同时覆盖 docs/private 的本机留档；
                                               # 悬空判定不适用于留档层，仍只查上传层

跳过围栏代码块内的内容；外部链接、纯锚点链接不检查。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# [文本](目标) 与 ![替代文本](目标)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
# 行内代码：其中的方括号圆括号是代码，不是链接
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def tracked_markdown() -> list[Path]:
    """要检查的 .md：已跟踪 + 未跟踪但未被忽略（新文件在被 commit 前也是盲区）。"""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(p) for p in out.splitlines() if p.strip()]


def ignored(paths: list[str]) -> set[str]:
    """返回其中被 git 忽略的路径。

    悬空的判据是"永远不会被上传"，也就是被忽略；仅未跟踪不算。
    """
    if not paths:
        return set()
    out = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(paths),
        capture_output=True,
        text=True,
    ).stdout
    return {p for p in out.splitlines() if p.strip()}


def targets_of(path: Path) -> list[tuple[int, str]]:
    """取出 (行号, 原始目标)。跳过围栏代码块与行内代码。"""
    found: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(
        (ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_CODE_RE.sub("", line)
        for raw in LINK_RE.findall(line):
            found.append((lineno, raw.strip()))
    return found


def normalize(raw: str) -> str | None:
    """归一化链接目标；返回 None 表示不检查。"""
    target = raw.split()[0] if raw.split() else ""
    if not target:
        return None
    if target.startswith(("http://", "https://", "mailto:", "tel:")):
        return None
    if target.startswith("#"):
        return None
    target = target.split("#", 1)[0]
    if not target or target.startswith("//"):
        return None
    return target


def local_records() -> list[Path]:
    """本机留档区的 .md。

    `docs/private/` 被 git 忽略，`tracked_markdown` 取不到它；这一层只在
    `--local-records` 下参与断链检查，不参与悬空判定。
    """
    root = ROOT / "docs" / "private"
    if not root.is_dir():
        return []
    return sorted(p.relative_to(ROOT) for p in root.rglob("*.md"))


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    with_local = "--local-records" in sys.argv
    broken: list[str] = []
    dangling: list[str] = []
    candidates: list[str] = []
    checked = 0

    docs = tracked_markdown()
    checked_docs = docs + local_records() if with_local else docs
    for doc in checked_docs:
        base = (ROOT / doc).parent
        for lineno, raw in targets_of(doc):
            target = normalize(raw)
            if target is None:
                continue
            checked += 1
            resolved = (base / target).resolve()
            try:
                rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                broken.append(f"{doc}:{lineno}  指向仓库外: {raw}")
                continue
            if not resolved.exists():
                broken.append(f"{doc}:{lineno}  -> {raw}")
                continue
            candidates.append(rel)

    # 悬空的判据是"目标永远不会被上传"，也就是被 git 忽略。
    # 仅未跟踪不算——新文件与被它引用的文件会一起提交。
    # 悬空只对上传层判定：留档层的引用本来就是本机的。
    ignored_targets = ignored(candidates)
    for doc in docs:
        base = (ROOT / doc).parent
        for lineno, raw in targets_of(doc):
            target = normalize(raw)
            if target is None:
                continue
            resolved = (base / target).resolve()
            try:
                rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if not resolved.exists():
                continue
            # 目录：目录自身被忽略，或子树里另有被引用的忽略路径，都算悬空。
            if resolved.is_dir():
                prefix = rel.rstrip("/") + "/"
                if rel in ignored_targets or any(
                    p.startswith(prefix) for p in ignored_targets
                ):
                    dangling.append(f"{doc}:{lineno}  -> {raw}")
            elif rel in ignored_targets:
                dangling.append(f"{doc}:{lineno}  -> {raw}")

    if broken:
        print("断链（目标不存在）：")
        for item in broken:
            print(f"  {item}")
    if dangling:
        print("悬空（文档引用了被 git 忽略、永远不会上传的文件）：")
        for item in dangling:
            print(f"  {item}")
    if verbose and not broken and not dangling:
        print(f"检查通过：{checked} 条相对链接，全部存在，且没有一条指向被忽略的文件。")

    return 1 if (broken or dangling) else 0


if __name__ == "__main__":
    raise SystemExit(main())

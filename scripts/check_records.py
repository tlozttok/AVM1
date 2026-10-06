#!/usr/bin/env python3
"""开发记录检查（固定动作）。

检查 `docs/private/Agents` 下的开发记录：

1. **frontmatter**：已分类的记录必须有 YAML frontmatter，且含 `type` 与 `description`。
2. **type 合法**：`type` 必须在声明的五类之内。
3. **前缀匹配**：文件名前缀必须与 `type` 对应。
4. **位置匹配**：记录所在的生命周期目录必须与它的状态一致。
5. **草稿字段**：位于 `proposed/` 的草稿类记录（question/discussion/design）必须含
   owner/created/expires/promotes-to。批准之后这四个字段不再要求。
6. **类别标记行**：`archived/` 必须有 `**归档**:` 行，`rejected/` 必须有 `**废弃**:` 行。
7. **sidecar**：若存在 `.refs.yaml`，必须能解析且每条引用四要素齐全。
8. **删除区**：`.deleted/` 不被读取——只统计数量，不打开文件。

未分类的记录（无 frontmatter 的历史文档）只计数，不算错误：它们是迁移欠账。

用法：

    python scripts/check_records.py
    python scripts/check_records.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS_ROOT = ROOT / "docs" / "private" / "Agents"

# type -> 文件名前缀
TYPES = {
    "question": "question-",
    "discussion": "discussion-",
    "design": "design-",
    "plan": "plan-",
    "report": "report-",
}

DRAFT_TYPES = {"question", "discussion", "design"}
DRAFT_FIELDS = ("owner", "created", "expires", "promotes-to")

# 生命周期目录（.deleted 只统计不读取）
CLASSES = ("proposed", "implemented", "rejected", "archived")
DELETED_DIR = ".deleted"

# 每个类的记录头部允许出现的日期行
CLASS_MARKERS = {
    "archived": "**归档**:",
    "rejected": "**废弃**:",
}


def parse_frontmatter(text: str) -> dict[str, str] | None:
    """解析极简 YAML frontmatter（仅 key: value）。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return data
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = value.strip().strip('"').strip("'")
    return None


def check_sidecar(path: Path) -> list[str]:
    """检查 .refs.yaml 的四要素。"""
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "citations:" not in text:
        return problems
    # 逐条引用检查四要素；用缩进块粗略切分。
    current: dict[str, str] = {}

    def flush() -> None:
        if not current:
            return
        for field in ("target", "rev", "span", "hash"):
            if field not in current:
                problems.append(f"{path.name}: 引用缺少 `{field}`（target={current.get('target', '?')}）")
        if current.get("state") == "lagging" and "reason" not in current:
            problems.append(f"{path.name}: state=lagging 但缺 `reason`")

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- target:"):
            flush()
            current = {"target": line.split(":", 1)[1].strip()}
        elif ":" in line and current and not line.startswith("#"):
            key, _, value = line.partition(":")
            current[key.strip()] = value.strip()
    flush()
    return problems


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    if not RECORDS_ROOT.is_dir():
        print(f"记录根目录不存在：{RECORDS_ROOT}")
        return 1

    errors: list[str] = []
    classified = 0
    unclassified = 0

    deleted = RECORDS_ROOT / DELETED_DIR
    deleted_count = len(list(deleted.rglob("*.md"))) if deleted.is_dir() else 0

    for path in sorted(RECORDS_ROOT.rglob("*.md")):
        rel = path.relative_to(RECORDS_ROOT)
        if rel.parts and rel.parts[0] == DELETED_DIR:
            continue  # 删除区不读取

        text = path.read_text(encoding="utf-8")
        data = parse_frontmatter(text)
        if data is None:
            unclassified += 1
            continue
        classified += 1

        rtype = data.get("type")
        if rtype is None:
            errors.append(f"{rel}: frontmatter 缺 `type`")
        elif rtype not in TYPES:
            errors.append(f"{rel}: 未知 type `{rtype}`（合法值：{', '.join(TYPES)}）")
        else:
            prefix = TYPES[rtype]
            if not path.name.startswith(prefix):
                errors.append(f"{rel}: type={rtype} 但文件名前缀应为 `{prefix}`")
            # 草稿四字段只约束提案中的记录；批准之后不再要求。
            if rtype in DRAFT_TYPES and rel.parts and rel.parts[0] == "proposed":
                for field in DRAFT_FIELDS:
                    if field not in data:
                        errors.append(f"{rel}: 提案中的草稿类记录缺字段 `{field}`")

        if "description" not in data or not data["description"]:
            errors.append(f"{rel}: frontmatter 缺 `description`")

        # 位置一致性：已分类记录应位于某个生命周期目录下。
        if rel.parts and rel.parts[0] not in CLASSES:
            errors.append(f"{rel}: 已分类但仍平铺在记录根，应移入 {', '.join(CLASSES)} 之一")
        # 类别标记行：归档与废弃各带一行日期。删除区不读取，故不校验。
        elif rel.parts and rel.parts[0] in CLASS_MARKERS:
            marker = CLASS_MARKERS[rel.parts[0]]
            if marker not in text:
                errors.append(f"{rel}: {rel.parts[0]} 类记录缺 `{marker}` 行")

        sidecar = path.with_suffix("").with_suffix(".refs.yaml")
        if sidecar.exists():
            errors.extend(check_sidecar(sidecar))

    if errors:
        print("发现的问题：")
        for item in errors:
            print(f"  {item}")
    if verbose:
        print(
            f"已分类 {classified} 份；未分类（迁移欠账）{unclassified} 份；"
            f"删除区 {deleted_count} 份（未读取）。"
        )
    if not errors and not verbose:
        print(f"检查通过：已分类 {classified} 份，未分类 {unclassified} 份。")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

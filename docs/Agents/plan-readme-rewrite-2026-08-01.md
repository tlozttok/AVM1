# 计划：README 全部重写

**时间**: 2026-08-01
**状态**: 已实施（README.md 重写完成，作者说明已留档）
**背景**: README.md 沿袭早期阶段，与当前代码、词汇、设计严重脱节；README_EN.md 同源。
**起因**: 用户明确"这个项目的 readme 需要全部重写了"。审计发现：
1. 词汇过期：标题仍为"Agent 虚拟机"；大量使用 Agent/系统提示词等 CONTEXT.md 明确 Avoid 的词。
2. 已废弃内容仍存在：`$MEM` 递归解引用、`&` 单层解引用（已删除）；`create` 指令（已拆分 create_cmd/create_sub）；AVM_MEMDUMP/AVM_MEMSOCK/memshell（已移除）；config.json 的 log_level/STEP DEBUG/prompt（已被系统镜像取代）。
3. 缺失当前内容：完整指令集（register_service/call_service/transfer_service/return_result）、调度状态机与交互结束语义、链接设备与 capability、事件/回调设计、系统镜像框架（main.py/debug.json/docs/system-image.md）。
4. 项目哲学问题：文末整段"AI 批注、优化语言、扩写思维分析"是 AI 扩写内容，恰是项目"AI 倾向于 AI 内容"要防范的东西。

**行动**（提案，待确认）:
1. 全部重写 README.md：以当前 CONTEXT.md 词汇为准，精简事实型，砍掉 AI 批注与营销性宣传词。
2. 新结构：项目定位 → 核心概念（术语表，指向 CONTEXT.md）→ 指令集 → 内存模型 → 调度模型 → 系统镜像 → 快速开始 → 设计哲学 → 当前状态与路线 → 文档索引。
3. README_EN.md 是否同步重写（待用户定）。
4. 作者的说明（个人叙述）→ 已移至 docs/record/author-notes.md 留档；"AI 批注、优化语言、扩写思维分析"整节删除。

**影响**: 文档与代码/设计对齐；旧文档中的废弃设计随重写一并清除。README 属于对外门面，词汇必须与 CONTEXT.md 一致；不新增 AI 扩写内容。

**废弃标记**: README 中的"引用语法（$ / &）"、"运行（AVM_MEMDUMP/memshell）"、"配置文件（config.json）"、"指令集（create）"、"运行对话程序（programs/deepseek_simplified.json）"章节内容及"AI 批注"整节。

**实施记录（2026-08-01）**:
- README.md 全部重写（99 行）：事实部分（计算模型 / 指令集 / 内存 / 调度 / 系统镜像 / 快速开始）+ 决策与动机部分（每个决策给出推理过程，不出现"这是有目的的设计"这类断言）+ 当前状态 + 文档索引。删除了宣传词、对比章节、AI 批注；全篇避免"活在"类拟人隐喻；"Agent"仅出现在"不是 Agent 框架"与目录名中。
- 作者的说明移入 docs/record/author-notes.md（原文留档，加注旧词汇不代表当前设计）。
- 标注：**按用户计划改动**（用户确认结构与内容取舍）。
- 待办：README_EN.md 仍是旧版，待用户确认是否同步重写。

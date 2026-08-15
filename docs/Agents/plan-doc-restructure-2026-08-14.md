# 计划：活跃文档重构（CONTEXT 修正 + README 拆分职能）

**时间**: 2026-08-14
**状态**: 已实施
**背景**: 活跃文档（CONTEXT.md、README.md、README_EN.md）存在遗留问题与自相矛盾的逻辑：CONTEXT 标题「Agent 虚拟机」与 README「运行时」不一致、且用了项目第一句否定的「Agent」；未实现功能（事件、内核态/用户态、工作空间、capability、文件安全性、链接）用现在时描述、与 README「当前状态」冲突；「对话」的 _Avoid 含「进程」，与设计根基「对话作为进程」矛盾；「程序活在内存中」是拟人隐喻。README 348 行，把事实规格、决策动机、编程导引混在一篇，其中「投递是简单投递」与「简单投递的动机」两节重复。
**起因**: 用户指示：改 CONTEXT 这类活跃文档的遗留问题与不合理逻辑；改 README 并拆分职能，把详细介绍和思想/哲学弄成独立文档链接到 README；用对话内聊出的实质想法充实哲学文档，但 README 不写宣传。
**行动**:
1. CONTEXT.md：标题去「Agent 虚拟机」；「活在内存中」→「存放在内存中」；「对话」_Avoid 删「进程」；未实现概念（内核态、用户态、链接、事件、工作空间、Capability 模型、文件安全性）加「（设计中，未实现）」；「休眠」「就绪」与 README 排队模型对齐（删「回调队列」）。
2. 新建 docs/runtime.md：运行时详细介绍（计算模型、指令集、内存模型、对话间通信、Python 程序、系统镜像、调试观测），从 README 迁移。
3. 新建 docs/programming-guide.md：如何设计 AVM 程序，从 README 迁移，删「（本节是设计导引，用户后续会在此修改。）」占位。
4. 新建 docs/philosophy.md：设计哲学，汇总 record/10.md（起源、kernel、shell、编译器）、record/3.md（控制论与管理学取代软件工程、中立指令分发层）、record/8.md（环境主义、微架构）、record/9.md（谁是 OS、数据即程序、事件=中断、不插件哲学）、record/1.md（主对话偏见）、record/4.md（AI 倾向 AI 内容、随机集成增强）与本次对话（误差是调节而非消除、简单内核 vs 重协议、简化指令的动机 + 只读设备搜索另一对话上下文）。
5. README.md / README_EN.md 瘦身为：定位 → 文档索引（前移）→ 快速开始 → 当前状态；删除重复的「简单投递」节；英文版顺带修掉「Multicast and two return modes supported」死表格行（08-13 删 wait 的遗留）。
6. 留档本计划。
**影响**: README 大幅瘦身；详细介绍、设计哲学、编程导引分散到三份独立文档；CONTEXT 词汇与真实设计对齐（未实现功能明确标注）。纯文档改动，未触碰 avm/ 代码、测试、镜像。
**废弃标记**: README 中「投递是简单投递」「简单投递的动机」两节中的重复段；README_EN 中「send_instruction … Multicast and two return modes supported」死表格行；「（本节是设计导引，用户后续会在此修改。）」占位注；CONTEXT 标题「Agent 虚拟机」与拟人「活在」。

**实施记录（2026-08-14）**:
- CONTEXT.md：11 处定向修正（标题、拟人、进程 Avoid、6 个未实现标记、休眠/就绪对齐）。
- docs/runtime.md / docs/programming-guide.md / docs/philosophy.md 新建。
- README.md / README_EN.md 瘦身重写（文档索引前移）。
- 验证：grep 无「Agent 虚拟机」「活在」「Multicast and two return modes」残留；链接可达（见下）；`pytest tests/` 未运行——纯文档改动不触及代码，测试数量与结果不受影响。
- 遗留：三份新文档为中文（与 CONTEXT.md、docs/ 现有约定一致）；若需要英文版另开计划。

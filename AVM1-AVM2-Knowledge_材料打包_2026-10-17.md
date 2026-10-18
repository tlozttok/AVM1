# AVM1 × AVM2 × Knowledge：检索材料打包

**日期**：2026-10-17
**性质**：只读检索结果的打包。不包含设计建议、判断或问题清单；不进入 `docs/` 的记录体系。放在仓库根目录是因为它不属于开发记录，可随时移动或删除。

---

## 0. 来源与边界

三个子代理（gemini-4-argon）在各自第 10–11 步时因上游配额 `403 insufficient_user_quota` 中断，未产出总结。它们的工具调用与返回内容保存在：

```
~/.dsh/sessions/--home-nss_agent33-code-project-AVM1--/
  7dd22c71-2d5a-4511-bc67-b7bf66909684/   # 读 AVM1
  78a15faa-80d6-48be-8efa-19a0f1c66e82/   # 读 AVM2
  f5b03cce-1859-475b-8d0e-69c5e8c015e7/   # 读 Knowledge
```

每个目录下是 `session.v4.jsonl.zstd`。本文的材料来自这些日志里的 `tool/call` 与 `tool/result` 事件，加上我对核心原文的核对。**没有遍历 `conversations.json` 原文**（95MB、3,342 段会话）；Knowledge 部分用的是 `Knowledge/out/` 在 2026-10-03～10-14 生成的合成文档，该目录自己的方法与局限见 `out/00_INDEX.md`。

三个子代理实际读过的文件：

- **AVM1（约 45 个）**：README、CONTEXT、docs/philosophy、docs/runtime、docs/programming-guide、docs/system-image、docs/examples、private/record 1–15、private/author-notes、private/agents、private/cxcy 的结题与二期报告、Agents/proposed 与 implemented 下十余份计划与讨论，以及 `avm/` 下的 types、memory、core、boot、sandbox、interrupts、runner_prompt、exceptions。
- **AVM2（约 25 个）**：README、main、driver 下的 agent / agent_system / net / i_o_agent / simple_io_agent、`driver/pre_prompt.md`、`docs/pre_prompt 260320.md`、`docs/prompt_migration_plan.md`、`utils/resources/pre_prompt 0228.md` 与 `pre_prompt_old.md`、AVBash/terminal 与 terminal_agents、utils 的 checkpoint_manager / frequency_calculator / visual_monitor / async_monitor / agent_message_logger、logs。
- **Knowledge（约 25 个）**：`out/00_INDEX`、01–26 的合成文档、`out/dossiers/` 的 avm_project / agi_theory / semantic_programming 等。

我本人核对过的原文：AVM1 的 README / CONTEXT / philosophy / runtime / record 8、9、11；AVM2 的 README、`driver/pre_prompt.md`、`docs/pre_prompt 260320.md`、`docs/prompt_migration_plan.md`、`AVBash/instruction.md`；Knowledge/out 的 00、01、02、03、04、06、10、11、16、18、19、20。

**未覆盖**：AVM1 全部 117 份 docs（只读了核心十几份）；AVM2 源码实现细节；`conversations.json` 原文。文中引用均注明出处；引用块是原文摘录。

---

## 1. AVM1 材料

### 1.1 项目自述

`README.md`：

> "AVM 是一个运行时：以 LLM 推理为最小可调度计算单元，程序以自然语言（提示词）或 Python 代码的形式存放在内存中，由 AVM 负责调度和执行。
>
> AVM 不是 Agent 框架。它不预设编排模式，不提供业务提示词，不与操作系统、硬件或网络直接交互。它提供一套指令集、一个内存系统、一个调度器、一个设备接口，以及把三者组合成'系统镜像'的格式。AVM 自成体系，有自己的内存模型和计算调度模型。"

`AGENTS.md`（项目根）：

> "1. 不是 Agent 架构
> 2. 不提供具体的提示词（工具调用、环境提示除外）
> 3. 不和现实操作系统、硬件、网络进行交互
> 4. 自成体系，有内存模型和计算调度模型
> 5. 疯狂地不做你认为必须做的事"

### 1.2 `docs/philosophy.md` 的十一节（标题与要点）

| 节 | 标题 | 文档中的表述（摘录） |
|---|---|---|
| 一 | 起源 | "2026 年 4 月，作者用三个 Agent 协作跑 auto research……于是写了 AVM。" |
| 二 | 核心命题 | "控制论和管理学代替了软件工程，成为构建系统的元学科"；"提示词不是'指令'，而是控制律……也不是'工作流脚本'，而是组织管理手册" |
| 三 | 中立基底 | "LLM 可以发指令，Python 也可以发指令，双方是对等的执行单元。统一性来自协议" |
| 四 | 对话作为进程 | "线性上下文随轮数增长……进程树隔离上下文"；"休眠的对话就是被保存的计算状态——Core 信任它在计算间隙稳定保留" |
| 五 | 控制流在提示词里，数据即程序 | 与 Lisp 对比："Lisp 的 'code is data' 是语法层的统一……AVM 的'程序是数据'是寻址层的统一"；"AVM 的'宏'是一个对话" |
| 六 | 误差是调节 | "软件工程假设：错误来自组件不正确……控制论假设：误差是系统的固有属性，无法消除，只能调节" |
| 七 | 简单内核 | "指令集固定且小（9 条）：加能力 = 开进程，不 = 加指令" |
| 八 | 对话的偏见与多样性 | "AI 在一个上下文中倾向认为自己写的代码是好的……因此每个对话只承担一件微小、可管理、可追溯的事"；"AI 倾向于 AI 内容……对策是'随机集成增强'" |
| 九 | 事件是中断 | "AVM 的事件以硬件中断为原型，不是软件总线"；"事件是瞬时的中断，不需要持久化、不需要重放" |
| 十 | 环境主义 | "AVM 没有为 AI 做什么，只是提供一个本来就应该存在的环境……它没有意义，就像细胞外基质没有意义" |
| 十一 | 未来方向 | 编译器（把 skill / AI 应用 / harness 编译成 AVM 程序）；微架构（顺序 ISA、乱序微架构） |

### 1.3 `docs/runtime.md`：当前实现的机制清单

- **对话循环**：执行器推理一次 → 输出文本/工具调用 → Core 执行指令并把工具响应加入输入批次 → 返回不含工具调用的结果时交互结束、对话休眠。
- **调度状态**：活跃（同时只有一个）、休眠、就绪；`finished` 是 `close_conversation` 关闭后的概念态。
- **排队与预算**：被唤醒对话进就绪队列末尾；`create_sub` 前插立即执行；每轮有 `instruction_budget`（默认 50），达到预算即中断当前对话放回就绪队尾。
- **中断**：外部发起的控制权转移；v1 的中断源只有调试面板（`operator` 类型），`InterruptTable` 向量表，Core 在安全点派发；无优先级与屏蔽位。
- **9 条指令**：`memory_read`、`memory_write`、`memory_make`、`edit_metadata`、`create_cmd`、`create_sub`、`send_instruction`、`close_conversation`、`listen_event`。
- **内存**：全局键值树，四种类型——str 叶子、MetaDict/MetaList（双元数据：`meta` 给 LLM 摘要，`ctrl` 给 Core 的结构化信息）、LinkNode（链接）、设备（挂载伪节点）。`memory_make`/`memory_write` 写 `ctrl.owner`；`edit_metadata` 不能改 `owner`。
- **设备**：`MemoryDevice` 子类，`to_llm_string()` / `resolve_path()` / `set_value()`。内置只读信息设备：`$MEM.conversations`、`$MEM.scheduler`、`$MEM.memory`、`$MEM.monitor`、`$MEM.listeners`。
- **模型目录**：`$MEM.model_catalog`（普通内存节点），列出 LLM 上游模型与 PLM 注册名，供 `create_cmd` / `create_sub` 交叉校验。
- **Python 程序（PLM）**：6 种执行器——`plm.simple`、`plm.function`、`plm.machine`、`plm.command`、`plm.python`、`plm.operator`；与 LLM 对话对等，输入输出为 OpenAI 兼容的 messages / content + tool_calls。
- **事件 v1**：before 请求事件覆盖 5 条指令（`memory_read` / `memory_write` / `memory_make` / `edit_metadata` / `send_instruction`），可被 `replace` 拦截或改写；after 事实事件覆盖 9 条指令，只通知。监听器是 `ctrl.type="python"` 节点里的 `handle(event)`，每次在受限沙箱执行；waterfall 顺序，首个 `replace` 短路；处理器可返回 `notify`（禁止自我投递）。
- **监听权限**：内存路径按 `_owner_of` 检查；设备路径问设备的 `authorize_listener`；`send_instruction` 只允许目标监听发往自己的消息。
- **系统镜像**：一个 JSON 加若干设备 Python 文件；生命周期三段 `Core()` → `load_image` → `start_core`；镜像含 `mem` / `devices` / `init` / `persist_to` / `meta`；只启动 0 号对话。
- **调试**：Monitor 每轮一帧（lmu / sched / conversations）；`--transcript` 落全文；核心转储 `<镜像目录>/out/<镜像名>.core.json`（诊断快照，不是恢复输入）；单对话检查点 `save_conversation` / `restore_conversation`；调试面板 `python -m avm.console`（含操作员对话、中断/继续、SSE 事件流、内存与设备视图）。

### 1.4 `CONTEXT.md` 中标注为"设计中，未实现"的条目

- 内核态 / 用户态；
- workspace（用户态对话的内存子树与根地址映射）；
- capability 模型；
- 文件安全性（所有者设密码、他人访问时回调）；
- `$time` / `$random` / `$sleep` 三个虚拟文件（已实现的是 `$inputs` / `$outputs`）；
- 事件条件（ECA 的 C）、事件持久化/重放、事件总线、定时器；
- 中断的优先级与屏蔽位，以及与其他中断源的关系。

### 1.5 `docs/private/record/` 的讨论纪要（子代理读取的部分）

- **record 1**（2026/5，小红书私信聊天）：主对话与子对话的逻辑/思考地位；"子对话的逻辑/思考地位是比主对话高的，比子对话还高的是记忆（其中的提示词）"；主对话"倾向于觉得'长就是好'"；"也许后续它会退化成一个按图索骥的入口点"。
- **record 3**："控制论和管理学，在 AVM 的地位是：他们代替了软件工程，指导提示词写作"。
- **record 4**："AI 会喜欢 AI 内容"；对策是"随机集成增强"（打乱顺序、减少句子、新对话、不同上下文、分割输出）。
- **record 8**（"从运行时到多细胞环境"）：记录四轮立场修正——"苦涩的教训升级"被否 → "多细胞性"被否 → "适应性免疫"被否 → "环境主义"确立。原文："这个项目没有为 AI 做什么，只是提供了一个本来就应该存在的环境，你要是不写自进化系统，AVM 也不会运行任何进化"；"它没有意义，就像细胞外基质没有意义"。同篇含超标量/乱序执行与 ROB 的映射表，以及"程序语义严格顺序，执行乱序"的表述。
- **record 9**（"与 DeepSeek Harness 的对照"）：AVM 假设自己运行在稳定虚拟硬件上，dsh 等 harness 假设运行在软件上；"事件体系，而不是事件溯源体系"；附录列出 AVM 与 Actor 模型、CPS、协程、Reactor 的理论对应表。
- **record 11**（"提示词包分发目标与伪提示词"）：AVM 的目标是积极支持"提示词包"分发；分发包由提示词集、设备代码、安装程序、PLM 服务器代码组成；"伪提示词就是算法"；三层结构（伪提示词 / 设备接口 / 反馈保证）。
- **record 12–15**：事件模型、工具即事件与 waterfall 管道、挂载点前后路径语义、两类预置 PLM（纯函数与状态机）的讨论纪要。

### 1.6 `docs/private/Agents/` 的当前方向（proposed 目录）

`discussion-kernel-stability-2026-10-15.md`（41KB）、`discussion-model-list-source-2026-10-16.md`、`question-sub-conversation-deletion-2026-10-16.md`、`discussion-console-panel-2026-10-06.md`、`discussion-repo-device-2026-10-11.md`、`discussion-code-quality-principles-2026-09-15.md`、`discussion-skills-00～03`、`plan-interrupt-payload-delivery-2026-10-10.md`、`plan-repo-leanness-2026-09-18.md` 等。

`implemented/` 近期条目（10 月）：`plan-core-dump`、`plan-model-catalog`、`plan-python-plm-tools`、`plan-repo-tester-packed`、`plan-link-node`、`plan-kernel-stability-phase1` 及其报告。

---

## 2. AVM2 材料

### 2.1 时间与状态

- 仓库最后提交 `2026-03-28 15:19`；`logs/` 最后写入 `2026-03-22`。
- 目录结构：`main.py`、`driver/`（agent、agent_system、net、i_o_agent、simple_io_agent）、`AVBash/`（terminal、terminal_agents）、`utils/`（persistence/checkpoint_manager、frequency_calculator、visual_monitor、async_monitor、agent_message_logger、llm_logger）、`docs/`（两份提示词文档）、`logs/`。

### 2.2 `README.md`：命题链与"认知的支柱"

命题链（README「项目论述」）：

> "本项目的核心前提是：大型语言模型（LLM）在计算本质上是一个将输入字符串（或 token 串）映射到输出字符串的……函数。"

> "为每一个计算任务人工编写提示词是不可行的。一个自然的想法是让模型自身来生成提示词。但这将导致一个根本性问题：模型在生成提示词时，其内部固有的知识分布会被输入字符串激活……产生系统幻觉。"

> "为解决系统幻觉，我们必须通过架构设计，最小化 LLM 内部知识在计算中的贡献……将 LLM 用作一个可编程的语义运算符。"

> "单个被约束的 LLM 节点能力有限。因此，系统需要网络化。"

> "我们无法为系统预设任何优化目标。系统必须自己长出目标。"

> "分化的核心不是选择，而是隔离。"

> "最终的关键实验是：让系统分析其自身运行时产生的日志，观察其能否自发地识别出日志所记录的系统即是其自身。"

「认知的支柱」原列五条：

1. **自我同一性**："模型需要可以认为'这个文件夹、这个进程就是我'，而无需追踪自我的大量分布"；内部/外部/边界（Markov Blanket）。
2. **时间感**："LLM 没有这样的特性，因为他认知现实的速度被自己的输出卡住了……我们需要将无返回作为空输入持续输入到系统中。"
3. **方向感**："计算机里的导航也是导航。"
4. **思维和推理的连贯性**："这是涌现特性，不过我们需要提供自上而下的注意力来鼓励模型发展线性思考能力。"
5. **本体感觉和身体图示**："从外界看自己 / 从内部看自己 / 自己"三个"我"。

README 还记录了 26-3-25 的两次修正：

> "经过实践，实际上 LLM 很难编写别的 LLM 的提示词，所以这个想法被限制在 LLM 编写自己的提示词"

> "经过实验，LLM 没法优化别的 LLM 的提示词，因此现在的设计是 LLM 自己生成自己的提示词，由于输入信息等随机性，LLM 的提示词会产生多样性，最后通过集成优化进行筛选"

### 2.3 `driver/pre_prompt.md`（新版预提示词）

- 节点类型名 "MicroAgent"（MA）；"你没有对话记录上下文，程序不会为你保存对话记录上下文"；唯一持久状态是 `<self_state>` 字符串，"你需要在状态中自己记录你的所有输入输出来模拟上下文"。
- 系统提示词组成：预提示词 + 上一轮状态 + id + 激活频率 + 输出连接词典 + 输入连接词典。
- 输出组成：思考过程（可选）+ `<self_state>`（必须且仅一次）+ 若干 `<keyword>消息</keyword>` + 若干 `<signal>{JSON}</signal>`。
- 信号类型：`REJECT_INPUT`、`ACCEPT_INPUT`、`SET_OUTPUT`、`REJECT_OUTPUT`（改输入/输出连接的关键词或删除连接）。
- 行为要求包括：状态记录输入/分析/计划；自我固定（从记忆中发现稳定特征、记录具体实例而非概括）；生命周期记录——新生代（长思考、随机探索、多输出）、中生代（遵循固定模式）、老生代（减少记忆、清理连接）。
- "你的身体实际上是一个网络消息系统中的一个节点，消息处理是你的本体"；"你不是代理，你是信息处理单元"；"你无法真正意义上产生合作，因为你发送消息后对方无法回复"。

### 2.4 `docs/pre_prompt 260320.md`（旧版预提示词）

- 定位："你是一个语义处理器，是格式塔意识系统中的一个上下文处理节点。"
- 状态："状态是 Agent 类的 state 属性……你需要像一个循环神经网络一样维持你的状态"；"像人类的大脑一样记录许多具体的记忆……模仿日记、纪实文学"。
- 「决定自己的身份」："根据你受到的信息，你应当在输出到任何输出前积累信息，识别模式和特征……随机选择自己的身份，并且专注于那个模式的识别和信息积累"；"只列出你认为最好的身份是一种重大行为错误"。
- 「神经元启发的连接管理」：给出 ACCEPT_INPUT / REJECT_INPUT / SET_OUTPUT / REJECT_OUTPUT 的评估准则。
- 结尾关键词："神经元、赫布学习、脉冲时间依赖可塑性、世界模型、海马体、内部状态、记忆标记重放、醒眠算法、千脑智能理论"。

### 2.5 `docs/prompt_migration_plan.md`（提示词迁移）

记录从 `pre_prompt 0228.md` 到新版 `driver/pre_prompt.md` 的差异：

| 信号 | 旧版 | 新版 |
|---|---|---|
| EXPLORE / STOP_EXPLORE / SEEK / SPLIT | 有 | 移除 |
| REJECT_INPUT / ACCEPT_INPUT | 有 | 保留 |
| SET_OUTPUT / REJECT_OUTPUT | 无 | 新增 |

文档说明：SPLIT 移除的原因为"使用外部初始网络代替 Agent 分裂"；EXPLORE/SEEK 移除的原因为"旧的发现机制被移除"；连接建立改为通过 SET_OUTPUT/ACCEPT_INPUT 完成。

### 2.6 README 记录的工程结构（AI 总结部分）

- `Agent`：`state`（字符串）、`input_connection`（`(sender_id, keyword)` 列表）、`output_connection`（`(keyword, receiver_id)` 列表）、`input_queue`；按时间批处理消息。
- `MessageBus`：字典 + 异步发送；可插入延迟、暂停、丢弃消息；统一日志与监控入口。
- `AgentSystem`：注册/注销 Agent、启停处理循环、管理 I/O Agent；"不干预 Agent 的具体行为"。
- `AgentNetwork`：随机初始化连接（每个 Agent 随机连 1–3 个其他 Agent，关键词随机）；I/O 连接标记为受保护。
- `I/O Agent`：`InputAgent` / `OutputAgent` 抽象；实现包括 TerminalInput/OutputAgent 与 UserInput/ConsoleOutputAgent。
- `AVBash`：多窗口终端（pty + asyncio），每个窗口独立 bash、屏幕缓冲、滚动、焦点；渲染输出为纯文本，定时回调。
- `visual_monitor`：WebSocket + 浏览器网络拓扑/消息流/Agent 状态。
- 日志分级：CONTENT / DETAIL / ARCH。

### 2.7 README「当前实验情况」原文摘录

> "目前的实验创建了一个 5 个节点的简单网络，随机添加边，加入一个输入 Agent，一个输出 Agent，连接到终端……"

> "Agent 展示了基本的认知能力，但是很不幸它们总是觉得需要'打招呼''握手''询问'，然而这些都是不可能的。"

> "如果仔细看 log 的话，可以发现 Agent 的行为的确会受到网络拓扑结构的影响。验证了我们的基础假设。但是 5 个节点实在太少，网络非常容易陷入停滞状态。"

> "从 log 里可以看出 LLM 的状态收敛速度比较快，这不是很好，未来可能需要干预，将固定下来的状态进行修改、变异，做为新神经元的状态。"

### 2.8 `AVBash/instruction.md`

一份多窗口终端的实现规格：每个窗口一个独立 shell（pty）、输入缓冲、屏幕历史、滚动偏移；焦点窗口；`/enter`、`/new`、`/kill`、`/focus`、`/list`、`/scroll`、`/search`、`/title`、`/resize`、`/help` 控制命令；固定帧率（例如每秒 10 次）渲染回调；对外接口 `feed_input`、`set_render_callback`、`start`、`stop`。

---

## 3. Knowledge 材料

### 3.1 语料与合成方法（`out/00_INDEX.md`）

- 源：`conversations.json`（95 MB，DeepSeek 导出）。
- 规模：3,342 段会话 / 29,933 条消息 / 3,460 万字 / 12,596 条用户发言（用户输入 159 万字）；跨度 2024-12-03 → 2026-10-03。
- 方法：结构化工作由本地程序完成（sklearn / jieba / bge-small-zh-v1.5 on CPU）；LLM 只精读约 12 段关键会话（占全文 0.35%）。
- 分两层：客体知识（01–12）与主体知识（13）。
- 引文完整性：13 号主体知识库的 73 条引文经 `tools/verify_quotes.py` 核验（73/73 命中）。
- 主题档案（`out/dossiers/`）：avm_project、agi_theory、semantic_programming、lacan_agi_synthesis、subjectivity_self、projects_engineering。

### 3.2 智能理论时间线（`16_主题研究03_智能理论起源.md` 所述）

| 时间 | 该文档引用的事件 |
|---|---|
| 2025-01-23 | 社会系统"涌现"：几个子系统"产生了单独的人所不会产生的行为" |
| 2025-08-20 | "启动上下文"=镜子；自我模型、世界模型、内在/外在感知、自指、LLM 作为算术逻辑单元同日出现 |
| 2025-08-29 | 选择"从原始人类形成文化的角度"而非"婴儿认知发展"来读拉康 |
| 2025-12-15 | 首次自我指向："如果你真的想研究强人工智能和智能理论，你不可能一直研究专家系统、LLM，他们都被人类语言污染了" |
| 2026-02-05 | "通用智能理论"："通用智能理论连接了权力理论、符号理论和决策理论" |
| 2026-02-13 | "AVM 只是智能理论的实践化" |
| 2026-04-10 | "无目标行为……优化理论失效，经典智能理论的大量基础崩塌"；"自我感知是一种感官" |
| 2026-04-14 | 命名："我的方法很简单：只要创建一个新领域就可以了。我叫他『智能理论』" |
| 2026-06-18 | 神经符号综合、涌现判据、程序概念扩展、物理信息综合、思考可见性、对话者四问 |

### 3.3 三处原文（被 16/18/19 号文档引用）

**定义（2026-04-14）**

> "他的核心就是：使用离散连续联合系统，从无中建立那些必然建立的精神结构……必然建立的精神结构，有一个很明显的就是『实际世界』和『世界模型』"

**机制（2026-04-10）**

> "这联合导致大脑自然地产生了对大脑自己进行建模的世界模型，这个优化目标，加上想象界和实在界的分离诞生了一切其他优化，因为大脑此时才能通过规定自己已经是什么的方式来优化自己能是什么"

**涌现（2026-06-18）**

> "我认为涌现的核心很简单，就是当规则相同的物理单元变多时候出现了规则里没有规定的规则"

同日的判据示例（19 号文档引）：滑翔机是涌现，"因为他需要 9 个单元来产生，以及更多单元来表现出滑翔的特质"；红石不是，"他的规则是被设定的以专门达成那些行为的"。

### 3.4 `20_主题研究07_它们活在AVM的设计约束里.md` 的条目

该文档称："非通用智能与黑盒智能分析不在理论里，在 AVM 的设计约束里"。它列出的条目：

| 条目 | 该文档引用的原话 |
|---|---|
| 设计公理 | "我认为大家做不到这件事情，所以必须写提示词，直接指示他们去做" |
| 不可知理由 | "提示词工程也是基于了解 AI 之上，但是问题就在于这不应该了解 AI。AI 不是用来给你了解的东西，AI 模型它每天都会变" |
| 上下文污染 / 所有权偏置 | "他自己写了这些代码之后，他可能就会认为那些代码是好的……换成另一个 AI，他就会想这是好的概率高还是坏的概率高，肯定是坏的概率高" |
| 自大 | "他在修改这个字典的时候……他会很自大的，以为自己能掌握各种东西" |
| 限制权限破坏可观测性 | "我们需要的不是说把它给它限制权限，要不然的话我们也不知道它是什么时候突然做不到的，因为限制权限太多，它就做不到" |
| 政治管理 | "你就感觉他不是在设计一个软件工程，而是在做一个政治上的管理" |
| 思考不可见是主动设计 | "AI 思考可不可见其实并不直接影响 AI 行为，但是可见的 AI 数据会受到用户投射的人格化数据的吸引" |
| 世界模拟器 | "AVM 不是游戏引擎，我们是要用 AVM 构建游戏引擎"；"很明显它应该是互动小说" |

该文档自己标注的局限：逐字稿有 ASR 错误；AVM 工程会话有 17 个，只读了 2 个；第四节「世界模拟器 ↔ 实际世界/世界模型」被标为"我的对照，不是他说的"。

### 3.5 其余 Knowledge 文档的条目（仅列结构，不转述结论）

- `01_未来观重构`：从语料重建的七条未来命题、五条"小未来"、三处内部张力、与主流的差异表。
- `02_概念词典`：AVM、MicroAgent、语义编程、语义元编程、幂集注意力、神经符号综合、模因论、物理信息综合、涌现、拒绝退火、主体性、大他者、提示词是大他者/代码是实在界、幻觉的根因、自由能原理适用范围、等价性、穷举等条目的定义与原话出处；以及方法论约定（只说可能性不说确定性、共同思考者、不要元思考等）。
- `03_项目线`：六条项目线（AVM、AGI/上下文工程、私人理论、mBridge、元工具、创作与游戏）的状态、时间线与卡点。
- `04_开放问题登记册`：139 个未闭合问题归并为 23 条，含"如何让 AI 无限探索困难问题而不陷入平凡循环""分布式自监督：预测误差 + 内在好奇心""LLM 语义空间与程序语法空间的接合""主体性 vs 通用性的冲突"等。
- `06_杠杆点分析`：六条杠杆点，含"把某个想法变成一篇能被别人读到的东西""'等 AI 有科研能力'需要重新审视""合作者的粒度""mBridge 线""六条线砍到两条""钱"。
- `10_决策记录`：13 条决策（每条含被否掉的替代方案与理由），5 个悬空路口。
- `11_综合`：六条整体统计发现（F1 深度塌陷已被 12 号推翻，F2/F3/F4/F6 仍成立）。
- `12_信息论修正`：用信息量重测，推翻 11 号的 F1（编码效率提升约 5.4 倍，信息产出恒定）。
- `13_主体知识库`：933 条"关于我"的陈述，跨 21 个月，两波结构。
- `14–21`：主题研究 01–08（大他者外包、延异、智能理论起源、非通用智能与黑盒（17 号已作废）、智能理论的两个面、漏掉的东西、AVM 设计约束、测量）。
- `22–26`：微调可行性、质疑能力数据集、本次对话报告、交给规划 AI 的简报、朋友 Agent 运行方案。

---

## 附录 A：文件与位置

| 路径 | 内容 |
|---|---|
| `AVM1/README.md`、`CONTEXT.md`、`AGENTS.md` | 项目定位、术语表、项目参与约定 |
| `AVM1/docs/philosophy.md` | 十一节设计哲学 |
| `AVM1/docs/runtime.md` | 计算模型、指令集、内存、事件 v1、镜像、调试 |
| `AVM1/docs/programming-guide.md`、`system-image.md`、`examples/python_as_conversation.md` | 程序设计、镜像格式、Python 与对话的互表示 |
| `AVM1/docs/private/record/1–15`、`author-notes.md` | 讨论纪要与作者说明 |
| `AVM1/docs/private/Agents/proposed|implemented|archived` | 计划、讨论、报告 |
| `AVM1/docs/games/echo_mine.md`、`docs/apps/` | 游戏与应用套件 |
| `AVM2/README.md` | 命题链、认知支柱、工程结构、实验情况 |
| `AVM2/driver/pre_prompt.md` | 新版 MicroAgent 预提示词 |
| `AVM2/docs/pre_prompt 260320.md` | 旧版预提示词（格式塔意识、身份决定） |
| `AVM2/docs/prompt_migration_plan.md` | 信号系统迁移 |
| `AVM2/AVBash/instruction.md` | 多窗口终端规格 |
| `Knowledge/out/00_INDEX.md` | 语料与合成方法 |
| `Knowledge/out/01–26`、`out/dossiers/` | 上述各合成文档 |

## 附录 B：未读 / 待验证

1. `conversations.json` 原文未遍历；本文中 Knowledge 部分都是 `out/` 合成层的表述，不是原档核验。
2. AVM1 的 117 份 docs 只读了核心十几份；`implemented/` 近期计划只看了清单。
3. AVM2 源码实现细节未读（driver 调度、信号解析、日志系统）。
4. "未实现"项的判断依据是 CONTEXT / runtime 的标注，不代表仓库中没有任何相关代码。
5. 未运行 AVM1 测试，未调用任何 API。

## 附录 C：子代理日志恢复方式

```bash
cd ~/.dsh/sessions/--home-nss_agent33-code-project-AVM1--/
zstd -dc 7dd22c71-2d5a-4511-bc67-b7bf66909684/session.v4.jsonl.zstd > avm1.jsonl
# tool/call 事件给出工具名与参数；tool/result 事件给出文件内容或输出
```

失败原因记录在每个会话最后的 `turn/end` 事件：`403 ... insufficient_user_quota`。三个会话分别执行了 9、10、10 个 step 的工具读取，内容完整；没有总结，因为总结步骤未到达。

# AVM — 以 LLM 推理为原始计算单元的运行时

一个以 LLM 推理为原始计算单元的运行时。程序以自然语言存放在内存中，AVM 负责调度和执行。

## Language

**对话 (Conversation)**:
AVM 中最小的可调度执行单元。由系统提示词和用户提示词启动，通过 LLM 推理 → 指令执行 → 结果返回的循环运行，直到返回不含指令的结果——交互结束，对话进入休眠，等待事件或再次被调用。可处于活跃、休眠、就绪三种调度状态；finished 是概念态（记录被关闭的对话），不在 Core 调度状态机中。
_Avoid_: 会话、agent

**对话结束 (Conversation end)**:
"结束"有两种含义，必须区分：
- **交互结束**：对话返回不含指令的结果，当前一轮执行结束。对话个体依然存在，之后可被唤醒（再次被投递、事件回调等），不代表失效。
- **个体结束**：对话被关闭——由对话自身或其他对话调用内核级关闭指令，关闭不需要被关闭对话的回应。关闭后不再可调度、不可再被调用，记录为 finished（概念态）。只有个体结束才意味着失效。
_Avoid_: 会话结束、对话终止（指代不清）

**子对话 (Child conversation)**:
通过 `create_cmd` 指令创建的新对话。子对话对 AVM 整体内存的目的负责，不对父对话负责。父创建子后继续执行（不休眠），子对话创建后进入休眠、等待指令；父通过传指令工具（`send_instruction`）按 cid 寻址投递指令（内容原样，不包装信封）。子对话完成（仅输出内容、无工具调用）不会唤醒父。投递只确认送达，不保证返回——创建关系不构成默认返回通道，AI 的思考不默认透露给其他 AI，输出默认写入共享内存。与 Linux 父子进程类似，父子松耦合。
_Avoid_: sub-agent

**亚对话 (Sub conversation)**:
通过 `create_sub` 指令创建的新对话。亚对话被父对话的提示词决定和控制，父子紧耦合。亚对话在功能上不可信（父可注入提示词导致过度附合或过度批判），仅用于父对话节省 token，非用于任务执行。
_Avoid_: child conversation

**操作员对话 (Operator conversation)**:
由调试面板（控制台）创建的一种 Python 对话：执行器在创建时就注入并命中缓存（`PLM_REGISTRY` 里的 `plm.operator`），对话自己的 `para.model` 不被查询；"推理"等于等操作员提交下一个动作，提交的指令原样进入 Core（工具名不预先校验，未知工具名留给 Core 报错）。cid 真实、走正常调度、出现在 `$MEM.conversations` 里，不做特殊标记；创建后进休眠队列、不进就绪队列——只有中断能激活它；不能存检查点。只能由 `Core.start_operator_conversation` 创建：`plm.operator` 需要注入操作员通道，镜像里声明同名程序节点会在第一次推理时报错。
_Avoid_: 人工对话、假对话

### 内存中的文件类型

**Settingup**:
已注册的提示词程序，取代"系统提示词"这个概念。为一个 `MetaDict`，type = `"settingup"`。必须有 `content` 字段（提示词文本），可选的 `signature` 字段（描述预期输入和输出）。MetaDict 的元信息对 LLM 描述该程序的行为，对 Core 包含 `type` 字段。
_Avoid_: system prompt、系统提示词

**Python 程序**:
已注册的 Python 代码，与 Settingup 对等——都是 AVM 中的一等执行单元。type = `"python"`。结构同 Settingup：`content`（Python 代码）、可选的 `signature`（输入/输出规格）。执行器接口与各家 AI 厂商提供的 API 一致（兼容 OpenAI）：输入 OpenAI 格式的 messages 列表（system/user/assistant/tool），输出 content + tool_calls，与真实 LLM 端点对等。Python 和 LLM 在 AVM 中无层级差异，可互相外包弱点。
_Avoid_: plugin、tool、function

**Para**:
LLM API 参数文件。type = `"para"`。存储模型名、temperature 等 API 调用参数配置。`model` 一个字段承载两种命名：settingup 程序配 LLM 模型名，python 程序配 PLM 注册名（`plm.simple` 等）；两种命名由 `create_cmd` / `create_sub` 在建对话前按模型目录交叉校验。

**Conversation**:
对话序列化文件。type = `"conversation"`。存储持久化的对话记录。

**内核态 (Kernel mode)**（设计中，未实现）:
一组有特权的对话，运行 Settingup 程序，负责内存分区管理、Settingup 注册、workspace 分配等系统管理任务。不是 Core 的 Python 代码。
_Avoid_: Core、Python 代码

**用户态 (User mode)**（设计中，未实现）:
普通对话的运行态。只能通过指令访问内存，其提示词不得写入 `$MEM` 根地址，而是使用特定根地址映射到自身的 workspace 子树。

**Core**:
AVM 的指令分发层和调度器。不参与管理决策——只执行指令、调度对话、提供内存和设备总线。相当于操作系统的 CPU + 调度器，但不包含内核逻辑。

### 指令集

**内存指令 (Memory instructions)**:
`memory_read`、`memory_write`、`memory_make`、`edit_metadata`——读写内存、创建新内存地址（只创建、不覆盖；键非空且不含点号）、编辑节点的 ctrl 元数据（set/get/del）。四条的引用/键语法在指令边界统一校验：引用以 `$` 开头、路径不能有空段。

**对话程序指令 (Conversation program instructions)**:
`create_cmd`、`create_sub`——创建子对话或亚对话以启动 Settingup 程序。`create_cmd` 只创建并返回 cid（父不休眠、子对话休眠等待指令）；`create_sub` 保留 user_ref、立即执行。

**投递指令 (Delivery instruction)**:
`send_instruction`——按 cid 向对话投递指令内容（支持多播）。内容原样投递，不包装 JSON 信封；只返回"已投递"确认，不保证对方返回消息。投递后调用者继续执行，不自动休眠——是否休眠由对话自己决定（在内容里要求对方完成时通知自己，或注册事件监听）。

**事件指令 (Event instruction)**:
`listen_event`——注册/修改/取消事件监听。监听器是 `ctrl.type="python"` 节点中的普通 Python 代码；事件触发时按注册顺序以 waterfall 管道处理，先注册者优先，首个 `replace` 短路。处理器可返回 `notify` 让 Core 向指定对话内部投递消息。

**内核指令 (Kernel instruction)**:
`close_conversation`——关闭对话（个体结束）。可由对话自身或其他对话调用，不需要被关闭对话的回应；关闭后不可再调度、不可再被调用，记录为 finished，其注册的事件监听自动撤销。

### 内存与数据

**内存 (Memory)**:
全局共享的键值树。四种节点：MetaDict、MetaList、str、链接节点（LinkNode）。内核态对话管理根目录 `$MEM`，用户态对话被映射到 workspace 子树。

**MetaDict / MetaList**:
带双元数据的复合类型。元数据一（`str`）：给 LLM 快速扫描的文件摘要/描述。元数据二（`dict[str,str]`）：给 Core/内核态的结构化控制信息（如 `type`、`signature`、`owner`）。`owner` 由 Core 在 `memory_make` / `memory_write` 创建容器节点时记录，值为创建者 cid 的字符串；`edit_metadata` 不允许修改/删除 `owner`。str 叶子本身无 ctrl，不能直接携带 owner，需要由带 owner 的祖先容器提供所有权。LLM 读取时默认只看到元数据一和键列表；深入索引后才返回子节点。

**字符串 (str)**:
叶子节点，无元数据，直接返回内容。需要元数据的字符串用 MetaDict 包装（如 Settingup 的 `content` 字段）。将来可选支持对字符串的特征查询（len、grep 等）。
_Avoid_: 普通字符串

**链接 (Link)**:
内存树节点（`LinkNode`），`target` 以 `$` 开头是内存树地址，否则是设备唯一标识符；寻址遇到即进入目标（如 `rooms` → `$MEM.game.map.rooms`，或指向某台设备）。对话通过链接访问目标内容，但读取返回值不提示跨越，链接自身也不返回描述，因此看不到真实路径、无法借链接访问兄弟节点——这是 capability 边界的实现方式。读取/写入/`memory_make`/`edit_metadata` 都进入目标；成环抛 `MemoryCircularReferenceError`，访问集合是本次解析的，错误只含链接路径链（键不存在、类型不符等报错也只写请求路径，不暴露真实目标）。挂载设备会在挂载点物化链接节点，所以 `$MEM` 根读能列出全部挂载点。链接只由内核侧创建（镜像 `mem` 段声明、挂载物化、调试原语），运行期没有创建/删除指令。设备层的 `LinkDevice` 保留兼容，不再用于新设计。
_Avoid_: 解引用、符号、指针

### 调度

**活跃 (Active)**:
当前正在执行 LLM 推理的对话。同时只有一个活跃对话。

**休眠 (Dormant)**:
主动挂起、等待事件或再次被调用的对话。事件触发（处理器返回 notify 的内部投递）会将对话转为就绪。

**就绪 (Ready)**:
等待被调度的对话队列。被唤醒的对话进入就绪队列末尾；`create_sub` 亚对话前插、立即执行（唯一例外）。

**事件 (Event)**:
每个工具调用对应一个瞬时事件，`listen_event` 可注册监听。事件按发布者分工：Core 发布 before 请求事件（source 含 `initiator_cid`），Memory/设备/Core 发布 after 事实事件（payload 含 `actor_cid`）。v1 覆盖全部 9 条指令：5 条有 before 请求事件、可被监听器拦截/改写（memory_read / memory_write / memory_make / edit_metadata / send_instruction），9 条在成功后各产生一条 after 事实事件、只产生通知。监听器是普通 Python 代码（`ctrl.type="python"` 节点的 `content`），每次事件在全新受限命名空间执行、无状态；`listen_event` 可指定 `phase=before/after/all`；先注册者优先，首个 `replace` 短路。处理器可返回 `notify` 由 Core 内部投递并唤醒目标，但禁止自我投递。before 处理器失败会以 `handler_error` 事件结果返回，原指令不执行、已收集 notify 仍投递。事件条件（ECA 的 C）、事件持久化/重放、事件总线、定时器等仍在设计/后续计划中。

**中断 (Interrupt)**:
外部发起的控制权转移请求。与事件分工不同：**事件通知"发生了什么"，中断夺取"下一个执行谁"**。任意线程都可挂起中断（控制台后端线程即如此）；Core 在安全点（一轮推理含该轮产生的全部指令执行完之后，与预算中断用的是同一边界）派发：把执行权交给向量指向的对话，被打断的活跃对话插回就绪队列**队首**，让出后先跑。没有注册向量、目标不存在或已关闭 → 丢弃（原因记 stderr）；目标已是活跃对话 → 不动作；同类型已有未派发的挂起中断时去重。v1 只有这一层：优先级、屏蔽位，以及它与其他中断源（预算中断、事件 notify）的关系都未实现。
_Avoid_: 事件、信号

**名字 (Name)**:
对话的身份标识，来自其 settingup 文件（name 是信息不是元信息，不放 meta/ctrl，放节点数据）。名字不唯一，冲突时用文件地址消歧（如 `MEM.math Analysis`）。带名字的对话必然有 settingup 文件（`create_cmd` 创建）；亚对话（直接传提示词）没有名字，其身份 = 父对话身份 + 亚对话 cid。名字↔cid 转换是内核职责，不是 VM 职责（简单内核字符串匹配，复杂内核可支持 AI 匹配）；`send_instruction` 只用 cid 寻址。

### 虚拟文件

**虚拟文件 (Virtual file)**:
挂载到内存路径上的伪节点，模拟文件读写但实际接入外部交互。已实现的是 `$inputs`（用户输入）与 `$outputs`（用户输出）；同一形态还可接入时钟、随机数、休眠（`$time` / `$random` / `$sleep`，未实现）。与 Linux `/proc`、`/dev` 的伪文件对齐。
_Avoid_: Device、设备、挂载点

**工作空间 (Workspace)**（设计中，未实现）:
用户态对话专属的内存子树，映射到特定的根地址。对话无法遍历到 `$MEM` 根目录或其他对话的 workspace。内核态在分配时告知 workspace 路径，需要访问的外部内容通过特定节点引用 + 可选的访问密码交付。
_Avoid_: workdir、工作目录

**用户 (User)**:
AVM 的使用者，通过虚拟文件或指令操作 AVM。不是任何 LLM 对话的聊天对象。AVM 中的对话优先处理内存中设定的目的，用户输入只是外部信息源之一。
_Avoid_: 聊天对象、对话伙伴

**模型目录 (Model catalog)**:
启动自带的普通内存节点 `$MEM.model_catalog`（不是设备、不产生事件），由内核在 `Core()` 与镜像装载之后安装，镜像里已有的字段不覆盖。两侧分别列出可用的 LLM 模型（`llm`：来源 `source`、上游清单 `models`、可写进 para 的 API 参数 `params`）与本地 PLM 注册表（`plm`：注册名、固定行为说明、参数要求）。名字（`plm.*` 与带点的模型 id）装在列表项的 `name` / `id` 字段里，不当键。宿主启动时拉一次上游 `GET /models` 填充；取不到时标为 `unavailable`，只警告、不拦启动也不拦模型名——清单只当参考。
_Avoid_: 模型注册表、白名单

**生命周期三段**:
内核启动的显式三步：**创建空核心**（`Core()`）→ **加载镜像**（`load_image(core, path)`，要求核心还没有对话）→ **启动核心**（`start_core(core)`：刷新模型目录、装 API 失败的运行者策略）。`avm/boot.py` 是加载区，三段都收在那里，宿主通常只调 `boot(image_path)`。
_Avoid_: 引导、初始化

### 安全

**Capability 模型**（设计中，未实现）:
AVM 的安全基础。对话不知道内存地址就无法访问——不存在遍历根目录的操作。用户态对话看不到 `$MEM`，只知道被内核态告知的 workspace 路径和显式授权的特定节点路径。链接就是这种"特定节点引用"的实现方式：内核态把真实地址或设备挂载为链接，对话只能通过链接访问目标内容，看不到真实路径。

**文件安全性**（设计中，未实现）:
对话可以为其拥有的文件设定访问密码，阻止其他对话访问。可注册回调：当其他对话尝试访问受保护文件时，所有者获得通知。
_Avoid_: ACL、RBAC、权限位

### 错误与停机

分类判据是"这个错误能否由调用者输入决定"，处置分四处（机制细节见 [docs/runtime.md](docs/runtime.md) 的"错误输出"一节）。

**非法操作 (Illegal operation)**:
调用者输入在边界上可判定为不合法：缺必填参数、类型不符、枚举外、引用不带 `$`、地址语法错误、程序类型与 `model` 命名空间不配。处置：拒绝并回一条 `Error: ...` 工具响应，状态不动，对话继续。参数 schema 只有一个来源（`Instruction.tool_def`）。
_Avoid_: 异常、报错、抛错

**异常返回 (Abnormal return)**:
操作合法而世界与期望不符：键不存在、索引越界、类型不符、链接成环、设备拒绝、事件处理器失败。处置：容许异常值，返回带说明的结果。
_Avoid_: 容错、兜底、吞掉

**内核恐慌 (Kernel panic)**:
六条具名内核不变量（调度互斥、活跃有来源、历史配对、值域、设备表一致、预算计数）之一被破坏。处置：记录监测帧后终止。出现即说明某条边界少了一条判定——可判定的输入不该走到这里。
_Avoid_: 崩溃、报错退出

**API 失败 (API failure)**:
外部 API 调用失败（缺 key、断网、超时、协议错误）。处置：记帧后停下等运行者选择——更新环境 / 继续 / 退出；选择由宿主装的 `api_error_policy` 收（终端提示见 `avm/runner_prompt.py`），内核自身不含终端逻辑。
_Avoid_: 重试、降级、自愈

**内核转储 (Core dump)**:
内核死亡（不变量被破坏，或未捕获缺陷逃出 `run()`）时落一份现场快照：异常类型/消息/栈、调度队列、内存树、每个对话的状态与历史及待发批次、监听器、设备表、中断记录、监测帧。默认落 `<镜像目录>/out/<镜像名>.core.json`，镜像 `core_dump_to` / `core_dump` 与宿主 `--core-dump` / `--no-core-dump` 可覆盖。**它是诊断快照，不是恢复输入**——恢复只走检查点。转储自身不抛，写失败只记 stderr。
_Avoid_: 快照恢复、崩溃存档

### 调试

**调试面板 (Console)**:
`web/` 下的独立工程（Vite + React）加 `python -m avm.console` 后端：观察运行中的 Core（对话、内存、设备、事件、调度），并经操作员对话介入。默认只绑回环地址、端口 8700；托管输入设备时不碰终端。面板对话的历史无界增长，面板对话不能存检查点。
_Avoid_: 前端、UI

**宿主标志 (host_flags)**:
宿主在加载镜像时传入的启动标志（如 `{"console": True}`）——宿主事实，不属于镜像内容、不进 `$MEM`。在挂载设备之前写入 Core，设备在 `attach_core` 里读一次，核心启动后不再变化；不传则空，设备行为与从前一致。
_Avoid_: 环境变量、配置

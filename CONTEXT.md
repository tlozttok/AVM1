# AVM — Agent 虚拟机

一个以 LLM 推理为原始计算单元的运行时。程序以自然语言活在内存中，AVM 负责调度和执行。

## Language

**对话 (Conversation)**:
AVM 中最小的可调度执行单元。由系统提示词和用户提示词启动，通过 LLM 推理 → 指令执行 → 结果返回的循环运行，直到返回不含指令的结果——交互结束，对话进入休眠，等待事件或再次被调用。可处于活跃、休眠、就绪三种调度状态；finished 是概念态（记录被关闭的对话），不在 Core 调度状态机中。
_Avoid_: 会话、进程、agent

**对话结束 (Conversation end)**:
"结束"有两种含义，必须区分：
- **交互结束**：对话返回不含指令的结果，当前一轮执行结束。对话个体依然存在，之后可被唤醒（服务再次调用、事件回调等），不代表失效。
- **个体结束**：对话被关闭——由对话自身或其他对话调用内核级关闭指令，关闭不需要被关闭对话的回应。关闭后不再可调度、不可再被调用，记录为 finished（概念态）。只有个体结束才意味着失效。
_Avoid_: 会话结束、对话终止（指代不清）

**子对话 (Child conversation)**:
通过 `create_cmd` 指令创建的新对话。子对话对 AVM 整体内存的目的负责，不对父对话负责。父创建子后进入休眠；子对话完成（仅输出内容、无工具调用）不会唤醒父。子对话可通过返回工具（`return_result`）显式将结论投递给父的 tool response，投递即直接唤醒父（消息已可用就不留在休眠队列）；子对话在下一轮以无工具结果收尾后进入休眠。与 Linux 父子进程类似，父子松耦合。
_Avoid_: sub-agent

**亚对话 (Sub conversation)**:
通过 `create_sub` 指令创建的新对话。亚对话被父对话的提示词决定和控制，父子紧耦合。亚对话在功能上不可信（父可注入提示词导致过度附合或过度批判），仅用于父对话节省 token，非用于任务执行。
_Avoid_: child conversation

**服务 (Service)**:
通过 `register_service` 注册的已存在对话。其他对话可通过 `call_service` 调用并等待返回——服务用 `return_result` 工具返回结果；或通过 `transfer_service` 移交控制权不等待返回。服务对话交互结束（进入休眠）后仍保持注册，可被再次调用；对话个体结束（被关闭）后服务才失效。

### 内存中的文件类型

**Settingup**:
已注册的提示词程序，取代"系统提示词"这个概念。为一个 `MetaDict`，type = `"settingup"`。必须有 `content` 字段（提示词文本），可选的 `signature` 字段（描述预期输入和输出）。MetaDict 的元信息对 LLM 描述该程序的行为，对 Core 包含 `type` 字段。
_Avoid_: system prompt、系统提示词

**Python 程序**:
已注册的 Python 代码，与 Settingup 对等——都是 AVM 中的一等执行单元。type = `"python"`。结构同 Settingup：`content`（Python 代码）、可选的 `signature`（输入/输出规格）。Python 和 LLM 在 AVM 中无层级差异，可互相外包弱点。
_Avoid_: plugin、tool、function

**Para**:
LLM API 参数文件。type = `"para"`。存储模型名、temperature 等 API 调用参数配置。

**Conversation**:
对话序列化文件。type = `"conversation"`。存储持久化的对话记录。

**内核态 (Kernel mode)**:
一组有特权的对话，运行 Settingup 程序，负责内存分区管理、Settingup 注册、workspace 分配等系统管理任务。不是 Core 的 Python 代码。
_Avoid_: Core、Python 代码

**用户态 (User mode)**:
普通对话的运行态。只能通过指令访问内存，其提示词不得写入 `$MEM` 根地址，而是使用特定根地址映射到自身的 workspace 子树。

**Core**:
AVM 的指令分发层和调度器。不参与管理决策——只执行指令、调度对话、提供内存和设备总线。相当于操作系统的 CPU + 调度器，但不包含内核逻辑。

### 指令集

**内存指令 (Memory instructions)**:
`memory_read`、`memory_write`、`memory_make`——读写内存和创建新内存地址。

**对话程序指令 (Conversation program instructions)**:
`create_cmd`、`create_sub`——创建子对话或亚对话以启动 Settingup 程序。

**调度模型指令 (Scheduling model instructions)**:
`register_service`、`call_service`、`transfer_service`、`return_result`——对话间的调度关系管理，控制服务注册、调用、移交和结果返回。

### 内存与数据

**内存 (Memory)**:
全局共享的键值树。三种数据类型：MetaDict、MetaList、str。内核态对话管理根目录 `$MEM`，用户态对话被映射到 workspace 子树。

**MetaDict / MetaList**:
带双元数据的复合类型。元数据一（`str`）：给 LLM 快速扫描的文件摘要/描述。元数据二（`dict[str,str]`）：给 Core/内核态的结构化控制信息（如 `type`、`signature`）。LLM 读取时默认只看到元数据一和键列表；深入索引后才返回子节点。

**字符串 (str)**:
叶子节点，无元数据，直接返回内容。需要元数据的字符串用 MetaDict 包装（如 Settingup 的 `content` 字段）。将来可选支持对字符串的特征查询（len、grep 等）。
_Avoid_: 普通字符串

**链接 (Link)**:
挂载在内存路径上的设备节点，指向真实路径的别名（如 `$rooms` → `$MEM.game.map.rooms`）。对话通过链接访问目标内容，但看不到真实路径，无法借链接访问兄弟节点——这是 capability 边界的实现方式。读取时跟随目标；链接成环抛 `MemoryCircularReferenceError`。
_Avoid_: 解引用、符号、指针

### 调度

**活跃 (Active)**:
当前正在执行 LLM 推理的对话。同时只有一个活跃对话。

**休眠 (Dormant)**:
主动挂起、等待所注册事件的对话。事件触发后 core 将对话转为就绪（回调激活）。

**就绪 (Ready)**:
等待被调度的对话队列。调度优先级：主对话 > 子对话调用队列 > 回调队列。

**事件 (Event)**:
对话可显式注册监听的事件。VM 提供的事件源：定时、内存地址更改、其他对话注册的自定义信号。事件触发后 core 将监听者从休眠转为就绪。返回工具（`return_result`）的投递不需要注册——投递即直接唤醒调用者（消息已可用就不留在休眠队列）。组合事件与事件总线由 AI 程序（提示词）实现，不属于 VM。

### 虚拟文件

**虚拟文件 (Virtual file)**:
挂载到内存路径上的伪节点，模拟文件读写但实际接入外部交互。如 `$inputs`（用户输入）、`$outputs`（用户输出）、`$services`（服务列表）、`$time`（时钟）、`$random`（随机数）、`$sleep`（休眠）。与 Linux `/proc`、`/dev` 的伪文件对齐。
_Avoid_: Device、设备、挂载点

**工作空间 (Workspace)**:
用户态对话专属的内存子树，映射到特定的根地址。对话无法遍历到 `$MEM` 根目录或其他对话的 workspace。内核态在分配时告知 workspace 路径，需要访问的外部内容通过特定节点引用 + 可选的访问密码交付。
_Avoid_: workdir、工作目录

**用户 (User)**:
AVM 的使用者，通过虚拟文件或指令操作 AVM。不是任何 LLM 对话的聊天对象。AVM 中的对话优先处理内存中设定的目的，用户输入只是外部信息源之一。
_Avoid_: 聊天对象、对话伙伴

### 安全

**Capability 模型**:
AVM 的安全基础。对话不知道内存地址就无法访问——不存在遍历根目录的操作。用户态对话看不到 `$MEM`，只知道被内核态告知的 workspace 路径和显式授权的特定节点路径。链接就是这种"特定节点引用"的实现方式：内核态把真实路径挂载为别名设备，对话只能通过别名访问目标内容，看不到真实路径。

**文件安全性**:
对话可以为其拥有的文件设定访问密码，阻止其他对话访问。可注册回调：当其他对话尝试访问受保护文件时，所有者获得通知。
_Avoid_: ACL、RBAC、权限位

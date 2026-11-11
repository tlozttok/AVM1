# AVM 运行时详细介绍

本文是 AVM 运行时的详细说明：计算模型、指令集、内存模型、对话间通信、Python 程序、系统镜像与调试观测。术语以 [CONTEXT.md](../CONTEXT.md) 为准，本文只在使用含义明确的地方简述。

## 计算模型

### 对话循环

对话（Conversation）是最小可调度执行单元。它由一个 Settingup 程序（提示词文本，或 type=`python` 的代码节点）和一条启动用户消息组成，循环执行：

1. 执行器推理一次（LLM 调用，或 Python 执行器调用）；
2. 输出为文本、工具调用，或两者都有；
3. 有工具调用时，Core 执行指令，把每个工具响应加入该对话的输入批次（batch）；
4. 返回不含工具调用的结果时，当前交互结束，对话进入休眠。

工具响应、被投递进来的用户消息、以及对话自己附加的用户内容，都会在下一次推理前合并进消息列表。

### 调度状态

对话处于三种调度状态之一：

- **活跃（Active）**：正在执行推理。同时只有一个活跃对话。
- **休眠（Dormant）**：主动挂起，等待事件或再次被调用。
- **就绪（Ready）**：等待被调度的队列。

"finished" 是概念态：记录被 `close_conversation` 内核级关闭的对话（个体结束）。Core 的调度状态机只有活跃、休眠、就绪三态。

### 排队调度与核心中断

调度采用排队模型：

- 被唤醒的对话进入就绪队列**末尾**（`send_instruction` 唤醒目标、事件 notify 唤醒等）；
- `create_sub` 是唯一例外：亚对话前插，立即执行；
- 亚对话完成唤醒父对话时直接激活（同一调用链）；
- **同一轮可声明多个 `create_sub`**：父对话要等**全部**亚对话返回后才恢复；每个亚对话返回后删除个体，早完成的不会提前唤醒父。
- **亚对话用完即删**：完成写回父、或返回前被关闭，个体立即从对话表移除（级联删除它的子亚对话），监听器与执行器一并清理；`send_instruction` 之后报「对话不存在」。
- **等待亚对话返回的对话不参与调度**：它的历史里有一条未答的 tool 调用（`create_sub`），在被外部消息唤醒时只把内容放进待发批次，不进入就绪队列；调度器选中它也会退回休眠。亚对话结束后父对话立即运行（关闭未返回的亚对话同样先给父补一条 `Error:` 响应并把它插到就绪队首）——父子之间不会插入其他对话。

为防止单个对话占用过多执行权，每轮激活有指令预算（`instruction_budget`，默认 50，镜像 `meta` 可配）。Core 在**工具返回加入批次之后**检查预算：达到预算就中断当前对话，把它放回就绪队列末尾，让出执行权。预算按调用链继承：亚对话与父对话共享同一预算（`is_sub` 且 parent 匹配），`create_cmd` 创建的子对话有独立预算。调用链断开（选中了另一个无关对话）时预算重新计数。

### 中断（外部控制权转移）

中断是**外部**发起的控制权转移请求，与事件是两回事：事件通知"发生了什么"，中断夺取"下一个执行谁"。v1 的中断源只有调试面板——面板的暂停按钮挂起 `operator` 类型中断，向量指向面板的操作员对话（`avm/interrupts.py` 的 `InterruptTable`）。

- 任意线程都可挂起中断（控制台的后端线程就是这么做的）；同类型已有未派发的挂起中断时去重——连点两次暂停与点一次等价。
- Core 在**安全点**派发：每轮推理（含该轮产生的全部指令）执行完之后、下一次推进之前——与预算中断用的是同一边界，状态完整、可恢复。
- 派发把执行权交给向量指向的对话；被打断的活跃对话插回就绪队列**队首**，让出后先跑。目标从就绪/休眠队列中摘除；调用链断开时重置指令预算。
- 没有注册向量、目标对话不存在或已关闭 → 丢弃，原因记 stderr；目标已是活跃对话 → 不动作。最近一次非空派发的记录留在 `InterruptTable.last_dispatch`，经 `GET /api/state` 给面板看（监测帧不含中断数据）。
- 优先级、屏蔽位，以及它与其他中断源（预算中断、事件 `notify`）的关系都不在 v1。

### 交互结束与个体结束

"对话结束"有两种含义，必须区分：

- **交互结束**：对话返回不含指令的结果，当前一轮执行完毕。对话个体依然存在，可被再次唤醒（再次被投递、事件回调等）。交互结束进入休眠。
- **个体结束**：对话被内核级关闭指令 `close_conversation` 关闭（可由对话自身或其他对话调用，不需要对方回应）。关闭后不可再调度、不可再被调用，记录为 finished；其 PLM 状态释放。关闭时目标若还有待执行批次（已投递未执行的指令、未处理的通知），批次一并丢弃，工具响应里会说明。

## 指令集

Core 只执行指令，不做业务决策。全部指令如下：

| 指令 | 作用 | 关键参数 |
|---|---|---|
| `memory_read` | 读取内存，返回给 LLM 的表示 | `ref` |
| `memory_write` | 写入内存（只改已存在路径；列表 `ref` 末段为 `-1` 时追加） | `ref`, `content` |
| `memory_make` | 在父容器里创建新键（`ref` 是父路径、`key` 是键名；只创建、不覆盖；`ref` 可建链；列表 `key="-1"` 追加；响应里的地址才是最终路径） | `ref`, `key`, `mem_type`(str/dict/list) |
| `memory_delete` | 删除内存节点（只删叶子与空容器；挂载点与根不能删；指向树节点的链接可删；整棵子树删除保留给内核态对话） | `ref` |
| `edit_metadata` | 编辑节点的控制数据（ctrl）：set 写入 / get 读取（key 省略返回整个 ctrl）/ del 删除。**只编辑取值，不做语义校验**：`type` 是否合法由 `create_cmd` 消费时判定 | `ref`, `type`(set/get/del), `key`?, `value`? |
| `create_cmd` | 创建子对话（`system_ref` 必须指向 settingup 或 python 程序节点且 `content` 是字符串；只创建，返回 cid；子对话休眠等待指令；本对话保持活跃） | `system_ref`, `para_ref` |
| `create_sub` | 创建亚对话（立即执行；完成自动写回并立即唤醒父；父子之间不插其他对话；**完成或关闭后立即删除个体**；`system_ref`/`para_ref` 要求同 `create_cmd`，`user_ref` 必须指向 str 节点） | `system_ref`, `user_ref`, `para_ref` |
| `send_instruction` | 按 cid 投递指令内容（支持多播；只确认投递成功，不保证返回；投递后本对话继续执行，不自动休眠） | `cid`(int/列表), `content` |
| `close_conversation` | 内核级关闭：关闭指定 cid 的对话（自身或他人，不需要对方回应）；关闭后不可再调度、不可再被调用，其注册的事件监听自动撤销。亚对话不能关闭主对话（`Illegal:`）；亚对话被关闭时个体直接删除，父对话收到 `Error:` 响应 | `cid` |
| `listen_event` | 注册/修改/取消事件监听；监听条件是 handler 节点 `condition` 数据叶子里声明的声明式条件；可带 `priority`/`tier` | `action`, `handler_ref`?, `listener_id`?, `priority`?, `tier`? |

**工具面按对话裁剪**：每个对话有一份工具集（指令名集合，见[对话能力与权限](#对话能力与权限)）。模型看到的工具 schema、PLM 注入的工具函数与 Core 实际接受的指令都按它过滤；宿主控制台 `/api/tools` 始终是全量 10 条。旧对话（无能力声明）默认全部 10 条。

### 地址与键的语法（指令边界校验）

指令在落笔前先校验引用与键的语法；不合法输入得到 `Illegal: ...` 工具响应，内存与事件都不动：

- **引用**（`memory_read` / `memory_write` 的 `ref`）必须以 `$` 开头。不以 `$` 开头的字符串不是内存引用，直接拒绝。
- **引用第一段是根名**：按当前对话的根表精确匹配（`MEM` 是内建根；用户态对话只有软件授予的根与只读 `caps` 自述根）。根表里没有的名字得到统一 `Error: 引用根不可用…`，不区分拼错与未授权；根目标本身缺失时报"引用根无效"。根表里含 `MEM` 的对话保留设备路径简写（如 `$inputs.next`）与 `$conv`/`$batch` 伪根；用户态对话两者都不可用。
- **路径不能有空段**：`$MEM..x` 一类输入报"引用不能有空段"，读侧与写侧同一套语法；`$MEM` 本身是合法的根。
- **键**（`memory_make` 的 `key`）必须是非空字符串、不能含点号——点号是路径分隔符，带点的键建得出来却永远读不回来。空键同样拒绝（它会让 `$MEM.` 与读根混淆）。
- **`memory_write` 不建节点**：写入只修改已存在的地址（中间段或末段缺失都返回 `Error: 路径不存在…——memory_write 不建节点，先用 memory_make 创建`）。新地址一律由 `memory_make` 创建；`memory_make` 的 ref 可以建链，所以"先 make 再 write"是两步。
- **`memory_delete` 的边界**：只删 str 叶子、空容器（dict/list 长度 0）与指向 `$MEM` 的链接节点（删链接本身，目标保留）；`$MEM` 根、设备挂载点、指向设备的链接、设备空间路径、非空容器都返回 `Error:`（整棵子树删除保留给内核态对话，尚未实现）。中间段跟随链接，最终段不跟随。列表元素删除后后续下标前移，响应给出被删位置。
- **`memory_make` 只创建、不覆盖**：目标键已存在（包括设备挂载点物化的链接节点）时返回 `Error: 地址已存在…`，原内容与 ctrl 不变。覆盖会丢整棵子树；覆写设备挂载点还会让设备表与内存树不一致（下一个安全点内核恐慌）。
- **`-1` 是追加位置，不是下标**：`memory_write` 的 `ref` 末段是 `-1` 且父是列表 → 追加一个 str；`memory_make` 的 `key="-1"` → 追加一个 `mem_type` 类型的节点。**读侧拒绝所有负下标（含 `-1`）**，`-2` 及以下在写侧也拒绝。字典上的 `-1` 在读、写、建三处报同一句"字典里不能用 -1（-1 只表示列表追加位置）"。响应给出真实下标（`Success appended str at $MEM.list.3`、`Success created dict at $MEM.list.3`），否则追加出来的节点无法被寻址。
- **同类规则（write 与 make 一致）**：列表已有值时，追加的节点必须与已有元素同类型；列表为空时是新列表——`memory_write` 追加默认 str，`memory_make` 的末尾段按 `mem_type` 建、ref 链中间段追加 dict。旧数据里的混合类型列表没有唯一"同类"依据，追加报 `列表元素类型不一致…`，需重建。
- **`memory_make` 的 `ref` 可以建链**：ref 上缺失的名字段由 make 补建——下一段是 `-1` 建列表，否则建字典；`-1` 段表示在列表末尾追加一个字典节点并继续。例：`memory_make(ref="$MEM.a.-1", key="b")` 在 `a` 是字典列表时追加一个 dict 再建 `b`，响应报 `$MEM.a.3.b`。链上任何一步失败都不留半成品；`key` 仍是单段（名字或 `-1`）。
- **读最新一条**：读容器返回 `list[len=N,elem=类型,metadata=…]`（`elem` 是元素类型：空列表默认 `str`、同类列表给类型名、混合列表给 `mixed(...)`），模型读容器拿 `N` 再读 `N-1` 即可；读侧没有负下标。`memory_make` 成功响应回显键（`Success created str at $MEM.a, key='a b'`），含空格的键也能看清是哪一段。

## 内存模型

内存是全局共享的键值树，真实路径形如 `$MEM.a.b.c`；对话看到的引用以根名开头（`$lib.a`、`$repo.readme`），根名到真实段的映射是对话能力的一部分（见[对话能力与权限](#对话能力与权限)）。四种数据类型：

- **str（StrNode）**：叶子节点，直接返回内容；只有控制数据 `ctrl`（如 `type`），没有描述数据 `meta`。
- **MetaDict / MetaList**：带元数据（描述 + 控制数据）的复合类型。列表用 `-1` 位置追加（规则见上）；可删 str 元素与空列表，删除用 `memory_delete`（删元素后后续下标前移）。
- **链接节点（LinkNode）**：`target` 以 `$` 开头是内存树地址，否则是设备唯一标识符；寻址遇到它即进入目标（见下文）。
- **设备（device）**：挂载到路径上的伪节点（见下文）。挂载点本身在树里就是一个链接节点。

### 元数据：描述与控制数据

MetaDict / MetaList 有两份元数据；字符串叶子只有控制数据：

- 描述（`meta`，字符串；镜像与代码字段名 `metadata`）：给 LLM 快速扫描的文件摘要/描述。LLM 读取时默认只看到摘要和键列表，深入索引后才返回子节点。
- 控制数据（`ctrl`，`dict[str,str]`）：给 Core / 内核态的结构化控制信息，如 `type="para"`（模型参数）、`type="settingup"`（LLM 提示词程序）、`type="python"`（Python 程序）、`type="handler"`（事件处理器）。

两者统称**元数据**：`memory_write` 对容器改的是描述，`edit_metadata` 改的是控制数据。

Core 不写节点所有权：`memory_make` 只创建节点，`memory_write` 不建节点、写已有字符串只改内容并保留它的 `ctrl`，`edit_metadata` 可自由 set/get/del `ctrl` 的任意键。字符串叶子只接受 `ctrl`，镜像里给它写 `meta` 会报 `ImageError`。内存监听注册不按节点所有权授权（见[监听注册权限](#监听注册权限)）。

### 字符串与链接

**str 永远是 str**：字符串值不会被当作引用解引用。`$` 递归解引用与 `&` 单层解引用是已废弃的设计，不存在于当前代码中。

**链接（Link）是内存树节点**（`LinkNode`）：寻址遇到它就进入 `target` 指向的地址。`target` 以 `$` 开头（`$MEM.a.b`）是内存树地址，否则是设备唯一标识符；被指向的设备登记在设备表里。读、写、`memory_make`、`edit_metadata` 都进入目标；`memory_delete` 的中间段进入目标、最终段不进入（删的是链接节点本身）；链接自身不返回描述；目标被删除后链接节点保留但访问报 `MemoryLinkBrokenError`（链接无效），指向未登记设备的链接同样按链接无效处理；成环抛 `MemoryCircularReferenceError`，访问集合是本次解析的，错误只含链接路径链。链接只由内核侧创建（镜像 `mem` 段声明、设备挂载物化、调试原语）；运行期没有创建链接的指令，指向树节点的链接可以用 `memory_delete` 删除（挂载点与指向设备的链接不行）。

### 设备

设备是 `avm.memory_device.MemoryDevice` 的子类，通过接口与对话交互：

- `to_llm_string()`：LLM 读取该设备时看到的表示；
- `resolve_path(path)`：子路径访问（可选，默认抛错）；
- `set_value(value)`：写入（可选）。

读设备本身（`$MEM.<dev>`）得到的是设备自述（`to_llm_string()`）；镜像自带的设备对缺少必需参数的动作返回该动作的用法说明。约定见[系统镜像规范](system-image.md)的 devices 一节。

**设备按名称读写**：设备的子路径用命名子键（如 `next`、`out`、`append`、`latest`），由设备类自己定义；数字下标只用于列表式设备的内容读取（非负）。设备没有"`-1` 是最后一条/追加位置"的语义——`-1` 只属于内存列表的追加位置；设备写入只走命名接口（例如输出写 `$MEM.outputs.out`、读下一行输入用 `$MEM.inputs.next`）。

**设备来源表**（`Core.device_specs`）：挂载地址 → `{path, file, class, args, id, builtin, sha256}`；装载设备（loader）热装的条目另带 `approved_sha256` / `approved_at` / `approved_via`。`sha256` 是设备文件内容哈希，恢复与重放只用它与文件核对。镜像装载与运行期热装（程序包里的设备、装载设备）都会写一条；卸载时（`Core.unmount_device`）与设备表一起删。设备表本身只有实例，没有"这是从哪来的"，重启快照靠来源表在重启后重新加载设备文件、实例化并挂载。

**哈希校验只有一个落点**：`avm/image.py` 的 `mount_device_entry`。文件型设备在 import 之前重算 sha256：请求带 `approved_sha256` 时必须与它相等，否则必须与来源表记录的 `sha256` 相等（镜像首次冷装载两者都没有时，计算并记录实际哈希）；不相等抛 `ImageError`，不 import、不挂载。镜像装载、程序包装载、日志重放、快照恢复共用这一条路径。

运行期装卸设备是纯内存操作（`Memory.mount`/`unmount` 只改设备表与挂载点链接节点，不发事件），所以在指令执行中途调用是安全的。热装设备的重启复原依赖设备文件留在原路径；文件缺失时恢复失败并明确报错。

Core 内置只读信息设备：`$MEM.conversations`（对话列表与单对话详情）、`$MEM.scheduler`（活跃/就绪/休眠与预算）、`$MEM.memory`（内存树摘要）、`$MEM.monitor`（监测帧摘要）、`$MEM.listeners`（事件监听器列表）。镜像可挂载自己的设备，例如 `$inputs`、`$outputs`（见[系统镜像](#系统镜像)）。这些挂载点和镜像设备一样在内存树里有链接节点，因此读 `$MEM` 根目录能看到它们。

### 挂载与链接节点

挂载设备（`Memory.mount`）做三件事：把地址写进设备表、按设备唯一标识符登记设备、在挂载路径物化一个链接节点。标识符缺省取挂载地址，镜像 `devices` 条目可用 `"id"` 覆盖；标识符重复或以 `$` 开头报错。

- 挂载路径的父段不存在时自动物化空 `MetaDict`；父段是链接节点、或该路径已有数据节点时报错（挂载点与数据节点不能同名）。
- 卸载（`Memory.unmount`）同时清掉地址、标识符与链接节点；控制台的设备屏蔽走的就是卸载与重新挂载。
- 整树替换（镜像装载 `core.mem._data = root`）会抹掉先前物化的链接，装载器随后按设备表重建（`Memory.materialize_mount_links`）。
- 读 `$MEM` 根目录列出全部挂载点：键就是链接节点，不再出现"根目录列出了键、读到的却是别的东西"的遮盖。
- 解析权威是内存树遍历：按地址前缀命中设备表只用于事件边界、调试原语（`mounted_devices`、屏蔽）与 `split_path` 的段序列规则。
- `save` 把链接节点按 `{"kind": "link", "target": ...}` 写出（设备本身不落盘）；重载后若没有同名设备，读该链接报"设备标识符未知"。`MetaDict.to_dict` / `MetaList.to_list` 不跟随链接，写 `{"$link": target}` 占位。
- 报错只含请求路径：寻址按链接目标走，错误信息（键不存在、类型不符）用对话请求的原段序列，不暴露真实目标路径；成环错误只含链接路径链。

### 链接设备（LinkDevice，保留兼容）

`LinkDevice` 是设备层的别名（把 `$MEM` 下的真实路径挂载为别名），在内存树节点链接落地后**不要用于新设计**，只保留兼容与既有测试：`devices` 条目 `{"path": "rooms", "class": "LinkDevice", "builtin": true, "args": {"target": "$MEM.game.map.rooms"}}`。

- 读/写/`memory_make`/`edit_metadata` 对别名透明，等同于操作真实目标；挂载它时同样会在挂载点物化链接节点。
- 链接目标只允许 `$MEM` 开头的路径；允许链接指向链接（链式跟随），成环抛 `MemoryCircularReferenceError`，错误只含别名链。
- 错误隐藏：LLM 看到的错误只包含别名路径；真实路径/异常详情输出到 stderr 给运行者。镜像加载期校验失败属于运行者诊断，可显示真实路径。
- 事件透明：注册监听 target 为别名时，权限检查跟随真实目标（memory 目标不做所有者判定；device 走对方 `authorize_listener`）；after 事实事件由真实执行组件发布，但事件 target/payload 仍写别名。

### 模型目录（`$MEM.model_catalog`）

启动自带的普通内存节点（不是设备、不产生事件），由 `Core.install_model_catalog()` 在 `Core()` 与镜像装载之后安装，镜像或检查点里已有的字段不覆盖。作用是让对话读得到"有哪些模型可用、各自要什么参数"，并给 `create_cmd` / `create_sub` 的命名空间校验提供依据。

- `llm`：`source`（`"api"` 已从上游 `/models` 取到 / `"unavailable"` 没取到 / `"image"` 镜像自带）、`reason`（取不到时的原因）、`required`（`"model"`）、`params`（可写进 para 的 API 参数，来自 `LMU._API_PARAM_KEYS`，标注数值型或字符串型）、`models`（列表，每项 `id` 加可选的上游元数据：`name` / `owned_by` / `context_window` / `max_output_tokens` / 输入输出模态 / `reasoning_effort_levels` / `default_effort_level`）。
- `plm`：列表，每项 `name`（`plm.simple` 等）、`required`、`params`（`只有 model 有效；tools 由内核注入；LLM 参数无效`）与来自类文档串首行的说明；宿主独占的 PLM（如 `plm.operator`）不列进这个列表——模型目录是对话可见面。
- 名字（`plm.*` 与带点的模型 id）装在列表项的 `name` / `id` 字段里，不当键——`$MEM` 寻址按点号切段，带点的键读不回来。
- `Core.refresh_model_catalog(timeout)` 拉一次上游 `GET {base_url}/models` 填充 `models`：自己校验信封（`object="list"` 且 `data` 是数组）、除 `id` 外全部按可选字段读、任何失败都收敛成 `source="unavailable"` 并只打一行 stderr。原始响应的解析兼容 openai 各版本（1.x 的 `.json()`、2.x 的 `.http_response`/`.text`），不依赖 SDK 的强类型页面对象。清单只当参考：取不到时不拦启动，也不拦模型名。宿主在启动时调一次（`avm/boot.py` 的 `start_core`）。

### 命名空间校验（输入边界）

`model` 一个字段承载两种命名：LLM 模型名与 PLM 注册表名。`create_cmd` / `create_sub` 在建对话前交叉校验：

- `system_ref` 指向 `python` 程序 → `model` 必须是 `plm` 列表里的注册名，否则 `Illegal:` 工具响应、不创建对话；
- 宿主独占的 PLM（不在对话可见的 `plm` 列表里）→ 统一回 `Illegal: 操作被外部原因拒绝`，不创建对话；拒绝理由只写 stderr（对话不接触宿主/调试实现）；
- `system_ref` 指向 `settingup` 程序 → `llm.source == "api"` 时 `model` 必须在 `models` 里，取不到清单时只记 stderr 不拦；`para` 缺 `model` 与上游清单无关，一律 `Illegal:`。

漏网的错配（例如创建后把 para 的 `model` 改坏）由 `_get_executor` 兜底：它现在在 `advance_conversation` 的 `try` 之内，会留下监测帧再上抛。

## 对话能力与权限

完整的能力模型、事件权限机制与镜像层安全程序写法见 [docs/capabilities.md](capabilities.md)。本节只列运行时要点。

Core 提供权限机制的构件：每个对话的能力（`caps`）、地址空间、监听器分级与 10 条指令的 before/after 事件。**权限策略**（用监听器做 ACL、用控制面调整根、用 Guard 裁决监听注册）在镜像/软件层，Core 不内置 ACL 规则，也不预置保护监听器。

### 对话能力（caps）

`Conversation.caps` 是 `avm/capabilities.py` 的 `Capability`：

| 字段 | 含义 |
| --- | --- |
| `mode` | `kernel`（内核态，根表含 `MEM`）或 `user`（用户态，只有软件授予的根与只读 `caps` 自述根） |
| `groups` | 组标签（镜像层策略使用；进事件载荷 `source.conversation.groups`） |
| `roots` | 根表：根名 → `$MEM` 之后的真实段序列 |
| `tools` | 工具集：本对话可调用的指令名集合 |

默认值：宿主侧创建的对话（`Core.start`、操作员对话、旧检查点恢复）是内核态全能力（`MEM` 根 + 10 条工具）。检查点保存 `caps` 与 `attrs`（自由元数据）。

### 程序声明能力：`value.caps`

程序节点（`ctrl.type` 为 `settingup` 或 `python` 的 dict）用数据节点 `value.caps` 声明子对话能力（`ctrl` 只描述类型与数据组织）：

```json
{"kind": "dict", "value": {
  "mode": {"kind": "str", "value": "user"},
  "groups": {"kind": "list", "value": [{"kind": "str", "value": "worker"}]},
  "roots": {"kind": "list", "value": [{"kind": "str", "value": "lib"}]},
  "tools": {"kind": "list", "value": [{"kind": "str", "value": "memory_read"}]}
}}
```

`ctrl.caps` 不再支持：镜像装载与创建对话都会拒绝。

- `mode` 缺省继承父模式；用户态父对话不能创建内核态子对话；亚对话不能跨模式；镜像 init 强制内核态。
- `tools` 缺省继承父工具集；显式声明时必须 ⊆ 父工具集；未知工具名报错。
- `groups` 缺省继承；用户态父对话只能给子对话父已有的组。
- `roots` 是**根注册表里的名字列表**（见下），只能由内核态父对话授予；同域创建（亚对话、用户态父对话建子对话）只能沿用父根表。
- `workspace` 已移除：程序声明它会得到「`value.caps.workspace` 已移除」错误。内核不再分配工作目录——目录由软件用 `memory_make` 建，根在创建之后写 `$MEM.conversations.<cid>.roots` 动态加（见下）。
- 没有 `caps` 的程序：继承父能力。

### 根注册表与动态授予根（镜像 `security` 段）

镜像顶层可选 `security` 段只含根注册表：

```json
"security": {
  "roots": { "lib": "$MEM.lib", "repo": "$MEM.AVMRepo" }
}
```

- `roots` 是可授予根的注册表（名字 → `$MEM` 路径），装载时校验名字规则与目标**两两前缀无关**，并拒绝落在内核控制面设备路径的目标。`workspace_region` 已移除，出现即报 `ImageError`。
- 用户态对话创建时，`caps.roots` 只能引用注册表里的名字；创建不产生任何目录或子树。
- **运行期授予/撤销**走控制面 `$MEM.conversations.<cid>.roots`：写入是「根名 → `$MEM` 路径」的 JSON 对象，**整体替换该对话的非 core 根表**（省略即撤销，不是合并）。Core 自述根 `caps` 保留、不可覆盖；内核态对话的 `MEM` 根表不能通过控制面改写。
- 写入校验：根名规则、目标非空且不是 `$MEM` 根本身、不落在内核控制面设备路径、与其他存活对话根及根注册表两两前缀无关（相同目标允许共享）、显式授予设备本身允许。
- 每次成功写入记日志效果 `conversation_roots`（含 cid 与完整非 core 根表），随日志重放；`reboot` 丢弃对话态效果，由 init 里的软件重新授予。
- 目录与回收都是软件责任：软件用 `memory_make` 建子树、加根；整棵子树的删除仍未实现。

### 用户态对话看到什么

- 引用：软件授予的根（如 `$lib.greeting`）与只读 `caps` 自述根；`$MEM`、设备路径简写、`$conv`/`$batch` 都得到统一错误。没有内核分配的默认根。
- 工具：只有 `caps.tools` 里的指令；未授权调用返回 `Error: 工具 <名> 未授权给本对话`。
- 创建：`create_sub` 在同域内可用（沿用父根表、工具只收窄）；能否 `create_cmd`/`close_conversation`/`listen_event` 由镜像授予的工具集决定。
- `$MEM.conversations` 单对话详情增加 `mode` / `groups` / `roots` 名字 / 工具数；控制台 `/api/conversations/<cid>` 增加 caps 摘要（根只给名字，不给真实目标）。

### 当前边界

- `.mode` / `.tools` 仍只读；组、根与自由元数据可经对话设备控制面运行期写入（见[内核信息与控制设备](#内核信息与控制设备)）。
- 事件层权限机制：before 覆盖全部工具、载荷分 normal/kernel 两级、监听器 priority/allow/replace、`ctrl.type="handler"`（见[事件模型 v1](#事件模型-v1)）。
- 节点级内存访问权限（读写规则、密码、访问回调）未实现；监听注册的内存目标授权由软件层 Guard 负责（见[监听注册权限](#监听注册权限)）。

## 对话间通信

### 简单投递

`send_instruction` 是对话间通信的唯一原语：按 cid 投递指令内容，内容**原样**放入目标对话的输入批次并唤醒它（进就绪队列末尾）。调用者立即收到确认：单目标为 `已投递到 <身份>（cid=N）`，多播为 `已投递到 cid=[…]`——**只保证送达，不保证对方返回消息**。投递后调用者继续执行，不自动休眠。多播在投递前**整体校验**所有目标的存在性与开启状态：任一不存在或已关闭则整体失败、一条都不投；全部通过后，单个目标被事件拦截/处理失败不阻断其他目标。

想要结果，由对话自己决定等待方式：

- **在内容里要求对方完成时通知自己**：对方把结论作为消息投递回来（仍走 `send_instruction`），收到时被唤醒；
- **注册事件监听**：对"预期行为"（如某内存地址变化、某对话完成）注册监听，处理器返回 `notify` 时 Core 自动唤醒目标（见[事件模型 v1](#事件模型-v1)）。

**父对话不是默认接收者**：创建关系不构成返回通道，AI 的思考不默认透露给其他 AI，输出默认写入共享内存。`create_cmd` 子对话结束时的纯文本输出没有落点（不写回父对话、不自动进 `$MEM.outputs`）；要留痕由程序自己 `memory_write`。

### 消息内容

投递的内容就是普通字符串，**不包装 JSON 信封**（无 `from`/`to`/`icc_id` 字段）。对话需要知道谁发来的，自己在内容里说明。

### 名字与 cid

- 名字来自 Settingup 节点的 `name` 字段（是节点数据，不是元信息）。名字不唯一，冲突时用文件地址消歧。
- 带名字的对话必然有 Settingup 文件；亚对话（直接传提示词）没有名字，其身份 = 父对话身份 + 亚对话 cid。
- `send_instruction` 只用 cid 寻址。名字 ↔ cid 转换是内核职责（工具未实现）。
- 身份回退：没有 `name` 的对话身份为「父身份#cid」（亚对话、无 name 的 python 子对话）；根对话无 name 时身份 = cid。`$MEM.conversations` 列表与投递确认总是带 cid，供名字冲突时消歧。

## Python 程序

Python 程序与 LLM 对话对等：都是 AVM 中的一等执行单元，通过 OpenAI 兼容的消息格式交互（输入 messages 列表，输出 `content` + `tool_calls`）。

内存节点 `ctrl.type="python"` 声明 Python 程序，`content` 是代码。`para` 的 `model` 参数选择 PLM（Python Language Model）类型：

| model | 行为 |
|---|---|
| `plm.simple` | 计算器：单行表达式求值。仅测试调试，暂存、不推荐、不维护。 |
| `plm.function` | 纯函数：system 定义 `handle(input)`，user 传 JSON，返回可 JSON 序列化的值；每次调用全新受限命名空间，无副作用、无跨调用状态。 |
| `plm.machine` | 状态机：system 是状态机 DSL，user 是 JSON 事件；状态/变量跨消息保留在 PLM 实例内，可输出全部 10 条指令的 tool calls。 |
| `plm.command` | 字符串 IO 命令行：system 定义 `handle(cmd)`，user 是命令与参数（整行字符串）；`handle` 可用 `yield from` 调用注入的工具函数，工具响应回到断点继续执行。 |
| `plm.python` | 图灵完备兜底：Jupyter 笔记本式持久命名空间；系统/用户消息都进同一个命名空间；工具调用写成 `yield from`，挂起后从断点继续（不重放前缀）；程序结束后返回 stdout，`return` 字符串追加。 |
| `plm.operator` | 操作员：执行器由调试面板持有而不是模型——每次"推理"等操作员提交一个动作，工具调用原样透传（未知工具名不在这里拦，留给 Core 报未知工具），或提交结束动作让本轮交互结束。只能由 `Core.bind_console_conversation` 创建（需要注入操作员通道）；镜像里声明同名程序节点会在第一次推理时报错。 |

Python 对话的 `model` 必须是注册表（`PLM_REGISTRY`）中的名字，否则报错，不静默回退到 LLM。

**恢复约定**：PLM 的命名空间不序列化。重启快照恢复后执行器重建，Core 会重放 system 消息（setup 代码）让程序重新进入可用状态，并把已有历史标记为已处理——历史里的 user/tool 消息不会被重新执行（副作用不能重复发生）。因此 **Python 程序的跨轮状态必须落在 `$MEM`**，不能只存在 PLM 命名空间的变量里；镜像里以命名空间存状态的程序在恢复后会丢状态。生成器挂起状态同样不序列化：崩溃或快照恢复发生在一次工具调用中途时，当前程序丢失。

### plm.function

- system 消息是 Python 代码，需定义可调用 `handle(input)`；
- user 消息是 JSON 文本，`json.loads` 的结果作为 `input`；
- 代码编译一次；每次调用使用全新受限命名空间（可用 `math` / `json` / `re`），stdout 丢弃；
- 返回值必须可 JSON 序列化，`json.dumps(..., ensure_ascii=False)` 作为 content；
- 不产生 tool_calls，工具消息返回 `""`；错误以 `错误: FunctionPLM: <类型>: <信息>` 文本返回。

### plm.machine

system 消息是状态机定义：

```text
initial: idle

state idle:
  on start { set counter = 0 } -> running

state running:
  on tick if counter < 3 { set counter = counter + 1 } -> running
  on stop -> idle | {"content": "stopped", "tools": [
      {"name": "memory_write", "arguments": {"ref": "$MEM.log", "content": "stopped"}}
  ]}
```

- user 消息是 JSON：`{"event": "<str>", "data": {...}}`；
- 当前 state 的转移按定义顺序匹配 event（支持 `*` 默认）与可选 guard；
- guard/set 表达式在受限沙箱中求值，可用 state 变量、`event_name`、`event_data`、`last`；
- set 块事务化：全部求值成功才写回；
- response JSON 的 `content` 与 `tools` 均可省略；`tools` 的 name 必须是 10 条指令之一；
- tool calls 的 id 形如 `fsm-0`；tool 返回内容追加到 `last`；
- 无匹配转移返回错误文本，状态与变量不变；定义解析错误不抛穿 Core。

### plm.command

system 消息是命令处理程序（Python），需定义 `handle(cmd)`；user 消息是整行命令字符串，程序自己解析命令与参数：

```python
def handle(cmd):
    op, _, rest = cmd.partition(' ')
    if op == 'bump':
        cur = yield from memory_read('$MEM.n')
        yield from memory_write('$MEM.n', str(int(cur) + 1))
        return 'n=' + str(int(cur) + 1)
    return 'unknown: ' + op
```

- `handle` 返回字符串即命令输出（返回 `None` 视为空输出，其它类型报错）；stdout 丢弃；
- `handle` 可以是生成器：注入的工具函数用 `yield from` 产生工具调用，Core 执行后把工具响应字符串送回断点继续（一条命令内可完成「读→算→写」往返）；
- 注入的工具函数有 6 个：`memory_read(ref)`、`memory_write(ref, content)`、`memory_make(ref, key, mem_type)`、`memory_delete(ref)`、`send_instruction(cid, content)`、`close_conversation(cid)`；工具响应原样返回（包括 `Illegal: ...` / `Error: ...` 文本），由程序决定如何处理；其它指令名不可调用；
- 每条命令在全新受限命名空间中执行（程序与辅助代码重新执行），无跨命令状态——要持久化就写 `$MEM`；
- tool calls 的 id 形如 `cmd-0`，每条命令内递增；一条命令最多挂起一个工具调用，命令未结束就投递新命令会丢弃旧生成器；
- 错误以 `错误: CommandPLM: <类型>: <信息>` 文本返回。

### plm.python

system 消息是 setup 代码，user 消息是程序代码，两者执行进同一个持久命名空间（跨轮次保留）。程序里的工具调用写成 `yield from`：

```python
def pack():
    text = yield from memory_read('$MEM.AVMRepo.read."CONTEXT.md".0')
    yield from memory_write('$MEM.system.content', text)
    print('完成：' + str(len(text)) + ' 字符')

yield from pack()
```

- 顶层代码由 PLM 包进一个生成器函数，所以顶层 `yield from` 与 `return` 合法；
- 注入的工具是 10 条指令：名字与参数同指令，返回工具响应字符串（包括 `Illegal: ...` / `Error: ...` 文本），由程序自己处理；
- **挂起与继续**：一次工具调用返回 `tool_calls`（id 形如 `py-<n>`，一次只挂起一个）；工具响应回来后从断点继续，不重放前缀代码；
- 输出：每段 stdout 随该回合返回；程序结束时 `return` 字符串追加到最终输出；`return` 为 `None` 时只返回 stdout，其它类型报错；
- 工具调用必须是 `yield from tool(...)` 的直接形式；直接调用或嵌套在其它表达式里，在编译检查时以 `错误: SecurityError: ...` 拒绝；setup 里会被执行的代码不能调用工具，但 setup 定义的函数体里可以写 `yield from`（不执行）；
- 新的 user 消息会关闭仍挂起的旧程序，从新程序开始（确定性边界）；
- 沙箱：安全 builtins 白名单 + AST 检查（禁 import、禁双下划线属性、禁危险调用），另有工具调用位置检查。

### 确定性

Python 程序输出是确定性的：执行器通过比较上一次输入与本次输入，只处理新增的消息；只有最后一条消息的返回被保留（对同样的输入序列，只有同样的输出）。消息内容直接作为代码/表达式执行，不解析信封。

### 沙箱

PLM 代码运行在受限环境：安全 builtins 白名单 + AST 检查（禁止 import、禁止双下划线属性访问、禁止危险调用名如 open/eval/exec）。它防止 Python 程序访问宿主系统（文件、网络、进程、时钟、熵源）；AVM 内部行为不受限。真正强隔离（子进程 + OS 级隔离）超出 VM 范围。

白名单（`avm/sandbox.py` 的 `SAFE_BUILTIN_NAMES`）：`abs all any bin bool chr dict divmod enumerate filter float format frozenset hash hex int isinstance issubclass callable iter next len list map max min oct ord pow range repr reversed round set slice sorted str sum tuple zip print`，以及常用异常类（`ArithmeticError AssertionError AttributeError Exception KeyError IndexError NameError RuntimeError StopIteration TypeError ValueError ZeroDivisionError`）。`isinstance` / `issubclass` / `callable` / `iter` / `next` 是类型与迭代工具，不构成逃逸面。

白名单**不包含 `BaseException`**：Python 程序（PLM 代码、事件处理器）不能触发"紧急停止"——`raise BaseException("x")` 只是一句普通的 `NameError`。停机信号只来自宿主：人按 Ctrl+C、宿主主动退出，宿主在 `finally` 里写回 `persist_to` 之后退出。

## 事件模型 v1

事件是瞬时 dict，不落盘；由 Core 单调递增生成 `event_id`。**发布者分工**：before 请求事件由 Core 发布（source 为 `{"kind":"core","initiator_cid":cid}`），after 事实事件由实际执行组件发布（Memory 发布 `{"kind":"memory"}`，设备发布 `{"kind":"device","path":"$MEM.<mount>"}`，Core 发布 `{"kind":"core"}`）。

### 事件载荷

每次派发（有监听器匹配时）在事件顶层构造：

- `source.conversation`：发起者（before 的 `initiator_cid` 或 after 的 `payload.actor_cid`）的对话快照：`cid`、`identity`、`name`、`mode`、`groups`、`is_sub`、`is_python`、`parent_cid`、`state`、`para_ref`。
- `target_info`：目标简单信息——`kind`（memory/device/conversation/listener/wildcard）、`ref`（请求别名）、`exists`、`type`、`meta`、`ctrl`、`ancestors`（从根到父的简单信息 `[{depth,type,meta,ctrl}]`，不含真实路径，深度上限 32）、`parent`（目标存在或最长存在前缀下的直接父节点信息）；目标不存在时（make/追加）给 `longest_existing_prefix`（最近存在前缀的别名、depth 与信息）；device 目标给 `device`（设备自述）、`mount`、`remainder`；conversation 目标给对话快照；listener 目标给监听器摘要。
- 内部 `target_ref`：canonical 目标（跟随根别名、链接与 LinkDevice 透明挂载后的真实段序列），只用于匹配与 canonical 比较，不放进处理器载荷。
- kernel 级追加（只有匹配到 kernel 级监听器时构造）：`target_full`（目标节点完整 JSON 安全数据）、`source_context`（来源对话消息列表，保最后 20 条，带 `total`/`truncated`）。

### before 事件（10 种工具全部可拦截）

| event_type | source | target 格式 | 基础 payload |
|---|---|---|---|
| `memory_read` | core | 原 `ref` | `{"ref"}` |
| `memory_write` | core | 原 `ref` | `{"ref", "content"}` |
| `memory_make` | core | 新建完整路径 `ref.key` | `{"ref", "key", "mem_type"}` |
| `memory_delete` | core | 原 `ref` | `{"ref"}` |
| `edit_metadata` | core | 原 `ref` | `{"ref", "type", "key", "value"}` |
| `send_instruction` | core（每个目标一条） | `"cid:<cid>"` | `{"cid", "content"}` |
| `create_cmd` | core | `system_ref` 程序节点（别名） | `{"system_ref", "para_ref"}` |
| `create_sub` | core | `system_ref` 程序节点（别名） | `{"system_ref", "user_ref", "para_ref"}` |
| `close_conversation` | core | `"cid:<目标 cid>"` | `{"cid"}` |
| `listen_event` | core | action=listen：handler 条件的 canonical 目标（无 target 时 `"*"`，预设目标为 `"cid:<自己>"`）；modify/cancel：`"listener:<id>"` | action=listen：`{"action", "handler_ref", "condition", "priority", "tier"}`（`condition` 是**解析后**的条件：events/phases/target/source，target 已 canonical 化；`priority`/`tier` 是请求值，可为 null）；modify/cancel：`{"action", "listener_id", "handler_ref", "priority"}` |

before 语义：操作被请求，不保证到达执行组件；可被 `allow` 放行短路、`replace` 阻断/改写或 `handler_error` 阻断。before 在指令参数校验**之前**分发（例如 `edit_metadata` 的 `type` 非法时也会先经过监听器）：监听器看到的是请求，不是一定会执行的合法操作。这四类生命周期事件的 before 可变 payload 不开放改写，只允许决策动作。

### after 事件（10 种工具，成功时）

| event_type | source | target | 基础 payload |
|---|---|---|---|
| `memory_read` | memory 或 device | 原 `ref` | `{"ref", "actor_cid", "result"}` |
| `memory_write` | memory 或 device | 原 `ref`（追加写入的 target 仍是原 `ref`，真实地址在 `payload.address`） | `{"ref", "content", "actor_cid", "address"?}` |
| `memory_make` | memory | 新建完整路径 | `{"ref", "key", "mem_type", "path", "actor_cid"}` |
| `memory_delete` | memory | 原 `ref`（删除目标） | `{"ref", "deleted_type", "actor_cid"}` |
| `edit_metadata` | memory | 原 `ref` | `{"ref", "type", "key", "value", "actor_cid"}` |
| `send_instruction` | core | `"cid:<目标>"` | `{"cid", "content", "actor_cid"}` |
| `create_cmd` | core | `"cid:<新 cid>"` | `{"cid", "actor_cid"}` |
| `create_sub` | core | `"cid:<新 cid>"` | `{"cid", "actor_cid"}` |
| `close_conversation` | core | `"cid:<被关闭 cid>"` | `{"cid", "actor_cid"}` |
| `listen_event` | core | `"listener:<id>"` | `{"listener_id", "action", "actor_cid"}` |

规则：失败、被 replace、handler_error 的操作不产生 after；after 的 action 固定为 `execute`，决策动作被忽略；after 阶段对 payload 的改写会并入事件，但操作已经执行完毕，不改变任何结果。

### 监听器：类型、级别与顺序

- 监听器是 `ctrl.type="handler"` 节点：`content` 是处理器代码，`condition` 是声明式条件数据叶子。每次事件在全新受限沙箱命名空间执行（只提供 `json` 和受限 builtins），不创建对话、不保留处理器实例状态。**处理器代码与条件在注册时抄副本**：之后改内存里的节点不会自动生效，必须 `listen_event action=modify` 一次重新解析。
- 条件 schema（handler 节点 `condition`，JSON 字符串叶）：`events`（事件类型列表）、`phase`（before/after 列表）、`target`（`exact`/`subtree` + `$` 引用，或 `cid_owner`/`self_lifecycle`）、`source`（`mode`/`groups` 过滤）。匹配用 canonical 目标。
- 用户态对话只能用命名条件子集（preset）：`inbox`（发往自己的 `send_instruction`）、`self.lifecycle`（自己相关的创建/关闭/监听生命周期事件）；自定义条件被拒绝。内核态可以写完整条件，`subtree` 或省略 target 用于全局监听（例如禁止访问表）。
- 已删除的条件形态：`target.kind="owned_by_owner"`、`source.cid`、preset `own_work.memory` / `own_nodes.memory`。handler 节点使用它们时注册报「已移除」（使用错误，不是兼容入口）。
- 旧式 `listen_event` 的 `event_type`/`target`/`phase` 参数已移除；`action=modify` 可更新 handler_ref 与/或 priority，并重新解析 handler 的条件。
- 处理器必须定义可调用 `handle(event)`，返回 dict。返回的 `action` 在 before 阶段可为 `execute`（继续）、`replace`（拒绝/改写，原指令不执行）、`allow`（接受并短路，原指令照常执行）。
- **级别（tier）**由注册者的模式决定：内核态对话注册的监听器可声明 `tier=kernel`；普通对话一律 `normal`，不采信 handler 节点自报。kernel 级拿正常载荷 + `target_full` + `source_context`；normal 级拿不到这两个重块。
- **顺序**：`priority` 降序，同优先级按 `listener_id`（注册顺序）升序。`priority` 是非负整数，缺省 0；`priority>0` 与 `tier=kernel` 只有内核态对话可设置。首个 `allow`/`replace` 短路；`execute` 继续；管道结束仍无决定则执行原指令。
- `allow` 只有 kernel 级监听器可返回；normal 级返回 `allow` 视为处理器错误（before fail closed：原指令不执行、调用者收到 Error）。
- 处理器可改的 payload 字段：`memory_write.content`、`send_instruction.content`、`edit_metadata.value`（仅原操作 `type=set`）。身份字段不可改。
- 处理器可返回 `notify`：`[{"cid": int, "content": str}]`，由 Core 在管道结束后内部投递（与 `send_instruction` 相同语义，但不产生事件、不递归）。**不允许自我投递**：before 阶段视为处理器错误；after 阶段丢弃该条并记入 `handler_errors`。**notify 的目标授权按监听器所有者模式限制**：用户态监听器只能通知注册者自己（否则处理失败）；内核态监听器可通知任意存活对话，由镜像层策略负责。`send_instruction` 不受这条限制。
- before 处理器错误返回 `action="handler_error"` 的事件结果：管道停止、原指令不执行、已累积的 notify 仍投递。after 处理器错误 fail open：记录 stderr、跳过该处理器的 notify、继续后续监听器。监听器失败计数与最近错误写进 `$MEM.listeners`。

### 事件匹配与 canonical 目标

- 事件对象内部有 canonical `target_ref`：按发起对话根表展开别名、跟随链接与 LinkDevice 后的真实目标（`("memory", 段序列)` / `("device", 挂载键, 剩余段)` / `("cid", id)` / `("listener", id)` / `("*",)`）。
- `_match` 用 canonical 目标比较：用户态对话在根 `lib`（→ `$MEM.lib`）下的 `$lib.x` 上产生的事件，会命中内核态安全对话注册在 `$MEM.lib.x` 上的监听器；同一节点用别名与真实路径注册的监听器互相命中。处理器看到的 `target` 与 `target_info.ref` 仍是请求别名。

### 监听注册权限

- 内存目标：Core **不做节点所有者判定**（节点上没有 `ctrl.owner`，也没有别的所有权数据）。监听注册策略由软件层的 Guard 监听器负责：`listen_event` 的 before 事件携带解析后的条件（canonical target 与 `source`），Guard 可以 `replace` 拒绝或 `allow` 放行。镜像没有装 Guard 时注册不受限——这是默认行为，不是漏洞，策略由镜像作者负责装上。
- 设备目标：`listen_event` 的新式条件路径同样不做目标授权——设备 `authorize_listener(event_type, remainder, owner_cid)` 只在宿主侧注册/更新路径上调用；设备路径的监听注册由 Guard 按 canonical target 裁决（待作者确认，见 [docs/capabilities.md](capabilities.md) 的讨论区）。
- `send_instruction`：普通对话只能监听发往自己的消息（`cid:<自己>`）；内核态可监听任意 cid。
- `close_conversation`：普通对话只能监听关闭自己的事件；内核态可监听任意 cid。
- `create_cmd` / `create_sub`：生命周期事件不做所有者判定；普通对话只能监听自己相关的事件，内核态可监听任意程序节点。
- `listen_event` 自身：普通对话只能使用命名条件子集；内核态可以用完整条件与 subtree/任意事件范围。
- 条件 target 为 `subtree` 或省略 target 时可用于任意范围监听；这只对内核态开放（用户态受 preset 限制）。
- `tier=kernel` 与 `priority>0` 只有内核态对话可注册；`listen_event` 的 before 事件再给 Guard 一次否决机会。
- `modify` / `cancel` 只允许注册该监听器的对话执行；modify 可改 handler_ref 与 priority，tier/event_type/target/phase 不可改。

### Guard 监听器

Guard 是软件层（镜像/OS 程序）注册的一个普通内核态监听器，不是 Core 的内置组件：

- 处理器节点 `condition` 形如 `{"events": ["listen_event"], "phase": ["before"]}`，以 `tier=kernel`、`priority>0` 注册，因此先于普通监听器的裁决执行。
- 处理器从 `event.payload.condition` 读解析后的条件（`target.kind` / `target.refs` / `source`），从 `event.source.conversation` 读注册者的 `mode`/`groups`；返回 `{"action": "replace", "response": "..."}` 拒绝，或 `{"action": "allow"}` 放行。
- `listen_event action=listen` 的 before 事件目标就是条件目标的 canonical ref（无 target 时 `"*"`，`cid_owner`/`self_lifecycle` 为 `"cid:<cid>"`），所以 Guard 也可以只用条件注册，不解析 payload。
- `EventSystem.register` 是宿主/内部受信任入口，不经过 Guard；Guard 只管对话的 `listen_event` 指令。

### 监听器生命周期

- **监听器生命周期跟随对话、不跟随 target**：关闭对话时撤销它的全部监听；目标路径后来被 `memory_delete` 删除时监听器悬空（不删、不报错），路径重建后会重新生效。
- 条件同时包含 before 与 after 时，一次成功操作会调用处理器两次；after 阶段返回决策动作被忽略。
- 目标对话可以注册发往自己的 `send_instruction` before 监听并 `replace`，效果是静默丢弃自己的收件箱。
- 亚对话被关闭时自动撤销其监听。

### 内核信息与控制设备

内核设备按用途分四类：

| 类 | 设备 | 语义 |
| --- | --- | --- |
| 观测面（只读） | `$MEM.scheduler`、`$MEM.memory`、`$MEM.monitor` | 读运行时状态；写入拒绝 |
| 控制面（可写） | `$MEM.conversations`、`$MEM.listeners`、`$MEM.roots` | 读实时状态；写固定字段直接改 Core 运行时对象，不在内存树生成节点 |
| 能力自述（只读） | `$MEM.caps` | 按 cid 读对话能力摘要；用户态经保留根 `caps` 读自己 |
| 镜像设备 | 镜像 `devices` 段挂载的设备 | 各自定义，与内核控制面分开 |

控制面共同规则：

- **真相在 Core 运行时对象**：写 `$MEM.conversations.<cid>.groups` 改的是 `Conversation.caps`，不写内存树；读同一路径返回实时值。这些字段不进 `persist_to`，由检查点/内核转储单独保存。
- **写是整体替换**；字段名固定，未知字段与只读字段拒绝；结构校验在设备内完成：对话存在且未关闭、类型正确、根名/目标合法、根目标前缀无关、不占保留区。
- **授权由镜像层策略做**：设备接口拿不到调用者身份，所以设备不判断"谁可以改"；写入在 Core 里是普通 `memory_write`，before 事件 target 是设备路径，镜像层的安全监听器可以 `allow`/`replace`（引导顺序：先注册保护监听器，再开放写入）。
- **只有根表含 `MEM` 的内核态对话能命名这些设备**；用户态对话看不到控制面。
- **写入不改变内存树的值域**：调用者得到普通 `Success set` 响应；after 事件最小化发布（`cid`/`listener_id`、字段、新值），不给完整对象。
- 控制面设备的 `authorize_listener` 允许内核态对话注册监听，观测设备继续拒绝。

**`$MEM.conversations`**

| 路径 | 读 | 写 | 语义 |
| --- | --- | --- | --- |
| `$MEM.conversations` | 按状态分组的列表 + mode/groups 摘要 | — | — |
| `$MEM.conversations.<cid>` | 详情（含 mode/groups/根名/工具数/meta 键） | — | — |
| `...<cid>.groups` | JSON 字符串数组 | 开（v1） | 整体替换组；后续事件载荷立即反映新组 |
| `...<cid>.roots` | JSON 对象 `{根名: {"target": "$MEM...", "core": bool}}` | 开（v1） | 整体替换非 core 根表；Core 自述根 `caps` 保留并标 `core: true` |
| `...<cid>.meta` | JSON 对象（字符串→字符串） | 开（v1） | 自由元数据进 `Conversation.attrs`（与内部 `metadata` 分开） |
| `...<cid>.meta.<key>` | 字符串 | 开（v1） | 单键写入；删除用整体替换 `.meta` |
| `...<cid>.mode` / `.tools` | 字符串 / JSON 数组 | 只读 | 运行期改模式与工具集留后续 |

`.roots` 校验：根名规则；目标以 `$MEM` 开头且非空；内核态对话写 `.roots` 一律拒绝（用户态 `MEM` 根不存在，无法授予）；`caps` 自述根不可覆盖；新目标不落在内核控制面设备路径（显式授予设备本身允许）、与其他存活对话的根目标及根注册表前缀无关（相同目标允许共享）。成功写入记 `conversation_roots` 日志效果。

**`$MEM.listeners`**

| 路径 | 读 | 写 | 语义 |
| --- | --- | --- | --- |
| `$MEM.listeners` / `.<id>` | 列表与详情（含 priority/tier） | — | — |
| `...<id>.priority` | 十进制字符串 | 开（v1） | 非负整数，上限 10000；下一次派发按新优先级排序 |
| `...<id>.enabled` | `true` / `false` | 开（v1） | 启停监听器 |
| `...<id>.<其他字段>` | 事件类型/目标/phase/tier/handler_ref | 只读 | 改 handler_ref 要重解析代码副本，仍用 `listen_event action=modify` |

**`$MEM.roots`**

| 路径 | 读 | 写 | 语义 |
| --- | --- | --- | --- |
| `$MEM.roots` | 根注册表 JSON | 开（v1） | 整体替换 JSON 对象 `根名 -> "$MEM..."` |
| `$MEM.roots.<name>` | 该根目标别名 | 开（v1） | 单条新增/替换；删除用整体替换 |

运行期注册的根在后续创建对话的 `caps.roots` 中可按名字引用；校验与镜像 `security.roots` 共用：根名、目标前缀无关、保留区（内核控制面设备路径）、与其他存活对话根表前缀无关。只有根表含 `MEM` 的内核态对话能命名。

**`$MEM.caps`**（只读）：`<cid>` 给能力摘要（mode/groups/tools/roots 名字与 core 标记）；用户态对话根表自动带保留根 `caps -> $MEM.caps.<cid>`，可读自己的 mode/groups/roots 名字/tools；写入拒绝。

控制面失败行为：对话/监听器不存在或已关闭 → `Error:`；字段未知或只读、JSON 形状不对、组名/根名/目标非法、priority 不是非负整数或超上限、enabled 不是 true/false → `Illegal:`；被安全监听器 replace → `Interrupted:`，值不变。

### 设备自改

- 事件本身瞬时，不提供 `$MEM.events` 历史、不写盘。
- `Core.emit_device_event(device_path, target, payload)` 是设备自改事件入口：只支持 `memory_write` before，source 为设备自身路径，target 必须属于 `$MEM.<device_path>`（用 `split_path` 判定，避免字符串前缀误匹配）；返回处理后的 event，设备据此决定是否执行自改。
- 设备路径的 after 事件不由 `emit_device_event` 产生；设备可在操作成功后通过 `after_event(event_type, remainder, result)` 返回纯 payload，由 Core 统一包装成 after 事件。v1 不实现设备异步自发 after 事件。

## 系统镜像

一个 JSON 文件 + 若干设备 Python 文件 = 一个标准 AVM 系统镜像。规范见 [docs/system-image.md](system-image.md)，要点：

**生命周期三段**（`avm/boot.py` 是加载区，把三段串起来给宿主用）：**创建空核心**（`Core()`）→ **加载镜像**（`load_image(core, path)`，要求核心还没有对话）→ **启动核心**（`start_core(core)`：刷新模型目录、装 API 失败的运行者策略）。宿主通常只调 `boot(image_path)`。

- **显式节点**：内存每个节点都是 `{ "kind": "str"|"dict"|"list"|"device"|"link", "meta"?, "ctrl"?, "value" }`，无任何隐式转换（str 也必须显式包装）。`kind="device"` 是任意层级都可出现的可读性标记，实际挂载在 `devices` 段；`kind="link"` 用 `target` 声明链接（`$MEM` 地址或设备标识符），不写 `value`。
- **para**：`ctrl.type="para"` 的 dict 节点存放模型调用参数（`model`、`temperature`、`use_tool`、`extra_body`、`reasoning_effort` 等），均以字符串存储，每次调用 API 前把数值型参数转回数值。
- **devices**：数组，每项 `{ path, file, class, args, id?, sha256? }`。`file` 是相对镜像文件的 Python 模块路径，`class` 是其中的设备类，**必须继承** `MemoryDevice`，否则加载报错；`id` 可选，覆盖缺省的设备标识符（缺省取 `path`）；`sha256` 可选，给出时装载前核对设备文件内容哈希。这样"一个 JSON + 几个定义设备的 Python 文件"就是一个完整的系统镜像。
- **security**：可选安全段，只含 `roots` 根注册表（`workspace_region` 已移除，出现即报错），见[对话能力与权限](#对话能力与权限)。
- **init**：启动对话配置（`name` 默认 `"init"`，`system`/`system_ref`、`user`/`user_ref`、可选 `para_ref`）。`system_ref` 指向 `ctrl.type="settingup"` 的 LLM 程序节点，或 `ctrl.type="python"` 的 Python 程序节点（PLM 入口：0 号是 Python 对话）；`system` 字面量仍只表示 LLM 入口。入口的 para 按 `init.para_ref` → 程序节点 `value.para_ref` → 镜像顶层 `para_ref` 取，Python 入口在前两者都缺时报错（顶层 para 是 LLM 参数）；`user_ref` 指向 str 节点。只启动 0 号对话，不预置其他对话；其他对话由对话运行期自己 `create_cmd` / `create_sub` 创建。init 强制内核态：程序 `caps.mode` 声明 `user` 时装载报 `ImageError`。
- **persist_to**：运行结束后内存写回路径（异常也写）。
- **meta**：`version` 必须为 1；`instruction_budget` 可配核心中断预算；`snapshot_every` 配定时快照；`guest_programs` 声明 guest 角色（见下）。

### 宿主标志（host_flags）

`load_image(core, image_path, checkpoint_in=None, host_flags=None)` 的 `host_flags` 是宿主在加载时传入的启动标志：不是镜像内容，镜像 JSON 里没有对应的键，标志也不进 `$MEM`。它在挂载设备之前写入 Core，设备可在 `attach_core` 里读一次；核心启动后不再变化。不传则空，设备行为与从前一致。必须是对象、键必须是非空字符串，否则抛 `ImageError`。

当前唯一的标志是 `console`：控制台以 `{"console": True}` 加载镜像，输入设备据此从宿主 IO 总线取行、不再读终端（面板的调试抽屉与当前输入框都喂这条通道；正式 UI/IO 通道按操作面板计划改造为可扩展 IO，尚未实现）。

### 启动模式与 guest 程序

`boot(image, mode="auto"|"resume"|"reboot"|"reset", restore=..., checkpoint_in=...)` 与 CLI `--mode auto|resume|reboot|reset`（默认 `auto`）选择世界从哪来：

| 模式 | 世界来源 | 对话/监听器/调度 | 日志 |
|---|---|---|---|
| `resume` | 最近快照 + 日志尾部重放 | 从快照恢复，不跑 init | 续当前代际 |
| `reboot` | 最近快照的世界段（内存/安全配置/para/设备来源表）+ 日志尾部里的世界类效果 | 全部丢弃，init 重跑，`next_cid` 来自快照 | 归档当前代际日志，开新代际 |
| `reset` | 原始镜像 | init 重跑 | 归档当前日志与快照，开新代际（没有旧输入时留在当前代际） |
| `auto`（默认） | 快照可读则等价 `resume`，否则等价 `reset` | 与等价模式相同 | 与等价模式相同 |

- `resume` 与 `reboot` 都**需要可读快照**：没有快照直接拒绝并提示 `reset`（日志不能单独重建对话）。`restore=<path>` 等价于显式给出快照的 `resume`；`checkpoint_in` 只用于冷启动（`reset`）。`auto` 没有"镜像 + 日志重放"的冷启动路径。
- `reboot` 只重放世界类效果：`memory_set` / `memory_make` / `memory_delete` / `memory_attach` / `memory_detach` / `edit_metadata` / `device_mount` / `device_unmount`；消息、批次、调度、对话创建/关闭/根授予、投递、监听器效果一律跳过。
- **恢复报告**：`boot` 把 `recovery` 写到 `core.recovery` 并打印一行——`mode`、`source`（cold / snapshot）、`world_source`（image / snapshot / snapshot+journal）、`snapshot_seq`、`replayed`（重放记录数）、`world_effects`（世界效果数）、`generation`、`next_cid`、`guest_programs`（角色 → cid）、`interrupt_vectors`、`last_exit`（`normal` / `forced` / `unknown`）、`restored`（重建数量，仅展示）与 `warnings`。`last_exit` 只是数据：VM 不替 OS 写 `$MEM.sys.boot.*`。

**guest 程序**（镜像 `meta.guest_programs`，可选）：把角色名映射到程序节点引用，角色限定 `boot` / `idle` / `shutdown`，宿主的中断向量名就是角色名。

- 冷启动在 init 之后按固定顺序 `boot` → `idle` → `shutdown` 创建内核态对话，写 `attrs["guest_role"]`（resume 靠它找回），注册同名中断向量；`resume` 不重建对话，只按 `attrs` 重新注册向量。声明的角色在快照里找不到、或角色重复时报错——不允许在 resume 时补建（新 cid 会让后续日志错位），只能 `reboot` / `reset`。
- 入口：Python 程序定义与角色同名的生成器，payload 是 `yield from <role>()`；`settingup` 程序的 payload 是角色名加数据行。`boot` 角色在 resume 时由宿主用中断唤醒，payload 带 `last_exit`。
- 声明了 `guest_programs` 的镜像启用**严格调度**：活跃为空、就绪为空、无待派发中断即「空闲」。空闲是 guest 活体错误，Core 向 `idle` 中断向量派发：console 宿主把该向量注册为面板对话（面板在操作员通道上阻塞等待，系统即处于空闲态）；镜像也可声明 `idle` guest 作为无面板宿主的 fallback。派发后仍无法离开空闲（无向量/处理程序没让系统离开）都抛 `KernelPanic`。嵌入式宿主未接入生命周期管理（镜像没有声明 guest 程序、也没有注册 idle 中断向量）时，队列空即 `run()` 返回。
- 没有声明 `guest_programs` 的镜像不启用严格调度：队列空了就是跑完了，`run()` 返回。

**OS 电源与关机约定**（数据面在 `$MEM.sys`，程序实现见 [images/base/README.md](../images/base/README.md)）：

- 操作员的关机请求经 `shutdown` **中断**通知镜像声明的处理程序（宿主挂中断，payload 是角色入口正文）；没有声明该角色时请求停在挂起、VM 不自行断电。OS 软件负责收尾：停服务、写 `clean`，最后写 `$MEM.sys.power.state = "off"`。
- `$MEM.sys.power.state` 变为 `"off"` 的**边沿**是正常断电边界：`run()` 在安全点返回，宿主写快照与 `persist_to` 后退出。`power.state` 是 VM 唯一读取的 OS 节点（断电寄存器）；VM 不写 `clean`、不改 `$MEM.sys.boot.*`——脏关闭判定属于 OS 数据面。

## 持久运行（重启恢复 / 回合日志 / 程序包）

持久运行回答两个问题：进程死后上一轮做到哪里（回合日志），以及重启后怎么接着跑（对话由内核代存，运行态重建）。两者都是宿主层机制，Core 只提供记录点与恢复 API；无日志镜像的行为不变（所有 hook 在 `journal is None` 时是空操作）。

### 承诺

分工来自"谁能持久化"：程序的状态由程序自己写进 `$MEM`；对话没有办法把自己的上下文写进 `$MEM`，所以内核代存它的消息历史、身份与待发批次；运行态（执行器、监听器、设备实例、调度队列）按数据重建。

**硬承诺**：已提交回合对 `$MEM` 的变更（数据、程序、包与安装记录），以及各对话的消息历史、身份（cid / name / para / caps / attrs）与待发批次，重启后一致。依据是回合日志与快照；快照是日志的缓存。

**软承诺（重建结果，不是契约）**：监听器、调度队列、设备表按日志与快照重建；设备实例重新构造；LLM 对话沿用消息历史，Python 对话重放 system 消息（命名空间变量不恢复）。

**明确不承诺**：设备内部状态与设备副作用；未提交回合的副作用（可能已经发生，日志里没有）；宿主特权操作（控制台屏蔽、直接改监听器、外部输入中未读的部分）；调度队列逐字段一致。

日志只对产生它的镜像有效：换镜像视为新的世界。显式 `--restore` 严格——读不出来就报错；`auto` 在镜像声明了 `snapshot_to` 且文件可读时等价 `resume`，否则等价 `reset`；快照能读但引用的设备文件缺失时明确失败。

### 回合日志（journal）

一轮 = 一次推理 + 该轮产生的全部指令。执行完回到 `Core.advance_conversation` 的安全点（`monitor.record` 之后）记一条记录；API 失败重试与未捕获异常不提交，该回合视为未发生。

一条记录是**有序变化序列**（`v=3`，完整形状见 `avm/journal.py` 模块 docstring）：`{"v": 3, "generation", "seq", "cid", "at", "changes": [...]}`。`changes` 按执行顺序列出：

- `message`：本轮真正追加进历史的消息（上一轮指令产生的 tool 响应、投递进来的 user 内容、assistant 消息）——按实际追加记，不按生成它的效果推导：before 事件处理器可以改写 payload，指令最终写入的值才是状态；
- `effect`：语义效果——内存写/建/删、`edit_metadata`、对话创建/关闭/删除、投递、监听器注册/修改/取消、设备装卸。重放期间不调模型、不派发事件、不访问设备；设备写只记审计条目（`device_write`），重放时跳过；
- `batch`：待发批次；只在相对上次提交有变化时写（带 `cid`，可能落在别的对话上——投递给目标加输入）；
- `scheduler`：调度增量（活跃对话、就绪/休眠的增删、finished、预算计数、下一个 cid），只在有变化时写。增量基线是该对话上次提交后的调度状态；装载/快照恢复后对齐，所以第一次提交包含 `run()` 之前的活跃选择。

记录格式显式版本化为 `v=3`。旧格式（带 `history` / `call` / `batch` / `scheduler` / `effects`、没有 `v`）与任何 `v != 3` 的记录都不读取也不迁移：启动时直接报错并提示 `reset` 重建世界。

**代际**：每条记录带 `generation`（本次启动所处的日志代际）；新代际从 `seq=0` 开始。`resume` 续用当前代际；`reboot` / `reset` 由宿主把当前日志改名为 `<base>.<generation><ext>` 归档，再开新代际。记录缺少代际按格式错误拒绝。

记录点：内存写入口在指令执行体里记（带调用者地址空间与事件处理后的真实值）；对话生命周期在 `_build_conversation` 的指令入口、`_close_conversation`、`_delete_conversation` 与亚对话返回/关闭时记投递；动态授予根记 `conversation_roots`；监听器在 `EventSystem.register/register_condition/modify/cancel/cancel_all_for_owner` 记。宿主侧特权操作（控制台屏蔽设备、就地编辑监听器）不记录。

重放（`Core.replay_journal`）：按 seq 连续应用，不连续直接报错；流式逐条读、逐条应用（内存只留一条记录）；应用位置 `_journal_seq` 推进，同一段日志重放两次第二次是空操作（幂等）。日志是 JSONL，每行 flush+fsync；半行/坏行之后的内容按未提交处理——半截日志重放等于上一条提交后的状态。

恢复路径另有断言：`resume` 重放结束的 seq 必须同时等于「快照 seq + 尾部条数」与日志末 seq；快照代际与日志代际不一致时拒绝重放，提示 `reboot` 或 `reset`。`reboot` 重放世界尾部时同样校验 seq 连续与代际一致。

### 重启快照（snapshot，缓存）

`avm/snapshot.py`；`format="avm.snapshot"`、`version=2`（显式版本化，旧版本拒绝恢复并提示 `reset`），`generation` 必须存在且为正整数。形状与内核转储同源（调度、内存、安全段、对话、监听器、设备表），差别：不截断、带 `seq`（已提交的日志位置）与 `generation`（日志代际）、带 `device_specs`（设备来源表）、对话额外带 cid/parent_cid/待发批次。它是加速重启的缓存；没有快照就没有可恢复的对话态。

`Core.snapshot_to(path)` 只在安全点调用（宿主正常退出路径，以及定时快照触发）。`Core.restore_from(path)` 顺序：清空调度/对话/监听器 → 换内存树 → 按来源表重建设备并重新物化挂载点 → 按 cid 重建对话与父子引用 → 恢复监听器 → 安全配置/para_ref/下一个 cid → 跑安全点自检。对话身份与历史按硬承诺保留；设备内部状态不序列化，按 spec 重建实例。

**设备哈希校验**：恢复文件型设备时用来源表记录的 `sha256` 重新核对设备文件，不一致拒绝恢复（`SnapshotError`，含设备名、路径与期望/实际哈希）；来源表缺少 `sha256` 的快照直接拒绝恢复并提示重新审批装载。快照里的 `approved_sha256` 只是历史审批事实，不参与本次核对（程序包的批准哈希是包文件哈希，不是设备文件哈希）。设备文件缺失同样明确失败。

**定时快照**：镜像声明 `meta.snapshot_every` 为正整数时，每 N 个已提交回合在安全点写一次快照（写失败只警告，日志仍是依据）。恢复时从最近快照 + 尾部重放，崩溃后最多重放 N 个回合；镜像不声明时只在正常退出写快照。

重启恢复的入口与语义见[启动模式与 guest 程序](#启动模式与-guest-程序)：`resume` = 快照 + 日志尾部，`reboot` = 快照世界段 + 日志尾部世界类效果，`reset` = 原始镜像；`auto`（默认）在快照可读时等价 `resume`，否则等价 `reset`（进程崩溃、快照落后或损坏时没有可恢复的对话态，日志不单独重放）。

### 程序包（packages）

`avm/packages.py`：一个 JSON 程序包 = 一段内存子树 + 若干设备，运行期安装到 `$MEM.packages.<name>.<package_version>`，同时写安装记录 `$MEM.package_records.<name>.<package_version>`：来源路径、包文件 `sha256`、安装时间、`package_version`、设备清单（每项含设备 `mount`/`class`/`path`/`sha256`，以及审批字段）；经 loader 安装时顶层另带 `approved_sha256` / `approved_at` / `approved_via`。包格式见模块 docstring。

- 安装：校验 → 建子树 → 挂设备 → 写记录 → 更新 `$MEM.packages.<name>.current` 指针；任一步失败回滚已挂设备与已建节点（回滚不进日志，只有成功安装才提交效果）。
- 卸载：卸设备 → `detach` 整棵子树 → 删记录；包内路径的悬空引用不主动清理，访问得到既有的链接无效/路径不存在错误。
- 升级 = 安装新版本后 `current` 指向新版本；不做包依赖解析、版本约束与签名校验。
- 设备条目的 `path` 是相对包子树的挂载路径（`io` → `$MEM.packages.<name>.<version>.io`），卸载时挂载点链接随子树回收。
- **版本号是内存键，不能含点号**（点号是路径分隔符；引号段寻址未实现）。示例：`1_0_0`。
- 装载设备（`images/packages/loader.py` 是演示实现）把"装包/卸包/热装设备"变成对挂载路径的一次 `memory_write`。`install_package` 与 `load_device` **必须带 `approved_sha256`**（操作员审批得到的文件内容哈希），`password` 字段一律拒绝；装载器把哈希透传给 `install` / `mount_device_entry`，内核在 import 前重算并核对，失败不 import、不挂载。`uninstall_package` 不需要审批。设备不发事件、错误信息不回显写入原文。
- 审核的强制力是人的仪式：内核不检查发起者——任何能写装载设备路径的对话，知道哈希就能装载；谁能写那个路径由镜像层监听器决定。

### 边界

- 不做日志自动压缩；代际归档由 `reboot`/`reset` 在启动时完成，同一代际的尾部长度仍由定时快照 + 手动 `truncate(seq)` 控制。不做并发与调度状态机改动。
- 宿主侧外部输入目前只有调试通道 `POST /api/io/input`：发布到宿主侧 IO 总线，不进日志、不产生事件、不重放。正式输入/输出按操作面板计划改成可扩展 IO 底层（宿主 IO 总线 + `$MEM` 里的设备前端），尚未实现。宿主直接改批次/监听器同样不在日志里。

## 调试与观测

运行入口（`python main.py <image.json>`）支持观测参数：`--frames` 在运行结束后打印监测帧摘要（stderr），`--transcript <path>` 把 API 调用全文落盘；`--mode auto|resume|reboot|reset` 选择启动模式（见[启动模式与 guest 程序](#启动模式与-guest-程序)），`--restore <snapshot>` 等价于显式给出快照的 `resume`。这些参数也可写在 `debug.json`（`image` / `frames` / `transcript` / `checkpoint_in` / `restore` / `mode`），`--debug-config <file>` 换一份配置。

### 监测器（Monitor）

每轮对话推进结束后生成一帧（Frame），第 0 帧为基线。帧内容：

- `lmu`：最近一次执行的结果（`result`）、思维链内容（`reasoning`，与 result 同截断策略）、工具调用摘要（`tool_calls`）、事件摘要（`events`，每条含 event_id/event_type/phase/target/source/action，失败时含 error/handler_errors）、耗时、错误；
- `sched`：活跃 cid、就绪队列、休眠队列、下一个 cid；
- `conversations`：每个对话的状态、消息条数、输入批次内容。

提供 `trail()`（单变量跨帧轨迹）、`diff()`（两帧间变化）、`find()`（按谓词找帧）等查询；`--frames` 在运行结束后打印帧摘要（stderr），`--transcript` 把每次 API 调用全文写入文件（写完一帧就落盘；全文序列化失败时该帧降级为占位并在 stderr 报一行，观测面不会因为不可序列化的值终止内核）。

### 内核转储（core dump）

内核死亡时落一份现场快照，**只作诊断，不是恢复输入**（恢复依据是回合日志，快照是它的缓存，见[持久运行](#持久运行重启恢复--回合日志--程序包)）。

- **触发**：`KernelPanic`（不变量被破坏）与未捕获异常逃出 `run()`；`RunnerStop`（运行者选退出）与 Ctrl+C 不转储。
- **落点**：`<镜像目录>/out/<镜像名>.core.json`；镜像顶层 `core_dump_to` 可覆盖，`core_dump: false` 关闭；宿主 `--core-dump <path>` / `--no-core-dump` 优先。
- **内容**：`format`/`when`/`image`/`host_flags`、`error`（类型、消息、traceback、`panicked`）、`scheduler`、`memory`（与 `persist_to` 同格式，不截断）、每个对话的状态与历史（保最后 200 条，`messages_truncated` 标出）与待发批次、`listeners`、`devices`、`interrupts`、监测帧（保最后 200 帧）、`truncated`。
- **转储自身不抛**：任何一段序列化失败降级为占位；写文件失败只记 stderr，不改变原始异常与退出码。
- 文件含提示词与 API 参数等敏感内容：只落在本机镜像目录，不自动上传。

### 单对话检查点（调试子集）

重启续跑走[持久运行](#持久运行重启恢复--回合日志--程序包)的重启快照；本节是调试用的子集，只保存单个对话，调用方自己决定何时保存、恢复后分配新 cid。

`Core.save_conversation(cid, path)` / `Core.restore_conversation(path)` 保存/恢复单个 LLM 对话：序列化无损、确定性；检查点包含 `caps` 与 `attrs`，旧检查点恢复为内核态全能力，恢复后发送给 API 的消息前缀与保存时逐字节一致（可命中厂商前缀缓存）。历史末尾是未配对的工具调用时（如等待输入响应），保存会自动截到最后一个完整交换。v1 不支持 Python 对话。

输入设备支持操作者命令 `/save`（可带路径 `/save <path>`）：在输入前截取，保存当前活跃对话的检查点后继续等待真实输入——对话感知不到该命令。恢复在 host 侧进行（`restore_conversation`），不做 AVM 内 `/load`。

启动时加载：`python main.py <image.json> --checkpoint-in cp.json`（或 debug.json 的 `checkpoint_in` 字段）会**跳过 init**，把检查点恢复为 cid 0 的就绪对话直接续跑——这是"上次保存的对话下次启动接着用"的入口。它只用于冷启动（`reset`；`auto` 在无快照时等价 `reset`）：`resume` / `reboot` 与检查点同时给出会报错。

终端输入：等待输入时显示正在等待的对话身份提示（如 `[tester（cid=0）等待输入，/save 保存检查点]`），不再是黑屏裸等；已启用 readline 行编辑（历史/方向键，中文输入法组合友好）；终端非 UTF-8 编码会警告一次——中文乱码通常源于终端编码而不是程序。

### 调试面板（Console）

`python -m avm.console <image.json>`（等价入口 `scripts/run_console.py`，后者先清洗本进程的 `NO_PROXY` / `no_proxy`）起一个 HTTP 服务：默认 `127.0.0.1:8700`（`--port` / `--host` / `--checkpoint-in` / `--mode` / `--restore` 可改），只绑回环。面板凭证是启动时生成的一次性 token（打印到 stderr 并写入镜像目录 `out/console.token`，登录本身即凭证）：`/api/*` 做同源检查（跨源 403）与 token 校验（缺失/错误 401），`--allow-multiple-clients` 关闭单客户端准入（默认第二个客户端 409，`/api/session/*` 例外）。非 `/api/` 的 GET 都按面板产物处理：`web/dist` 构建过就是面板，没构建时 `/` 给构建提示页。镜像以 `host_flags={"console": True}` 加载。

**当前输入通道**：只有 `POST /api/io/input` 调试通道——发布到宿主侧 IO 总线、唤醒正在等 `$MEM.inputs.next` 的读，不进日志、不产生事件、响应带 `debug: true`。正式输入/输出要按操作面板计划改成可扩展 IO 设备（宿主 IO 总线 + `$MEM` 设备前端 + 可寻址输入/结构化输出/手动打断），该改造尚未实现，输入路径保持冻结。

**启动与构建**：

```bash
python scripts/run_console.py images/demo.json   # 起后端；前端构建过的话 / 就是面板，没构建则给出构建提示
cd web && npm install && npm run build   # 构建面板到 web/dist（不入库）
cd web && npm run dev                    # 开发期：Vite 代理 /api 到 AVM_CONSOLE_URL（默认 http://127.0.0.1:8700）
```

`scripts/run_console.py` 与 `python -m avm.console` 等价，只是先给这个进程一份清洗过的 `NO_PROXY` / `no_proxy`：装了代理工具的机器上，变量里已经带方括号的 IPv6 项（如 `[::1]`）会让 httpx 建不出客户端，控制台会在第一轮模型调用前崩掉。脚本只改自己进程的那份环境，父进程与 `.env` 都不动。

面板自身：`cd web && npm run test:unit` 跑纯逻辑用例，`cd web && npm run e2e` 起一个脚本化 MockLMU 的控制台并在浏览器里跑五个场景（观测、工具选择、三步交互、喂输入、产物由 console 托管）。

- **操作员对话（保留 cid -1）**：cid 是作者规定的保留值，不参与正常分配；控制台在 `boot` 返回后、`core.run()` 之前把 `plm.operator` 通道绑到它，没有就创建。对话随快照与日志正常恢复：日志里出现它的回合而它还不存在时（只靠日志恢复），重放先补建它，回合按普通记录应用；快照里它与普通对话一起保存和恢复。**隐藏标记在对话 attrs**（`host: "console"`、`hidden: "1"`），不是内存树节点：对普通对话隐藏——`$MEM.conversations` / `$MEM.scheduler` / `$MEM.caps` 列表与监测帧过滤，`send_instruction` / `close_conversation` 与 `$conv.<cid>` / `$batch.<cid>` 一律按「不存在」处理；隐藏只由对话 `attrs` 决定，cid 不参与判定。隐藏只是显示与寻址策略，不影响恢复，宿主控制台自己仍能看到它（`/api/state` 与 `/api/conversations/-1`）。初始进休眠队列、不进就绪队列——只有中断能激活。它活跃就等于全局暂停（其他对话都不推进）。`POST /api/interrupt` 挂起 `operator` 中断（在下一个安全点生效；`core.run()` 已返回时不再生效），`POST /api/resume` 让它结束本轮、让出执行权。`POST /api/action` 提交一条动作，未暂停时先自动中断；未知工具名有意允许（响应里有 `known_tool` 与提示）。
- **读取端点**：`GET /api/state`（运行状态、调度、对话（不过滤 cid -1）、中断、屏蔽账本、生命周期、恢复报告、IO 总线状态）、`/api/tools`（十条指令表）、`/api/conversations/<cid>`、`/api/memory`（可带 `ref`，整树无分页）、`/api/memory/diff?from=&to=`、`/api/listeners`、`/api/devices`、`/api/events`（SSE：面板输出与调度变化的有序流，按 `seq` 增量读）、`/api/io/stream?after=`（SSE：宿主 IO 输出流）。
- **介入端点**：`POST /api/action`（面板对话执行一条工具调用）、`POST /api/io/input`（调试输入 → 宿主 IO 总线，200 返回 `debug: true`）、`POST /api/system/shutdown`（挂 shutdown 中断通知 OS，见下）、`POST /api/system/force-stop`（操作员强制退出，见下）、`POST /api/device/inspect`（只读检查设备文件：路径/大小/sha256/类名/挂载/args/现有来源表条目，不 import、不挂载、不写内存）、`POST /api/device-action`（调设备自己声明的调试动作，如给 `$inputs` 投喂一行）、`/api/memory/snapshot`、`PATCH /api/listeners/<id>`（就地启停监听器、改 priority/tier）、`POST|DELETE /api/mask/memory`、`/api/mask/device`（临时摘下一个内存节点或卸载一台设备，`DELETE` 还原；调试原语不产生事件）、`POST /api/session/acquire|release`（单客户端席位）。
- **关机与强制退出**：`POST /api/system/shutdown` 不执行关机语义，只挂 `shutdown` 中断通知镜像声明的处理程序（中断是 VM 通知软件的通道）；面板占执行权（系统空闲态）时，关机按钮同时在操作员通道提交一次 `{"end": True}`，让阻塞的面板回合回到安全点才能派发中断。OS 软件收尾后写 `$MEM.sys.power.state = "off"`，宿主在安全点看到该边沿后写快照并退出。没有声明 shutdown 处理程序时请求停在挂起（warning），VM 不自行断电；请求关机 `force_stop_after` 秒后可用 `POST /api/system/force-stop` 强制退出——**不写快照、不写 clean**，响应 `{"status": "force-stopped", "snapshot_written": false}`。
- **界面当前接的**：三栏界面（对话列表 / 内存索引与对话·内存·设备详情 / 暂停与动作·事件流）；对话列表与详情带 caps 摘要（mode/groups/根名/工具集）、暂停与继续、十条指令的结构化动作表单（下拉 +「其它（手输）」+ 原始 JSON 兜底）、事件流、主输入框与调试抽屉（当前都走调试 IO 总线；正式可扩展 IO 待实现，标注不重放）、设备审批面板（先用 `/api/device/inspect` 取哈希，再带 `approved_sha256` 提交装载）。内存索引照树画：链接节点一行，没有子节点可展开；挂载点那行标出设备类型（`device InputsListDevice`），点它打开设备视图，描述与动作来自 `GET /api/devices`。监听器编辑、屏蔽还原、快照与 diff 视图、设备动作全量都已有后端端点、界面未接。
- **边界**：面板对话的消息随每次动作增长，且不能存检查点（v1 检查点不支持 Python 对话）；面板输出是容量 4000 条的有界环形缓冲，丢弃数记在状态里；`GET /api/devices` 的动作账本只留最近 50 条；阻塞读的路径末段是已知会阻塞的命名子键（当前只有 `next`，如 `$MEM.inputs.next`）时面板先标出、点击才发请求（8 秒超时）。

### 思维链内容

LLM 响应中的 `reasoning_content`（思维链）被完整存入对话历史，并在后续请求中随 assistant 消息回传（无值则空字符串）——这是厂商协议的硬性要求，也是"回传所有推理内容"的既定策略。监测器记录它（与普通内容相同的截断策略）；`$MEM` 信息设备不暴露它（外泄控制靠禁止内部访问，不靠不记录）。

### 错误输出

错误的落点按"能否由调用者输入决定"分四类（依据见 `docs/private/Agents/proposed/discussion-kernel-stability-2026-10-15.md`）：

- **非法操作**：参数在指令入口可判定为不合法（缺必填、类型不符、枚举外、`ref` 不带 `$`、地址语法错误、字典上用 `-1`、删除根/挂载点、程序类型与 `model` 命名空间不配、`para` 缺 `model`、向自己投递、向主对话发送、创建宿主独占 PLM）→ 拒绝，回一条 **`Illegal: ...`** 工具响应，状态不动，对话继续——调用者改参数可重试。schema 只有一个来源（`Instruction.tool_def`），仲裁层从它派生校验；边界内的策略性拒绝同样用 `Illegal:`。
- **异常返回**：操作合法而世界与期望不符（键不存在、越界、类型不符、非空容器、链接无效/成环、设备拒绝）→ 返回 **`Error: ...`** 工具响应——换办法，而不是重试同一参数。**两档前缀是给调用者的唯一区分依据**（LLM 据此选择改参数重试还是改道）。
- **协议层拒绝**：工具名未知、参数不是合法 JSON、`call_id` 不可用 → 细节只写给运行者（stderr），对话侧回一条中性占位（`（该工具调用未执行）`），保住 assistant 的 `tool_calls` 与 tool 消息一一配对。
- **内核恐慌**：六条具名不变量（调度互斥、活跃有来源、历史配对、值域、设备表一致、预算计数）在每轮安全点自检，违反即 `KernelPanic` 并留监测帧。可判定的输入不该走到这里——出现说明某条边界少了一条判定。
- **内核缺陷**：约定错误类之外的异常不再被读路径吞掉；记录监测帧后向上抛，进程退出。

API 调用失败（缺 key、断网、超时）不再结束进程：内核记帧后停下问运行者，终端给三个选择——更新环境（重建客户端后重试本轮）、继续（直接重试）、退出（写回内存后干净退出）。宿主通过 `Core.api_error_policy` 安装策略，默认实现见 `avm/runner_prompt.py`；EOF 与非终端输入按退出处理。

**一个工具调用只写一条工具响应**。after 事实事件的发布失败只记 stderr（fail-open），不改调用者看到的返回值——发布属于观测面，让它再写一条同 `call_id` 的响应会破坏"assistant 的 `tool_calls` 与 tool 消息一一配对"，下一次请求会被 API 拒绝。同理，亚对话的返回通道只在第一次完成时使用：再次完成不写回父对话（亚对话本不该被再次唤醒，这条是兜底）；亚对话在返回前被关闭时，内核给父补一条 `Error: 亚对话（cid=N）在返回结果前被关闭`。**亚对话不能向主对话（含祖先）发送消息**：`send_instruction` 会返回 `Illegal:`，返回只能走 `create_sub` 的通道；对等待返回的父对话，`send_instruction` 与事件 `notify` 的内容只进批次、不唤醒它。

Python 程序（PLM 代码、事件处理器）手里的"紧急停止"按钮已收掉：沙箱不再发 `BaseException`，`raise BaseException("x")` 变成普通错误文本；停机信号只来自宿主（Ctrl+C 写回内存后退出）。

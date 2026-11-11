# 如何设计 AVM 程序

本文是设计 AVM 程序的导引。术语以 [CONTEXT.md](../CONTEXT.md) 为准；运行时的机制细节见 [docs/runtime.md](runtime.md)。

## 程序放在内存里

一个程序由两部分组成：

- **Settingup 节点**：`content` 是程序文本（提示词，或 type=`python` 的代码）。节点的 `name` 字段是对话的身份。
- **para 节点**：`model`、`temperature`、`use_tool` 等调用参数。`model` 有两种命名：配 settingup 程序时是 LLM 模型名，配 python 程序时是 PLM 注册名（`plm.simple` 等）。两种命名由 `create_cmd` / `create_sub` 在建对话前按 `$MEM.model_catalog` 交叉校验，配错会得到 Error 工具响应而不是等第一次推理报错。

对话由 `system_ref` 指向的程序节点（或入口对话的字面量）和一条用户消息启动。`create_cmd` / `create_sub` 的 `system_ref` 必须指向 `ctrl.type="settingup"` 或 `"python"` 的程序节点，指向 str 节点或无类型节点会报错。程序文本在内存中，对话可以读取、检查、修改其他节点——AVM 的元编程基础在这里。

手工用 `memory_make` + `memory_write` + `edit_metadata` 逐个拼出合法节点容易漏字段或写错 `ctrl.type`。base 镜像预置了构造器 PLM `plm.builder`（入口 `$MEM.sys.programs.builder`，para `$MEM.para.builder`）：把一组节点规格包成一条 JSON 交给它，它按「探测 → 构造/更新 → 直通指令 → 验证」发完工具调用并返回 JSON 摘要；`settingup` / `python` / `handler` / `para` 四类节点的字段形状、`create` 与 `update` 语义、`calls` 直通与摘要格式见 [images/base/README.md](../images/base/README.md) 的「构造器 PLM（plm.builder）」一节。

## 声明对话能力（内核态 / 用户态）

程序节点用数据 `value.caps`（一个 dict 数据节点）声明由它创建的对话的能力：

```json
{"kind": "dict", "ctrl": {"type": "settingup"},
 "value": {"content": {"kind": "str", "value": "..."},
           "caps": {"kind": "dict", "value": {
             "mode": {"kind": "str", "value": "user"},
             "groups": {"kind": "list", "value": [{"kind": "str", "value": "worker"}]},
             "roots": {"kind": "list", "value": [{"kind": "str", "value": "lib"}]},
             "tools": {"kind": "list", "value": [{"kind": "str", "value": "memory_read"}]}}}}}
```

`ctrl.caps` 不再支持；能力属于数据，`ctrl` 只描述节点类型与数据组织。

- `mode` 缺省继承父对话。内核态父对话可以创建内核态或用户态子对话；用户态父对话只能创建用户态子对话，亚对话不能跨模式。
- `roots` 是镜像 `security.roots` 注册表里的名字列表，只由内核态父对话授予；用户态对话拿到的是根名到真实子树/设备的绑定，自己看不到真实路径。
- `tools` 缺省继承父工具集；显式声明时只能是父工具集的子集。模型看到的工具表、PLM 注入的函数与 Core 实际接受三处一致。
- `workspace` 已移除：内核不分配工作目录。程序创建出用户态对话后，由软件用 `memory_make` 建目录，再把「根名 → `$MEM` 路径」全量写进 `$MEM.conversations.<cid>.roots` 授予根（整体替换，省略即撤销；Core 的 `caps` 自述根不可覆盖）。
- 没有 `caps` 的程序继承父能力（旧镜像行为不变）。

写用户态程序时：提示词只使用软件授予的根名（例如 `$lib`），不要写 `$MEM`；工具集按需要声明得尽量小；需要长期个体时由内核态对话创建（用户态是否能用 `create_cmd` 取决于工具集）。写内核态程序时：它持有 `$MEM` 与默认全部工具，负责建立权限层、建目录并授予根、注册程序。

能力在创建时声明；创建之后，组与根可经控制面 `$MEM.conversations.<cid>.groups` / `.roots` 运行期改写（`.mode` / `.tools` 仍只读），监听器的 priority/启停可经 `$MEM.listeners.<id>` 改写。

## 一个对话做一件小事

每个对话只承担一个微小、可管理、可追溯的任务。从概率模型角度：模型倾向认可自己刚产出的内容（否则不会写），而另一个模型看到旧内容倾向质疑（否则不会被派来）。小的任务边界是隔离这两类偏差的物理手段；对话之间的结论靠显式消息传递，不靠共享上下文。

## 用指令表达控制流

对话不直接"执行"，只做两件事：输出文本（结果）或调用工具（指令）。Core 执行指令并把响应放回批次。控制流是提示词里写好的循环协议：读输入 → 处理 → 调用工具 → 读返回 → 写输出。行为边界靠提示词约束，不靠代码强制。

## 选择调用关系

按三个维度选：

| 需求 | 用 | 说明 |
|---|---|---|
| 创建独立执行单元，之后按需发指令 | `create_cmd` | 父不休眠；子休眠等指令；投递靠 `send_instruction` |
| 立刻执行一件小事并要结果 | `create_sub` | 父休眠；亚对话完成自动写回唤醒父 |
| 给已有 cid 的对话发消息 | `send_instruction` | 支持多播；只确认送达，等待方式由对话自决 |

## 记住返回通道不是默认的

创建关系不构成返回通道。想让对方把结果发回来，在指令内容里说明（对方完成后用 `send_instruction` 投递回来），或注册事件监听（事件模型 v1 已实现，用 `listen_event` + Python 处理器 + `notify` 内部投递）。其他对话要知道结果，可以监听生命周期/内存事件并让 Core 用 `notify` 唤醒自己。

## 用 listen_event 表达等待

- 监听器是 `ctrl.type="handler"` 节点：`content` 是处理器代码，`condition` 数据叶子声明条件（events/phase/target/source）；每次事件在全新受限沙箱中执行，定义 `handle(event)` 返回 dict。
- 用户态对话只能用命名条件子集（`inbox` / `self.lifecycle`），内核态可以写完整条件。内存监听注册不做所有者判定：策略由软件层的 Guard 监听器裁决（它监听 `listen_event` 的 before 事件，读解析后的条件）。
- 处理器可返回 `action="replace"` 拦截 before 事件，或改写允许的 payload 字段；返回 `notify` 可让 Core 向指定对话投递消息并唤醒它。`priority` 只有内核态能设大于 0；`tier=kernel` 只有内核态可注册。
- `notify` 不允许投递给当前对话（禁止自我投递）。
- 注意：before 事件只表示「放行/改写」，不保证结果发生——后续处理器可能失败，因此 before 事件不是操作已完成的承诺。

更多细节见 [docs/runtime.md#事件模型-v1](runtime.md#事件模型-v1)。

## 消息是字符串

投递的内容就是普通字符串，不包装信封。Python 程序把它当代码/表达式执行；LLM 用提示词约束自己按内容行事。对话需要知道谁发来的，自己在内容里说明。

## 确定性

Python 程序对同样的输入序列只有同样的输出；非最后一条消息的返回被忽略。LLM 程序没有这种保证，提示词要写清楚分支和终止条件，避免无限循环（预算中断会强制让出，但那是兜底不是常规路径）。

## 持久运行与跨轮状态

镜像可以声明 `journal_to`（回合日志）与 `snapshot_to`（重启快照）：每个提交回合记一条状态增量，正常退出时写快照；快照只是加速缓存。重启用 `--mode resume|reboot|reset`（`resume` = 快照 + 日志尾部，`reboot` = 快照世界段 + 世界类效果重放，`reset` = 原始镜像），或直接启动——不传 mode 时 `boot` 自动使用可读快照。机制细节见 [docs/runtime.md 的「启动模式与 guest 程序」](runtime.md#启动模式与-guest-程序)与[「持久运行」](runtime.md#持久运行重启恢复--回合日志--程序包)。

对写程序的人来说只有一条硬约定：**Python 程序的跨轮状态必须落在 `$MEM`**。PLM 的命名空间不序列化——恢复后执行器重建，Core 重放 system 消息（setup 代码），但历史里的 user/tool 消息不会被重新执行（副作用不能重复发生）。把状态存在命名空间变量里，恢复后就丢了。

- 需要跨轮保留的中间结果写进内存树（普通节点或设备），这是唯一会随快照/日志恢复的状态；
- 内存写、对话创建/关闭、投递、监听器变化会进回合日志；宿主侧特权操作（控制台屏蔽设备、就地编辑监听器）不在日志里；
- 恢复后的 cid 与快照一致；单对话检查点（`--checkpoint-in`）是调试子集，恢复后分配新 cid。

## 预算意识

一轮激活内的工具调用计入调用链预算（默认 50）。一个循环协议里别堆太多工具调用；达到预算会被中断并排到队尾。

## 观测运行

每轮推进都有帧：活跃/就绪/休眠、工具调用摘要、结果、思维链、错误；信息设备让对话也能查询调度状态（细节见 [docs/runtime.md 的「监测器」](runtime.md#监测器monitor)）。

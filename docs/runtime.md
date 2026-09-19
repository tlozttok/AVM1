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

- 被唤醒的对话进入就绪队列**末尾**（`send_instruction` 唤醒目标、亚对话完成唤醒父等）；
- `create_sub` 是唯一例外：亚对话前插，立即执行；
- 亚对话完成唤醒父对话时直接激活（同一调用链）。

为防止单个对话占用过多执行权，每轮激活有指令预算（`instruction_budget`，默认 50，镜像 `meta` 可配）。Core 在**工具返回加入批次之后**检查预算：达到预算就中断当前对话，把它放回就绪队列末尾，让出执行权。预算按调用链继承：亚对话与父对话共享同一预算（`is_sub` 且 parent 匹配），`create_cmd` 创建的子对话有独立预算。调用链断开（选中了另一个无关对话）时预算重新计数。

### 交互结束与个体结束

"对话结束"有两种含义，必须区分：

- **交互结束**：对话返回不含指令的结果，当前一轮执行完毕。对话个体依然存在，可被再次唤醒（再次被投递、事件回调等）。交互结束进入休眠。
- **个体结束**：对话被内核级关闭指令 `close_conversation` 关闭（可由对话自身或其他对话调用，不需要对方回应）。关闭后不可再调度、不可再被调用，记录为 finished；其 PLM 状态释放。

## 指令集

Core 只执行指令，不做业务决策。全部指令如下：

| 指令 | 作用 | 关键参数 |
|---|---|---|
| `memory_read` | 读取内存，返回给 LLM 的表示 | `ref` |
| `memory_write` | 写入内存 | `ref`, `content` |
| `memory_make` | 创建新内存地址 | `ref`, `key`, `mem_type`(str/dict/list) |
| `edit_metadata` | 编辑节点的 ctrl 元数据：set 写入 / get 读取（key 省略返回整个 ctrl）/ del 删除 | `ref`, `type`(set/get/del), `key`?, `value`? |
| `create_cmd` | 创建子对话（`system_ref` 必须指向 settingup 或 python 程序节点；只创建，返回 cid；子对话休眠等待指令；本对话保持活跃） | `system_ref`, `para_ref` |
| `create_sub` | 创建亚对话（立即执行；完成自动写回并唤醒父） | `system_ref`, `user_ref`, `para_ref` |
| `send_instruction` | 按 cid 投递指令内容（支持多播；只确认投递成功，不保证返回；投递后本对话继续执行，不自动休眠） | `cid`(int/列表), `content` |
| `close_conversation` | 内核级关闭：关闭指定 cid 的对话（自身或他人，不需要对方回应）；关闭后不可再调度、不可再被调用，其注册的事件监听自动撤销 | `cid` |
| `listen_event` | 注册/修改/取消事件监听；监听器是 `ctrl.type="python"` 节点的 `content`，每次事件在受限沙箱中执行 | `action`, `event_type`?, `target`?, `handler_ref`?, `listener_id`? |

## 内存模型

内存是全局共享的键值树，路径形如 `$MEM.a.b.c`。三种数据类型：

- **str**：叶子节点，直接返回内容。
- **MetaDict / MetaList**：带双元数据的复合类型。
- **设备（device）**：挂载到路径上的伪节点（见下文）。

### 双元数据

MetaDict / MetaList 有两份元数据：

- 元数据一（`meta`，字符串）：给 LLM 快速扫描的文件摘要/描述。LLM 读取时默认只看到摘要和键列表，深入索引后才返回子节点。
- 元数据二（`ctrl`，`dict[str,str]`）：给 Core / 内核态的结构化控制信息，如 `type="para"`（模型参数）、`type="settingup"`（LLM 提示词程序）、`type="python"`（Python 程序）、`owner`（节点所有者 cid，由 Core 维护）。

`memory_make` 创建 dict/list 节点、`memory_write` 隐式创建中间容器或新建最终容器时，Core 会写入 `ctrl.owner`；覆盖已有节点不改 owner；str 叶子没有 ctrl。`edit_metadata` 对 `owner` 键的 set/del 一律拒绝，get 允许。无主路径在事件模型 v1 中不能注册 before 内存监听；镜像作者可在 `ctrl` 中显式预置 `"owner": "0"`。

### 字符串与链接

**str 永远是 str**：字符串值不会被当作引用解引用。`$` 递归解引用与 `&` 单层解引用是已废弃的设计，不存在于当前代码中。

**链接（Link）是设备**（LinkDevice）：把真实 `$MEM` 路径挂载为别名，如 `$rooms` → `$MEM.game.map.rooms`。对话通过链接访问目标内容，但看不到真实路径，因此无法借链接访问兄弟节点——这是 capability 边界的实现方式。读取/写入/`memory_make`/`edit_metadata` 对链接透明；链接成环抛 `MemoryCircularReferenceError`；错误信息只暴露别名，真实错误详情输出到 stderr。v1 为静态链接（镜像 `builtin` 设备声明），运行期无创建/删除指令。

### 设备

设备是 `avm.memory_device.MemoryDevice` 的子类，通过接口与对话交互：

- `to_llm_string()`：LLM 读取该设备时看到的表示；
- `resolve_path(path)`：子路径访问（可选，默认抛错）；
- `set_value(value)`：写入（可选）。

Core 内置只读信息设备：`$MEM.conversations`（对话列表与单对话详情）、`$MEM.scheduler`（活跃/就绪/休眠与预算）、`$MEM.memory`（内存树摘要）、`$MEM.monitor`（监测帧摘要）、`$MEM.listeners`（事件监听器列表）。镜像可挂载自己的设备，例如 `$inputs`、`$outputs`（见[系统镜像](#系统镜像)）。

### 链接设备（LinkDevice）

`LinkDevice` 是内建设备：把 `$MEM` 下的真实路径挂载为别名（如 `rooms` → `$MEM.game.map.rooms`）。

- 读/写/`memory_make`/`edit_metadata` 对别名透明，等同于操作真实目标。
- 链接目标 v1 只允许 `$MEM` 开头的路径；允许链接指向链接（链式跟随），成环抛 `MemoryCircularReferenceError`，错误只含别名链。
- 错误隐藏：LLM 看到的错误只包含别名路径；真实路径/异常详情输出到 stderr 给运行者。镜像加载期校验失败属于运行者诊断，可显示真实路径。
- 事件透明：注册监听 target 为别名时，权限检查跟随真实目标（memory 按 owner，device 走对方 `authorize_listener`）；after 事实事件由真实执行组件发布，但事件 target/payload 仍写别名。
- 声明方式：`devices` 条目 `{"path": "rooms", "class": "LinkDevice", "builtin": true, "args": {"target": "$MEM.game.map.rooms"}}`。旧镜像不声明链接，行为不变。

## 对话间通信

### 简单投递

`send_instruction` 是对话间通信的唯一原语：按 cid 投递指令内容，内容**原样**放入目标对话的输入批次并唤醒它（进就绪队列末尾）。调用者立即收到"已投递到 <身份>"确认——**只保证送达，不保证对方返回消息**。投递后调用者继续执行，不自动休眠。

想要结果，由对话自己决定等待方式：

- **在内容里要求对方完成时通知自己**：对方把结论作为消息投递回来（仍走 `send_instruction`），收到时被唤醒；
- **注册事件监听**：对"预期行为"（如某内存地址变化、某对话完成）注册监听，处理器返回 `notify` 时 Core 自动唤醒目标（见[事件模型 v1](#事件模型-v1)）。

**父对话不是默认接收者**：创建关系不构成返回通道，AI 的思考不默认透露给其他 AI，输出默认写入共享内存。

### 消息内容

投递的内容就是普通字符串，**不包装 JSON 信封**（无 `from`/`to`/`icc_id` 字段）。对话需要知道谁发来的，自己在内容里说明。

### 名字与 cid

- 名字来自 Settingup 节点的 `name` 字段（是节点数据，不是元信息）。名字不唯一，冲突时用文件地址消歧。
- 带名字的对话必然有 Settingup 文件；亚对话（直接传提示词）没有名字，其身份 = 父对话身份 + 亚对话 cid。
- `send_instruction` 只用 cid 寻址。名字 ↔ cid 转换是内核职责（工具未实现）。

## Python 程序

Python 程序与 LLM 对话对等：都是 AVM 中的一等执行单元，通过 OpenAI 兼容的消息格式交互（输入 messages 列表，输出 `content` + `tool_calls`）。

内存节点 `ctrl.type="python"` 声明 Python 程序，`content` 是代码。`para` 的 `model` 参数选择 PLM（Python Language Model）类型：

| model | 行为 |
|---|---|
| `plm.simple` | 计算器：单行表达式求值。仅测试调试，暂存、不推荐、不维护。 |
| `plm.function` | 纯函数：system 定义 `handle(input)`，user 传 JSON，返回可 JSON 序列化的值；每次调用全新受限命名空间，无副作用、无跨调用状态。 |
| `plm.machine` | 状态机：system 是状态机 DSL，user 是 JSON 事件；状态/变量跨消息保留在 PLM 实例内，可输出全部 9 条指令的 tool calls。 |
| `plm.command` | 字符串 IO 命令行：system 定义 `handle(cmd)`，user 是命令与参数（整行字符串）；`handle` 可用 `yield from` 调用注入的工具函数，工具响应回到断点继续执行。 |
| `plm.python` | 图灵完备兜底：Jupyter 笔记本式语义。系统消息与用户消息都执行进同一个持久命名空间（跨轮次保留）；多个 print 全部执行，输出按顺序拼接为整体返回。 |

Python 对话的 `model` 必须是注册表（`PLM_REGISTRY`）中的名字，否则报错，不静默回退到 LLM。

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
- response JSON 的 `content` 与 `tools` 均可省略；`tools` 的 name 必须是 9 条指令之一；
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
- 注入的工具函数有 5 个：`memory_read(ref)`、`memory_write(ref, content)`、`memory_make(ref, key, mem_type)`、`send_instruction(cid, content)`、`close_conversation(cid)`；工具响应原样返回（包括 `Error: ...` 文本），由程序决定如何处理；其它指令名不可调用；
- 每条命令在全新受限命名空间中执行（程序与辅助代码重新执行），无跨命令状态——要持久化就写 `$MEM`；
- tool calls 的 id 形如 `cmd-0`，每条命令内递增；一条命令最多挂起一个工具调用，命令未结束就投递新命令会丢弃旧生成器；
- 错误以 `错误: CommandPLM: <类型>: <信息>` 文本返回。

### 确定性

Python 程序输出是确定性的：执行器通过比较上一次输入与本次输入，只处理新增的消息；只有最后一条消息的返回被保留（对同样的输入序列，只有同样的输出）。消息内容直接作为代码/表达式执行，不解析信封。

### 沙箱

PLM 代码运行在受限环境：安全 builtins 白名单 + AST 检查（禁止 import、禁止双下划线属性访问、禁止危险调用名如 open/eval/exec）。它防止 Python 程序访问宿主系统（文件、网络、进程、时钟、熵源）；AVM 内部行为不受限。真正强隔离（子进程 + OS 级隔离）超出 VM 范围。

## 事件模型 v1

事件是瞬时 dict，不落盘；由 Core 单调递增生成 `event_id`。**发布者分工**：before 请求事件由 Core 发布（source 为 `{"kind":"core","initiator_cid":cid}`）；after 事实事件由实际执行组件发布（Memory 发布 `{"kind":"memory"}`，设备发布 `{"kind":"device","path":"$MEM.<mount>"}`，Core 发布 `{"kind":"core"}`）。对话身份只出现在 before 的 `source.initiator_cid` 与 after 的 `payload.actor_cid`。

### before 事件（5 种可拦截工具）

| event_type | source | target 格式 |
|---|---|---|
| `memory_read` | `{"kind":"core","initiator_cid":cid}` | 原 `ref` |
| `memory_write` | 同上 | 原 `ref` |
| `memory_make` | 同上 | 新建完整路径 `ref.key` |
| `edit_metadata` | 同上 | 原 `ref` |
| `send_instruction` | 同上（每个目标一条） | `"cid:<cid>"` |

before 语义：操作被请求，不保证到达执行组件；可被 `replace` / `handler_error` 阻断。

### after 事件（9 种工具，成功时）

| event_type | source | target | 基础 payload |
|---|---|---|---|
| `memory_read` | memory 或 device | 原 `ref` | `{"ref", "actor_cid", "result"}` |
| `memory_write` | memory 或 device | 原 `ref` | `{"ref", "content", "actor_cid"}` |
| `memory_make` | memory | 新建完整路径 | `{"ref", "key", "mem_type", "path", "actor_cid"}` |
| `edit_metadata` | memory | 原 `ref` | `{"ref", "type", "key", "value", "actor_cid"}` |
| `send_instruction` | core | `"cid:<目标>"` | `{"cid", "content", "actor_cid"}` |
| `create_cmd` | core | `"cid:<新 cid>"` | `{"cid", "actor_cid"}` |
| `create_sub` | core | `"cid:<新 cid>"` | `{"cid", "actor_cid"}` |
| `close_conversation` | core | `"cid:<被关闭 cid>"` | `{"cid", "actor_cid"}` |
| `listen_event` | core | `"listener:<id>"` | `{"listener_id", "action", "actor_cid"}` |

规则：失败、被 replace、handler_error 的操作不产生 after；after 的 action 固定为 `execute`，处理器不能 replace。

### 监听器与 waterfall

- `listen_event` 支持 `phase=before/after/all`，缺省 `all`；`$MEM.listeners` 会显示 phase。
- 监听器是 `ctrl.type="python"` 节点中的 `content`，每次事件在全新受限沙箱命名空间执行（只提供 `json` 和受限 builtins），不创建对话、不保留处理器实例状态。
- 处理器代码必须定义可调用 `handle(event)`，返回 dict；返回的 `action` 在 before 阶段可为 `execute` / `replace`，首个 `replace` 短路，后续监听器不再执行。
- `replace` 后原指令不执行，调用者收到 `response`（缺省为 `事件被监听器拦截`）。
- 处理器可改的 payload 字段：`memory_write.content`、`send_instruction.content`、`edit_metadata.value`（仅原操作 `type=set`）。身份字段不可改。
- 处理器可返回 `notify`：`[{"cid": int, "content": str}]`，由 Core 在管道结束后内部投递（与 `send_instruction` 相同语义，但不产生事件、不递归）。**不允许自我投递**：before 阶段视为处理器错误；after 阶段丢弃该条并记入 `handler_errors`。
- before 处理器错误不再抛异常，而是返回 `action="handler_error"` 的事件结果：管道停止、原指令不执行、调用者收到 Error 工具响应；此前已累积的 notify 仍会投递。after 处理器错误 fail open：记录 stderr、跳过该处理器的 notify、继续后续监听器。

### 监听权限与 owner

- 内存类 before/after 监听（`memory_read` / `memory_write` / `edit_metadata` / `memory_make`）按 `split_path` 区分：
  - memory 路径：`_owner_of(target)` 必须等于注册者 cid；`memory_make` 要求父路径所有者等于注册者 cid；
  - device 路径：询问设备 `authorize_listener(event_type, remainder, owner_cid)`，设备返回 `False` 则拒绝。
- `send_instruction` before/after 监听只允许目标对话监听发往自己的消息（`target="cid:<自己>"`）。`send_instruction` 本身不允许对话向自己投递；多播中单个目标被拦截或处理失败不阻断其他目标。
- `create_cmd` / `create_sub` / `close_conversation` / `listen_event` 只有 after，允许 `target="*"` 或精确 target。
- `modify` / `cancel` 只允许注册该监听器的对话执行。
- `close_conversation` 自动撤销被关闭对话注册的全部监听。
- `_owner_of` 只处理 memory kind；device kind 返回 None（设备授权交给 `authorize_listener`）。`split_path` 按最长挂载键匹配，权限检查不会调用设备 `resolve_path`。

### 信息设备与设备自改

- `$MEM.listeners` 只读列出 `listener_id / owner_cid / event_type / phase / target / handler_ref`；子路径 `<listener_id>` 返回单条详情；不暴露 handler 代码全文。
- 事件本身瞬时，不提供 `$MEM.events` 历史、不写盘。
- `Core.emit_device_event(device_path, target, payload)` 是设备自改事件入口：只支持 `memory_write` before，source 为设备自身路径，target 必须属于 `$MEM.<device_path>`（用 `split_path` 判定，避免字符串前缀误匹配）；返回处理后的 event，设备据此决定是否执行自改。
- 设备路径的 after 事件不由 `emit_device_event` 产生；设备可在操作成功后通过 `after_event(event_type, remainder, result)` 返回纯 payload，由 Core 统一包装成 after 事件。v1 不实现设备异步自发 after 事件。

## 系统镜像

一个 JSON 文件 + 若干设备 Python 文件 = 一个标准 AVM 系统镜像。规范见 [docs/system-image.md](system-image.md)，要点：

- **显式节点**：内存每个节点都是 `{ "kind": "str"|"dict"|"list"|"device", "meta"?, "ctrl"?, "value" }`，无任何隐式转换（str 也必须显式包装）。`kind="device"` 是任意层级都可出现的可读性标记，实际挂载在 `devices` 段。
- **para**：`ctrl.type="para"` 的 dict 节点存放模型调用参数（`model`、`temperature`、`use_tool`、`extra_body`、`reasoning_effort` 等），均以字符串存储，数值型加载时自动转回。
- **devices**：数组，每项 `{ path, file, class, args }`。`file` 是相对镜像文件的 Python 模块路径，`class` 是其中的设备类，**必须继承** `MemoryDevice`，否则加载报错。这样"一个 JSON + 几个定义设备的 Python 文件"就是一个完整的系统镜像。
- **init**：启动对话配置（`name` 默认 `"init"`，`system`/`system_ref`、`user`/`user_ref`）。`system_ref` 必须指向 `ctrl.type="settingup"` 的 LLM 程序节点（`system` 字面量可用于匿名入口程序）；`user_ref` 指向 str 节点。只启动 0 号对话，不预置其他对话；其他对话由对话运行期自己 `create_cmd` / `create_sub` 创建。
- **persist_to**：运行结束后内存写回路径（异常也写）。
- **meta**：`version` 必须为 1；`instruction_budget` 可配核心中断预算。

## 调试与观测

### 监测器（Monitor）

每轮对话推进结束后生成一帧（Frame），第 0 帧为基线。帧内容：

- `lmu`：最近一次执行的结果（`result`）、思维链内容（`reasoning`，与 result 同截断策略）、工具调用摘要（`tool_calls`）、事件摘要（`events`，每条含 event_id/event_type/phase/target/source/action，失败时含 error/handler_errors）、耗时、错误；
- `sched`：活跃 cid、就绪队列、休眠队列、下一个 cid；
- `conversations`：每个对话的状态、消息条数、输入批次内容。

提供 `trail()`（单变量跨帧轨迹）、`diff()`（两帧间变化）、`find()`（按谓词找帧）等查询；`--transcript` 把每次 API 调用全文写入文件。

### 单对话检查点（调试用）

`Core.save_conversation(cid, path)` / `Core.restore_conversation(path)` 保存/恢复单个 LLM 对话：序列化无损、确定性，恢复后发送给 API 的消息前缀与保存时逐字节一致（可命中厂商前缀缓存）。历史末尾是未配对的工具调用时（如等待输入响应），保存会自动截到最后一个完整交换。v1 不支持 Python 对话。

输入设备支持操作者命令 `/save`（可带路径 `/save <path>`）：在输入前截取，保存当前活跃对话的检查点后继续等待真实输入——对话感知不到该命令。恢复在 host 侧进行（`restore_conversation`），不做 AVM 内 `/load`。

启动时加载：`python main.py <image.json> --checkpoint-in cp.json`（或 debug.json 的 `checkpoint_in` 字段）会**跳过 init**，把检查点恢复为 cid 0 的就绪对话直接续跑——这是"上次保存的对话下次启动接着用"的入口。

终端输入：等待输入时显示正在等待的对话身份提示（如 `[tester（cid=0）等待输入，/save 保存检查点]`），不再是黑屏裸等；已启用 readline 行编辑（历史/方向键，中文输入法组合友好）；终端非 UTF-8 编码会警告一次——中文乱码通常源于终端编码而不是程序。

### 思维链内容

LLM 响应中的 `reasoning_content`（思维链）被完整存入对话历史，并在后续请求中随 assistant 消息回传（无值则空字符串）——这是厂商协议的硬性要求，也是"回传所有推理内容"的既定策略。监测器记录它（与普通内容相同的截断策略）；`$MEM` 信息设备不暴露它（外泄控制靠禁止内部访问，不靠不记录）。

### 错误输出

工具调用参数 JSON 解析失败、未知工具名等错误输出到 stderr，面向运行者，不喂回给 LLM 让它自纠。整体错误处理机制仍在设计中（见 [README 当前状态](../README.md#当前状态)）。

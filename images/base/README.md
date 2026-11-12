# base 母板镜像

第一块长期运行的镜像：不带业务，只带持久化、装载设备、OS 程序（引导/关机监督/
设备事件钟）、统一命令入口 shell 与空分区。空闲由宿主控制台的面板对话承担，镜像
不再声明空闲处理程序。其余基础由作者在系统内手动搭建。

```bash
python images/base/build_image.py          # 重新生成 base.json（首次生成后 base.json 是种子）
python -m avm.console images/base/base.json
```

装载设备/程序包不再使用密码：命令必须带操作员审批得到的 `approved_sha256`（文件内容哈希）。

## 镜像里有什么

| 路径 | 内容 |
|---|---|
| `$MEM.sys.power.state` | 断电寄存器（`on` / `off`）；只有 OS 写 `off`，宿主才正常断电退出。这是 VM 在安全点唯一读取的 OS 约定节点 |
| `$MEM.sys.boot.state` / `.prior` / `.id` / `.report` | 启动标记（`running` / `clean`）、上次状态、上电序号、最近一次恢复报告 |
| `$MEM.sys.boot.manifest` | 常驻服务清单（JSON 字符串，初始 `[]`）；boot 程序按它创建缺失对话 |
| `$MEM.sys.registry` | 名字 → cid；boot 程序重写（只记 manifest 服务；OS 服务的 cid 不落持久 MEM） |
| `$MEM.sys.programs.*` | OS 程序节点（`boot` / `supervisor` / `clock_a` / `clock_b`）、构造器命令程序（`builder`）与 shell 程序（`shell`），可在系统内改写 |
| `$MEM.sys.shell` | 统一命令入口：`args`（分发暂存位，str）与 `commands`（命令注册表：名字 → `{program, para}` 引用） |
| `$MEM.sys.outbox` | 持久输出（作者约定怎么用） |
| `$MEM.sys.clock` | 设备事件钟：`tick` / `target` / `interval` / `timeouts` / `state` |
| `$MEM.sys.handbook` | 分区与自举约定 |
| `$MEM.para.python` / `.command` / `.builder` / `.shell` | 内建 PLM 的参数档（`.builder` / `.shell` 用 `plm.command`） |
| `$MEM.packages` / `package_records` | 由装载设备管理 |
| `$MEM.workshop` / `$MEM.home` | 试验区 / 个人数据区（空） |
| `$MEM.loader` | 装载设备（程序包、设备） |
| `$MEM.AVMRepo` | 仓库只读设备（查文档用） |
| `$MEM.inputs` / `$MEM.outputs` | 标准 IO 设备；操作面板宿主会替换成总线设备 |

OS 程序节点里 `ctrl.type="python"`、`value.content` 是程序源、`value.para_ref` 指向
`$MEM.para.python`；服务节点的 `value.name`（`clock_a` / `clock_b` / `shell`）会成为
对话身份。构造器与 shell 程序节点同样是 `ctrl.type="python"`，`value.content` 是
`plm.command` 程序正文，`value.para_ref` 分别指向 `$MEM.para.builder` /
`$MEM.para.shell`（`model` 为 `plm.command`）。顶层声明
`journal_to` / `snapshot_to` / `persist_to`，落点在 `images/base/out/`；
`meta.snapshot_every=200`。

## 构造器命令程序（plm.command）

构造器是 base 预置的一个普通 `plm.command` 程序节点，不是镜像声明的自定义 PLM：
程序正文就是 `$MEM.sys.programs.builder` 的 `value.content`（运行时的 system 消息），
`$MEM.para.builder` 的 `model` 是 `plm.command`。它把「创建一个合法内存节点」的多步
指令（探测、`memory_make`、`memory_write`、`edit_metadata`）打包成一条 JSON 请求：
作者只描述节点规格，不用手拼每一步工具调用。程序正文是内存里的数据，可以在系统内
直接读、改、复制；运行时只注入 `math` / `json` / `re` 与工具函数，程序不能 import。

入口已预置。用法是在内核态对话里创建它、再投递一条 JSON：

```python
cid = yield from create_cmd("$MEM.sys.programs.builder", "$MEM.para.builder")
yield from send_instruction(cid, json.dumps(spec, ensure_ascii=False))
```

规格（system 消息是程序正文；用户消息必须是 JSON 对象）：

```json
{
  "mode": "create",
  "ops": [
    {"type": "settingup", "path": "$MEM.programs.hello", "name": "hello",
     "content": "...", "signature": "...", "meta": "一句描述"},
    {"type": "python", "path": "$MEM.programs.py", "name": "py", "content": "..."},
    {"type": "handler", "path": "$MEM.handlers.h", "name": "h",
     "content": "def handle(event): ...",
     "condition": {"events": ["memory_write"], "phase": ["before"]}},
    {"type": "para", "path": "$MEM.para.fast",
     "fields": {"model": "plm.python", "temperature": "0.2"}}
  ],
  "calls": [{"tool": "memory_read", "arguments": {"ref": "$MEM.x"}}]
}
```

| 字段 | 说明 |
|---|---|
| `mode` | `create`（缺省）或 `update` |
| `ops[].type` | `settingup` / `python` / `handler` / `para` |
| `ops[].path` | `$` 开头、至少写到根下第一层的目标地址（如 `$MEM.programs.hello`） |
| `ops[].name` / `content` / `signature` | 程序节点子字段；`content` 必填，其余可选 |
| `ops[].condition` | handler 必填；JSON 字符串，给对象/数组时自动 `json.dumps` |
| `ops[].fields` | para 必填；每个键值写成 str 子节点（数值/布尔按 JSON 字面量转字符串） |
| `ops[].meta` | 可选；写成节点描述（dict 节点用 `memory_write` 写 metadata） |
| `calls` | 可选；本命令程序注入的指令（见下），按给出顺序原样执行 |

语义：

- **create**：目标节点不存在时建 dict、写字段、用 `edit_metadata` 设 `ctrl.type`、
  最后写描述；缺失的祖先自动用 `memory_make(..., "dict")` 建出。目标已存在时
  记一条错误并跳过（不覆盖）。字段存在时直接写，缺失时先 `memory_make(..., "str")`。
- **update**：目标必须已存在，否则记错误并跳过；字段存在用 `memory_write` 覆盖，
  缺失用 `memory_make` + `memory_write`；`ctrl.type` 用 `edit_metadata` 设；
  `meta` 覆盖原描述。
- **calls**：`tool` 必须是本命令程序可调用的指令——`memory_read` / `memory_write` /
  `memory_make` / `memory_delete` / `edit_metadata` / `send_instruction` /
  `close_conversation`；`arguments` 必须是对象，结果原样记进摘要的 `calls`。
  `plm.command` 不注入 `create_cmd` / `create_sub` / `listen_event`，这三个不能直通。
- **验证**：每个 op 的 `path` 都发一次 `edit_metadata(type="get")`，解析 ctrl JSON
  并检查 `type`，结果写进摘要的 `verified`。

返回的正文是 JSON 字符串：

```json
{"ok": true, "mode": "create",
 "built": ["$MEM.programs.hello"], "updated": [], "skipped": [], "errors": [],
 "calls": [{"tool": "memory_read", "arguments": {"ref": "$MEM.x"}, "result": "..."}],
 "verified": {"$MEM.programs.hello": {"expected": "settingup", "actual": "settingup", "ok": true}}}
```

`ok` 为真要求没有 `errors`（含重复/缺失目标与失败的直通指令）且全部 `verified`
通过。规格本身非法（不是 JSON、缺必填字段、未知 `type`、不可直通的工具等）时不产生
任何工具调用，对话直接收到 `错误: builder: ...`。摘要就是该对话最后一条 assistant
消息；要让结果进内存，可在 `calls` 里写节点，或改用 `create_sub`（把规格先写进一个
str 节点作为 `user_ref`，亚对话完成时自动把摘要写回父对话）。

## shell（统一命令入口）

shell 是 base 预置的第二个普通 `plm.command` 程序节点（`$MEM.sys.programs.shell`，
para `$MEM.para.shell`），给作者一个统一命令入口：一条命令按注册表分发给对应的
命令程序，不用为每个功能记一组 cid。它不是新 PLM 类型，程序正文同样可以在系统内
直接读改。

**怎么找到 shell 服务。** boot 冷启动/恢复时创建（或复用）名字为 `shell` 的常驻
对话，规则与 `clock_a` / `clock_b` 相同：运行期按名字查 `$MEM.conversations`，cid
不写进持久 MEM（`$MEM.sys.registry` 只记 manifest 服务）。

```python
listing = yield from memory_read("$MEM.conversations")
match = re.search(r"cid=(\d+)\s+shell\b", str(listing))
shell_cid = int(match.group(1)) if match else None
yield from send_instruction(shell_cid, "builder {...JSON 规格...}")
```

`send_instruction` 只确认投递、不返回命令结果；命令的输出是 shell 对话自己的
assistant 消息（例如 builder 直接返回的 JSON 摘要）。需要同步拿结果的调用方可以把
`$MEM.sys.programs.shell` 当普通命令程序用 `create_sub` 启动（`user_ref` 指向一个
装了整行命令的 str 节点），shell 会把子命令结果作为返回通道写回。注意
`$MEM.sys.shell.args` 是共享暂存位：同一时刻只应有一条命令在分发。

**命令格式。** `<命令名> <参数...>`：第一个空格前是命令名，其余整段原样作为参数
（参数可以含空格）。整行只有空白时返回 `错误: shell: 空命令`。

**内建命令。**

| 命令 | 行为 |
|---|---|
| `help` | 列出已注册命令、`summary`，以及每个命令在 shell 里的调用形式（`usage`） |
| `help <名字>` | 先显示 shell 调用形式，再把 `help` 分发给该命令程序，由程序回答自己的参数格式 |

shell 不猜用法：列表里的说明来自注册项 `summary`，shell 调用形式来自 `usage`，
详细参数格式由命令程序自己的 `handle("help")` 返回。命令程序应同时提供
`summary`、`usage` 并处理 `help` 输入，否则 `help <名字>` 只会得到该程序对
`help` 这个参数的正常处理结果。

在 shell 里调用 builder 的完整命令（整行就是投递给 shell 的字符串）：

```
builder {"mode":"create","ops":[{"type":"python","path":"$MEM.workshop.hello","name":"hello","content":"..."}]}
```

命令名之后的内容原样作为参数传给命令程序：空格、引号、换行都不被 shell 处理
（JSON 里的空格不影响解析）；分发是同步的，返回的就是命令程序的输出。

**分发机制。** 从 `$MEM.sys.shell.commands.<名字>` 读 `program` 与 `para` 两个 str
子节点；读不到（或读出来是 Error/Illegal）时返回
`错误: shell: 未知命令 '<名字>'（用 help 查看）`。否则把剩余参数原样写入
`$MEM.sys.shell.args`，再：

```python
result = yield from create_sub(program, "$MEM.sys.shell.args", para)
return str(result)
```

`create_sub` 是同步的：shell 服务在亚对话返回前休眠，亚对话完成时输出自动写回，
shell 把那条工具响应原样返回——子命令返回的 JSON 摘要、错误文本都直接成为 shell
的输出，不再包装。程序内部异常不抛穿 Core，统一返回 `错误: shell: ...`。

**加新命令。**

1. 建一个 `ctrl.type="python"` 命令程序节点（正文是 `plm.command` 程序，定义
   `handle(cmd)`）；para 的 `model` 为 `plm.command`，可以复用 `$MEM.para.shell`。
2. 在 `$MEM.sys.shell.commands` 下建同名 dict，写 `program`、`para` 两个 str
   子节点指向上面两个节点，再写可选的 `summary`（列表一行说明）与 `usage`
   （shell 调用形式）两个 str 子节点；命令程序自己处理 `help` 输入返回详细用法。
3. 无需重启：下一条命令即按新注册表分发。注册表只存引用、不存 cid，命令程序
   按需现建现用。

作者可以用构造器（`builder`）一次建出程序节点与注册项，也可以用 `memory_make` /
`memory_write` 手工建，或直接改 `base.json` 再冷启动。

## OS 程序与自举

`meta.guest_programs` 声明两个 guest 角色（内核创建对话并注册同名中断向量）：

| 角色 | 程序节点 | 入口 | 何时运行 |
|---|---|---|---|
| `boot` | `$MEM.sys.programs.boot` | `yield from boot()` | 冷启动由就绪队列；resume 时宿主用中断唤醒并把 `last_exit` 放进入口 payload |
| `shutdown` | `$MEM.sys.programs.supervisor` | `yield from shutdown()` | 宿主挂 `shutdown` 中断时由内核把执行权交给它 |

`idle` 中断向量由宿主控制台注册为面板对话（cid -1）：系统没有活跃/就绪对话时
执行权交给面板，面板在自己的推理里阻塞等操作员动作，这就是空闲态。镜像不再声明
`idle` 角色、不预置 inbox 与面板服务对话；非 console 宿主仍可由镜像声明
`meta.guest_programs.idle` 提供空闲处理程序（内核只派发中断，不读任何空闲节点）。

- **init** 是 `yield from init()`：boot 程序里的一个空入口，真正的引导在 guest
  `boot` 角色；作者可以在系统内改写它加早期步骤。
- **boot**：写 `running` / 递增 `boot.id` / 复位 `power.state = on`（否则 resume
  后第二次关机没有边沿）/ 按 manifest 创建缺失服务 / 创建缺失的 OS 服务对话
  （`clock_a` / `clock_b` / `shell`）/ 写 `boot.report` / 休眠。
  服务 cid 不写持久 MEM：运行期通过 `$MEM.conversations` 按名字查找。
- **shutdown（supervisor）**：检查 `$MEM.conversations` 里是否还有非 OS 对话处于
  active / ready；有就启/保持设备事件钟、写等待/超时记录，**不写 off**；全部收尾
  才停钟 → 写 `boot.state = "clean"` → 写 `power.state = "off"`。宿主在安全点看到
  off 才写快照并退出。面板对话对普通对话隐藏，不出现在这个视图里。
- **clock_a / clock_b**：两个互相激活的 `plm.python` 乒乓；每轮一个逻辑 tick，
  跨过 `target` 就把超时通知投给 supervisor 并把 target 后移。只有 supervisor
  能启停（`$MEM.sys.clock.state`）。

## 输入

输入路径冻结：UI 是设备，输入格式待作者定义。`POST /api/io/input` 只发布到宿主侧
IO 总线，是不进日志的调试通道。base 不预置 inbox/面板输入服务。

操作面板的主入口是命令台：它经面板对话（cid -1）向 name=shell 的对话发
`send_instruction`，随后让出执行权；这不是 UI 输入设备，也不写 `$MEM.sys.inbox`。

## 关机

`POST /api/system/shutdown` 由宿主直接挂 `shutdown` 中断（中断是 VM 通知软件的
通道），不再写世界节点。镜像声明了 `shutdown` 角色时，supervisor 被中断唤醒、
收尾后写 `power.state = "off"`，宿主在安全点写快照并退出；镜像没有声明
`shutdown` 角色时请求返回 warning 并保持 stopping，不自行断电。没收尾或 supervisor
卡住时进程保持运行，操作员可在 `force_stop_after` 秒后走 `/api/system/force-stop`
强制退出（不写快照、不写 clean）。

## 还没接上的部分

- boot-check 的业务报告（作者写个人系统时完善）；base 只保留 `boot.report` 约定。
- 配置层/前端对关机中、超时次数的展示。
- 设备审批的强制力是人的仪式：内核不检查发起者（任何能写 `$MEM.loader` 的对话知道哈希即可装载）。

## 测试

```bash
python -m pytest tests/test_base_image.py tests/test_os_shutdown.py \
    tests/test_builder_command.py tests/test_shell_command.py -q
```

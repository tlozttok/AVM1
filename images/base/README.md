# base 母板镜像

第一块长期运行的镜像：不带业务，只带持久化、装载设备、OS 程序（引导/关机监督/
设备事件钟）与空分区。空闲由宿主控制台的面板对话承担，镜像不再声明空闲处理程序。
其余基础由作者在系统内手动搭建。

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
| `$MEM.sys.programs.*` | OS 程序节点（`boot` / `supervisor` / `clock_a` / `clock_b`）与构造器入口（`builder`），可在系统内改写 |
| `$MEM.sys.outbox` | 持久输出（作者约定怎么用） |
| `$MEM.sys.clock` | 设备事件钟：`tick` / `target` / `interval` / `timeouts` / `state` |
| `$MEM.sys.handbook` | 分区与自举约定 |
| `$MEM.para.python` / `.command` / `.builder` | 内建 PLM 的参数档（构造器用 `.builder`） |
| `$MEM.packages` / `package_records` | 由装载设备管理 |
| `$MEM.workshop` / `$MEM.home` | 试验区 / 个人数据区（空） |
| `$MEM.loader` | 装载设备（程序包、设备） |
| `$MEM.AVMRepo` | 仓库只读设备（查文档用） |
| `$MEM.inputs` / `$MEM.outputs` | 标准 IO 设备；操作面板宿主会替换成总线设备 |

OS 程序节点里 `ctrl.type="python"`、`value.content` 是程序源、`value.para_ref` 指向
`$MEM.para.python`；服务节点的 `value.name`（`clock_a` / `clock_b`）会成为对话身份。
构造器入口同样是一个 `ctrl.type="python"` 节点，它的 `value.para_ref` 指向
`$MEM.para.builder`。顶层声明 `journal_to` / `snapshot_to` / `persist_to`，落点在
`images/base/out/`；`meta.snapshot_every=200`。

## 构造器 PLM（plm.builder）

镜像顶层 `plms` 段声明了自定义 PLM `plm.builder`（实现：`plms/builder.py` 的
`NodeBuilderPLM`，继承 `avm.python_server.PLMServer`）。它把「创建一个合法内存节点」
的多步指令（探测、`memory_make`、`memory_write`、`edit_metadata`）打包成一条 JSON
请求：作者只描述节点规格，不用手拼每一步工具调用。

入口已预置：`$MEM.para.builder` 是它的 para 档，`$MEM.sys.programs.builder` 是
`ctrl.type="python"` 的入口程序。用法是在内核态对话里创建它、再投递一条 JSON：

```python
cid = yield from create_cmd("$MEM.sys.programs.builder", "$MEM.para.builder")
yield from send_instruction(cid, json.dumps(spec, ensure_ascii=False))
```

规格（`system` 消息不生效；用户消息必须是 JSON 对象）：

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
| `calls` | 可选；任意打包指令，按给出顺序原样执行 |

语义：

- **create**：目标节点不存在时建 dict、写字段、用 `edit_metadata` 设 `ctrl.type`、
  最后写描述；缺失的祖先自动用 `memory_make(..., "dict")` 建出。目标已存在时
  记一条错误并跳过（不覆盖）。字段存在时直接写，缺失时先 `memory_make(..., "str")`。
- **update**：目标必须已存在，否则记错误并跳过；字段存在用 `memory_write` 覆盖，
  缺失用 `memory_make` + `memory_write`；`ctrl.type` 用 `edit_metadata` 设；
  `meta` 覆盖原描述。
- **calls**：`tool` 必须是 10 条指令之一（`memory_*` / `edit_metadata` /
  `create_*` / `send_instruction` / `close_conversation` / `listen_event`），
  `arguments` 必须是对象；结果原样记进摘要的 `calls`。
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
通过。规格本身非法（不是 JSON、缺必填字段、未知 `type` 等）时不产生任何工具调用，
对话直接收到 `错误: BuilderPLM: ...`。摘要就是该对话最后一条 assistant 消息；
要让结果进内存，可在 `calls` 里写节点，或改用 `create_sub`（把规格先写进一个
str 节点作为 `user_ref`，亚对话完成时自动把摘要写回父对话）。

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
  （`clock_a` / `clock_b`）/ 写 `boot.report` / 休眠。
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
python -m pytest tests/test_base_image.py tests/test_os_shutdown.py -q
```

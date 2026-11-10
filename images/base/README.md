# base 母板镜像

第一块长期运行的镜像：不带业务，只带持久化、装载设备、OS 程序（引导/空闲/关机
监督/设备事件钟/面板服务）与空分区。其余基础由作者在系统内手动搭建。

```bash
python images/base/build_image.py          # 重新生成 base.json（首次生成后 base.json 是种子）
python -m avm.console images/base/base.json
```

装载设备/程序包不再使用密码：命令必须带操作员审批得到的 `approved_sha256`（文件内容哈希）。

## 镜像里有什么

| 路径 | 内容 |
|---|---|
| `$MEM.sys.power.request` / `.state` | OS 关机请求（写 `shutdown`）/ 断电标记（`on` / `off`）；只有 OS 写 `off`，宿主才正常断电退出 |
| `$MEM.sys.boot.state` / `.prior` / `.id` / `.report` | 启动标记（`running` / `clean`）、上次状态、上电序号、最近一次恢复报告 |
| `$MEM.sys.boot.manifest` | 常驻服务清单（JSON 字符串，初始 `[]`）；boot 程序按它创建缺失对话 |
| `$MEM.sys.registry` | 名字 → cid；boot 程序重写（只记 manifest 服务；OS 服务的 cid 不落持久 MEM） |
| `$MEM.sys.programs.*` | OS 程序节点（`boot` / `idle` / `supervisor` / `panel_shell` / `clock_a` / `clock_b`），可在系统内改写 |
| `$MEM.sys.inbox` / `outbox` | 面板输入的唯一落点（str 列表，元素是 JSON）/ 持久输出 |
| `$MEM.sys.idle.state` | `waiting` = OS 在空闲等待宿主中断；内核不判严重错误 |
| `$MEM.sys.clock` | 设备事件钟：`tick` / `target` / `interval` / `timeouts` / `state` |
| `$MEM.sys.handbook` | 分区与自举约定 |
| `$MEM.para.python` / `.command` | 两个内建 PLM 的参数档 |
| `$MEM.packages` / `package_records` | 由装载设备管理 |
| `$MEM.workshop` / `$MEM.home` | 试验区 / 个人数据区（空） |
| `$MEM.loader` | 装载设备（程序包、设备） |
| `$MEM.AVMRepo` | 仓库只读设备（查文档用） |
| `$MEM.inputs` / `$MEM.outputs` | 标准 IO 设备；操作面板宿主会替换成总线设备 |

程序节点里 `ctrl.type="python"`、`value.content` 是程序源、`value.para_ref` 指向
`$MEM.para.python`；服务节点的 `value.name`（`panel_shell` / `clock_a` / `clock_b`）
会成为对话身份。顶层声明 `journal_to` / `snapshot_to` / `persist_to`，落点在
`images/base/out/`；`meta.snapshot_every=200`。

## OS 程序与自举

`meta.guest_programs` 声明三个 guest 角色（内核创建对话并注册同名中断向量）：

| 角色 | 程序节点 | 入口 | 何时运行 |
|---|---|---|---|
| `boot` | `$MEM.sys.programs.boot` | `yield from boot()` | 冷启动由就绪队列；resume 时宿主用中断唤醒并把 `last_exit` 放进入口 payload |
| `idle` | `$MEM.sys.programs.idle` | `yield from idle()` | 系统空闲时内核挂 `idle` 中断 |
| `shutdown` | `$MEM.sys.programs.supervisor` | `yield from shutdown()` | 宿主看到 `power.request == "shutdown"` 后挂中断 |

- **init** 是 `yield from init()`：boot 程序里的一个空入口，真正的引导在 guest
  `boot` 角色；作者可以在系统内改写它加早期步骤。
- **boot**：写 `running` / 递增 `boot.id` / 清掉上次关机留下的
  `power.request` 与 `power.state = off`（否则 resume 后第二次关机没有边沿）/
  按 manifest 创建缺失服务 / 创建缺失的 OS 服务对话（`panel_shell` / `clock_a` /
  `clock_b`）/ 写 `boot.report` / 休眠。
  服务 cid 不写持久 MEM：运行期通过 `$MEM.conversations` 按名字查找。
- **idle**：读 `$MEM.sys.inbox`，有 `status == "new"` 的输入就唤醒 `panel_shell`
  （`send_instruction`）；没有输入就把 `idle.state` 写成 `waiting` 并结束本轮，
  内核停在安全点等宿主中断，不 panic。
- **panel_shell**：消费 inbox 的新条目，把结果写到 outbox / outputs，条目标记
  `done`。它是 boot 创建的普通服务对话，不是内核保留对话。
- **shutdown（supervisor）**：检查 `$MEM.conversations` 里是否还有非 OS 对话处于
  active / ready；有就启/保持设备事件钟、写等待/超时记录，**不写 off**；全部收尾
  才停钟 → 写 `boot.state = "clean"` → 写 `power.state = "off"`。宿主在安全点看到
  off 才写快照并退出。
- **clock_a / clock_b**：两个互相激活的 `plm.python` 乒乓；每轮一个逻辑 tick，
  跨过 `target` 就把超时通知投给 supervisor 并把 target 后移。只有 supervisor
  能启停（`$MEM.sys.clock.state`）。

## 面板输入

`POST /api/input` 由保留对话把条目写进 `$MEM.sys.inbox`（进日志、可重放），
宿主随后给 `idle` 挂中断；inbox 写入的 after 事件也会再挂一次，保证 idle 在
写入之后才被唤醒去消费。`POST /api/io/input` 仍是不进日志的调试通道。

## 关机

`POST /api/system/shutdown` 只写 `$MEM.sys.power.request = "shutdown"`；真正的
断电由 supervisor 写 `$MEM.sys.power.state = "off"`。没收尾或 supervisor 卡住时
进程保持运行，操作员可在 `force_stop_after` 秒后走 `/api/system/force-stop`
强制退出（不写快照、不写 clean）。

## 还没接上的部分

- boot-check 的业务报告（作者写个人系统时完善）；base 只保留 `boot.report` 约定。
- 配置层/前端对关机中、超时次数的展示。
- 设备审批的强制力是人的仪式：内核不检查发起者（任何能写 `$MEM.loader` 的对话知道哈希即可装载）。

## 测试

```bash
python -m pytest tests/test_base_image.py tests/test_os_shutdown.py -q
```

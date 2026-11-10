# base 母板镜像

第一块长期运行的镜像：不带业务，只带持久化、装载设备、可编辑的引导与空分区。
其余基础由作者在系统内手动搭建（见「手动搭建」）。

```bash
python images/base/build_image.py          # 生成 base.json
python -m avm.console images/base/base.json
```

装载设备/程序包不再使用密码：命令必须带操作员审批得到的 `approved_sha256`（文件内容哈希）。

## 镜像里有什么

| 路径 | 内容 |
|---|---|
| `$MEM.sys.boot.state` / `prior` | 启动标记与上次状态；boot 程序每次启动把它改成 `running` |
| `$MEM.sys.boot.manifest` | 常驻服务清单（JSON 字符串，初始 `[]`）；boot 程序按它创建对话 |
| `$MEM.sys.registry` | 名字 → cid；每次启动由 boot 程序重写 |
| `$MEM.sys.programs.boot` | 最小引导程序（`plm.python`，可编辑）；镜像 `init` 指向它 |
| `$MEM.sys.inbox` / `outbox` | 面板输入的唯一落点（空 list）/ 持久输出（空 list） |
| `$MEM.sys.handbook` | 分区与自举约定 |
| `$MEM.para.python` / `.command` | 两个内建 PLM 的参数档 |
| `$MEM.packages` / `package_records` | 由装载设备管理 |
| `$MEM.workshop` / `$MEM.home` | 试验区 / 个人数据区（空） |
| `$MEM.loader` | 装载设备（程序包、设备） |
| `$MEM.AVMRepo` | 仓库只读设备（查文档用） |
| `$MEM.inputs` / `$MEM.outputs` | 标准 IO 设备；操作面板宿主会替换成总线设备 |

顶层声明 `journal_to` / `snapshot_to` / `persist_to`，落点在 `images/base/out/`；`meta.snapshot_every=200` 表示每 200 个已提交回合写一次定时快照。

## 自举

- boot 程序是内存节点：在系统内改写 `$MEM.sys.programs.boot` 的 content，下一次启动就运行新内容。
- 常驻服务清单也是内存节点：给 `$MEM.sys.boot.manifest` 写入 JSON 数组，下一次启动 boot 程序按它创建对话并写 registry。
- 装载设备让作者在系统内装入新设备代码；持久化保证这些改写跨重启存活。

## 手动搭建第一个服务（示例）

1. 在系统内创建一个程序节点，例如 `$MEM.sys.programs.echo`：
   - `ctrl.type = "python"`，`content` 是 `plm.command` 程序；
   - 或 `ctrl.type = "settingup"`，`content` 是提示词。
2. 建一个 para 节点，比如 `$MEM.para.echo`（`ctrl.type="para"`，`model=plm.command`）。
3. 往 `$MEM.sys.boot.manifest` 写：
   ```json
   [{"name": "echo", "program": "$MEM.sys.programs.echo", "para": "$MEM.para.echo"}]
   ```
4. 用面板的调试抽屉提交动作：`memory_write $MEM.sys.boot.manifest`（content 是上面的 JSON）。
5. 重启（`python -m avm.console ...` 自动恢复快照；或 `--restart` 类入口）后，boot 程序创建它并写入 `$MEM.sys.registry.echo`。

## 装载新设备（今天的路）

面板的调试抽屉里提交一个动作：

- 工具：`memory_write`
- 参数：`{"ref": "$MEM.loader", "content": "<JSON 命令>"}`

JSON 命令：

```json
{"action": "load_device", "approved_sha256": "<设备文件 sha256>",
 "path": "echo", "file": "devices/echo.py", "class": "EchoDevice", "args": {}}
```

`path` 是挂载点（`$MEM.echo`），`file` 相对镜像目录（或绝对路径）。装载成功后设备在
`$MEM.echo`，对话用 `memory_read` / `memory_write` 读写它。装载是纯内存操作、进回合日志，
重启后按来源表重新加载；设备文件必须留在原路径。

> 操作面板的「装载设备（审批）」先只读检查文件哈希，再由操作员在全屏弹窗里允许；
> 批准值随 `memory_write $MEM.loader` 命令进内核，装载时重新核对。

## 还没接上的部分

- 输入消费者：`$MEM.sys.inbox` 现在只有落点，没有 handler；消费者由作者手动搭建。
- boot-check：脏关闭的恢复程序由作者在系统内写（启动生命周期计划提供标记约定）。
- 启动模式（复苏/重启/重置）与日志代际：见 `plan-boot-lifecycle-2026-11-08.md`。
- 设备审批的强制力是人的仪式：内核不检查发起者（任何能写 `$MEM.loader` 的对话知道哈希即可装载）。
- 输入路径的正式化（宿主对话从隐藏 cid -1 换成普通 cid）：见 `plan-panel-input-path-2026-11-08.md`。

## 测试

```bash
python -m pytest tests/test_base_image.py -q   # base 冒烟（不调真实 API）
```

# sysinfo（示例 OS 包）

宿主系统信息只读设备。这是 AVM 内 assistant（cid 6，由面板 cid -1 指挥）在 2026-11-14 写出的设备代码，从 generation-9 回合日志还原后转正的示例包；AVM 运行时本身不认识"包"，这个目录的格式与安装流程是 **OS 层约定**（见 [docs/porting-contract.md](../../docs/porting-contract.md) 的「包数据怎么进 AVM」）。

## 目录

| 文件 | 内容 |
|---|---|
| `manifest.json` | OS 包清单：`name` / `package_version` / `device` 条目 |
| `data.json` | 显式节点格式的包数据（安装程序把它搬进 `$MEM`） |
| `install.py` | 安装程序：`plm.command` 程序正文，读 `$MEM.pkgrepo`、搬数据、提交装载申请、批准后完成挂载 |
| `devices/sysinfo.py` | 设备代码（宿主 Python，`SysInfoDevice`）；只读，不 import 包内其他代码 |

## 前置

- base 镜像启动的运行实例（`python -m avm.console images/base/base.json`）；
- `$MEM.pkgrepo` 已挂载（base 自带），默认 `root` 是仓库根，所以本例的包路径是 `packages/sysinfo`；
- 设备代码经 `$MEM.device_loader` 挂载：申请写 `$MEM.device_loader.request.<名字>`，操作员在面板设备动作里批准，再写 `$MEM.device_loader.resume.<名字>` 完成挂载。审批值由设备在批准时生成，安装程序不提供。

## 手动安装流程

下面每一步都是一条普通 AVM 指令；仓库没有 OS 包管理器，实际操作可由你的 OS 软件/assistant 执行，或经面板原始动作一条条提交。

**1. 装载包目录**

```text
memory_write $MEM.pkgrepo "packages/sysinfo"
```

成功后可检查：

```text
$MEM.pkgrepo.manifest     # 清单原文
$MEM.pkgrepo.files        # manifest/data/install/devices 的大小与 sha256
$MEM.pkgrepo.data         # 数据包原文
$MEM.pkgrepo.installer    # 安装程序正文
```

**2. 把安装程序装成程序节点 + para**

把 `$MEM.pkgrepo.installer` 的全文作为程序节点 `content`（`ctrl.type="python"`），`para` 的 `model` 为 `plm.command`。可以用 base 的构造器一步建出来，也可以手写：

```text
memory_make  $MEM.workshop  programs  dict
memory_make  $MEM.workshop.programs  pkg_installer  dict
memory_make  $MEM.workshop.programs.pkg_installer  name     str
memory_write $MEM.workshop.programs.pkg_installer.name      pkg_installer
memory_make  $MEM.workshop.programs.pkg_installer  content  str
memory_write $MEM.workshop.programs.pkg_installer.content   <install.py 全文>
edit_metadata $MEM.workshop.programs.pkg_installer type=set key=type value=python
memory_make  $MEM.workshop  paras  dict
memory_make  $MEM.workshop.paras  pkg_install  dict
memory_make  $MEM.workshop.paras.pkg_install  model  str
memory_write $MEM.workshop.paras.pkg_install.model  plm.command
edit_metadata $MEM.workshop.paras.pkg_install type=set key=type value=para
```

**3. 建安装对话**

```text
create_cmd $MEM.workshop.programs.pkg_installer $MEM.workshop.paras.pkg_install
```

得到 `cid=N`。

**4. 投递安装命令（提交挂载申请）**

`device_file` 是相对 base 镜像目录（`images/base`）的宿主路径；`mount` 是单段设备名：

```json
{"action": "install",
 "device_file": "../../packages/sysinfo/devices/sysinfo.py",
 "mount": "sysinfo",
 "device_class": "SysInfoDevice",
 "device_args": {"probe_versions": ["git", "python3", "pip3"]},
 "dest_parent": "$MEM.workshop.installed",
 "dest_name": "sysinfo"}
```

```text
send_instruction <cid> '<上面的 JSON>'
```

安装程序会依次：读 `$MEM.pkgrepo.data` 把字段搬进 `$MEM.workshop.installed.sysinfo`（`memory_make`/`memory_write`）→ 写 `$MEM.device_loader.request.sysinfo` 提交挂载申请。返回 JSON 摘要，`pending=true` 表示数据已搬入、设备**尚未挂载**，等操作员批准。

**5. 在面板批准装载**

打开面板，选中 `$MEM.device_loader` 设备节点，在「设备声明的调试动作」里点「批准装载 sysinfo」。批准动作校验设备文件自申请以来没有变化，并记录 `approved_sha256` / `approved_at` / `approved_via`；它不挂载。

**6. 完成挂载**

再向安装程序投递：

```json
{"action": "resume", "mount": "sysinfo"}
```

安装程序写 `$MEM.device_loader.resume.sysinfo`，设备在这一条指令里 import 设备代码、挂载并记 `device_mount`；重启时按记录重放，不重新审批。

**7. 验证**

```text
$MEM.sysinfo                 # 说明书 + 一行摘要
$MEM.sysinfo.os              # 发行版/内核/架构/WSL/容器/init
$MEM.sysinfo.python          # 宿主 Python 运行时
$MEM.sysinfo.resource        # CPU/负载/uptime/内存/磁盘容量
$MEM.sysinfo.proc            # pid/uid/cwd/镜像目录
$MEM.sysinfo.env             # 白名单环境变量（疑似机密键已屏蔽）
$MEM.sysinfo.tools           # PATH 中的关键命令
$MEM.sysinfo.disk."/tmp"     # 指定路径的容量
$MEM.sysinfo.json            # 以上合成的 JSON
```

`$MEM.workshop.installed.sysinfo` 是本次安装搬进来的包数据；`core.device_specs["sysinfo"]` 里有设备来源、文件哈希与审批字段。

## 安全与审批

- `devices/sysinfo.py` 是宿主 Python 代码，能读平台/进程/环境/磁盘信息，代码里用 `subprocess`/`shutil` 做探测；挂载前应人工审查。
- 审批走 `$MEM.device_loader` 的申请/批准/挂载三步：申请只登记，设备文件哈希由设备在批准动作里复核并记录；请求方（安装程序或 AI）无法携带审批值。文件在批准后变化时 `resume` 会失败并要求重新申请。
- 设备只读：根地址写入、子路径写入都会得到 `MemoryTypeError`。

## 清理

安装没有对应的 guest 卸载指令。需要摘掉设备时，用控制台的设备屏蔽/卸载路径（宿主调试原语）或重启宿主；包数据节点在 `$MEM.workshop.installed.sysinfo` 下，删掉它们即回到安装前。

## 来源

设备代码来自 AVM 内的 assistant 对话（cid 6）在 2026-11-14 20:37 通过 `memory_make`/`memory_write` 写入 `$MEM.workshop.devices.sysinfo`；进程没有留下对应 generation 快照，代码与说明从当前 `base.journal` 的 `memory_set` effect 还原，语法检查通过、未执行。

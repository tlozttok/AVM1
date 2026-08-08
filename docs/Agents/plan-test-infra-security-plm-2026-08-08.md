# 计划：测试基础设施（信息查询设备 + 安全管理 + 图灵完备 PLM）

**时间**: 2026-08-08
**状态**: 已实施（2026-08-08，用户确认四个设计点后）
**背景**: AVM 功能变复杂（调度、ICC、多播、PLM、镜像），需要更复杂的测试。用户计划写一个 AI 对话尽情测试（提示词用户负责），需要：①信息查询设备；②安全管理（防止 AI 及 AI-Python 交互访问虚拟机外部）；③可执行图灵完备行为的 PLM。
**起因**: 用户明确三个需求。
**行动**（提案）:
1. **信息查询设备**（只读，挂载到 $MEM，类似 /proc）：`$MEM.conversations`（cid/名字/状态/para_ref）、`$MEM.scheduler`（活跃/就绪/休眠队列 + 预算计数）、`$MEM.services`（已有）、`$MEM.icc`（待处理 ICC 记录）、`$MEM.memory`（内存树摘要）、`$MEM.monitor`（监测帧摘要）。支持子路径查询。
2. **安全管理**：双面——AI 侧工具集固定（无 OS 工具），设备挂载受控，信息设备只读；Python 侧受限执行：安全 builtins 白名单 + 封锁模块导入（无 __import__）+ AST 检查（禁用 import 节点与双下划线属性访问，堵 __class__ 逃逸）。最佳努力沙箱（真正强隔离需子进程/OS 隔离，超出 VM 范围，注明）。
3. **图灵完备 PLM**：新增 `plm.python`——系统提示词为 setup 代码（受限执行进持久命名空间，跨轮次保留）；用户提示词为代码，执行并捕获 stdout（print 即返回）；信封带 icc_id 时自动构造 return_result 返回（同 plm.simple 模式）。持久命名空间 + 循环/函数 → 图灵完备。
**影响**: 测试 AI 可查询 VM 状态并驱动 Python 对话；安全沙箱防外部访问；plm.python 提供通用计算能力。镜像 meta 可配 `instruction_budget` 已支持，测试对话可配合。
**待用户定夺**: ①信息设备集合与挂载路径；②沙箱强度（builtins 白名单 + AST 检查是否够）；③plm.python 行为（setup 代码 + stdout 即返回 + 自动 return_result）；④信息设备是否全局开放（内核态/用户态隔离是后续）。

**实施记录（2026-08-08）**:
- 安全边界明确：只防宿主系统访问（文件/网络/进程/时钟/熵）；AVM 内部（内存、调度、设备）测试 AI 随便操作（压力测试），信息设备只读。
- `avm/sandbox.py`：安全 builtins 白名单 + AST 检查（禁 import/from-import、禁双下划线属性如 `().__class__`、禁危险调用名 open/eval/exec 等）。
- `avm/info_devices.py`：六只读设备挂载 $MEM（conversations 列表+单对话、scheduler 活跃/就绪/休眠+预算、services 已有、icc、memory 摘要、monitor 帧），写操作报错。
- `plm.python`：Jupyter 笔记本式——系统/用户提示词都执行进同一个持久命名空间；多个 print 全部执行、输出按顺序拼接为整体返回；信封带 icc_id 自动 return_result；图灵完备（循环/函数/递归）。`plm.simple` 同步加固（过 AST 检查 + 安全 builtins）。
- 测试：新增 15 个（沙箱拦截、plm.python 共享命名空间/多 print 拼接/图灵完备/自动返回/逃逸报错、信息设备只读）。

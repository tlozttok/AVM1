# 计划：AVM 应用套件 —— 用三个可运行镜像实测运行时完整性

**时间**: 2026-08-16
**状态**: 已实施（按用户 2026-08-16 指令直接落地：3 个镜像、自定义设备/PLM 服务器代码、3 份应用说明、1 组集成测试；见文末实施记录。四部分计划供用户评分，评分或修订意见回填本文件）
**背景**: 运行时已有 9 条指令、三态调度、内存树、事件模型 v1、Python 执行器（plm.simple / plm.python）、信息设备、监测器与镜像加载。现有测试 279 个全部通过，但都以单元测试和机制测试为主；`images/` 下只有 demo / calc_driver / tester 三个基础镜像，缺少一组有明确业务形状、能同时压住多条机制边界的应用。
**起因**: 用户提出「用更有实际性的方式测试 AVM 是否已经完善」，要求先理解文档与架构，再构思若干 AVM 应用，写说明/计划文档（架构 + LLM 式算法/控制流），最后落成镜像文件、设备代码和/或 PLM 服务器代码。
**约束**: 本计划只新增应用、设备、PLM 服务器与测试，不修改 `avm/` 运行时本体；遇到必须由 VM 提供的缺口只记录 issue，不自行扩核。

---

## 一、目标与判定标准

1. **目标**：用三个业务形状不同、可独立启动的应用镜像，实证 AVM 当前指令集能否支撑：
   - 异步子对话（create_cmd）+ 简单投递（send_instruction）往返；
   - 亚对话（create_sub）与 Python PLM 计算外包；
   - 自定义设备挂载与设备路径写入；
   - 事件模型 v1 的 before 拦截/改写、after 通知、notify 唤醒、owner 权限；
   - 镜像外自定义 PLM 服务器（OpenAI 兼容 chat 服务器）接入。
2. **通过标准**：
   - 每个镜像可通过 `load_image` 严格校验；
   - 每个应用有一份说明文档，明确架构、内存布局、消息协议与 LLM 式控制流；
   - 每个应用有确定性集成测试（MockLMU 脚本 + 真实 PLM/设备），不依赖真实 LLM API；
   - 全量 `pytest tests/` 通过，原有 279 个测试不回退。
3. **不通过即缺口**：若某条指令无法支撑应用协议，不在应用里绕行，而是记为 issue；本计划不含内核态/用户态、workspace、链接设备、事件条件等已知「设计中」能力。

## 二、应用集与覆盖矩阵

| 应用 | 业务形状 | 主要压测点 | 产物 |
|---|---|---|---|
| courier_office（传令站） | 操作者 → LLM 调度员 → LLM 工人 → plm.simple 计算 → 记录板 | create_cmd / send_instruction / create_sub / memory_make / close_conversation / plm.simple / 自定义 BoardDevice / 三态调度 | `images/apps/courier_office/` |
| vault_guard（金库守卫） | 操作者 → 守卫打标并注册监听 → 工人写金库 → before 守卫拦截/改写 → after 通知唤醒 | listen_event / edit_metadata / owner / before replace / payload 改写 / after notify | `images/apps/vault_guard/` |
| plm_ledger（账本服务） | 操作者 → LLM 柜员 → 自定义 LedgerPLM（OpenAI 兼容）→ memory_write 持久化 | PLMServer 契约 / 工具调用往返 / create_sub 自动返回 / 镜像外 PLM 注册 | `images/apps/plm_ledger/` |

应用不修改 `avm/`，只使用镜像格式允许的扩展点：设备插件与宿主侧 PLM 注册（对应 docs/record/11.md 分发包四部分中的「设备代码」与「PLM 服务器代码」）。

## 三、架构与 LLM 式算法/控制流（摘要）

每个应用都遵守同一条控制律：**提示词只写状态机与协议，不写业务知识**。LLM 对话的输出只有两种：文本（交互结束/休眠）或工具调用（请求 Core 执行指令）。详细说明在：

- `docs/apps/courier_office.md`
- `docs/apps/vault_guard.md`
- `docs/apps/plm_ledger.md`

### 3.1 courier_office

```
inputs ──> dispatcher(LLM) --create_cmd--> worker(LLM)
             ^  |                                |
             |  | send_instruction(JOB)          | create_sub(calc/plm.simple)
             |  v                                v
           sleep <---- RESULT(send_instruction) calc(Python)
             |
             +---> $MEM.board.append（BoardDevice 追加）---> outputs
```

控制流是三段式：dispatcher 创建 worker 后不休眠，投递 JOB 后主动结束交互；worker 用 create_sub 把确定计算外包给 Python 对话（紧耦合自动返回）；worker 把结论追加到设备并显式 `send_instruction` 回 dispatcher（创建关系不构成返回通道）。

### 3.2 vault_guard

```
guardian(init) --listen_event--> handlers(deposit_pipe, phase=all)
             \--listen_event--> handlers(audit_tail, phase=after)
              \--create_cmd----> worker(LLM)
worker --memory_write $MEM.vault.deposit--> before: replace 或 SAFE:改写
worker --memory_write $MEM.vault.audit ---> after: notify cid=0
worker --send_instruction RESULT----------> guardian
```

事件语义被做成业务规则：before 处理器是守卫（可拒绝），after 处理器是审计（只通知）。`$MEM.vault` 预置 `ctrl.owner="0"`，证明内存 before 监听的 owner 权限链路可用。

### 3.3 plm_ledger

```
teller(LLM) --memory_write $MEM.request--> create_sub(LedgerPLM, para model=plm.ledger)
LedgerPLM: system=JSON配置 --> user=命令 --> 返回 memory_write 工具调用 --> Core 执行
           --> tool 消息 --> 返回最终答复 --> create_sub 自动写回 teller
```

LedgerPLM 是 `PLMServer` 的镜像外子类，与真实 LLM 端点同形：输入 OpenAI messages，输出 content + tool_calls。宿主 runner 先把它注册进 `PLM_REGISTRY` 再 `load_image`。这验证了「一个 Python 对话可以不是 LLM，而是任何 OpenAI 兼容服务器」的执行器边界。

## 四、验收、影响与发现

**行动清单**：
1. `docs/apps/` 新建应用说明与索引；
2. `images/apps/<app>/` 新建镜像 JSON、设备文件、PLM 服务器文件与启动脚本（仅 plm_ledger 需要宿主 runner）；
3. `tests/test_app_images.py` 新建镜像加载、设备行为、事件链路、PLM 契约与全流程集成测试；
4. 全量测试后回填本计划实施记录；README 文档索引与测试计数同步。

**影响**：
- `avm/`、`main.py`、现有镜像与既有测试零改动；
- 新增文件不影响 `python -m avm.image images/demo.json` 等既有路径；
- plm_ledger 必须通过 `python images/apps/plm_ledger/run_ledger.py` 启动（或等价地先在宿主注册 PLM），这是当前镜像格式没有 PLM 声明段的既有边界，不以改 VM 方式绕过。

**发现的设计缺口（记录为 issue，不在本计划实现）**：
1. 镜像格式没有「随镜像声明并注册 PLM 服务器」的段；自定义 PLM 目前只能由宿主 runner 注册。分发包四部分中的 PLM 服务器代码因此缺一层安装协议（与 docs/record/11.md 遗留问题呼应）。
2. `init` 只接受 `settingup`，不能把入口设为 Python 对话；全 Python 的确定性镜像仍需要一个 LLM 入口（或宿主直接 `core.start`）。
3. `$MEM.inputs.-1` 在自动化测试中会阻塞宿主终端；应用测试用 MockLMU 脚本与直接向 input 设备预置数据绕过，说明测试宿主侧需要更可控的输入注入方式。
4. 路径语法已按用户定夺精确化：`BoardDevice` 写 `$MEM.board.append` 追加、读 `$MEM.board.latest` 取最新；数字索引（含 -1）保持普通列表语义，-1 不再承担追加职责。规范化 append 设备基类仍未进入 `avm/memory_device.py`，若后续要把该语法收进运行时，需另立计划。

**实施记录（2026-08-16）**:
- 修订：按用户定夺，courier_office 的记录板从 `$MEM.board.-1` 追加改为 `$MEM.board.append` 追加、`$MEM.board.latest` 读最新；-1 恢复普通列表语义。
- 新增 3 个应用镜像、2 个设备文件、1 个自定义 PLM 服务器文件、1 个宿主启动脚本；
- 新增 `tests/test_app_images.py`，全量 pytest 通过，测试计数见 README；
- 本文档状态由「计划」回填为「已实施」。

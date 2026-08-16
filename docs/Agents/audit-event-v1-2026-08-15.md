# 审计：事件模型 v1 实施结果与待收尾清单

**时间**: 2026-08-15
**状态**: 审计完成；P0 已由本代理直接修复，P1/P2 已由收尾计划实施（见 plan-event-v1-cleanup-2026-08-15.md）
**背景**: 用户告知计划已由另一模型实施，要求复查还有哪些未完成，以判断是否可以开始 AVM 软件与 OS 工作。
**起因**: 按项目规则，对已实施计划做对照审计并留档。
**行动**:
1. 运行 `pytest tests/`：254 个全部通过（审计前 252，P0 修复后 +2）。
2. 运行三镜像加载：tester / demo / calc_driver 均正常，工具集为 9 个。
3. 通读 `avm/events.py`、`avm/core.py`、`avm/memory.py`、`avm/info_devices.py`、`avm/monitor.py`、`tests/test_events.py` 及文档 diff，与 `docs/Agents/plan-event-v1-2026-08-15.md` 逐项对照。
4. 额外做了边界探测（设备路径权限查询、设备事件路径前缀、处理器失败路径）。

## 已按计划完成

- 第 9 条指令 `listen_event` 与 9 种工具事件。
- 5 种 before 事件（memory_read / memory_write / memory_make / edit_metadata / send_instruction）与 4 种 after 事件。
- Python 处理器：`handle(event)->dict`，每次全新沙箱命名空间，无状态。
- 注册顺序即优先级、首个 replace 短路。
- `ctrl.owner` 直录：memory_make 创建 dict/list、memory_write 隐式中间容器写 owner；str 叶子无 ctrl；`edit_metadata` 拒绝 set/del owner。
- notify 内部投递（不经事件管道）。
- `$MEM.listeners` 信息设备、monitor 帧 lmu.events、`main.py --frames` 事件计数。
- `close_conversation` 自动撤销监听；`Core.emit_device_event` 入口。
- 文档：CONTEXT / README / README_EN / runtime / system-image / tester.json 已同步。

## 审计发现（按优先级）

### P0（开始 OS 前应修）

1. **权限检查会触发设备子路径副作用**。`EventSystem._owner_of` 直接 `core.unwrap(完整路径)`；若目标是设备路径（如 `$MEM.inputs.-1`），`resolve_path` 会执行设备逻辑。对 `$MEM.inputs.-1` 注册 before 监听会在权限检查阶段阻塞等待操作者输入。修复方向：在 `_owner_of` 里先判断路径任何前缀是否命中 `core.mem._devices`，命中即视为无主路径返回 None（v1 不监听设备路径）。
2. **`emit_device_event` 的前缀边界错误**。当前校验 `target.startswith("$MEM." + device_path)`，设备路径 `"box"` 会放过 `"$MEM.box2"`。修复：`target == "$MEM.box"` 或 `target.startswith("$MEM.box.")`。

### P1（语义/状态需要定夺或补强）

3. **notify 到当前活跃对话会同时进 ready 队列**。例如 create_cmd 的 after 通配监听 notify 自己，结果 active=0 且 ready=[0]，三态互斥被破坏。此行为继承自 send_instruction 的 self 投递，但 notify 让它更容易出现。需要定义 self 投递语义：要么 active 目标只进 batch 不入 ready，要么显式接受 active+ready 并在文档说明；修复后补测试。
4. **before 处理器失败时事件摘要不进入 monitor**。`dispatch` 在失败时抛出前不执行 `_append_summary`，`_turn_events` 为空；同时 stderr 会打印两次（dispatch 一次、指令 execute 的 `_report_tool_error` 一次）。应记录失败摘要并只报一次。
5. **before 链中途失败时，先前处理器累积的 notify 被整体丢弃**。fail-closed 丢弃原操作合理；是否丢弃已收集的 notify 需定夺（建议：保留并投递，因为它们是已成功的处理器的输出）。
6. **多播 send_instruction 的事件处理器失败可造成部分投递**。当前每个目标独立 dispatch；先处理的目标已投递后，后一个目标处理器抛错则整体返回 Error，但前面的目标已收到消息。需要定夺全有或全无，或者明确允许部分投递并在响应中说明。
7. **测试覆盖缺口**：create_sub after 通知、memory_make / edit_metadata 的 before replace、send_instruction 改写 content、设备前缀边界、notify self 的队列状态均无直接测试。P0/P1 修复时应补齐。

### P2（文档与仓库卫生）

8. **`docs/programming-guide.md` 未更新**：仍写「注册事件监听（设计中，未实现）」「信号事件（设计中，未实现）」，与已实现的事件模型 v1 冲突。
9. **`AGENTS.md` 与当前工程脱节**：仍描述已删除的 register_service / call_service / transfer_service；引用不存在的 simple_main.py、web/server.py、scripts/run_dual_terminal.sh、config.json。OS 工作会依赖 AGENTS.md，应先修。
10. **`avm/__init__.py` 未同步**：docstring 仍为 `AVM — Agent Virtual Machine`；未导出 `ListenEventInstruction` / `EventSystem` / `ListenersDevice`。
11. **docs/record 未进版本控制**：`.gitignore` 的 `docs/record/*` 使 record/5 至 record/13（含本次事件设计纪要 12/13）全部为 ignored untracked；设计留档不在 git 中。需要改 .gitignore 或 `git add -f` 并提交，否则历史纪要不可随仓库交付。
12. **images/out 旧产物仍被 git 跟踪**：当前内容还是 12 工具/ICC 时代快照，与运行时不符。建议从跟踪中移除，或明确其运行产物身份后重新生成。
13. **计划文件状态未回填**：`plan-event-v1-2026-08-15.md` 顶部仍为「待实施」，没有实施记录；`plan-remove-service-tools-2026-08-13.md` 顶部仍为「提案（待审批）」，但其后已有实施记录。本次审计文件即回填动作的一部分，实施记录应补进对应计划。
14. **所有事件模型实现仍是未提交工作区改动**（git status 显示 M/??）。修复并复核后应作为一个或多个提交落盘。

### P3（后续设计与 OS 前置，不阻塞 v1 收尾）

15. **处理器无超时/预算**：before 处理器在活跃对话的指令路径内同步执行，死循环会挂起整个运行时。现有 PLM 同样无超时，但事件处理器在热路径上，后续应有预算或超时设计。
16. **owner 目前只约束监听注册，不约束写入与 edit_metadata**。任何对话仍可写任意路径、改任意节点的 type 等 ctrl（除 owner 键）。这是 OS/capability 层的主要缺口。
17. **str 叶子不能直接携带 owner**，只能靠祖先容器；直接在无主根下创建的 str 不可被监听。已在文档记录，OS 层若要更细粒度需新设计。
18. **owner 继承/转移语义未定义**：父容器 owner 与子节点 creator 不一致时，以子节点为准；对话关闭后其 owner 标记残留，无人能再注册监听。OS 层需要 owner 继承与清理规则。

## 结论

事件模型 v1 的代码主体与测试基本按计划完成，254 个测试通过。**P0 两项已直接修复；P1 的调度/失败语义、P2 的文档和仓库卫生由 plan-event-v1-cleanup-2026-08-15.md 交付实施。** OS 本身还需要内核态/用户态、workspace、链接设备、名字↔cid、文件密码等独立计划，不应与本次收尾混在同一计划里。

**影响**: 本次仅审计与留档，未触碰代码；后续实施者按本文档修复。


---

## P0 直接修复记录（2026-08-15，本代理执行）

1. `EventSystem._has_device_prefix` + `_owner_of` 重构：
   - 权限检查先用纯注册表判断路径是否含设备前缀；
   - 含设备前缀的层级绝不调用 `core.unwrap` / `resolve_path`，杜绝 `$MEM.inputs.-1` 阻塞等副作用；
   - 设备路径的所有权取自设备前缀之外的最近数据祖先（例如 `$MEM.box.dev` 看 `$MEM.box` 的 owner），保留设备事件可被祖先 owner 监听的能力。
2. `Core.emit_device_event` 前缀边界修复：target 必须等于 `$MEM.<device_path>` 或以 `$MEM.<device_path>.` 开头；`$MEM.box2` 不再被 `box` 放行。
3. 回归测试 +2：`test_owner_check_does_not_touch_device_subpaths`、`test_device_event_target_prefix_boundary`；全量 254 通过。
4. README.md / README_EN.md 测试数同步为 254。

**注意**：第 1 项中「设备路径取最近数据祖先 owner」是过渡约定，不是最终语义。挂载点前后路径语义分裂的影响见 docs/record/14.md；OS 工作前需要把挂载点边界设计为独立计划。

---

## 收尾实施记录（2026-08-15）

- P1 已按用户定夺实施：禁止自我投递、before 失败返回 `handler_error` 事件结果并保留 notify、多播部分投递、monitor 记录失败摘要、stderr 去重。
- P2 已处理：`docs/programming-guide.md` / `AGENTS.md` / `avm/__init__.py` / `.gitignore` 更新；`images/out/` 移除跟踪；计划状态回填。
- 验证：`pytest tests/` 266 个全部通过；三镜像加载正常；`compileall` 通过。

# 计划前勘察：AVM 项目下一计划的目标收集

**时间**: 2026-08-15
**状态**: 已收敛（用户选定事件模型方向；正式计划见 plan-event-model-2026-08-15.md）
**背景**: 用户要求为 AVM 项目专门制定计划，并指示先浏览项目。按项目规则（AGENTS.md），制定计划前先查阅并建立 docs/Agents 下的记忆文件；本文件是本次计划流程的起点记录，后续确定目标后另立正式计划文件，并回填本文件的定夺结果。
**起因**: 用户给出两段式指令：先浏览项目，再为项目做计划。为防止在未确认目标前直接产出计划，先建立勘察基线。
**行动**:
1. 查阅 docs/Agents 全部 20 个记忆文件，理清计划格式与已实施/待定状态；重点阅读最新文件：plan-remove-service-tools-2026-08-13（提案中但实施记录已完成）、plan-para-llm-node-2026-08-13（已实施）、plan-doc-restructure-2026-08-14（已实施）。
2. 查阅 docs/record/9.md（事件体系与设备化方向）、record/10.md（起源与编译器愿景）、record/11.md（提示词包分发与伪提示词），以及 docs/runtime.md、docs/programming-guide.md、docs/philosophy.md、docs/system-image.md。
3. 通读 avm/ 代码（core、types、memory、memory_device、image、info_devices、monitor、python_server、sandbox）、main.py、三个系统镜像与 tests/。
4. 验证基线：`pytest tests/` 220 个全部通过；`images/tester.json`、`images/demo.json`、`images/calc_driver.json` 均可加载（不触发真实 API）。
5. 检查 git 状态：当前分支 `simpliy_for_refactor`；HEAD 为 2f87116（简化对话间通信机制）；未提交变更为文档重构（CONTEXT.md、README.md、README_EN.md 修改 + docs/runtime.md、docs/programming-guide.md、docs/philosophy.md、plan-doc-restructure 新建）。
**勘察发现**:
- 运行时当前状态与 README「当前状态」一致：8 条指令、简单投递、三态调度 + 指令预算、镜像加载、PLM 执行器、信息设备、监测器、思维链回传、单对话检查点。
- 服务三工具与 ICC 机制已从代码、测试、镜像中删除；三个镜像源文件无 register_service / call_service / transfer_service / return_result 残留。
- docs/Agents/plan-remove-service-tools-2026-08-13.md 的「状态」字段仍写「提案（待审批）」，但实施记录已完整；属记忆文件自身状态标注滞后，待本次或后续文档计划顺带修正。
- AGENTS.md 与当前工程脱节：仍描述 call_service / transfer_service / register_service 等已删除机制；「运行」节引用的 simple_main.py、web/server.py、scripts/run_dual_terminal.sh、config.json 在仓库中不存在。
- avm/core.py 中 close_conversation 的工具描述仍含「其注册的服务注销……未完成调用清理」旧语义；avm/__init__.py 模块 docstring 仍为「AVM — Agent Virtual Machine」；images/out/ 下旧运行产物仍含 12 指令版快照（属运行产物，随下次运行重新生成）。
**候选计划域（未定夺，供用户选择）**:
- 文档/描述对齐：修正 AGENTS.md 与 core.py 工具描述、__init__.py docstring、记忆文件状态标注。
- 事件机制：docs/record/9.md 已确认「事件是中断，不是软件总线」的设计前提，是 README 当前状态中排序靠前的未实现项。
- 设备化（xx.xx.post / xx.xx.response 方向）：恢复被删除服务能力，并给「对话自决休眠 + 预期行为监听」提供载体。
- 提示词包分发与动态设备挂载：docs/record/11.md 的新方向，依赖 MEM 动态挂载设备能力与安装程序规范。
- 其他未实现项：内核态/用户态分离、链接设备、名字 ↔ cid 转换、文件密码与访问回调、错误处理机制细化。
**影响**: 本次只读勘察 + 新建本记忆文件，未触碰 avm/、tests/、镜像与既有活跃文档；为下一份正式计划提供基线。
**定夺结果（2026-08-15）**: 用户指定事件模型方向（每个工具调用对应一个事件）。已新建 docs/Agents/plan-event-model-2026-08-15.md 记录模型、现状对照与决策点；待用户对各决策点评分与选择后形成可实施的正式计划。

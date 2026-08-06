# 计划：PLMServer 三个抽象方法的实现方案

**时间**: 2026-08-04
**状态**: 基类机制与 SimplePLM 已实施（2026-08-05），其余待定
**背景**: `avm/python_server.py` 的 PLMServer 已具备入口骨架（消息转换、前后输入比较取增量、工具定义解析、make_response），三个抽象执行方法（system/user/tool）待实现。
**起因**: 用户询问三个抽象方法如何实现；需要确定基类提供什么机制、用户程序怎么写，以及"系统提示词中导入模块、设置可复用背景"和外部封锁如何落地。
**行动**（提案）:
1. 基类机制（让实现"不任意"）：
   - `parse_message(content)`：解析指令 JSON（from/to/icc_id/content），失败返回带 `__error__` 的字典（格式变换辅助，指令的语义解释是程序自己的）；
   - 状态约定：可复用背景放实例字段，`_execute_system_message` 负责初始化。
   - **工具调用行为归具体实现**（用户纠正）：工具调用的构造（何时调用、调用什么、传什么参数，按 `self._tools` 里的定义）、自己调用的记忆与返回关联——全部是子类的行为。就像 transformer 神经网络（模型）不保证工具调用行为一样，基类/运行时也不提供工具调用构造器、不保证工具调用行为；基类只做消息变换（转换、增量、派发、包装输出）。
   - **工具调用 id = 确定性 hash(发来的消息)**（用户建议）：id 是对被响应消息的确定性编码，同一消息必然产生同一 id；基类提供 hash 工具函数，程序构造工具调用时使用。
2. **在提示词中编程**（用户纠正，核心模型）：type="python" 节点内容（提示词）就是可执行的 Python 程序源码，系统提示词和用户提示词都是提示词。PLMServer 执行提示词来运行程序：
   - `_execute_system_message(content)`：受限执行提示词（import 白名单 + 禁止 os/time/random/网络/IO）——导入模块、设置可复用背景，这是外部封锁的执行点；
   - `_execute_user_message(content)` / `_execute_tool_message(content, call_id)`：由具体 PLM 类型的固定行为模式决定。
3. **PLM 类型 = 固定行为模式，由 para 的 `model` 参数选择**（用户确认的核心模型）：一个 Python 对话的 para 里存 `model`，指代 PLM 的具体类型（像 LLM 模型名一样）。每个 PLM 类型有自己的固定行为模式，例如：
   - SimplePLM：exec 系统提示词建立持久上下文；用户提示词直接 exec 并把 stdout 重定向到输出——代码里的 print 就是"AI"的返回；
   - 更复杂的 PLM：加安全检查、确定性检查等。
   - 三个抽象方法是 PLM 类型的实现，由 model 参数选择具体实例；不需要提示词里定义动态处理程序。
   - **状态与调用配对**（2026-08-05 用户补充）：具体 PLM 实例自己定义状态（如 SimplePLM 的 `namespace`），工具返回处理时读取状态 + 基类保证的调用记录；基类 `_call_records`（call_id → {name, arguments}）在 Response 带 tool_calls 时自动记录，保证返回与调用行为配对。工具调用与 AI 回复共享响应接口（`make_response`，函数即可）。
4. 测试：Mock 消息序列验证完整循环（system 初始化 → user 指令 → 工具调用/返回 → 结果）。
**影响**: 三个方法落地后 Python 执行器可跑通完整消息循环；受限 exec 是"只能经设备访问外部"的第一次实现；为 Core 接入（type="python" 检测 + 执行器切换）铺路。
**实施记录（2026-08-05）**: 基类新增 `_call_records`（Response 带 tool_calls 时自动记录）、`call_id_for(message)`（sha256[:16] 确定性 id）；`PLM_REGISTRY` + `create_plm(model)`（注册表在本模块）。`SimplePLM` 经用户简化后 = **计算器**：只接受一行表达式，eval 求值（受限命名空间、无内置函数、提供 math），结果直接返回，无状态；系统/工具消息不处理。

**实施记录（2026-08-06）**: `SimplePLM` 支持返回工具（用户要求）——用户消息是 JSON 信封，取出 content 求值；信封带 icc_id 时，构造 `return_result(content, icc_id)` 工具调用（id = `call_id_for(消息)` 确定性 hash）把结果投递给发起者；错误结果也通过工具返回。裸表达式（无 icc_id）仍直接返回内容。

**实施记录（2026-08-06，Core 接入）**: `Conversation` 增加 `para_ref`；新增 `PLMExecutor`（把 PLMServer 适配成与 LMU.exec 同形状：构建 OpenAI 格式消息 → 调 PLM → 解析 Response 为 return_calls → 提交历史）；`_build_conversation` 记录 `conv.para_ref`，system_ref 指向 MetaDict 时取其 content 作系统提示词；`advance_conversation` 用 `_get_executor(conv, para)` 选执行器；`start` 记录 para_ref。

**执行器归属修正（2026-08-06，用户纠正）**: 执行器**不挂在对话上**——对话只是状态（消息 + para 配置）；执行器由 Core 管理：`Core._executors` 按 cid 缓存 PLM 实例（PLM 状态跨轮次保留），`_get_executor` 按 para.model 在 `PLM_REGISTRY` 中查找并惰性创建，否则用 core.lmu。集成测试 2 个：计算器全流程（create → send → 求值 → return_result → 结果回到调用者）+ LLM 对话仍走 core.lmu。

**执行器选择修正（2026-08-06，用户纠正）**: 不静默回退 lmu——选择基于对话的**程序类型**（Conversation 增加 `is_python`，由 system_ref 节点的 `ctrl.type == "python"` 决定）：Python 对话的 model 必须在 `PLM_REGISTRY` 中，否则抛错（像 API 对错误模型报错一样）；LLM 对话才用 core.lmu（model 交给 API 校验）。新增测试：Python 对话 model 错误 → ValueError。

**已知小缺口**: Monitor 的 lmu 帧对 Python 对话仍读 `core.lmu.last_call`（未接 PLMExecutor.last_call），后续再处理。

**待用户定夺**: ①SimplePLM 的工具返回行为（当前：注入命名空间并原样返回 content）是否符合预期；②exec 模型里代码如何发出 tool_calls——SimplePLM 目前是纯 print 程序，不发出工具调用；共享接口 make_response 已支持 tool_calls，复杂 PLM 可加（如命名空间注入 call_tool 辅助函数）；③安全检查（复杂 PLM 的事，SimplePLM 直接 exec 不加）。

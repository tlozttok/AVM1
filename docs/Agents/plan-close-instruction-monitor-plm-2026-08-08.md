# 计划：内核级关闭指令（close_conversation）与监测器接入 PLM 执行器

**时间**: 2026-08-08
**状态**: 已实施
**标注**: 按用户计划改动（用户确认实施关闭指令、解决两个顺带发现的问题，并更新 tester 提示词）。
**背景**: 两个挂起项：① `_executors` 缓存只增不减——个体结束（内核级关闭指令）未实现，无法在关闭时释放 PLM 状态（变量空间）；② monitor 帧的 lmu 块只读 `core.lmu.last_call`，`PLMExecutor.last_call` 未接入，Python 对话的帧里没有 result/tool_calls。CONTEXT.md 已定义"个体结束"概念：由对话自身或其他对话调用内核级关闭指令，不需要被关闭对话的回应。
**起因**: 用户确认实施这两项并加入关闭指令，同时更新 tester 镜像提示词。
**行动**（提案，全部为"按用户计划改动"）:
1. **关闭指令**（avm/core.py）：新增 `CloseInstruction`（工具名 `close_conversation`，参数 `cid`，可关闭自身或其他对话）并注册。`Core` 增加 `_finished: set`；`_close_conversation(cid)` 做：标记 finished、移出就绪/休眠队列、释放 PLM 执行器、注销其注册的服务、清理未完成调用（其发起的 ICC 记录作废；发给它的 ICC 记录以 Error 通知发起者并唤醒，多播时错误并入组、组归零才唤醒）。新增 `_drop_icc`（移除记录并递减多播组计数，归零清理组记录）。
2. **调度守卫**（avm/core.py）：ICC 记录增加 `callee_cid`；`return_result` 对已关闭/不存在的发起者不投递（记录作废，报错）；`send_instruction` 对已关闭目标报错；`_resume` 拒绝激活 finished 对话；亚对话完成时父已关闭则无处写回、按普通交互结束进入休眠；`advance_conversation` 在本轮发生个体结束后选择下一个活跃对话。
3. **监测器接入 PLM**（avm/monitor.py + avm/core.py）：`Monitor.record` 增加 `last_call` 参数（None 时回退 `core.lmu.last_call`，兼容 MockLMU 与直接调用）；`advance_conversation` 把本轮执行器的 `last_call` 传入（LLM 与 PLM 共用一条路径）。
4. **信息设备**（avm/info_devices.py）：`$MEM.icc` 显示 `callee`。
5. **tester 提示词**（images/tester.json）：系统提示词工具列表加 `close_conversation`；手册指令集 10 → 11；instruction_ref 增加第 11 条；scheduling 一节把"当前没有内核级关闭指令"改为已实现的 finished/个体结束语义。
6. **文档**：CONTEXT.md 指令集增加"内核指令"分类；README.md / README_EN.md 指令表加行、finished/个体结束更新、当前状态改为 11 条指令并移除"设计中"的关闭指令项。
**影响**: 个体结束语义落地：关闭不可逆，closed 对话保留在 `_conv_by_cid` 供监测器/信息设备展示为 finished；服务关闭即注销并通知等待者；多播组中部分目标关闭时错误并入合并返回，组归零才唤醒发起者；Python 对话的执行结果进入监测帧。
**实施记录（2026-08-08）**:
- avm/core.py、avm/monitor.py、avm/info_devices.py、tests/test_avm.py、tests/test_monitor.py、images/tester.json、CONTEXT.md、README.md、README_EN.md 已改；194 个测试全部通过（原 184 + 10 新增）。

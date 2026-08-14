# 计划：删除服务工具与 ICC 机制（register / call / transfer / return_result + ICC 本体）

**时间**: 2026-08-13
**状态**: 提案（待审批）
**标注**: 按用户计划改动。用户已确定 ICC 代替方案方向：**简单投递，不保证有对话消息返回，只保证一个报告发送成功的工具返回，没有 return_result**。分支 `simpliy_for_refactor` 已创建，本计划在此分支实施。
**背景**: 与 DeepSeek Harness 对照后（docs/record/9.md），用户决定扩展 AVM 能力：大部分工具调用改为设备访问，Core 只保留最核心的对话推进、发送消息、创建对话。服务注册表是数据（写成设备无非是加权限控制和输入控制）；服务调用语义将被设备 post/response 取代；显式返回（return_result）被取消——"返回就是返回一个普通的指令"；ICC 记录机制（icc_id → 发起者映射）整体废弃，改为简单投递。
**起因**: 用户指示"先把多余的工具代码删掉才能好好思考之后要做什么"；用户随后明确 ICC 代替方案（简单投递 + 发送成功确认，无 return_result），本计划据此将 ICC 机制本体纳入删除。

## 决策点（需用户确认）

1. **close_conversation 保留**：删除其 ICC 清理分支（`_close_conversation` 中 `for icc_id, rec in list(self._icc.items())` 一段）后保留内核关闭能力。它是"对话推进"的一部分（个体结束），设备化关闭方案未定，本次不删。若用户要求删除需另定替代。
2. **send_instruction 新语义**（按用户描述）：`send_instruction(cid, content)`——投递 content 到目标对话（唤醒：dormant/ready → ready 队尾），立即给调用者返回"已投递到 <身份>"确认；**不建 ICC 记录、不等待返回、调用者不自动休眠**（休眠靠对话自己让出）。删除参数：wait、return_mode、format；删除多播工具返回合并（cid 列表仍可投递，每个目标一条消息，返回统一确认）。
3. **UserMessageBatch 结构化合并段**（`_merged`、`add_tool_response` 的 from_identity/cid/icc_id 参数）：多播工具返回合并的唯一消费者是 ICC 多播返回，一并删除。

## 行动（提案）

### 1. avm/core.py

**删除工具类**：
- `RegisterServiceInstruction`（279-305 行）
- `CallServiceInstruction`（308-364 行）
- `TransferServiceInstruction`（367-393 行）
- `ReturnResultInstruction`（396-471 行）

**`_instruction_registry()`**：12 条 → 8 条（memory_read / memory_write / memory_make / edit_metadata / create_cmd / create_sub / send_instruction / close_conversation）

**`SendInstruction` 简化**：
- tool_def：只保留 `cid`、`content` 参数（cid 支持 int 或列表；无 wait/return_mode/format）
- execute：解析目标（不存在/已关闭 → Error 工具响应）→ 投递 content 到每个目标 user_batch → 唤醒目标（dormant/ready → ready 队尾）→ 返回"已投递到 <身份>"确认；**不写 `_icc`、不写 `_icc_groups`、不挂起调用者**
- 投递内容：直接投递 content 字符串（不再强制 JSON 信封——用户："发送消息不再强制包装成 json"）

**删除 ICC 机制**：
- `Core.__init__`：删 `self._icc`、`self._icc_groups`；删 `self.mem.mount("icc", IccDevice(self))`；删 `self.mem.mount("services", _ServicesDevice(self))` 与 `self._services`
- `_ServicesDevice` 类（680-700 行）
- `_drop_icc` 方法（1072-1079 行）
- `_close_conversation`：删 ICC 清理循环（1029-1070 行）与服务注销分支（1025-1026 行）
- `_process_return_calls`：删 ICC 相关（return_result 路由逻辑随 ReturnResultInstruction 消失；多播 `_icc_groups` 处理一并消失）

### 2. avm/types.py

- `Conversation.service_desc` 字段、`__init__` 参数、`to_checkpoint`/`from_checkpoint` 序列化（243、248、256、381、414 行）
- `UserMessageBatch`：删 `_merged`、`add_tool_response` 的 from_identity/cid/icc_id 参数与结构化段逻辑（456-480 行）、`to_tool_messages` 的合并物化（509-511 行）
- 若 `metadata["returned"]` 只为 return_result 服务：删 `_on_conversation_finished` 中 `conv.metadata.get("returned")` 检查（985 行），亚对话完成总是自动写回唤醒父

### 3. avm/python_server.py（PLM）

- `SimplePLM._execute_user_message` / `PythonPLM._execute_user_message`：删 icc_id 分支（信封解析仍保留？——投递不再强制 JSON，PLM 按"消息即代码/表达式"处理，无 icc_id 概念）；删 `return_result` 工具调用构造（196-206、246-255 行）
- 文档字符串同步（"信封带 icc_id 时构造 return_result" 描述删除）

### 4. avm/info_devices.py

- 删 `IccDevice` 类（81-97 行）

### 5. 测试更新

- `tests/test_avm.py`：删除/重写服务、return_result、ICC、多播合并相关测试（约 27 处引用）；send_instruction 测试改为新语义（投递确认、不等待）；保留 create_sub 自动唤醒父的测试（去掉 returned 分支）
- `tests/test_plm.py`、`tests/test_python_executor.py`：删 return_result 断言，PLM 直接返回结果
- `tests/test_checkpoint.py`：删 service_desc 断言（43、53 行）
- `tests/test_monitor.py` / `test_main.py` / `test_image.py`：若引用 $MEM.icc / $MEM.services 则更新

### 6. 镜像更新

- `images/tester.json`：instruction_ref 删 register/call/transfer/return_result 条目并重编号；send_instruction 条目改为新语义；overview"指令集（12 个）"→ 8 个；observe 删 $MEM.services、$MEM.icc 设备条目；scheduling 节删服务/ICC 描述
- `images/demo.json`、`images/calc_driver.json`：提示词"可用工具"列表删服务工具（calc_driver 的 send_instruction(wait=true) 改为无 wait 投递）

### 7. 文档更新

- `README.md` / `README_EN.md`：指令表 12→8；ICC 协议节改为"简单投递"描述；删除服务术语段；决策与动机节"返回用显式工具"、"ICC 与消息格式"改写为简单投递的动机；当前状态节更新
- `CONTEXT.md`：删"服务（Service）"术语、"ICC id"术语、指令集三服务指令；"调度"节改"事件触发后 core 将对话转为就绪"保留（事件机制仍是设计方向）
- `docs/system-image.md`：80 行"服务由对话运行期自己 register_service"更新
- `docs/Agents/`：本计划；实施后追加实施记录

### 8. 验证

- `pytest tests/` 全量通过
- `python main.py`（debug.json）tester 镜像加载正常
- `grep -rn "register_service\|call_service\|transfer_service\|return_result\|_icc\|icc_id\|service_desc\|_services" avm/ tests/ images/` 无残留（docs/ 留档除外）

## 影响

- **行为变化**：服务三工具消失（设备化前服务能力不可用）；return_result 消失（返回 = 普通投递）；send_instruction 变简单投递（无等待、无返回合并、无 ICC 记录）；$MEM.services、$MEM.icc 设备消失；多播返回合并消失。
- **保留**：内存四指令、create_cmd / create_sub、close_conversation（内核关闭，去掉 ICC 清理）、指令预算、排队调度、检查点、监测器、PLM 执行器。
- **后续**：设备化方案（post/response）恢复服务能力；事件机制（docs/record/9.md）提供"预期行为监听"，是"对话自决休眠 + 对方完成唤醒"的载体。均不在本计划。

## 实施记录（2026-08-13）

- 用户确认 ICC 代替方案：**简单投递，无 return_result，无 ICC 机制**。计划据此扩展删除范围：ICC 本体（`_icc`/`_icc_groups`/`_drop_icc`/`IccDevice`/`$MEM.icc`/`$MEM.services`）、service_desc、多播返回合并、PLM 的 return_result 构造与信封解析全部删除。
- 分支 `simpliy_for_refactor` 实施。
- avm/core.py：删 4 个工具类（Register/Call/Transfer/ReturnResult）；指令集 12 → 8；SendInstruction 简化为简单投递（cid/content，原样投递 + 唤醒 + "已投递"确认，不建 ICC、不等待、调用者不休眠）；删 `_services`/`_icc`/`_icc_groups`/`_drop_icc`/`_ServicesDevice`/mount(services/icc)；`_close_conversation` 删服务注销与 ICC 清理；`_on_conversation_finished` 删 returned 分支（亚对话完成总是自动写回唤醒父）；import 清理 IccDevice。
- avm/types.py：删 service_desc 字段与检查点序列化；UserMessageBatch 删结构化合并段（`_merged`/from/cid/icc_id 参数）。
- avm/python_server.py：SimplePLM/PythonPLM 不解析信封，消息内容直接作为表达式/代码执行；删 return_result 构造、parse_message、call_id_for；清理 hashlib import。
- avm/info_devices.py：删 IccDevice。
- avm/__init__.py：移除 ReturnResultInstruction 导出。
- 测试：220 个全部通过（原 231 - 删除的服务/ICC/多播测试 + 新语义测试）。test_avm.py 重写 TestSendInstruction（4 个新语义测试）、删 TestIccRouting / TestServiceScheduling / 4 个 close 服务测试 / 多播合并测试；test_plm.py 信封测试改为"内容直接执行不解析信封"；test_python_executor.py 计算器流程改为简单投递语义；test_checkpoint.py 删 service_desc 断言。
- 镜像：tester.json 手册全部更新（指令集 8 个、observe 删 services/icc、instruction_ref 删服务/返回条目、scheduling 改简单投递语义、helper 提示词改直接执行）；demo.json / calc_driver.json 工具列表更新（calc_driver 去 wait=true，结果回传依赖未来设备化/事件机制）。三个镜像加载正常。
- 文档：README.md / README_EN.md / CONTEXT.md / docs/system-image.md 同步更新（指令表、对话间通信节改"简单投递"、决策动机、当前状态、术语表删服务与 ICC id）。
- 验证：`pytest tests/` 220 通过；三个镜像加载 OK；avm/ 与 tests/ 与 images/（源文件）grep 无残留（images/out/ 为运行产物，含旧版快照，随下次运行重新生成）。
- 遗留：calc_driver 示例的"结果回传"依赖后续设备化（post/response）或事件机制；事件机制（docs/record/9.md）是"对话自决休眠 + 预期行为监听"的载体，尚未实现。

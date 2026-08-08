# 计划：思维链内容（reasoning_content）回传支持

**时间**: 2026-08-08
**状态**: 已实施（2026-08-08，用户确认后实施，181 个测试全部通过）
**标注**: 本计划涉及的改动全部属于"按用户计划改动"（用户明确要求支持思维链内容回传）。
**背景**: DeepSeek 思考模式（thinking mode）官方文档（api-docs.deepseek.com/zh-cn/guides/thinking_mode/）规定：
- thinking 模式开关走 `extra_body={"thinking": {"type": "enabled"/"disabled"}}`，思考强度可用顶层参数 `reasoning_effort`（如 `"high"`）控制。
- 思考模式下，响应在最终回答（`content`）之前输出思维链内容 `reasoning_content`，与 `content` 同级返回。
- **硬性要求**：对携带 `tools` 参数的请求，发生过工具调用的轮次的 `reasoning_content` 必须在后续所有请求中完整回传；否则 API 返回 400（错误文本形如 "The reasoning_content in the thinking mode must be passed back to the API"）。
- 社区观测到的附加现象（DeepSeek V4）：一旦历史中任意 assistant 消息带 `reasoning_content`，后续所有 assistant 消息也必须带该字段（没有就是空字符串也要原样带），否则部分场景同样报 400。官方文档只强制"工具调用轮次必须回传"，未强制"所有轮次都带"；这一差异作为待定夺项④。
**起因**: 用户要求支持思维链内容回传，为复杂模型行为铺路。当前 AVM 的 LMU 不保留、不回传该字段：`AssistantMessage` 没有 `reasoning_content` 字段，`message_to_api_dict` 不输出它，`LMU.exec` 只存 content 和 tool_calls。结果：使用 thinking 模型 + 工具调用场景时，第二轮请求缺少 `reasoning_content`，API 返回 400。
**行动**（提案，全部为"按用户计划改动"）:
1. **类型层**（avm/types.py）：`AssistantMessage` 增加 `reasoning_content: Optional[str] = None`；`message_to_api_dict` 对 assistant 消息**总是**输出 `reasoning_content` 字段（有值带值，无值带空字符串）——这是"回传所有推理内容"的落点；`Conversation.append_assistant_message` 增加可选参数 `reasoning_content`。
2. **反向解析**（avm/python_server.py）：`message_list_from_api_dict` 解析 assistant 消息时读取并保留 `reasoning_content`，使它与 `message_to_api_dict` 保持互逆（往返不丢字段）。
3. **LMU.exec**（avm/core.py）：用 `getattr(message, "reasoning_content", None)` 防御性读取（openai SDK 对非标准响应字段的支持依赖 SDK 版本），与 content/tool_calls 一起存入 assistant 消息；`last_call` 增加 `reasoning` 字段（供监测器使用，见待定项③）。
4. **para 支持 thinking**（avm/core.py）：`_API_PARAM_KEYS` 增加 `reasoning_effort`（顶层参数，与 temperature 等同级）；`extra_body` 已在白名单，para 里直接写 `extra_body={"thinking": {"type": "enabled"}}` 即可开启 thinking，不需要新增字段（见待定项②）。
5. **monitor**（avm/monitor.py）：`_build_frame` 的 lmu 块增加 `reasoning` 项，用与 `result` 相同的 `_truncate` 截断策略记录 `last_call["reasoning"]`。**不**把 reasoning 暴露给 `$MEM` 信息设备（conversations 详情等）——外泄控制的落点是不让内部访问，而不是不记录；monitor 帧和全文文件只面向运行者（人），不算泄露给其他对话。
6. **测试**：现有 MockLMU 预设响应扩展为可携带 `reasoning_content` → 验证：a) 历史中 assistant 消息保留该字段；b) 下一次 `exec` 的 `messages` 带回该字段（有值带值）；c) `message_to_api_dict` / `message_list_from_api_dict` 往返一致；d) 无 reasoning 的 assistant 消息在 `message_to_api_dict` 输出 `reasoning_content: ""`（行为变化点是"总是带字段"，相关旧断言同步更新）。
**影响**:
- 兼容 DeepSeek thinking 模型：工具调用场景不再 400，AVM 的对话可使用思考模型执行复杂任务。
- 对话历史体积增大：`reasoning_content` 必须随历史回传，计入上下文 token（这是厂商协议的强制成本，不是 AVM 可选优化）。
- `message_to_api_dict` 对 assistant 消息总是输出 `reasoning_content`（无则空字符串）：对不认识该字段的厂商（如默认 gpt-4）会多带一个空字段。OpenAI 兼容端点通常忽略未知字段，但这是记录在案的风险；若未来遇到严格厂商报错，再收紧为"仅 thinking 会话带字段"，不预设实现。
- Python 对话不受影响（Python 程序无 thinking）；但 `message_list_from_api_dict` 需能往返该字段，避免 PLM 收到/返回时丢字段。
- monitor 帧的 lmu 块记录 reasoning，思维链可能很长，与 result 同用 `_max_item_len` 截断。
- 与 PLM、事件总线、错误处理等挂起项无交叉。
**待用户定夺**（2026-08-08 已答复，记录如下）:
① reasoning 可见性：**monitor 记录（与普通内容相同策略），不暴露给 AVM 内部（$MEM 信息设备）**。用户原话："reasoning_content是要被monitor记录的，和普通内容一样——思维链基本上和传统代码的调试过程一样重要。如果想不外泄，就不让内部访问即可。"——即外泄控制靠"禁止内部访问"，不靠"不记录"。
② para：`_API_PARAM_KEYS` 增加 `reasoning_effort`，thinking 开关复用 `extra_body`，不新增便捷字段（推荐方案，用户未反对）。
③ monitor：记录 reasoning，与 result 同截断策略。
④ 回传策略：**总是回传所有推理内容**（assistant 消息带 `reasoning_content`，无则空字符串）。用户原话："我倾向于回传所有推理内容，因为越来越多的模型正在回传所有思维链"。

**实施记录**（2026-08-08，全部为"按用户计划改动"）:
- avm/types.py：`AssistantMessage` 增加 `reasoning_content: Optional[str] = None`；`message_to_api_dict` 对 assistant 消息总是输出 `reasoning_content`（有值带值，无值空字符串）；`append_assistant_message` 增加可选参数 `reasoning_content`。
- avm/python_server.py：`message_list_from_api_dict` 解析 assistant 消息时保留 `reasoning_content`，与 `message_to_api_dict` 互逆。
- avm/core.py：`LMU.exec` 用 `getattr(message, "reasoning_content", None)` 读取并存入 assistant 消息；`last_call` 增加 `reasoning` 字段；`_API_PARAM_KEYS` 增加 `reasoning_effort`。
- avm/monitor.py：帧 lmu 块增加 `reasoning` 项，与 `result` 同 `_max_item_len` 截断；未暴露给任何 `$MEM` 信息设备（外泄控制 = 禁止内部访问）。
- 测试：test_messages.py 更新旧断言 + 新增往返/总是带字段断言；test_avm.py 新增 LMU.exec 历史保留与下一轮带回测试；test_monitor.py 新增 reasoning 进帧与截断测试。181 通过（原 175 + 6）。
- 记录在案的风险：`message_to_api_dict` 对不认识 `reasoning_content` 的厂商（如默认 gpt-4）会多带空字段；OpenAI 兼容端点通常忽略未知字段，若遇严格厂商报错再收紧为"仅 thinking 会话带字段"。

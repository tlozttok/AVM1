# 计划（待定夺）：单对话检查点（conversation checkpoint）

**时间**: 2026-08-10
**状态**: 已实施（用户确认方案后完成，223 个测试全部通过）
**标注**: 按用户计划改动（用户提出：读取手册消耗大量 token，希望复用已读取手册的对话，至少需要一种单对话检查点方法；可以不考虑在 AVM 内调用）。
**背景**:
- tester 镜像中 init 对话按提示词读取 $MEM.tester_manual（5 节参考手册），内容进入对话历史（memory_read 的工具响应）。
- 目前 Conversation 没有任何序列化机制：只有 `from_any_list`（构建用，且不保留 reasoning_content）；内存持久化只保存 mem 树，不保存对话。
- CONTEXT.md 预留了 "Conversation 文件类型：type = \"conversation\"，存储持久化的对话记录"，尚未实现。
- Python 对话的 PLM 实例持有不可 JSON 序列化的命名空间（PythonPLM.namespace），SimplePLM 无状态。
**起因**: 每次新建对话都要重读手册。**用户澄清核心目标：LLM API 提供前缀缓存，缓存命中的 token 价格约等于零——所以重点不是减少历史体积，而是保证发送给 API 的数据完全一致，让重复运行命中缓存。** 检查点因此必须是"逐字节可复现的对话快照"，而不是"可编辑的摘要"。
**关键事实与边界**:
1. **缓存命中的充分必要条件是发送给 API 的 messages 逐字节一致**（同一模型、同一工具集下）。因此检查点的设计要求是：保存无损、序列化确定性、恢复后 `to_api_messages()` 的输出与保存时逐字节相同。
2. **裁剪/摘要会破坏前缀缓存**（改动任何历史字节，前缀即不同，缓存失效）——上一版方案中"恢复前人工裁剪换摘要"的建议与缓存目标矛盾，已撤回。
3. 破坏"完全一致"的风险点：消息字段丢失（reasoning_content、tool_calls 的 id/arguments 原样、空字符串、tool_call_id）、JSON 键序不稳定、在 user_batch 非空时保存/恢复（请求会多出未消费消息）、模型（para/model）或工具集变更。工具集由代码固定；模型由 para_ref 指向的节点内容决定，恢复时需一致。
4. 对话自身状态 = messages（role/content/tool_calls/tool_call_id/reasoning_content，全部无损）+ name + para_ref + is_python + metadata + service_desc + is_sub/is_root。不随检查点走的：cid（恢复时重新分配）、parent 引用（不序列化父链）、Core 队列/ICC/服务表（恢复方决定是否注册）。保存点必须选在干净边界（user_batch 已清空，即交互结束/休眠时）。
5. Python 对话命名空间不可 JSON 序列化；PythonPLM 的 setup（system 消息）是确定性 exec，恢复时可重放 system 消息重建命名空间；运行时累积的用户变量无法恢复（记录为限制）。
**方案选项**:
- **A. 独立 JSON 检查点文件（推荐 v1）**：`Conversation.to_checkpoint()` / `from_checkpoint()`（消息逐条无损序列化，保留 tool_calls 原样、reasoning_content、空字符串、tool_call_id；键序固定）；`Core.save_conversation(cid, path)` / `Core.restore_conversation(path)`（恢复注册新 cid，休眠或就绪参数化）。host 侧 API 调用，不需要 AVM 内指令。**验收标准：save→restore 后 `to_api_messages()` 与保存时逐字节相同（测试断言字节一致）。**
- **B. CLI / debug.json 接入**：`python main.py image.json --checkpoint-out path`（运行结束/交互结束时写快照）、`--checkpoint-in path`（启动时用快照恢复的对话替代/补充 init）；debug.json 加 `checkpoint_in` / `checkpoint_out`。
- **C. 自动检查点**：镜像 meta 或独立段声明 `"checkpoint": {"path": ..., "on_finish": true}`，每轮交互结束自动覆盖写快照；重启用 `--checkpoint-in` 恢复。
- **D. 内存树节点（type="conversation"）**：把对话序列化写进 mem 树（落地 CONTEXT.md 预留类型），host 侧与未来 AVM 内读取都可用；作为 A 的存储后端选项而非替代。
- **Python 对话策略**：v1 只支持 LLM 对话检查点（Python 对话报错），或支持 Python 对话但恢复时重放 system 消息重建命名空间；给 PLMServer 加可选 save/restore 钩子留作扩展。
**影响**: tester 场景可读一次手册、存快照、反复恢复复用；同一快照恢复出的多个对话在相同输入下发送相同前缀，命中厂商前缀缓存（缓存命中价格约等于零）；恢复的对话若曾注册服务，服务表不随检查点恢复，需恢复方重新注册。**注意：代码演进（新增指令改变工具集）或更换模型会破坏缓存，这是接受的事实，不是检查点能解决的。**
**待定夺**:
**定夺结果（2026-08-10）**:
① 独立 JSON 文件（A）。
② 作为调试功能少暴露：不做 CLI/debug.json 参数；恢复走 host 侧 API。
③ 手动：不做自动检查点；手动触发方式为输入设备命令 `/save`（用户提议"在 input 前截取输入，如果是 /save 就保存"，已确认）。
④ Python 对话 v1 拒绝（用户："先拒绝 python"）。
⑤ 干净边界。实现方式：保存时若历史末尾是未配对的 assistant 工具调用（等待输入响应时正是如此），截掉该条，检查点 = 截至最后一个完整交换的干净前缀——与缓存目标一致（截完恰好等于上一次发出的请求，逐字节命中前缀缓存）。

**实施记录（2026-08-10）**:
- avm/types.py：`CHECKPOINT_VERSION`、`_tool_call_to_dict`（兼容 ToolCall 对象与 dict）、`Conversation.to_checkpoint()` / `from_checkpoint()`（无损、确定性，保留 tool_calls 原样、reasoning_content、tool_call_id）。
- avm/core.py：`Core.image_dir/image_name`（镜像加载时设置）、`checkpoint_path(cid)`（默认 `out/<镜像名>.conv.json`）、`save_conversation(cid, path)`（含末尾未配对工具调用截断；拒绝 Python 对话）、`restore_conversation(path, schedule)`（注册新 cid，dormant/ready）。
- avm/memory_device.py：`MemoryDevice.attach_core(core)`（调试用可选钩子）；`InputsListDevice` 覆写 attach_core（调试用）并在 `__getitem__(-1)` 拦截行首 `/save`（循环保存后继续等待真实输入，对话感知不到；`/save <path>` 可指定路径，缺省走 checkpoint_path）。
- avm/image.py：`_mount_devices` 挂载后调用 `device.attach_core(core)`（注释"调试用"）；`load_image` 设置 image_dir/image_name。
- tests/test_checkpoint.py：新增 11 个测试（无损往返字节一致、reasoning_content None/空串区分、版本拒绝、save/restore 新 cid 与调度、截断、Python 拒绝、/save 默认路径/自定义路径/无 Core）。全量 223 通过（原 212 + 11）。
- README.md / README_EN.md：调试与观测节新增"单对话检查点（调试用）"；当前状态补充检查点与测试数。

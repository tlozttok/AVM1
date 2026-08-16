# 计划：事件模型 v1 —— 工具即事件、waterfall Python 处理器、owner 直录

**时间**: 2026-08-15
**状态**: 已实施（含 P0/P1/P2 收尾；266 个测试通过，见 docs/Agents/plan-event-v1-cleanup-2026-08-15.md）
**标注**: 决策过程见 docs/Agents/plan-event-model-2026-08-15.md；设计讨论见 docs/record/12.md。实施者必须按本计划执行；遇到计划未覆盖的冲突或必要设计缺失，立即停止并回报 issue，不自行扩展。
**背景**: 事件机制是调度状态机的缺失半边（record/9.md）。当前 Dormant → Active 只有 send_instruction 唤醒一条路径；对话无法监听内存访问、消息投递或生命周期事实。用户确定模型：每个工具调用对应一个事件，设备事件是设备对自身的更改，监听器用普通 Python 代码以 waterfall 管道处理事件，先注册者优先；内存创建时由 Core 直接记录所有者。
**起因**: 用户要求形成详细计划，交给另一个成本较低的模型实施。

---

## 一、目标与范围（v1）

1. 新增第 9 条指令 `listen_event`（注册/修改/取消监听）。
2. 为全部 9 种工具调用建立事件：5 种可拦截（before + waterfall），4 种只产生 after 通知。
3. 处理器 = 普通 Python 代码（`type="python"` 节点中的 `content`），每次事件在全新沙箱命名空间执行，不创建对话、不保留处理器实例状态。
4. 监听器权限：内存类事件按 `ctrl.owner` 校验；`send_instruction` 事件只允许目标对话监听发往自己的消息；after 通知类事件允许通配监听。
5. 先注册者优先：管道按注册顺序执行，首个 `replace` 短路，后续监听器不再执行。
6. 处理器通过返回 `notify` 字段向对话发消息；Core 用内部投递（不经事件管道）唤醒目标，不重新引入 ICC/return_result。
7. `close_conversation` 自动撤销被关闭对话注册的全部监听。
8. 设备自改事件：新增 `Core.emit_device_event` 受约束入口；source 必须是设备自身路径，event_type 仅 `memory_write`；本计划不实现具体定时器设备。

**不在本计划内**：内核态/用户态、事件条件（ECA 的 C）、事件持久化/重放、具体定时器设备、`memory_make` 直接附加 ctrl 参数（另立 issue）、服务设备化、链接设备。

---

## 二、事件模型规格

### 2.1 事件对象

事件是瞬时 dict，不落盘；由 Core 单调查递增生成 `event_id`。

```python
{
  "event_id": "evt-3",
  "event_type": "memory_read",        # 9 种之一，恒等于工具名
  "phase": "before",                  # before 或 after
  "source": {"kind": "conversation", "cid": 0},   # 或 {"kind": "device", "path": "$MEM.timer"}
  "target": "$MEM.rooms.secret",      # 见 2.2
  "payload": {"ref": "$MEM.rooms.secret"},        # 各事件类型不同，见 2.3
  "action": "execute",                # execute / replace
  "response": "",                     # action=replace 时返回给调用者的文本
  "notify": []                        # 处理器累积的投递请求
}
```

- 管道只允许处理器修改：`action`、`response`、`notify`、以及 2.3 表列出的 payload 字段。其余字段不可变；处理器返回了也忽略。
- `notify` 元素格式：`{"cid": int, "content": str}`。

### 2.2 事件类型、phase 与 target

| 工具 / event_type | phase | 是否可 replace | target 格式 |
|---|---|---|---|
| memory_read | before | 是 | 原 `ref`（`$MEM...`） |
| memory_write | before | 是 | 原 `ref` |
| memory_make | before | 是 | 新建完整路径 `ref.key` |
| edit_metadata | before | 是 | 原 `ref` |
| send_instruction | before（每个目标一条） | 是 | `"cid:<cid>"` |
| create_cmd | after | 否 | `"cid:<新 cid>"` |
| create_sub | after | 否 | `"cid:<新 cid>"` |
| close_conversation | after | 否 | `"cid:<被关闭 cid>"` |
| listen_event | after | 否 | `"listener:<listener_id>"` |

规则：
- before 事件在指令执行前分发；处理器可把 action 改为 `replace`。`replace` 后原指令不执行，调用者收到 `response` 作为工具响应；首个 `replace` 短路。
- after 事件在指令成功后分发；处理器只能贡献 `notify`，不能 replace。after 处理器异常不影响已完成的指令，记录 stderr 并跳过该处理器的 notify。
- 9 种事件严格一一对应工具；不归并 memory_make / edit_metadata。

### 2.3 payload 与可变字段

| event_type | payload 内容 | 处理器可改的 payload 字段 |
|---|---|---|
| memory_read | `{"ref": ref}` | 无（只可 replace/notify） |
| memory_write | `{"ref": ref, "content": content}` | `content`（必须为 str） |
| memory_make | `{"ref": ref, "key": key, "mem_type": mem_type}` | 无 |
| edit_metadata | `{"ref": ref, "type": op, "key": key, "value": value}` | `value`（仅当原 type=set 时） |
| send_instruction | `{"cid": 该目标 cid, "content": content}` | `content`（必须为 str） |
| create_cmd / create_sub | `{"cid": 新 cid}` | 无 |
| close_conversation | `{"cid": 被关闭 cid}` | 无 |
| listen_event | `{"listener_id": id, "action": 本次动作}` | 无 |

身份字段（ref、key、cid、event_type、source、target）不可被处理器改变，防止守卫改道绕过另一路径的监听。

### 2.4 处理器契约（普通 Python 代码，无状态）

- `handler_ref` 必须指向 `ctrl.type="python"` 的 MetaDict 节点，其 `content` 为 str。
- 注册/修改时读取并缓存 `content`；运行期对节点的修改不自动生效，需 `modify` 重新指定。
- 每次事件执行：
  1. `namespace = safe_globals({"json": json})`（来自 `avm.sandbox`；全新命名空间，不跨事件保留）；
  2. `exec_safe(code, namespace)`；
  3. 取 `namespace["handle"]`，必须是可调用对象，否则视为处理器错误；
  4. `out = handle(dict(event))`；`out` 必须是 dict，否则视为处理器错误。
- 返回值处理：
  - `action`：before 阶段允许 `execute` / `replace`；after 阶段忽略。
  - `response`：仅 action=replace 时使用；缺失则用默认 `"事件被监听器拦截"`。
  - `payload`：只合并 2.3 表中允许的字段，其他字段忽略。
  - `notify`：必须是 list；每项为 `{"cid": int, "content": str}`，校验失败视为处理器错误。
  - 未知键忽略。
- before 阶段处理器错误 = **fail closed**：管道立即停止，原指令不执行，调用者工具响应为 `Error: 事件监听器 <listener_id> 处理失败: <类型>: <信息>`，并向 stderr 报告。
- after 阶段处理器错误 = fail open：stderr 报告，跳过该处理器的 notify，继续后续监听器。

### 2.5 监听注册与权限

`listen_event` 工具参数：

| 参数 | listen | modify | cancel |
|---|---|---|---|
| `action` | 必填 | 必填 | 必填 |
| `event_type` | 必填（9 种之一） | 不填 | 不填 |
| `target` | 必填 | 不填 | 不填 |
| `handler_ref` | 必填 | 必填 | 不填 |
| `listener_id` | 不填 | 必填 | 必填 |

- 工具定义 schema 以 `action` 为必填，其余按上表在 execute 中校验（与 edit_metadata 相同的「非法参数给 Error 工具响应」模式）。
- `listen` 成功返回 `listener_id=<id> 已注册`；`modify` 返回 `listener_id=<id> 已更新`；`cancel` 返回 `listener_id=<id> 已取消`。
- `modify` 只允许改 `handler_ref`；`event_type` / `target` 不可变，要改就 cancel 后重新 listen。
- `modify` / `cancel` 只允许注册该监听器的对话（owner_cid）执行。

权限表：

| event_type | 注册条件 |
|---|---|
| memory_read / memory_write / edit_metadata | target 路径存在；`_owner_of(target)` 必须等于注册者 cid |
| memory_make | target 的父路径存在；`_owner_of(父路径)` 必须等于注册者 cid |
| send_instruction | target 必须是 `"cid:<cid>"`，且该 cid == 注册者 cid（对话只能拦截发往自己的消息） |
| create_cmd / create_sub / close_conversation / listen_event | target 为 `"*"` 或精确 target；任何对话可注册（这些事实已通过 `$MEM.conversations` 可见） |

- after 事件匹配：`"*"` 匹配所有 target；精确 target 匹配相等。
- before 事件匹配：只支持精确 target；注册时拒绝 `"*"`。

### 2.6 所有者（owner）

- `ctrl` 是 dict[str,str]；owner 值 = `str(cid)`。
- `memory_make` 创建 dict/list 节点时写 `ctrl.owner`；创建 str 节点时 str 无 ctrl，不写（见限制）。
- `memory_write` 经 `set_by_path` 隐式创建的中间 MetaDict、或新建的最终 MetaDict/MetaList 节点，写 `ctrl.owner`；覆盖已有节点不改变 owner；新建 str 叶子无 ctrl，不写。
- `_owner_of(path)`：
  1. 解析 `$MEM...` 到节点；节点是 MetaDict/MetaList 且有 `ctrl.owner` → 返回；
  2. 节点是 str 或无 owner → 从最深祖先向上找第一个带 `ctrl.owner` 的 MetaDict/MetaList；
  3. 都没有 → 无主（None），不允许注册 before 监听。
- `edit_metadata` 的 `type=set` 或 `del` 若 key 为 `owner`，一律返回 Error `owner 由 Core 维护，不允许通过 edit_metadata 修改`；`get` 允许。
- 镜像预置节点没有 owner 即为无主；v1 无内核态，无主路径不可注册 before 监听。镜像作者可显式在 JSON 的 ctrl 里写 `"owner": "0"` 预置所有权（文档说明）。
- 已知限制（写进文档与代码注释）：str 叶子本身无 ctrl；要保护一个字符串路径，需先由该对话创建带 owner 的祖先容器，再把字符串放进容器。直接在无主 `$MEM` 根下创建的 str 在 v1 不可被监听。
- owner 进入 `Memory.save` 的 ctrl，自动随内存持久化；`load_image` 无需改动。

### 2.7 投递 notify

- 处理器返回的 notify 由 Core 在管道结束后投递；不经事件管道、不产生 send_instruction 事件、不递归。
- 投递语义与 send_instruction 相同：内容原样进入目标 `user_batch`，目标从 dormant/ready 移到 ready 队尾；目标不存在或 finished 时跳过并向 stderr 报告。
- 管道为 before 事件时，notify 在「执行原操作或返回 replace 响应」之前投递；after 事件在事件分发结束后投递。

### 2.8 设备自改事件入口

- `Core.emit_device_event(device_path: str, target: str, payload: dict) -> dict`：
  - `device_path` 必须已挂载在 `mem._devices`；
  - 只支持 event_type=`memory_write`、phase=before；
  - `target` 必须满足 `target.startswith("$MEM." + device_path)`；
  - `source = {"kind": "device", "path": "$MEM." + device_path}`；
  - 走正常 before 管道；返回处理后的 event（调用设备据此决定是否执行自改：action=replace 时应跳过自改，action=execute 时按 payload.content 执行）。
- v1 只提供入口与单元测试；不实现具体设备，不替设备代码强制隔离其宿主行为（设备代码信任边界另立 issue）。

### 2.9 可见性与监测

- 新增只读信息设备 `$MEM.listeners`：`to_llm_string` 逐行列出 `listener_id / owner_cid / event_type / target / handler_ref`；子路径 `<listener_id>` 返回单条详情；不暴露 handler 代码全文。
- 事件本身瞬时：不提供 `$MEM.events` 历史；不写盘。
- Monitor：Core 在每个 `advance_conversation` 周期收集 `events` 摘要（每条：event_id、event_type、phase、target、source、最终 action）；`Frame.lmu` 增加 `events` 字段，沿用现有截断策略。`$MEM.monitor` 最近帧摘要可增加最近一帧事件条数。

---

## 三、行动（按文件实施）

### 3.1 新增 `avm/events.py`

职责：事件构造、监听器注册表、权限校验、处理器执行、管道分发。建议类结构：

- `EVENT_TYPES`、`INTERCEPTABLE_TYPES`、`AFTER_ONLY_TYPES`、`MUTABLE_PAYLOAD_KEYS` 常量。
- `build_event(core, event_type, phase, conv, target, payload)`：生成事件 dict 并分配 event_id。
- `class EventSystem`：
  - `__init__(core)`：`self.listeners = []`（注册顺序即优先级），`self._next_listener_id = 1`，`self._next_event_id = 1`；
  - `register(owner_cid, event_type, target, handler_ref) -> int`：按 2.5 权限表校验；读取并缓存 handler code；追加记录 `{listener_id, owner_cid, event_type, target, handler_ref, handler_code}`；
  - `modify(listener_id, owner_cid, handler_ref)` / `cancel(listener_id, owner_cid)`：校验归属；
  - `cancel_all_for_owner(owner_cid)`：供 close_conversation 使用；
  - `_match(event) -> list`：先 event_type 相等，再按 2.5 匹配 target；
  - `dispatch(event) -> dict`：按注册顺序执行匹配监听器；before 的 replace 短路；after 只收 notify；汇总最终 event 与 notify；处理器异常按 2.4 fail closed/fail open；向 `core._turn_events` 追加首尾摘要；
  - `deliver_notifies(event)`：遍历 event["notify"]，按 2.7 内部投递；
  - `resolve_handler(core, handler_ref) -> str`：复用 Core.unwrap(for_llm=False)，校验 `MetaDict`、`ctrl.type == "python"`、`content` 是 str；
  - `_owner_of(core, path) -> Optional[int]`：按 2.6 实现。
- 沙箱调用：`from .sandbox import safe_globals, exec_safe`；提供 `json` 给处理器。处理器无法访问 Core / Memory / 任何工具，递归监听在 v1 结构上不可能发生。
- 事件 dict 用普通 dict；返回给处理器的 `dict(event)` 为副本，避免处理器改动原对象。

### 3.2 `avm/memory.py`

- `make(ref, key, mem_type, owner_cid: Optional[int] = None)`：创建 dict/list 节点后，若 `owner_cid is not None` 且节点有 `get_ctrl`/`set_ctrl`，把现有 ctrl 复制一份并写 `ctrl["owner"] = str(owner_cid)`；str 节点跳过。
- `set_by_path(path, value, owner_cid: Optional[int] = None)`：
  - 在普通路径处理分支中，跟踪新创建的中间 MetaDict/MetaList 节点并写 owner；
  - 最终新建的 MetaDict/MetaList 写 owner；最终新建 str 或覆盖已有节点不写/不改 owner；
  - 现有「终点是 MetaDict/MetaList 时写入 = 改 metadata」语义保持不变，且不覆盖 owner。
- `set(ref, value, owner_cid: Optional[int] = None)` 透传给 `set_by_path`。
- 保留原签名兼容旧调用（owner_cid 可选）。
- 注意 `_wrap_value` 返回的新节点没有 ctrl；owner 必须在 `_wrap_value` 之后由 set_by_path 写，并注意列表元素/字典子节点的递归处理范围（本次只处理路径上新建的容器节点，不递归给子节点写 owner）。

### 3.3 `avm/core.py`

- `Core.__init__`：新增 `self.events = EventSystem(self)`、`self._turn_events = []`；挂载 `$MEM.listeners`（`ListenersDevice(self)`）；`close_conversation` 旧描述同步改为「关闭后不可再调度、不可再被调用；其注册的事件监听自动撤销、PLM 状态释放」。
- 新增 `ListenEventInstruction`（tool_name=`listen_event`），按 2.5 实现；注册/修改/取消成功后分发 after 事件（event_type=listen_event，target=`"listener:<id>"`，payload 含 listener_id 与 action）。
- 修改 5 条 before 指令：
  - `MemoryReadInstruction.execute`：先构造并 dispatch before 事件；若最终 action=replace，投递 notify 并 `add_tool_response(response, call_id)` 返回；否则执行原读取。事件 target=ref，payload 见 2.3。
  - `MemoryWriteInstruction.execute`：先 dispatch；replace 同上；否则用最终 `payload["content"]` 调 `core.mem.set(ref, content, owner_cid=conv.cid)`，成功/失败路径与现在一致。notify 在管道后立即投递（无论原操作随后成败）。
  - `MemoryMakeInstruction.execute`：先 dispatch（target=`f"{self.ref}.{self.key}"`）；replace 同上；否则 `core.mem.make(ref, key, mem_type, owner_cid=conv.cid)`。
  - `EditMetadataInstruction.execute`：先 dispatch（payload 含解析出的 op/key/value；在参数合法性校验之前）；replace 同上；否则执行原逻辑，但 set/del 前加 owner 键保护（2.6）。若处理器改了 `payload.value`，set 分支使用改后的 value。
  - `SendInstruction.execute`：
    1. 保留现有「先解析全部目标，任一不存在/已关闭即整体失败」的检查；
    2. 对每个 target 构造 before 事件（target=`"cid:<cid>"`，payload 含该 cid 与最终 content，content 初始为 self.content）；
    3. 事件 replace → 该目标跳过投递，记录该 response 与 notify；action=execute → 按最终 content 投递该目标（复用现有唤醒/入队逻辑）；
    4. 全部目标处理完后：全部跳过时，工具响应 = 第一个 replace 的 response；部分跳过时 = `已投递到 <n>，<k> 个目标被拦截`；全部投递时保持现有 `已投递到 <身份>/<数量>` 文案；
    5. 每条事件的 notify 在管道后投递。
- 修改 4 条 after 指令：
  - `CreateInstruction.execute`：创建成功后 dispatch after 事件（target 新 cid），投递 notify；现有工具响应不变。
  - `CreateSubInstruction.execute`：同上。
  - `CloseInstruction.execute`：`core._close_conversation` 成功后 dispatch after 事件，投递 notify；现有工具响应不变。
  - `ListenEventInstruction.execute`：成功后 dispatch 自己的 after 事件。
- `_close_conversation`：调用 `self.events.cancel_all_for_owner(cid)`；保持 finished/队列/executor 清理。
- `advance_conversation`：每轮开头 `self._turn_events.clear()`；`monitor.record` 前若 `last_call` 存在且 `_turn_events` 非空，传 `last_call = {**last_call, "events": list(self._turn_events)}`（不要修改执行器原始 last_call）。
- 新增 `Core.emit_device_event(device_path, target, payload)`：按 2.8 校验与分发，返回最终 event。
- `_instruction_registry`：8 → 9，加入 `ListenEventInstruction`。
- 保持 `_process_return_calls` 的现有错误策略；json_error/unknown_tool 不产生事件。

### 3.4 `avm/info_devices.py`

- 新增 `ListenersDevice(MemoryDevice)`：`to_llm_string()` 与 `resolve_path([listener_id])`；只读；挂载在 Core 中。
- `MonitorDevice.to_llm_string` 可加最近帧事件条数（若 Monitor 帧 lmu 含 events）。

### 3.5 `avm/monitor.py`

- `Monitor._build_frame` 的 `lmu` 字典增加 `"events"` 键：值为 `last_call.get("events") or []` 的截断摘要（每条取 event_id/event_type/phase/target/source/action，target/source 过长截断）。
- 无事件时保持 `[]`，既有测试无需改默认形状。
- `--frames` 打印可在 tools 后追加 `events=<数量>`（可选，不改既有断言语义）。

### 3.6 文档

- `CONTEXT.md`：指令集增加 `listen_event`；「事件」术语从（设计中，未实现）改为已实现 v1，并保留未实现边界（条件、事件持久化等）；`ctrl.owner` 写入元数据二说明。
- `docs/runtime.md`：指令表 9 条；新增「事件模型 v1」节（事件对象、waterfall、处理器契约、权限、owner、`$MEM.listeners`、设备自改入口、与 record/9 的关系）；`$MEM` 信息设备列表加 listeners。
- `docs/system-image.md`：`ctrl.owner` 说明与镜像预置示例；说明无主节点在 v1 不可注册 before 监听。
- `README.md` / `README_EN.md`：当前状态「8 条指令」改 9 条；事件从「设计中」移到已实现，未实现项改为事件条件/总线等剩余部分；测试数按最终结果更新。
- `images/tester.json`：指令集从 8 更新到 9，overview/instruction_ref 增加 `listen_event` 与事件模型摘要；observe 增加 `$MEM.listeners`。运行产物 `images/out/` 不手工修改（下次运行生成）。
- `AGENTS.md`、`avm/__init__.py` 的旧描述不在本计划内；需要时另立文档清理计划。

### 3.7 测试

新增 `tests/test_events.py`，至少覆盖：

1. **工具集**：`_build_tools()` 含 9 个工具名，`listen_event` schema 存在。
2. **owner 记录**：`memory_make` 创建 dict/list 写 `ctrl.owner`；str 不写；`memory_write` 隐式中间容器写 owner、覆盖已有节点不覆盖 owner；`Memory.save` 持久化 owner、`load_image` 读回。
3. **权限**：非 owner 对内存路径 listen 被拒绝；owner 成功；`send_instruction` 只能监听 `cid:<自己>`；modify/cancel 非归属者被拒绝；无主路径 listen 被拒绝。
4. **memory_read waterfall**：owner 注册 handler 返回 replace `"请输入密码"`；调用者工具响应为该文本；后续监听器不执行（首个 replace 短路）；原读取不发生（可用不存在路径 + replace 成功来证明没有抛内存错误）。
5. **memory_write 改写**：handler 改 `payload.content`，内存最终写入改后的值；handler replace 时内存不变。
6. **顺序**：两个监听器都 replace 时，先注册者的 response 生效；先注册者 execute、后注册者 replace 时，后注册者 response 生效。
7. **notify**：handler 返回 notify；目标对话 user_batch 收到内容、进入 ready；不产生 send_instruction 事件。
8. **send_instruction 拦截**：目标对话监听发往自己的消息并 replace；发送方工具响应按 2.2/3.3 约定；目标 user_batch 无该内容；多播部分拦截、全部拦截两种路径。
9. **modify / cancel / close 清理**：modify 换 handler 生效；cancel 后不再触发；close 对话后其监听消失（`$MEM.listeners` 与行为都验证）。
10. **after 事件**：create_cmd / close_conversation / listen_event 成功后，通配监听器收到 notify（用 mock 对话注册后触发并检查 batch/ready）。
11. **处理器错误**：before 抛异常/返回非 dict → fail closed，调用者收到 Error，原操作不执行；after 抛异常 → 指令结果不受影响，stderr 有报告（capsys）。
12. **owner 键保护**：`edit_metadata` set/del `owner` 返回 Error；get owner 成功。
13. **设备入口**：dummy device 调用 `core.emit_device_event`，source/target 校验；跨路径 target 被拒绝；replace 返回 action=replace。
14. **信息设备与 monitor**：`$MEM.listeners` 列表与子路径；monitor 帧 lmu.events 有首尾摘要。
15. **既有回归**：全量 pytest；重点确认 memory_make/write 现有测试对 ctrl.owner 的新增不敏感，必要时只补断言不删旧语义。

### 3.8 验证清单

```bash
python -m pytest tests/ -q          # 全量通过，测试数按最终新增数量更新
python -m pytest tests/test_events.py -q
python - <<'PY'
from avm.image import load_image
for p in ["images/tester.json", "images/demo.json", "images/calc_driver.json"]:
    load_image(p)
print("images ok")
PY
python -m compileall -q avm
grep -n "listen_event" avm/core.py avm/events.py avm/info_devices.py docs/runtime.md CONTEXT.md README.md
grep -n "ctrl.owner\|owner" avm/memory.py avm/events.py docs/system-image.md
```

- 行为基线：未注册任何监听器时，8 条旧指令的输入输出必须与现在完全一致；`images/out` 旧快照不参与断言。

---

## 四、影响

- **指令集**：8 → 9；工具列表变化会影响 LLM 前缀缓存（检查点与 para 不变，工具集变了缓存必然失效，属已接受事实）。
- **调度**：新增 notify → 内部投递 → ready 唤醒路径；事件管道本身在一条指令内同步完成，不新增活跃对话切换。
- **内存**：新建容器节点开始携带 `ctrl.owner`；`memory_read` 的 LLM 表示不显示 ctrl，因此普通读取视野不变；持久化文件多出 owner 字段，旧格式仍可加载。
- **安全**：before 监听以 owner 为准；`edit_metadata` 的 owner 键被保护；处理器在沙箱中无 Core/工具访问；after 通配监听只暴露 `$MEM.conversations` 已可见的生命周期事实。
- **性能**：无监听器时仅增加事件构造与空管道遍历；有 Python 处理器时每条指令增加一次受限 exec。
- **冲突提示**：本计划与「先注册者优先」一致；与已删除的 ICC/return_result 无耦合；不恢复服务工具。
- **废弃标记**：无旧机制被废弃；`close_conversation` 工具描述中的「服务注销、未完成调用清理」旧文案随本次修改删除。
- **遗留 issue（实施中如必要用 TODO 标出，实施后另立计划）**：
  1. `memory_make` 应支持创建时直接附加 ctrl 元信息；
  2. str 叶子无法直接携带 owner，需祖先容器所有权兜底；
  3. 设备代码实际仍可绕过 `emit_device_event` 直接改自身路径，信任边界需要设备规范约束；
  4. 事件没有条件（ECA 的 C）、没有持久化/重放、没有事件总线；后续计划处理。

---

## 实施记录（2026-08-15）

- 由实施模型按本计划完成代码与文档改动（当前为未提交工作区改动）。
- `pytest tests/` 254 个全部通过（P0 修复 +2）；三镜像加载正常；工具集 9 个。
- 主要文件：新增 avm/events.py、tests/test_events.py；修改 avm/core.py、avm/memory.py、avm/info_devices.py、avm/monitor.py、main.py、CONTEXT.md、README.md、README_EN.md、docs/runtime.md、docs/system-image.md、images/tester.json。
- 审计结果：主体按计划完成；P0 已直接修复，P1/P2 待收尾项详见 docs/Agents/audit-event-v1-2026-08-15.md 与 docs/Agents/plan-event-v1-cleanup-2026-08-15.md。收尾前不进入 AVM 软件与 OS 计划。

---

## 收尾实施记录（2026-08-15）

- 按 `docs/Agents/plan-event-v1-cleanup-2026-08-15.md` 完成 P1/P2 收尾。
- P1：禁止自我投递；before 处理器失败改为 `handler_error` 事件结果；保留已收集 notify；多播 `send_instruction` 允许部分投递；monitor 记录 `error` / `handler_errors`；stderr 去重。
- P2：更新 `docs/programming-guide.md`、`AGENTS.md`、`avm/__init__.py`、`.gitignore`；`images/out/` 从跟踪移除；计划文件状态回填。
- `pytest tests/` 266 个全部通过；三镜像加载正常；`compileall` 通过。
- 主要改动文件：`avm/events.py`、`avm/core.py`、`avm/monitor.py`、`tests/test_events.py`、`README.md`、`README_EN.md`、`docs/programming-guide.md`、`AGENTS.md`、`avm/__init__.py`、`.gitignore`、`docs/Agents/*`。

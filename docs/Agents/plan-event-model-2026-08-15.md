# 计划（提案，待定夺）：以工具调用为事件源的事件模型

**时间**: 2026-08-15
**状态**: 决策闭合（第四轮确认；详细实施计划见 plan-event-v1-2026-08-15.md）
**背景**: 事件机制是 README「当前状态」中列出的未实现项，也是 docs/record/9.md 确认的调度状态机缺失的另一半：Dormant → Active 目前只有显式调用（send_instruction 唤醒）一条路径。record/9 已确定两个前提：事件以硬件中断为原型、不做事件溯源；并区分事实事件（emit，通知）与钩子事件（waterfall，可改写/拦截），后者的同步等待语义当时留为开放问题。
**起因**: 用户给出事件模型的新想法：不按传统事件源分类（定时、内存变化、信号），而是**每个工具调用对应一个事件**。本文件记录模型、对照与定夺过程；全部决策点闭合后，再形成可实施的正式计划。

## 用户提出的模型（2026-08-15，第一轮）

1. 每个工具对应一个事件：内存读写、消息发送、对话创建和关闭；此外事件监听本身也是一个工具，因此也有「事件监听事件」。
2. 事件种类少是刻意的：内存改变的事件可被视为内存写事件（事件源为设备自身）；定时器也可被视为「定时器设备启动」的事件。
3. 监听机制尚未确定；期望监听具备**可组合、可改变、可取消**的性质。用户给出目标形态：一个对话监听别人访问某个内存，把返回值改变为「请输入密码」。

## 用户补充（2026-08-15，第二轮）

4. **设备产生的事件就是设备对自己的更改**：源是 `self`；一个路径的设备不能更改其他路径的内部。
5. **waterfall 管道**：监听器使用一个程序处理事件；事件按顺序经过监听器被改变；全部处理完后按最终事件执行。事件可被改为「立刻返回拒绝访问」，原操作不执行；处理结果还可以向监听者发送指定消息，使其进入就绪状态。
6. **优先级 = 注册顺序**：先监听的对话有更高的权限；要求对话对自己创建的东西负责。
7. **未决**：处理器程序如何向对话返回消息。

## 第三轮定夺（2026-08-15）

8. **向对话返回消息：候选一（notify 字段）**。处理器不在管道中调用工具；处理器输出事件时携带 `notify: [{cid, content}]`，Core 用现有 send_instruction 的投递 + 唤醒机制交给目标对话。用户指出两个候选的差异是「尾递归 / 中途递归」，选择尾递归形态（候选一）。
9. **所有者：Core 直接记录，暂不写系统级对话 OS**。创建内存路径时把创建者 cid 写入节点 `ctrl.owner`；监听注册时由 Core 校验所有者。用户做过软件化思想实验：系统级对话监听 memory_make 保证创建者最先知道；系统对话在创建事件时给 payload 附加 ctrl 元信息；再拦截 listen_event 注册防越权。该路径说明架构灵活性，但用户尚未写好这样的 OS，当前直接记录所有者。
10. **连带发现（另立 issue，不在本计划内实施）**：`memory_make` 创建时应当能直接附加 ctrl 元信息。当前 `memory_make` 只创建空 str/dict/list，ctrl 需之后用 edit_metadata 补；本次只按第九条由 Core 写 owner，不扩展 memory_make 参数。

## 现状对照

- 现有 8 条指令均有确定的执行点：`memory_read` 在 `core.unwrap` 后立即 `add_tool_response`；`memory_write` 在 `core.mem.set` 后返回确认；`create_cmd`/`create_sub`/`send_instruction`/`close_conversation` 各自先改状态再写工具响应。事件分发要改变「调用者看到的结果」，必须插在 `add_tool_response` 之前（或包住指令执行）。
- 设备自改事件目前没有发射 API：MemoryDevice 没有向 Core 发事件的方法；需要受约束的钩子，并校验 source.path 是 target 的前缀。
- 「先注册者优先」已与所有者校验合并：第一版按注册顺序处理，但拦截型监听只能由目标节点的 owner 注册，抢注被 Core 拒绝。
- 刚完成的简化（plan-remove-service-tools-2026-08-13）删除了 ICC/return_result 延续机制。notify 走既有 send_instruction 投递，不重新引入返回通道。
- `ctrl` 是 dict[str,str]；owner 以字符串 cid 存储。现存问题：`edit_metadata` 目前可修改任意 ctrl 键（包括 owner），第一版是否封锁 owner 键待定；镜像预置节点没有 owner，其保护语义待定。

## 拟定的事件管道（草案，待评分）

事件对象（瞬时，不落盘）：

```json
{
  "id": "evt-...",
  "type": "memory_read",
  "phase": "before",
  "source": {"kind": "conversation", "cid": 0},
  "target": "$MEM.rooms.secret",
  "payload": {"ref": "$MEM.rooms.secret"},
  "action": "execute",
  "response": ""
}
```

- `source.kind = "device"` 时 `source.path` 必须是该设备自身挂载路径，且 target 在该路径之内。
- `action`：`execute`（按 payload 执行原操作）/ `replace`（不执行原操作，直接把 response 返回给调用者）。处理器不改 action 即继续流动；首个 replace 短路。
- 管道顺序：注册顺序即优先级；先注册者先处理。
- 处理器输出：原样或修改后的 payload 继续流动；或 `action="replace"` + `response`；可携带 `notify` 列表，由 Core 投递。

## 第四轮建议默认值（已由用户确认）

1. **事件类型表**：严格 9 类（memory_read / memory_write / memory_make / edit_metadata / create_cmd / create_sub / send_instruction / close_conversation / listen_event）；工具与事件一一对应。
2. **waterfall 范围**：memory_read / memory_write / memory_make / edit_metadata / send_instruction 可拦截；create_cmd / create_sub / close_conversation / listen_event 只产生 after 通知。
3. **处理器程序**：普通 Python 代码（`type="python"` 节点 content），每次事件在全新沙箱命名空间执行，不创建对话、不保留实例状态。
4. **管道短路**：先注册者输出 `replace` 后短路，后续监听器不可再改。
5. **owner 细节**：`edit_metadata` 禁止普通对话修改/删除 `owner`；镜像预置无主节点第一版不可注册拦截型监听。
6. **监听工具形状**：单条 `listen_event`，参数 `action=listen/modify/cancel`、`event_type`、`target`、`handler_ref`、`listener_id`；listen 返回 listener_id；modify/cancel 按 listener_id 操作；三种成功动作各自产生 `listen_event` after 事件。
7. **可见性**：注册表挂 `$MEM.listeners` 信息设备；事件瞬时、不落盘；monitor 记录事件进管道的首尾摘要。
8. **生命周期**：close_conversation 自动撤销该对话注册的全部监听；检查点不保存监听，恢复后重新注册。

## 影响预估（若实施）

- 指令集 8 → 9；Core 增加事件注册表、所有者记录、优先级链、指令执行点事件分发、设备自改事件发射钩子。
- 调度状态机补上事件唤醒路径（notify → send_instruction → ready）。
- 处理器限定 Python 时，waterfall 不产生额外 LLM 调用，也不产生调度嵌套。
- 检查点/恢复不持久化监听与 owner 之外的运行时注册状态。
- 与 capability/文件密码设计重叠：文件访问回调可被本事件模型吸收。
- `ctrl.owner` 进入内存持久化格式与镜像；镜像规范、CONTEXT、README 需同步。

**第四轮定夺（2026-08-15）**: 用户确认第 3 项改为「处理器程序就是普通 Python 代码，不是持久对话（每次事件执行于全新沙箱命名空间）」；其余默认值均接受。详细实施计划另立：docs/Agents/plan-event-v1-2026-08-15.md。

# AVM Agent 范式文档

> 本文档面向开发者和研究者，详细描述 AVM（Agent Virtual Machine）的信息流逻辑、调用返回机制，并与传统 Agent 范式进行对照分析，标注容易误解和思维转换失败的点。

---

## 1. 核心信息流全景

AVM 的信息流可以概括为：**一切交互都走工具调用通道**。这不是比喻，是字面意义上的全部。

**关键前提：系统的启动点是自发的**，和电脑开机一样，不是由"用户输入"触发。`main.py` 中压入初始 `create` 指令，是系统自身的启动序列。用户输入如果存在，只是后续事件系统中的一种事件类型，而非系统的原动力。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AVM 信息流全景图                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  [内存: $MEM.sys, $MEM.usr, $MEM.para]
         │
         ▼ unwrap
  ┌──────────────┐
  │ CreateInstruction │──call_id, utr_index──┐
  └──────────────┘                        │
         │                                │
         ▼                                │
  ┌──────────────┐    result ────────────┤ 回写父批次
  │  LMU.exec_crt  │───▼                 │
  └──────────────┘    return_calls ─────┤ 子指令
         │                                │
         ▼                                │
  ┌──────────────┐                       │
  │   OpenAI API   │◄──工具调用请求        │
  └──────────────┘                       │
         │                                │
         ▼                                │
  ┌──────────────┐                       │
  │  LLM 响应      │── result(纯文本)      │
  │  (二选一)      │── tool_calls(工具调用)│
  └──────────────┘                       │
                                         │
  有 tool_calls? ──Yes──► 分配新寄存器    │
         │                    │           │
         No                   ▼           │
         │            ┌──────────────┐   │
         │            │ ExecInstruction │  │
         │            └──────────────┘   │
         │                   │           │
         ▼                   ▼           │
  回写父批次           ┌──────────────┐  │
  (EXIT)              │  LMU.exec      │──┘
                      └──────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │   OpenAI API   │◄──conversation + tool_messages
                      └──────────────┘
```

### 1.1 关键数据通道

| 通道 | 方向 | 内容 | 生命周期 |
|------|------|------|----------|
| `command_stack` | VM → 指令 | 待执行指令序列（字符串或对象） | 动态压栈/出栈 |
| `last_msg_reg` | VM ↔ LMU | `Conversation` 对象（对话历史） | create 创建，exec 完成后 pop |
| `usr_tool_reg` | VM ↔ LMU | `UserMessageBatch`（工具响应批次） | create 创建，exec 完成后 pop |
| `mem` | VM ↔ 指令 | 全局共享内存（键值树） | 全局持久 |
| `result` | LMU → VM | LLM 纯文本输出 | 立即回写父批次 |
| `return_calls` | LMU → VM | 子指令列表（对象） | 压入 command_stack |

---

## 2. 调用链详解

### 2.1 发起调用：`CreateInstruction`

`create` 是**唯一能够创建新对话上下文**的指令。它的执行流程：

```python
system = core.unwrap("$MEM.sys")      # 解引用系统提示
user   = core.unwrap("$MEM.usr")      # 解引用用户消息
para   = core.unwrap("$MEM.para")     # 解引用参数

result, return_calls, conversation = core.lmu.exec_crt(system, user, para)
```

**`exec_crt` 内部**：
1. 构造 `messages = [system_msg, user_msg]`
2. 调用 OpenAI API（带 tools）
3. 解析 LLM 响应：
   - 如果有 `message.content` → 作为 `result` 返回
   - 如果有 `message.tool_calls` → 解析每个 tool_call，生成 `return_calls` 对象列表
4. 构造 `Conversation(system, user, assistant)` 返回

**关键事实**：`create` 的初始输入只有 `system` + `user` 两条消息。如果 LLM 有 tool_calls，它**不会**在本次响应中产生 content（`result` 为 None）。

### 2.2 继续对话：`ExecInstruction`

当 `create` 收到 tool_calls 时，栈顶被替换为 `exec`，同时子指令被压栈。`exec` 的执行流程：

```python
conversation = core.unwrap("$last_msg_reg.0")   # 复用已有对话
user_batch   = core.unwrap("$usr_tool_reg.1")   # 获取工具响应批次

result, return_calls, _ = core.lmu.exec(conversation, user_batch, para)
user_batch.clear()  # 清空已消费的批次
```

**`exec` 内部**：
1. `messages = conversation.to_api_messages() + user_batch.to_tool_messages()`
2. 如果有 `user_content`，追加 user 消息
3. 调用 OpenAI API（带 tools）
4. 更新 `conversation`：追加 user + assistant 消息
5. 返回 `result` 和 `return_calls`

**关键事实**：`exec` 复用已有的 `Conversation`，并把工具响应作为上下文传入。这使得 LLM 能看到之前工具调用的结果。

### 2.3 结果回写机制

```python
# CreateInstruction 中
if result and self.utr_index != -1:
    core.usr_tool_reg[self.utr_index].add_tool_response(result, self.call_id)

# ExecInstruction 中
if result and self.utr_index != -1:
    core.usr_tool_reg[self.utr_index].add_tool_response(result, self.call_id)
```

**注意 `self.utr_index` 指向的是父调用的批次，不是当前调用的批次。**

当前调用（create/exec）的工具调用结果写入父调用的 `usr_tool_reg`，父调用在后续 `exec` 时将这些结果作为上下文传给 LLM。

### 2.4 嵌套调用示例

```
用户触发：create root 0 $MEM.sys $MEM.usr $MEM.para
    │
    ▼
LMU.exec_crt() ──► LLM 返回 tool_calls:
    ├── create_cmd(tc2, sys2, usr2, para2)  ──► 子指令A
    └── create_cmd(tc3, sys3, usr3, para3)  ──► 子指令B
    │
    ▼
栈状态：[exec(root), 子指令B, 子指令A]  （反序压栈，A先执行）
    │
    ▼
子指令A.execute() ──► CreateInstruction(tc2, 1, sys2, usr2, para2)
    │
    ▼
LMU.exec_crt() ──► LLM 返回 result="done"
    │
    ▼
结果"done"回写到 usr_tool_reg[1]（exec(root)的批次）
EXIT，弹出子指令A
    │
    ▼
子指令B.execute() ──► 类似...
    │
    ▼
exec(root).execute()
    │
    ▼
LMU.exec() ──► conversation + usr_tool_reg[1] 的结果
    │
    ▼
LLM 看到 tc2/done 和 tc3/... 的结果，返回最终答案
    │
    ▼
结果回写到 usr_tool_reg[0]（父create的批次）
EXIT，弹出寄存器
```

---

## 3. 与传统 Agent 范式的对比

### 3.1 传统 ReAct / Tool-Using Agent

```
用户输入 ──► LLM ──► [思考] ──► 工具调用A ──► 执行A ──► 结果A ──► LLM
                                              │
                                              ▼
                                         [思考] ──► 工具调用B ──► ...
                                              │
                                              ▼
                                         [思考] ──► 最终答案
```

**特点**：
- 单一会话上下文
- 用户输入作为 `user` 消息启动对话
- 工具调用和结果在同一个对话中交替
- LLM 的 "思考" 过程可见（Chain-of-Thought）

### 3.2 AVM 范式

```
系统启动/初始指令 ──► create ──► LLM ──► tool_calls ──► 子指令执行 ──► 结果回写
                                                        │
                                                        ▼
                                                   exec ──► LLM ──► tool_calls ──► ...
                                                        │
                                                        ▼
                                                   exec ──► LLM ──► result
                                                        │
                                                        ▼
                                                   结果回写父批次
```

**特点**：
- **所有外部交互（包括用户输入）都通过工具响应传递**
- 可以嵌套创建新的子对话（`create_cmd`）
- 多工具调用通过**反序压栈**实现顺序执行
- 没有显式的 "思考" 步骤，所有逻辑隐含在工具调用链中
- `UserMessageBatch.user_contents` 用于 LLM 间内部通信，不是给用户输入用的

### 3.3 核心差异对照表

| 维度 | 传统 Agent | AVM |
|------|-----------|-----|
| **系统启动** | 由用户输入触发 | 自发启动（类似电脑开机），用户输入只是后续事件之一 |
| **"用户输入"的地位** | 基础元事件，不可再分 | 普通工具，和其他工具无本质区别 |
| **会话模型** | 单一对话 | 嵌套子对话 + 共享内存 |
| **工具结果传递** | 直接作为 `tool` 消息回传 | 先写入 `usr_tool_reg` 批次，再批量传入 |
| **多工具并行** | LLM 一次输出多个 tool_calls | 反序压栈，顺序执行 |
| **思考过程** | 显式 CoT | 隐式，无中间输出 |
| **状态管理** | 会话内隐式 | 显式寄存器（`last_msg_reg`, `usr_tool_reg`） |
| **嵌套能力** | 无 | create_cmd 可创建全新子对话 |

---

## 4. 容易误解和思维转换失败的点

### 4.1 "系统中没有'用户输入'这个特殊概念"

**传统思维**："用户输入"是一个基础元事件，是系统的原动力。就像物理定律一样不可再分，是 Agent 必须"被动接收"的东西。

**AVM 现实**：不用"用户输入"不是因为设计偏好，而是因为**内存读写的表达力明显优于聊天**。

聊天（对话）的局限：
- **线性**：消息按时间顺序排列，引用前面内容需要重复或摘要
- **模糊**：自然语言有歧义，LLM 对上下文的理解可能不准确
- **难以精确访问**：要获取对话中的某个特定信息，需要遍历整个历史
- **难以共享**：不同对话之间的信息传递依赖复制和转述

内存读写的优势：
- **结构化**：键值树、嵌套结构，通过路径精确访问 `$MEM.data.input`
- **随机访问**：不需要按顺序遍历，直接定位
- **全局共享**：多个子对话可以读写同一块内存
- **无歧义**：写进去的是什么，读出来就是什么

所以 AVM 选择把**内存读写作为主要的交互原语**，而不是聊天。工具调用只是实现内存读写的机制——LLM 通过调用 `memory_read`/`memory_write` 来精确地表达"我要访问内存的哪个位置"。

**后果**：如果你在心里把"用户输入"特殊化，你会不自觉地为它预留聊天通道（专门的 `user` 消息、特殊的事件处理逻辑）。这会削弱系统的表达能力，因为聊天不如内存读写精确。`UserMessageBatch.add_user_content()` 不是给用户输入用的——它是 LLM 间内部通信的通道。任何外部信息如果要进入系统，都应该走工具响应通道，最终落入可被精确读写的内存中。

### 4.2 `create` vs `exec` 的语义混淆

**传统思维**：`create` = 创建文件/对象，`exec` = 执行命令。

**AVM 现实**：
- `create` = 发起**全新对话**（从 system + user 开始）
- `exec` = **继续已有对话**（复用 Conversation + 传入工具响应批次）

`exec` 不是"执行命令"，而是"继续对话"。它的名字来源于 "execute the conversation"。

### 4.3 `result` 和 `return_calls` 的互斥性

**传统思维**：LLM 可以同时返回文本和工具调用。

**AVM 现实**：当 LLM 返回 `tool_calls` 时，`message.content` 几乎总是 `None` 或空字符串。`result` 和 `return_calls` 在实际运行中是互斥的。

**代码体现**：
```python
if result and self.utr_index != -1:
    # 只有纯文本响应时才会进入这里
    batch.add_tool_response(result, self.call_id)

if return_calls:
    # 有工具调用时走这里，result 通常为 None
    ...
```

### 4.4 `utr_index` 的递归回写方向

**传统思维**：指令执行的结果写回自己的输出缓冲区。

**AVM 现实**：`create`/`exec` 的结果写回 **`self.utr_index` 指向的父批次**，不是当前调用的批次。

```python
# 在 ExecInstruction 中
parent_batch = core.usr_tool_reg[self.utr_index]  # ← 父调用的批次！
parent_batch.add_tool_response(result, self.call_id)
```

当前调用的批次（`$usr_tool_reg.user_msg_idx`）只是用来收集子指令结果的，最终会被清空或弹出。

### 4.5 指令压栈顺序

**传统思维**：指令按正序执行（先输出的先执行）。

**AVM 现实**：`return_calls` 被**反序** extend 到栈中。LLM 先输出的指令最后被压入栈顶，因此**最先执行**。

```python
# LLM 输出: [callA, callB, callC]
# 反序压栈: [callC, callB, callA]
# 栈顶是 callA，先执行
for rc in reversed(return_calls):
    core.command_stack.append(instr)
```

### 4.6 寄存器清理时机

**传统思维**：每次 exec 结束都清理资源。

**AVM 现实**：`ExecInstruction` **只在无更多子调用时才 pop 寄存器**。如果 LLM 又返回了 tool_calls，寄存器保留供后续 exec 复用。

```python
if return_calls:
    # 有子调用：保留寄存器，压入新指令
    return CRT.CONTINUE
else:
    # 无子调用：清理寄存器
    core.usr_tool_reg.pop(user_msg_idx)
    core.last_msg_reg.pop(last_msg_idx)
    return CRT.EXIT
```

### 4.7 `command` 工具的占位符性质

**传统思维**：`command` 是一个通用工具，可以执行任意指令，应该完善其处理逻辑。

**AVM 现实**：`command` 是**未设计的占位符**。当前真正重要的是 `create_cmd`、`memory_read` 等单独工具。`command` 的存在是因为早期设想过通用命令通道，但后来发现核心指令都应该直接写成独立工具。

**后果**：过度设计 `command` 的处理逻辑（如解析 command 字符串、参数拆分等）是浪费时间。

### 4.8 `memory_read` 不是普通内存访问

**传统思维**：`memory_read` 读取内存，返回给调用者使用。

**AVM 现实**：`memory_read` 读取内存后，把结果作为**工具响应**写入 `usr_tool_reg`，回传给 LLM。它不是给代码内部使用的——它是给 LLM 提供上下文的。

```python
# 错误理解：
value = core.mem.unwrap(ref)  # 以为这就是目的

# 正确理解：
user_batch.add_tool_response(value, self.call_id)  # 目的是让 LLM 看到这个值
```

### 4.9 `current_utr_index` 已被移除

**历史包袱**：早期代码中 `current_utr_index` 被传入 LMU，但 LMU 并不使用它。后来通过对象式参数包装彻底消除了这个参数。

**当前设计**：
- `call_id` 由 LMU 从 `tool_call.id` 提取，放入 return_calls 对象
- `utr_index` 由 Core 在构造指令对象时填入
- 两者不需要通过字符串占位符传递

---

## 5. 设计理由

### 5.1 为什么全部走工具调用？

直接原因：**聊天的表达力不如内存读写**。

传统 Agent 用聊天（对话）作为主要的交互和信息传递方式。但聊天有几个工程上的硬伤：

| 维度 | 聊天 | 内存读写 |
|------|------|----------|
| 访问方式 | 线性遍历 | 随机访问（路径定位） |
| 精确度 | 自然语言有歧义 | 结构化数据无歧义 |
| 共享性 | 对话间隔离，需复制 | 全局共享 `$MEM` |
| 引用能力 | 需重复或摘要 | 直接路径引用 `$MEM.data.key` |
| 持久性 | 随对话结束而丢失 | 全局持久 |

AVM 的选择是：**用结构化的内存读写替代线性的聊天对话作为主要的信息交互方式**。

工具调用是实现这个选择的机制：
- LLM 调用 `memory_read` → 精确读取内存路径的值
- LLM 调用 `memory_write` → 精确写入内存路径
- LLM 调用 `create_cmd` → 创建新对话时从内存取 system/user/para

**这个设计带来的优势**：
1. **精确表达**：LLM 通过工具参数精确指定要访问的内存路径，不用在聊天中猜测
2. **全局状态**：所有信息落盘到 `$MEM`，子对话之间可以共享状态
3. **可组合**：内存操作可以嵌套、批量、条件执行，不受对话线性结构的限制
4. **显式寄存器**：`usr_tool_reg` 和 `last_msg_reg` 是显式的状态容器，不在对话中隐式积累
5. **子对话隔离**：`create_cmd` 创建全新对话，但状态通过 `$MEM` 共享，避免上下文污染

### 5.2 为什么是反序压栈？

OpenAI API 返回的 `tool_calls` 是一个列表。如果 LLM 同时请求两个操作：
- "先读取文件 A，再写入文件 B"

LLM 输出的顺序是 `[readA, writeB]`。但栈是 LIFO 结构，为了保证 `readA` 先执行，需要反序压栈。

### 5.3 `UserMessageBatch` 的双队列设计

```python
@dataclass
class UserMessageBatch:
    tool_responses: List[ToolCallResponse]   # 给 LLM 看的工具结果
    user_contents: List[str]                  # LLM 间内部通信
```

- `tool_responses` → 通过 `to_tool_messages()` 转为 API 的 `tool` 角色消息
- `user_contents` → 通过 `get_user_content()` 合并为一条 `user` 角色消息

这个设计允许在工具响应之外，额外注入 `role=user` 的内部消息（如 LLM 间的直接信息传递）。注意这里的 "user" 是消息角色（role=user），不是"来自用户的输入"。

---

## 6. 扩展指南

### 6.1 添加新工具

1. 在 `LMU.tools` 中定义工具 schema
2. 在 `LMU.exec_crt` / `LMU.exec` 中解析 tool_call，生成 return_calls 对象
3. 在 `_make_instruction` 中添加对应的指令构造逻辑
4. 实现对应的 `Instruction` 子类

### 6.2 添加新指令类型

1. 继承 `Instruction`，实现 `execute(self, core)`
2. 如果需要 LLM 触发，在 LMU 中添加工具定义和解析逻辑
3. 在 `_make_instruction` 中注册

---

## 7. 常见调试场景

| 现象 | 可能原因 | 检查点 |
|------|---------|--------|
| LLM 看不到工具结果 | `result` 没回写父批次，或 `user_batch.clear()` 太早 | `usr_tool_reg` 内容 |
| 子指令不按预期执行 | 压栈顺序理解错误 | `command_stack` 顺序 |
| 寄存器越界/泄漏 | exec 有子调用时提前 pop | `return_calls` 是否为空 |
| KeyError: 'MEM' | 解引用路径未去掉 MEM 前缀 | `memory.py` 的路径处理 |
| LLM 不输出工具调用 | `para` 中缺少 `"use_tool": True` | `main.py` 的 model_params |
| 嵌套 create 结果丢失 | utr_index 指向错误 | 子指令的 utr_index 是否等于父 exec 的 user_msg_idx |

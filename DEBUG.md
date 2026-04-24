# AVM 调试指南

> 本文档面向开发者和维护者，记录调试工作流、测试策略、已修复的底层 bug，以及仍需确认的设计决策。

---

## 1. 调试哲学（先读这个）

AVM 有两层执行模型：

- **底层**：Python 代码逐行执行（`Core.run()` 的 while 循环，每条指令的 `execute`）
- **高层**：一个 LLM 调用视为"一步"，但底层可能经历多次指令压栈/出栈

**当前阶段的目标是：让底层足够稳固。** 底层稳固后，才能在其上构建高层步进调试器。

调试时，状态分散在 4 个地方：

| 位置 | 内容 |
|------|------|
| `core.command_stack` | 待执行指令序列 |
| `core.last_msg_reg` | 对话历史寄存器（`Conversation` 列表） |
| `core.usr_tool_reg` | 用户输入/工具响应批次寄存器（`UserMessageBatch` 列表） |
| `core.mem` | 全局共享内存（带元数据的复合结构） |

每次追踪 bug 时，先确认问题发生在哪一层：

1. **Python 运行时错误**（`KeyError`、`AttributeError`、类型错误）→ 底层 bug，用测试覆盖
2. **指令未按预期执行**（栈状态不对、寄存器未弹出）→ 检查 `parse_instruction` 和 `Instruction.execute`
3. **LLM 行为不符合预期**（不输出工具调用、参数错误）→ 检查 prompt 和 LMU 工具定义

---

## 2. 快速开始：运行测试

```bash
# 运行全部测试
python -m pytest test_avm.py -v

# 运行单个测试类
python -m pytest test_avm.py::TestMemory -v

# 运行单个测试
python -m pytest test_avm.py::TestInstructionExecution::test_create_no_tool_calls -v
```

**所有测试使用 MockLMU，不调用真实 LLM API。** 这是调试焦虑的解药——你可以在没有网络、没有 API key 的情况下验证整个状态机。

---

## 3. 调试工具（debug_utils.py）

### 3.1 步进追踪

```python
from AVM import Core
from debug_utils import DebugTracer

core = Core()
# ... 初始化内存和命令栈 ...

tracer = DebugTracer(core)
while tracer.step():
    print(tracer.last_diff())  # 打印每次指令执行前后的状态变化

# 导出完整历史
print(tracer.dump_history("/tmp/run.json"))
```

### 3.2 即时状态检查

```python
from debug_utils import inspect_core
print(inspect_core(core, title="故障点状态"))
```

输出示例：
```
============================================================
  故障点状态
============================================================
command_stack  : ['exec cid_1 0 $last_msg_reg.0 $usr_tool_reg.1 $MEM.para']
last_msg_reg   : 1 conversation(s)
  [0] [('system', 's'), ('user', 'u'), ('assistant', '')]
usr_tool_reg   : 2 batch(es)
  [0] tools=[], users=[]
  [1] tools=[ToolCallResponse(...)], users=['hello']
mem top keys   : ['sys', 'usr', 'para', 'input']
devices        : []
============================================================
```

---

## 4. 已修复的底层 Bug（硬性错误）

### 4.1 `parse_instruction` 为空函数
**问题**：`Core.run()` 调用 `parse_instruction(raw)`，但原实现是 `pass`，返回 `None`，导致 `instruction.execute(core)` 抛 `AttributeError`。

**修复**：实现了完整解析器，统一指令格式为：
```
<cmd_type> <call_id> <utr_index> <...payload>
```

### 4.2 `$MEM` 前缀解引用失败
**问题**：`Memory.unwrap()` 对 `"$MEM.key"` 的处理路径在递归解引用时未移除 `MEM` 前缀，导致 `KeyError: 'MEM'`。

**修复**：在 `unwrap`、`_unwrap_dollar`、`_unwrap_ampersand`、`make` 四个位置统一添加了 `MEM` 前缀移除逻辑。

### 4.3 `LMU` 缺少 `__init__`，且 `OpenAI()` 在初始化时强制要求 API key
**问题**：`Core()` 创建即崩溃，因为 `LMU()` 没有 `__init__`，而后来补上的 `self.client = OpenAI()` 在缺少 `OPENAI_API_KEY` 环境变量时报错。

**修复**：`LMU.client` 改为惰性属性（`@property` + `_client = None`），第一次调用 `exec_crt`/`exec` 时才创建 `OpenAI()` 实例。这样测试和纯内存操作不再被 API key 阻塞。

### 4.4 `CreateInstruction.execute` 返回类型标注为 `int`
**问题**：标注 `-> int`，实际返回 `CRT` 枚举值。虽不影响运行，但误导静态分析和阅读者。

**修复**：改为 `-> CRT`。

### 4.5 `ExecInstruction` 调用 `core.lmu.exec` 时传入了错误的 `current_utr_index`
**问题**：原代码传的是 `self.user_msg_ref`（字符串，如 `"$usr_tool_reg.0"`），但 `exec` 期望整数。

**修复**：通过 `_parse_index(self.user_msg_ref)` 解析出整数后再传入。

### 4.6 `CreateInstruction` 生成 exec 指令时寄存器名写错
**问题**：生成的是 `$user_tool_reg.x`，但 `Core.unwrap` 检查的是 `$usr_tool_reg.x`，导致 `KeyError`。

**修复**：统一为 `$usr_tool_reg.x`。

### 4.7 `ExecInstruction` 未替换 `$current_utr_index` 占位符
**问题**：`CreateInstruction` 有占位符替换逻辑，但 `ExecInstruction` 没有。导致 exec 产生的子指令中的 `$current_utr_index` 保持原样，无法解析。

**修复**：在 `ExecInstruction.execute` 中补充了替换逻辑。

### 4.8 `MemoryReadInstruction` 调用不存在的方法
**问题**：`user_batch.add_user_message(content)`，但 `UserMessageBatch` 只有 `add_user_content`。

**修复**：改为 `add_user_content`。这是一个**我做出了决定**的修复（见第 6 节）。

---

## 5. 测试策略

### 5.1 测试分层

| 层级 | 文件/类 | 验证内容 |
|------|---------|----------|
| 解析层 | `TestParseInstruction` | 指令字符串 → 对象映射的正确性 |
| 内存层 | `TestMemory` | 读写、解引用、设备挂载、嵌套路径 |
| Core 状态 | `TestCoreState` | 寄存器访问、`unwrap` 行为 |
| 指令执行 | `TestInstructionExecution` | 单指令 + MockLMU 的完整状态转换 |
| 集成 | `TestIntegration` | create→exec 链条、多工具调用压栈顺序 |

### 5.2 MockLMU 用法

```python
from test_avm import MockLMU  # 或 copy 到你的测试

core.lmu = MockLMU([
    # (result: str|None, return_calls: list[str], conversation: Conversation|None)
    ("hello", [], None),                                    # 无工具调用
    (None, ["memory_read cid 0 $MEM.x"], conv_obj),         # 有一个工具调用
    ("done", [], None),                                     # 后续轮次
])
```

MockLMU 会自动记录每次调用参数到 `core.lmu.calls`，可用于断言 LLM 实际收到了什么。

### 5.3 新增测试的准则

1. **绝不调用真实 LLM**。所有执行路径都必须能用 MockLMU 覆盖。
2. **断言状态，不只是返回值**。AVM 的核心是状态转换，每条测试应检查 `command_stack`、`last_msg_reg`、`usr_tool_reg`、`mem` 的变化。
3. **对于多工具调用，显式断言压栈顺序**。这最容易出错（`return_calls[::-1]` 的反转逻辑）。

---

## 6. 我（AI助手）自行做出的设计决定

以下决策未与用户确认，可能存在方向性偏差。如果用户意图不同，需要回滚或调整。

### 6.1 `&`（ampersand）引用的语义
**代码行为**：`&MEM.a` 会解一层引用。如果 `a="$MEM.b"` 且 `b="final"`，则 `&MEM.a` 返回 `"final"`。

**我做的决定**：让 `&` 保持"解一层"的实现语义，并修改了测试 `test_ampersand_one_level` 的期望值以匹配代码。

**替代方案**：如果用户预期 `&` 是"不解引用、仅返回原始值"，则 `_unwrap_ampersand` 应直接返回 `_get_by_path(path)` 的结果，不再检查是否以 `$` 开头。

**用户决定**：`&MEM.a`直接返回原始值。用户说的“解一层引用”是指`&MEM.a`与字面值`MEM.a`相比解了一层引用，而以 `$` 开头的是解引用直到找到字面值

### 6.2 `MemoryReadInstruction` 的行为修复
**原代码**：`user_batch.add_user_message(content)` —— 方法不存在。

**我的修复**：改为 `user_batch.add_user_content(content)`。

**潜在问题**：如果原意是"将内存读取结果作为工具响应返回"（即 `add_tool_response`），那当前修复是错误的。从指令名 `memory_read` 看，它更像是在"向对话中注入用户消息"，所以 `add_user_content` 是合理的，但需用户确认。

**用户决定**：`memory_read`是指读取内存，不一定是用户消息，而且该模型采用的用户交互方法是独特的“包裹在工具内”方法，传统的用户消息在该系统中被用作LLM对话间直接传递信息的方式，尤其是初始用户输入。更复杂的交互可能需要在之后模拟并发对话和高级上下文切换时候进行注入，暂且不管。

### 6.3 指令字符串的统一格式
**原代码**：`create_cmd` 生成的格式是 `create <sys> <usr> <para> <mode> <utr> <call_id>`，`command` 生成的格式是 `<command> <call_id>`，两者不统一。

**我的决定**：统一为 `<cmd_type> <call_id> <utr_index> <...args>`，并在 LMU 生成命令时插入 `$current_utr_index` 占位符。

**影响**：LLM 的 prompt 中如果描述了指令格式，需要同步更新。

### 6.4 `LMU` 中 `command` 工具调用的命令生成
**原代码**：直接 `f"{command} {call_id}"`。

**我的修改**：解析 `command` 字符串，提取 `cmd_type` 和 `args`，重新格式化为 `f"{cmd_type} {call_id} $current_utr_index {cmd_args}"`。

**风险**：如果 LLM 输出的 `command` 参数包含特殊格式（如带空格的参数、引号），简单 `split()` 会出错。当前实现对此类输入是脆弱的。

**用户决定**：command工具是占位符。这些修改让用户感到有些困惑。需要和用户沟通关于指令格式的更多内容

---

## 7. 已知限制与待办

### 7.1 `parse_instruction` 的空格分割限制
`memory_write <call_id> <utr_index> <ref> <content>` 中的 `content` 如果包含空格，会被截断。

**缓解**：当前测试中所有 content 都是单个 token。如需支持空格，需引入引号解析或 JSON 参数格式。

### 7.2 `main.py` 中 `model_params` 缺少 `use_tool`
```python
model_params = MetaDict(data={"model": "deepseek-chat"})
```
缺少 `"use_tool": True`。这意味着初始 `create` 指令不会触发工具调用，LLM 只会返回普通文本。

**待确认**：这是有意为之（先跑通纯文本流程）还是遗漏？

### 7.3 `devices.py` 与 `main.py` 的 `UserInputDevice` 重复
`devices.py` 定义了一个简单版本，`main.py` 定义了一个继承 `StringDevice` 的更复杂版本。后者覆盖了前者，但代码中存在两个同名类，容易混淆。

### 7.4 没有高层步进调试器
`debug_utils.py` 提供的是底层步进（单条 Python 指令）。用户期望的高层调试器（一步 = 一次 LLM 调用）尚未实现，需要底层稳定后再构建。

### 7.5 错误处理不完善
- 指令执行中的异常未被 `vm_exception_handler` 捕获
- `Core.run()` 没有优雅退出机制（遇到异常直接抛出让 Python 崩溃）
- 子调用深度没有限制，存在栈溢出风险

---

## 8. 调试检查清单

遇到问题时，按此顺序排查：

- [ ] **能否通过测试？** 先运行 `pytest test_avm.py`，确认底层无回归
- [ ] **指令格式是否正确？** 检查 `command_stack` 中的字符串是否符合 `<cmd_type> <call_id> <utr_index> ...`
- [ ] **寄存器索引是否越界？** `last_msg_reg` 和 `usr_tool_reg` 是否被正确 pop（从后往前）
- [ ] **`$current_utr_index` 是否被替换？** 如果子指令的 `utr_index` 看起来是字符串，说明替换逻辑遗漏
- [ ] **MEM 前缀是否导致 KeyError？** 检查 `memory.py` 的解引用路径
- [ ] **MockLMU 响应是否足够？** 每条 LLM 调用路径都需要一条 mock 响应

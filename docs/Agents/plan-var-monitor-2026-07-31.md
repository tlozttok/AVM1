# 计划：变量监测型日志系统（Variable Monitor）

**时间**: 2026-07-31（第三次修订：已实施）
**状态**: 已实施（2026-07-31，用户批准后实现）
**背景**: 上一会话删除了 web/、日志、终端显示（随核心重写一并废弃）。用户需要新的日志系统，用于检查系统状态。
**起因**: 旧日志是"记录事件型"（谁在何时做了什么的线性流水）。用户希望新系统是"监测变量型"——以具名变量为基本单位，记录变量值的变化轨迹，而不是事件流水。评审后明确：内存写入监测不重要，当前重点是调试内核和 LMU 的稳定性，记录周期为内核行为循环（一次对话推进为一周期）。
**行动**: 已实现 `avm/monitor.py`（Frame + Monitor：帧序列、trail、diff、find、全文文件）；Core.advance_conversation 挂钩（异常传播不变，帧含 error 字段）；LMU.last_call 捕获每次 API 调用全文；`tests/test_monitor.py` 21 个测试（帧序列、diff、trail、环形上限、截断、异常路径、全文文件分段）。
**影响**: 取代旧 logging/dump_tree/query_path/memshell；Core 的 `advance_conversation` 与 `run` 增加监测挂钩，不改变调度行为、不吞异常；挂钩契约（"每轮推进生成一帧、每次 API 调用写一段全文"）为核心重写保留。

---

## 设计概要 v3（待评审后定稿）

### ~~v1 设计（废弃）~~

~~核心概念为 MonitorPoint（具名变量时间序列，值去重）。捕获来源含内存路径监测。~~ 已废弃：用户确认内存写入监测不重要。

### v2 步进帧模型（保留）

记录周期 = 内核行为循环的一轮 = 一次对话推进（`advance_conversation`，含一次 LMU.exec）。每轮推进结束后生成一帧，帧内是全部被监测变量在该轮结束后的值。第 0 帧为基线（启动后、第一轮推进前）。

### v3 修订（本版）

**决定 1 — LMU 异常：抛出，不吞。**
`advance_conversation` 中 LMU.exec 抛出的异常原样向上传播（与当前行为一致），run 终止。API 相关异常需要人工处理，系统不代为决策。
补充：异常抛出前，本轮请求全文仍写入全文文件（供人工处理时查看请求内容），帧也在 `finally` 中记录最后一帧（带 `error` 字段，保留崩溃点的状态快照）。传播行为不受影响。

**决定 2 — 双文件存储：帧截断 + API 调用全文。**
| 存储 | 内容 | 生命周期 |
|------|------|---------|
| 内存环形缓冲 | 帧序列，消息/输出截断（默认 200 字符/条，可配置） | 进程内，默认 2000 帧，可 dump |
| 全文文件（append-only） | 每次 API 调用的完整输入消息 + 完整输出（含 tool_calls），按帧号分段 | 落盘，默认 `log/api_transcript.txt`（gitignore 已覆盖 log/*.txt） |

帧 n 的 `lmu.result` 是截断摘要；全文文件第 n 段是完整内容。帧号 ↔ 全文段号一一对应，从帧跳到全文看完整上下文。

**决定 3 — `$monitor` 设备：暂不考虑。**
当前处于极早期开发，LLM 可读的监测接口留待核心重写后决定。

### 帧结构

```
Frame:
  seq: int                       # 帧号（= API 调用序号，0 为基线帧）
  ts: float
  lmu:
    result: str | None           # 截断
    tool_calls: list             # 指令名 + call_id + 参数摘要（截断）
    elapsed_ms: float
    error: str | None            # 异常信息（最后一帧）
  sched:
    active_cid, ready_cids, dormant_cids, next_cid
  conversations:                 # 每对话状态向量
    {cid: {state, msgs: int, batch: {tool_responses, user_contents}}}
```

### 稳定性调试场景

| 症状 | 表现 |
|------|------|
| LMU 挂起/慢 | 某帧 `lmu.elapsed_ms` 异常大；全文文件看该轮完整请求 |
| LMU 返回坏 JSON / 异常 | 异常传播终止 run；全文文件有该轮请求全文，最后一帧带 error |
| 无限循环 | 帧数激增且状态向量反复不变 |
| 状态错乱 | 帧 diff 中某对话 batch 出现不应有的内容 |
| 对话丢失 | conversations 里某 cid 消失 |

### 查询接口

- `frame(n)` / `frames()` / `trail(name)`（单变量跨帧轨迹）/ `diff(a, b)` / `find(条件)`
- 全文文件按帧号分段，头部含 seq、ts、model、elapsed_ms，人工直接翻阅

---

## 实施步骤

1. `avm/monitor.py`：Frame 模型 + 环形缓冲 + 查询 API（纯数据，可单测）
2. 全文文件写入器：按帧号分段的 append-only 写入
3. Core 挂钩：`advance_conversation` 内 — exec 后写全文段；结束后 `finally` 生成帧；异常时帧带 error 且异常继续传播
4. 状态向量提取 + 截断策略
5. 测试（MockLMU 驱动）：帧序列、diff、trail、环形上限、全文文件分段、异常路径

## 验证与影响

- `tests/test_monitor.py`：全部 Mock 验证，不调 API
- 影响面：新增模块 + Core 的 advance_conversation 挂钩，不改调度行为、不吞异常
- 挂钩契约（"每次推进生成一帧、每次 API 调用写一段全文"）写入文档，核心重写时保留

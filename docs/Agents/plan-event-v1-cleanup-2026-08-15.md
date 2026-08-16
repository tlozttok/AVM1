# 计划（待实施）：事件模型 v1 收尾（P1 定夺 + P2 仓库/文档）

**时间**: 2026-08-15
**状态**: 已实施（266 个测试通过；实施记录见文件末尾）
**背景**: 事件模型 v1 主体已实施（见 docs/Agents/plan-event-v1-2026-08-15.md），审计发现 P1 语义问题和 P2 仓库/文档问题（见 docs/Agents/audit-event-v1-2026-08-15.md）。
**起因**: 用户对 P1 逐项给出定夺，并要求把 P1 与 P2 写成计划书交给实施模型。

## 用户定夺（必须执行）

1. **禁止自我投递**：对话不能把消息投递给自己。
2. **monitor 失败记录**：由本计划按下文方案决定。
3. **before 链失败时保留已收集的 notify**：处理器失败不丢弃先前处理器产生的 notify。
4. **多播 send_instruction 允许部分投递**：一个目标失败不阻断其他目标。
5. **补测试**：按下文测试清单补齐。
6. **P2 一并处理**：文档与仓库卫生。

设计提醒（写入实施注释，也供未来软件设计使用）：**before 事件不保证结果发生**。处理器只决定事件是否放行/改写；管道可能在后续处理器处失败，因此通过 before 事件得到的「允许」不是操作已发生的承诺。

---

## 一、P1 行为规格

### 1.1 禁止自我投递

- `SendInstruction.execute`：解析目标后，若任一 `cid == conv.cid`（单目标或多播中任一项），整体失败：向 stderr 报告并返回工具响应 `Error: send_instruction 不允许自我投递（cid=<cid> 是当前对话）`；不投递任何目标。
- 事件处理器输出的 `notify`：`EventSystem._merge_output` 校验时，若 `notify` 项的 `cid` 等于事件 `source` 为 conversation 时的 `source["cid"]`，判定为处理器输出非法：
  - before 事件 → EventHandlerError，走 fail-closed；
  - after 事件 → 丢弃该条 notify 并记入 handler_errors，不投递。
- `Core.emit_device_event` 的 source 是 device，没有 cid，无自我投递概念。
- 更新 `send_instruction` 与 `listen_event` 工具描述，明确「不允许自我投递」。
- 现有测试需改：`tests/test_events.py` 中 `test_create_cmd_after_notifies`、`test_listen_event_after_notifies` 目前 notify cid 0（等于 source 对话），改为先创建一个休眠目标对话并 notify 该目标。

### 1.2 before 失败不抛异常，改为事件结果

当前 `dispatch` 在 before 处理器失败时抛 `EventHandlerError`。改为**返回事件结果**：

- before 处理器执行/输出校验失败时：
  1. 把错误文本写入 `event["response"] = f"Error: 事件监听器 <listener_id> 处理失败: <类型>: <信息>"`；
  2. `event["action"] = "handler_error"`；
  3. 停止后续监听器；
  4. 追加事件摘要（见 1.3）；
  5. **不投递、不抛异常**，直接 `return event`。
- 各 before 指令（memory_read / memory_write / memory_make / edit_metadata / send_instruction）不再捕获 `EventHandlerError`；统一改为：
  ```
  if event["action"] in ("replace", "handler_error"):
      core.events.deliver_notifies(event)
      conv.user_batch.add_tool_response(event["response"], self.call_id)
      return
  ```
  这同时满足「保留已收集 notify」：`event["notify"]` 在失败前已累积，照常投递。
- `EventHandlerError` 类可保留但不再用于 before 路径（若没有其他使用者则删除，并同步清理 core.py import）。
- after 阶段仍 fail-open、不抛异常，行为不变。

### 1.3 monitor 记录与 stderr 去重

- 每个事件在管道结束时必有一条摘要进入 `core._turn_events`。
- before 失败时摘要为：
  ```
  {
    "event_id": ..., "event_type": ..., "phase": "before",
    "target": ..., "source": ...,
    "action": "handler_error",
    "error": "<截断后的错误文本>"
  }
  ```
- 普通成功摘要保持 `action` 为 `execute`/`replace`；after 处理器失败不改变事件 action，但摘要增加 `handler_errors: [{"listener_id": ..., "error": ...}]`。
- `Monitor._summarize_event` 增加 `error` 与 `handler_errors` 字段（截断策略同现有 target/source）。
- stderr 只报一次：before 错误由指令执行方 `_report_tool_error` 报告（dispatch 不再打印）；after 错误由 dispatch 打印，指令执行方不重复打印。

### 1.4 多播 send_instruction 部分投递

- 对每个目标分别走 before 管道并独立结算：
  - `execute`：按最终 content 投递该目标；
  - `replace`：该目标被拦截；
  - `handler_error`：该目标处理失败。
- 不再因任一目标失败提前 return；所有目标的 notify 都照常投递。
- 最终工具响应：
  - 全部 `execute`：保持现状（`已投递到 <身份>/<数量>`）。
  - 有被拦截/失败：
    - 若 `delivered > 0`：`已投递到 <delivered>，<blocked> 个被拦截，<failed> 个处理失败`；
    - 若 `delivered == 0` 且只有一个目标：直接返回该目标的 response（被拦截返回拦截文本，处理失败返回 Error 文本）；
    - 若 `delivered == 0` 且多个目标：优先返回第一个处理失败的 Error 文本；没有处理失败时返回第一个拦截文本。
  - 每个被拦截/失败目标的细节仍按现状向 stderr 报告一次。

### 1.5 测试清单（P1）

在 `tests/test_events.py` 增加/修改：

1. `send_instruction` 单目标与多播目标含 self：返回 Error，无投递。
2. notify 含 self：before → handler_error、原指令不执行；after → 该 notify 被丢弃、其他 notify 照常投递。
3. after 测试改用非 self 目标后，断言 active 与 ready 不重复（目标为休眠对话时 ready，source 保持 active）。
4. before 处理器失败：`_turn_events` 含 `action="handler_error"` 摘要；monitor 帧 lmu.events 含 error；capsys 中错误只出现一次。
5. 两个监听器：第一个成功返回 notify，第二个抛错；断言原指令不执行、工具响应为 handler_error、第一个 notify 已投递。
6. 多播 send_instruction：三个目标分别 execute / replace / handler_error；断言 execute 目标收到改写后内容、响应文案含 1/1/1、其余目标队列状态正确。
7. 之前缺失的覆盖：
   - `create_sub` after 通知；
   - `memory_make` before replace（不创建节点）与 execute（正常创建）；
   - `edit_metadata` before replace（不改 ctrl）与 value 改写；
   - `send_instruction` before 改写 content 后目标收到改写内容。
8. 所有改动后全量 `pytest tests/` 通过，并更新 README/README_EN 测试数。

---

## 二、P2 文档与仓库卫生

### 2.1 docs/programming-guide.md

- 「记住返回通道不是默认的」一节：删除「注册事件监听（设计中，未实现）」「信号事件（设计中，未实现）」；改为事件模型 v1 的 `listen_event` + notify 描述，并链接 `docs/runtime.md#事件模型-v1`。
- 增加或并入「用 listen_event 表达等待」小节：内存 owner、before replace、after 通配、notify 唤醒、禁止自我投递、before 事件不保证结果发生。

### 2.2 AGENTS.md

修正与当前工程脱节的内容：
- 删除服务调用分类法中的 `call_service` / `transfer_service` / `register_service` 描述，替换为当前 9 指令与简单投递/事件模型。
- 「运行」节删除不存在的 `simple_main.py`、`web/server.py`、`scripts/run_dual_terminal.sh`、`config.json`，改为当前真实入口：`python main.py`、`python -m avm.image <image.json>`、`pytest tests/`。
- 保留项目宗旨、计划方法论、好计划标准等原有内容；只改事实性过时部分，不重写全文。

### 2.3 avm/__init__.py

- docstring 改为与项目定位一致（如 `"""AVM — 以 LLM 推理为原始计算单元的运行时"""`）。
- 导出补齐：`ListenEventInstruction`、`EventSystem`、`ListenersDevice`（以及经评估需要给外部使用的 `EVENT_TYPES`）。

### 2.4 docs/record 版本控制

- 修改 `.gitignore`：把 `docs/record/*` 替换为只忽略大二进制：
  ```
  docs/record/*.png
  docs/record/*.jpg
  ```
- 将已存在的文字纪要加入 git（实施模型按仓库实际权限执行，未授权提交则只改 .gitignore 并列出待加入文件）：
  - 5-6img2md.md、7img2md.md、8.md、9.md、10.md、11.md、12.md、13.md、author-notes.md、20260618160046-*.txt。
- 注意 2.jpg 已被 git 跟踪的历史文件不受影响。

### 2.5 images/out

- 这些是运行产物且已过时（12 工具/ICC 时代）。执行：
  ```
  git rm --cached images/out/mem.json images/out/tester.conv.json images/out/tester_mem.json
  ```
- `.gitignore` 增加 `images/out/`。
- 不手工伪造新快照；下次真实运行 tester 会重新生成（生成物仍不入库）。

### 2.6 计划文件状态回填

- `docs/Agents/plan-remove-service-tools-2026-08-13.md`：顶部状态由「提案（待审批）」改为「已实施」，标注实施记录见同文件。
- `docs/Agents/plan-event-v1-2026-08-15.md`、`docs/Agents/audit-event-v1-2026-08-15.md`：本收尾计划实施完成后追加实施记录（日期、改动文件、最终测试数、验证结果）。
- 若发现其它 `docs/Agents/*.md` 状态与实际不符，一并在实施记录中修正，但不得改写历史记录内容。

### 2.7 提交

实施完成、用户确认后，建议按逻辑分批提交：
1. P1 事件语义修复 + 测试；
2. P2 文档与仓库卫生；
3. 不要在一个提交里混入无关改动。

---

## 三、验证清单

```bash
python -m pytest tests/ -q
python -m compileall -q avm
python - <<'PY'
from avm.image import load_image
for p in ["images/tester.json", "images/demo.json", "images/calc_driver.json"]:
    load_image(p)
print("images ok")
PY
grep -RIn "设计中，未实现" docs/programming-guide.md
grep -n "register_service\|call_service\|transfer_service" AGENTS.md
git status --short
```

验收标准：
- 全量测试通过（README 测试数同步）；
- 三镜像加载正常；
- `docs/programming-guide.md` 无「事件监听（设计中，未实现）」残留；
- `AGENTS.md` 无已删除服务工具与不存在文件引用；
- `git status` 中 `images/out/` 不再出现在 tracked 变更（仅 working tree 删除缓存，文件本身可保留或删除）；
- `docs/record/*.md` 文字纪要成为可跟踪文件（若已获提交授权）。

## 四、影响

- `send_instruction` 增加自我投递限制：此前允许自我投递的对话会收到 Error，行为变化按用户定夺执行。
- before 事件失败从异常传播改为事件结果返回：调用方看到 handler_error 工具响应，已收集 notify 保留投递，monitor 记录失败。
- 多播部分投递成为明确语义。
- 文档与仓库卫生问题闭环；OS 计划不再被 AGENTS.md 的过时事实干扰。
- 不改变：事件类型表、owner 模型、处理器无状态、注册顺序优先、after fail-open、设备事件入口。
- 不在本计划：处理器超时/预算、owner 约束写入/edit_metadata、str 叶子 owner、OS 内核态等，仍按 P3 另立计划。
- **新增已知问题（不阻塞本收尾）**：挂载点前后路径语义分裂（设备内路径 vs 内存树路径）见 docs/record/14.md。P0 的 owner 过渡规则与设备事件权限需要在 OS 计划前单独定夺。
- **待用户定夺**：用户提出「设备的事件由设备发布」（docs/record/14.md 第七节）。若确认，本计划增加一项——对话的 `listen_event` 拒绝设备内路径；`emit_device_event` 作为 v1 唯一设备事件发布入口；删除「设备路径看最近数据祖先 owner」的过渡规则并改写相关测试。


---

## 实施记录（2026-08-15）

- 完成 P1 语义修复：禁止自我投递、before 失败改为 `handler_error` 事件结果、保留已收集 notify、多播 `send_instruction` 部分投递、monitor 记录 `error` / `handler_errors`、stderr 去重。
- 完成 P2 仓库/文档卫生：更新 `docs/programming-guide.md`、`AGENTS.md`、`avm/__init__.py`、`.gitignore`；`images/out/` 从跟踪移除；计划状态回填。
- 验证：`pytest tests/` 266 个全部通过；三镜像加载正常；`compileall` 通过。

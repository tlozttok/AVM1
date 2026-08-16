# 计划（待实施）：事件发布者分工——Core 发布请求，Memory/设备发布事实

**时间**: 2026-08-15
**状态**: 已实施（279 个测试通过；实施记录见文件末尾）
**前置**: `plan-event-v1-cleanup-2026-08-15.md` 已实施；当前基线 266 测试通过。
**背景**: 事件模型 v1 与收尾计划已实施。剩余结构问题是挂载点前后路径语义分裂（docs/record/14.md）。用户确定：事件按「是否已经到达执行组件」分发布者——before 请求事件由 Core 发布；after 事实事件由实际执行组件（Memory / 设备 / Core）发布。
**起因**: 用户对五个设计问题定夺完毕，要求形成实施计划。

## 一、目标

1. 把事件 source 从「对话发布」改为「执行组件发布」。
2. 所有 9 种工具调用成功时产生 after 事实事件；5 种可拦截工具保留 before 请求事件。
3. `listen_event` 增加 `phase=before/after/all`，缺省 `all`。
4. 设备路径在注册监听时由 `authorize_listener` 授权；权限检查不触发 `resolve_path`。
5. 设备路径的 after 事件通过 `device.after_event` 询问，设备只返回纯 payload，Core 统一加事件字段。
6. 不加入设备自有 event_handler。

## 二、事件 source 与 phase 规则（最终规格）

### 2.1 before 事件（仅 5 种可拦截工具）

- event_type：memory_read / memory_write / memory_make / edit_metadata / send_instruction
- phase：before
- source：`{"kind": "core", "initiator_cid": <发起对话 cid>}`
- 语义：操作被请求，不保证到达执行组件；可被 replace / handler_error 阻断。
- 不产生 before 的类型：create_cmd / create_sub / close_conversation / listen_event。

### 2.2 after 事件（9 种工具，成功时）

| event_type | 何时产生 | source | target | 基础 payload |
|---|---|---|---|---|
| memory_read | 读取成功返回内容后 | target 为 memory kind：`{"kind":"memory"}`；device kind：`{"kind":"device","path":"$MEM.<mount>"}` | 原 ref | `{"ref": ref, "actor_cid": cid, "result": <返回内容>}` |
| memory_write | 写入成功后 | 同上 | 原 ref | `{"ref": ref, "content": 最终写入内容, "actor_cid": cid}` |
| memory_make | 创建成功后 | `{"kind":"memory"}` | 新建完整路径 | `{"ref": 父 ref, "key": key, "mem_type": mem_type, "path": 新路径, "actor_cid": cid}` |
| edit_metadata | 操作成功后 | `{"kind":"memory"}` | 原 ref | `{"ref": ref, "type": op, "key": key, "value": value, "actor_cid": cid}` |
| send_instruction | 每个目标成功投递后各一条 | `{"kind":"core"}` | `"cid:<目标>"` | `{"cid": 目标 cid, "content": 最终内容, "actor_cid": 发送者 cid}` |
| create_cmd | 创建成功后 | `{"kind":"core"}` | `"cid:<新 cid>"` | `{"cid": 新 cid, "actor_cid": 发起者 cid}` |
| create_sub | 创建并调度成功后 | `{"kind":"core"}` | `"cid:<新 cid>"` | `{"cid": 新 cid, "actor_cid": 发起者 cid}` |
| close_conversation | 关闭成功后 | `{"kind":"core"}` | `"cid:<被关闭 cid>"` | `{"cid": 被关闭 cid, "actor_cid": 发起者 cid}` |
| listen_event | listen/modify/cancel 成功后 | `{"kind":"core"}` | `"listener:<id>"` | `{"listener_id": id, "action": action, "actor_cid": 发起者 cid}` |

规则：
- 失败、被 replace、handler_error 的操作不产生 after 事件。
- send_instruction 的 before 是每个目标一条；after 只给实际投递成功的目标，拦截/处理失败的目标无 after。
- after 事件的 action 固定为 `execute`，处理器不能 replace；处理器只可返回 notify（现有 after fail-open 语义不变）。

### 2.3 source 兼容性变更

- 删除 `{"kind":"conversation","cid":n}` 作为事件 source。
- 对话身份只出现在：
  - before 事件的 `source.initiator_cid`；
  - after 事件的 `payload.actor_cid`。
- monitor 摘要直接保存新 source 结构，不额外兼容旧结构（旧帧不迁移）。

## 三、listen_event 的 phase

- 工具参数增加 `phase`：enum `["before","after","all"]`，可选，缺省 `"all"`。
- 注册记录增加 `phase` 字段；`$MEM.listeners` 每行显示 phase。
- 匹配规则：`listener.event_type == event.event_type` 且 `listener.phase in ("all", event.phase)` 且 target 匹配（精确或 `*`）。
- 校验：
  - `create_cmd / create_sub / close_conversation / listen_event` 不允许 `phase="before"`（这些类型只有 after），给 Error；
  - 其余类型 before / after / all 均可。
- 权限沿用收尾后规则：
  - memory before/after：目标为 memory kind 时按 ctrl.owner；
  - device before/after：目标为 device kind 时调用 `authorize_listener`；
  - send_instruction before/after：仍只允许 `cid:<自己>`；
  - create/close/listen 的 after：允许 `*` 或精确 target。
- `authorize_listener` 当前参数只有 event_type / remainder / owner_cid，设备对 before 与 after 使用同一授权结论；如需按 phase 区分的策略，另立计划扩展参数。

## 四、设备接口与路径边界

### 4.1 split_path

`Memory.split_path(ref) -> ("memory", segments) | ("device", mount_key, remainder)`

- 段序列匹配 `_devices` 的 key；`box` 不匹配 `$MEM.box2`。
- remainder 对 Core 不透明；Core 只原样传给设备接口，不调用 `resolve_path`。
- `_owner_of` 只接受 memory kind；device kind 直接 None。

### 4.2 MemoryDevice 新接口

```python
class MemoryDevice:
    mount_path: Optional[str] = None   # 由 image._mount_devices 设置

    def authorize_listener(self, event_type: str, remainder: list, owner_cid: int) -> bool:
        """Core 在注册监听时询问：是否允许该对话监听此设备内路径。默认 False。"""
        return False

    def after_event(self, event_type: str, remainder: list, result: dict) -> Optional[dict]:
        """设备操作成功后 Core 询问是否有事实事件。返回纯 payload dict；None 表示无事件。"""
        return None
```

- 默认拒绝监听；默认无 after 事件。
- `after_event` 返回的 dict 只作为 payload 数据；Core 统一加 `event_id / phase / source / target`，并强制合并 `actor_cid` 与 `ref`（Core 值优先），防止设备 payload 覆盖。
- 权限检查期间禁止调用设备 `resolve_path`；测试必须断言这一点。

### 4.3 Core.emit_device_event

- 保留为设备异步自发事件的发布入口，现有 v1 语义：phase=before、source=device、只支持 memory_write、target 必须属于该设备 mount。
- 边界校验改用 `split_path`，不再使用字符串 startswith。
- 设备路径的 after 事件不由 `emit_device_event` 产生；它走 4.2 的同步询问接口。
- 设备异步 after 事实事件（如定时器到期）不在本计划，留 TODO/issue。

## 五、实施步骤

1. **avm/memory.py**
   - 新增 `split_path(ref)`，导出；不改 `resolve_path` 行为。
   - 补充测试确认与现有 `resolve_path` 前缀循环一致。

2. **avm/memory_device.py**
   - `MemoryDevice` 增加 `mount_path`、`authorize_listener`、`after_event`，docstring 写明权限边界。
   - 不增加 `event_handler`。

3. **avm/image.py**
   - `_mount_devices` 在 `attach_core` 前设置 `device.mount_path = path`。

4. **avm/events.py**
   - `build_event`：before + conv → source `{"kind":"core","initiator_cid":cid}`；after 调用方必须显式给 source。
   - listener 记录与 `_match` 增加 phase。
   - `_merge_output` 自我投递判定改为：
     - before：`source.initiator_cid`；
     - after：`payload.actor_cid`（无则不做 self 检查）；
     - 判定到 self 时沿用收尾计划：before 为处理器错误，after 丢弃该条并记 handler_errors。
   - 新增 `publish_after_event(event_type, conv, target, payload, source_kind/device)` 辅助函数，统一构造 after 事件。
   - `_check_before_permission` / 注册流程按第四节接入 `split_path` 与 `authorize_listener`；rename 或扩展为同时处理 before/after 权限（实现者可用 `_check_target_permission`）。
   - `_owner_of` 对 device kind 返回 None。

5. **avm/core.py**
   - 5 条 before 指令：before 事件构造改用新 source；成功后按 2.2 构造 after 事件并 `dispatch + deliver_notifies`。
     - memory_read：先有 content 结果，再发 after（source 按 split_path 目标 kind 决定）。
     - memory_write：成功写后用最终 content 发 after。
     - memory_make：成功创建后发 after。
     - edit_metadata：成功执行后发 after（get/set/del 都算成功；非法参数无 after）。
     - send_instruction：对每个 `delivered` 目标发 after；blocked / failed 不发。
   - create_cmd / create_sub / close_conversation / listen_event：现有 after 事件改为新 source，payload 按 2.2。
   - `ListenEventInstruction`：增加 phase 参数与校验。
   - `emit_device_event`：改用 `split_path` 边界；source 保持 device。
   - 所有 before 指令中 `build_event` 调用去掉旧 conversation source 假设，使用新签名。

6. **avm/info_devices.py**
   - `ListenersDevice` 输出增加 phase。

7. **avm/monitor.py**
   - `_summarize_event` 保持 source 原样；无需结构改动（source 变为新 dict）。
   - `$MEM.monitor` 事件计数自然包含 before+after。

8. **文档**
   - `docs/runtime.md`：source 模型、after 事件表、phase 参数、authorize_listener / after_event / split_path、emit_device_event 保留语义。
   - `CONTEXT.md`：事件术语更新「Core 发布请求事件，Memory/设备发布事实事件」；设备术语补 mount 边界。
   - `README.md` / `README_EN.md`：当前状态补「事件发布者分工 / after 事实事件」；测试数同步。
   - `images/tester.json`：listen_event 条目增加 phase；概述补 after 事件与 source 规则。
   - `docs/record/14.md` 追加实施记录。

9. **测试**
   - 更新现有 source 相关断言（tests/test_events.py 中 device source 断言保留，其余按新 source）。
   - 新增：
     a. `split_path` 段序列边界（box / box2 / box.dev / box.dev.sub）。
     b. memory after 事件：read/write/make/edit_metadata 成功后 phase=after、source=memory、payload 含 actor_cid；replace/handler_error 无 after。
     c. device after 事件：dummy 设备 `after_event` 返回 payload；Core 包装 source=device、强制 actor_cid/ref；返回 None 无 after。
     d. 设备注册授权：`authorize_listener` 被调用、False 拒绝、True 允许、全程不调用 `resolve_path`。
     e. phase 匹配：before-only、after-only、all 三种监听器的触发次数与事件相位。
     f. after-only 类型拒绝 phase=before。
     g. send_instruction 成功投递产生 after（source=core，target=cid），blocked/failed 不产生。
     h. 生命周期 after source=core。
     i. self notify 判定：before 用 initiator_cid；after 用 payload.actor_cid。
     j. `emit_device_event` 仍为 before/source=device；用 split_path 拒绝 `$MEM.box2`。
   - 全量 pytest；README 测试数最终同步。

## 六、验证

```bash
python -m pytest tests/ -q
python -m compileall -q avm
python - <<'PY'
from avm.image import load_image
for p in ["images/tester.json", "images/demo.json", "images/calc_driver.json"]:
    load_image(p)
print("images ok")
PY
grep -n '"kind": "conversation"' avm tests || true
grep -n "startswith(\"\$MEM\." avm/core.py || true
grep -n "authorize_listener\|after_event\|split_path" avm/memory_device.py avm/memory.py avm/events.py avm/core.py
```

验收：
- 全量测试通过；README 测试数同步；
- 没有旧 `kind: conversation` source 残留；
- `emit_device_event` 无字符串 startswith 边界；
- 设备 after 事件由 Core 统一加 event_id/phase/source/target；
- 设备授权检查不触发 resolve_path。

## 七、影响

- **兼容性变化**：事件 source 格式改变；`listen_event` 缺省 phase=all 会让既有监听器同时收到 before 与 after，既有 handler 需要在 after 阶段不产生副作用（after 的 replace 被忽略，notify 仍会投递）。
- **事件量**：成功指令从 1 条事件变为最多 2 条（before+after），monitor 帧与 `--frames` 数量翻倍；这是发布者分工的代价，不单独做事件合并。
- **内存读取结果进入 after payload**：只发给该路径的已授权监听器；不被授权者看不到。结果可能较大，先不截断（monitor 仍截断）。
- **设备语义**：设备自己决定「谁可以监听」和「成功操作是否产生事实事件」；Core 不再猜设备路径。
- **send_instruction after** 是用户标注的暂时结论，实施时在代码注释中标注 `TODO(issue): send_instruction 事实事件的语义需进一步研究`。
- **不在本计划**：设备异步 after 事件、设备自有 event_handler、authorize_listener 按 phase 区分的策略、事件条件（ECA 的 C）、事件持久化/重放。

## 八、实施者须知

- 严格按本计划实施；发现计划与代码冲突或缺少必要设计时，停止并以 issue 回报。
- 先跑通现有 266 个测试作为基线；本计划完成后测试数会变化，以实际全量通过为准。
- 不要改动与发布者分工无关的调度、检查点、PLM、沙箱逻辑。


---

## 实施记录（2026-08-15）

- 完成发布者分工：before 请求事件由 Core 发布，after 事实事件由 Memory/设备/Core 发布。
- `listen_event` 增加 phase=before/after/all；`$MEM.listeners` 显示 phase。
- 新增 `Memory.split_path`、`MemoryDevice.mount_path/authorize_listener/after_event`，Core 统一包装设备 after 事件。
- 验证：`pytest tests/` 279 个全部通过；三镜像加载正常；`compileall` 通过。

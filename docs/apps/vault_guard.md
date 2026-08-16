# AVM 应用：vault_guard（金库守卫）

**目标**: 用「守卫打标并注册监听 → 工人写金库 → before 拦截/改写 → after 通知唤醒」的链条，验证事件模型 v1 的 `listen_event`、`ctrl.owner` 权限、`edit_metadata` 运行期写 ctrl、waterfall 处理器、`replace` 短路、payload 改写与 `notify` 内部投递。

**不验证**: 事件条件（ECA 的 C）、事件持久化/重放、定时器（设计中）。

---

## 1. 架构

```
guardian(init, cid 0, $MEM.vault owner)
  |
  |-- listen_event(memory_write, $MEM.vault.deposit, handler=deposit_pipe, phase=all)
  |-- listen_event(memory_write, $MEM.vault.audit,  handler=audit_tail,  phase=after)
  |-- create_cmd($MEM.programs.worker)
  v
worker(LLM, cid N)
  |
  |-- memory_write($MEM.vault.deposit, text)
  |      before: deposit_pipe 检查禁止词
  |              -> 含 SECRET: action=replace，原写入不执行
  |              -> 否则:       payload.content 改写为 "SAFE:" + text
  |      after（仅写入成功时）: deposit_pipe 返回 notify cid=0
  |
  |-- memory_write($MEM.vault.audit, "TASK <id> 尝试写入 deposit")
  |      after: audit_tail 返回 notify cid=0
  |
  +-- send_instruction(cid=0, content="RESULT <id> done")
```

组件：

| 组件 | 类型 | 职责 |
|---|---|---|
| `$MEM.programs.guardian` | settingup | 入口守卫：注册监听、创建 worker、读输入、输出审计 |
| `$MEM.programs.worker` | settingup | 尝试写金库并回执 |
| `$MEM.handlers.deposit_pipe` | python | `handle(event)`；before 拦截/改写，after 通知 |
| `$MEM.handlers.audit_tail` | python | after 审计通知 |
| `$MEM.vault` | dict, ctrl.owner="0" | `deposit` 与 `audit` 两个 str |
| `$MEM.state.worker_cid` | str | worker cid |

## 2. LLM 式算法/控制流

### 2.1 guardian

```
初始化（首次激活）:
  edit_metadata(ref="$MEM.vault", type="set", key="policy", value="guard-redact")
  listen_event(action=listen, event_type=memory_write,
               target="$MEM.vault.deposit",
               handler_ref="$MEM.handlers.deposit_pipe", phase=all)
  listen_event(action=listen, event_type=memory_write,
               target="$MEM.vault.audit",
               handler_ref="$MEM.handlers.audit_tail", phase=after)
  worker_cid = create_cmd($MEM.programs.worker, $MEM.para.llm)
  memory_write("$MEM.state.worker_cid", str(worker_cid))

主循环:
  input = memory_read("$MEM.inputs.-1")
  if input == "exit":
      close_conversation(worker_cid); return "vault closed"
  send_instruction(cid=worker_cid,
      content="TASK <n>: <input> | 完成后 send_instruction(cid=0, content=RESULT <n> done)")
  return "waiting"                       # 无工具调用 => 休眠

被唤醒时:
  if 收到 AUDIT 消息:
      memory_read("$MEM.vault.audit"); memory_write("$MEM.outputs.-1", "AUDIT: " + 内容)
  if 收到 RESULT 消息:
      memory_read("$MEM.vault.deposit"); memory_write("$MEM.outputs.-1", "FINAL: " + 内容)
      goto 主循环
```

要点：
- 守卫注册的监听器缓存的 handler code 在注册时读取；镜像中的 `vault` 节点预置 `ctrl.owner="0"`，因此 cid0 能注册 deposit/audit 的 before/after 监听。
- before 处理器是同步 guard：它不产生 LLM 调用，不打断单活跃对话。
- after 处理器只贡献 `notify`；`notify` 的目标是 cid0，actor 是 worker，所以不触发「禁止自我投递」。

### 2.2 worker

```
收到 TASK <id>: <text>:
  memory_write("$MEM.vault.deposit", text)
  memory_write("$MEM.vault.audit", "TASK <id> 尝试写入 deposit")
  send_instruction(cid=0, content="RESULT <id> done")
  return "done"
```

worker 不知道守卫存在，也不知道 handler 内容。被拦截时它收到的第一个工具响应是守卫的 response；它不判断成败，继续执行剩余步骤——这正是 before 事件「只表示放行/改写，不保证结果发生」在应用层的体现。

### 2.3 处理器伪代码

`deposit_pipe`：

```python
def handle(event):
    content = event["payload"].get("content", "")
    if event["phase"] == "before":
        if "SECRET" in content.upper():
            return {"action": "replace", "response": "拦截：内容包含禁止词 SECRET"}
        return {"payload": {"content": "SAFE:" + content}}
    if event["phase"] == "after":
        return {"notify": [{"cid": 0, "content": "AUDIT deposit: " + content}]}
    return {}
```

`audit_tail`：

```python
def handle(event):
    return {"notify": [{"cid": 0, "content": "AUDIT log: " + event["payload"].get("content", "")}]}
```

语义：
- `phase=all` 的监听器会先后看到同一工具的 before 与 after 事件；`phase=after` 只看到 after。
- before 返回 `replace` 后原 `memory_write` 不执行，也不会产生 after 事件。
- 普通内容被改写成 `SAFE:` 前缀后写入；被禁止内容保持 `EMPTY` 不变。

## 3. 事件与状态轨迹

1. cid0 活跃：注册两个监听器（成功产生 after `listen_event` 事件）、创建 cid1。
2. cid0 投递 TASK 后休眠。
3. cid1 第一次 memory_write：Core 构造 before 事件 → deposit_pipe 处理：
   - 禁止词：action=replace，工具响应为拦截文本，`$MEM.vault.deposit` 不变，无 after；
   - 普通词：action=execute，payload.content 被改写，写入 `SAFE:...`，Core 发布 after，handler notify cid0。
4. cid1 第二次 memory_write：after audit_tail notify cid0。
5. cid1 send_instruction 再次唤醒 cid0，然后休眠。
6. cid0 被唤醒，处理 AUDIT 与 RESULT 消息后写 `$MEM.outputs`。

## 4. 验证

- `tests/test_app_images.py::TestVaultGuardImage`：
  1. 镜像严格加载；`$MEM.vault.owner == "0"`；两个 handler 是可加载 python 节点；
  2. 脚本化 MockLMU 跑含禁止词任务，断言 deposit 未写入、worker 收到拦截响应、audit 写入、guardian 被 notify 唤醒、outputs 有审计；
  3. 普通词任务走 before 改写，断言 `SAFE:` 前缀。
- 手工：`python main.py images/apps/vault_guard/vault_guard.json`，输入 `hello`、`SECRET`、`exit`。

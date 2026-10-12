# AVM 应用：courier_office（传令站）

**目标**: 用一个「调度员 → 工人 → 计算器」的三段式异步外包，验证 `create_cmd` 的松耦合语义、`send_instruction` 的双向投递、`create_sub` 的紧耦合返回、`plm.simple` 的确定性计算、`memory_make` 的运行期建目录、`close_conversation` 的内核级关闭，以及自定义内存设备作为结果记录板。

**不验证**: 内核态/用户态、workspace（设计中）；链接设备（已实现，但没有镜像使用它）；不把业务逻辑写进设备。

---

## 1. 架构

```
操作者 --($MEM.inputs)--> dispatcher(LLM, cid 0)
                            |
                            | 初始化：create_cmd(system_ref=$MEM.programs.worker,
                            |                     para_ref=$MEM.para.llm)
                            v
                        worker(LLM, cid N，创建后休眠)
                            |
                            | 收到 JOB 后：
                            |   memory_write($MEM.tasks.expr, expression)
                            |   create_sub(system_ref=$MEM.programs.calc,
                            |              user_ref=$MEM.tasks.expr,
                            |              para_ref=$MEM.para.simple)
                            v
                        calc(Python 对话, plm.simple)
                            |
                            | 紧耦合返回：结果自动写回 worker 的 tool response
                            v
worker --memory_write($MEM.board.append)--> BoardDevice（追加记录）
worker --send_instruction(cid=0)-----> dispatcher
dispatcher --memory_read($MEM.board.latest)--> $MEM.outputs
```

组件：

| 组件 | 类型 | 职责 |
|---|---|---|
| `$MEM.programs.dispatcher` | settingup | 入口对话：初始化、发任务、接收结果、输出 |
| `$MEM.programs.worker` | settingup | 单任务工人：解析 JOB、外包计算、记录、回执 |
| `$MEM.programs.calc` | python | plm.simple 计算器（content 可为空） |
| `$MEM.para.llm` | para | dispatcher/worker 的 LLM 参数 |
| `$MEM.para.simple` | para | 计算器 Python 对话参数，model=plm.simple |
| `$MEM.tasks` | dict | `current`（任务原文）与 `expr`（传给计算器的一行表达式） |
| `$MEM.state.worker_cid` | str | worker 的 cid（跨交互保留） |
| `$MEM.board` | device | `BoardDevice`：`.append` 写追加记录，`.latest` 读最新记录，数字索引（含 -1）是普通列表语义 |
| `$MEM.inputs` / `$MEM.outputs` | device | 操作者输入/输出 |

## 2. LLM 式算法/控制流

### 2.1 dispatcher 状态机

```
状态 等待输入:
  input = memory_read("$MEM.inputs.-1")
  if input == "exit":
      close_conversation(worker_cid)  # 如已创建
      return "office closed"          # 无工具调用 => 交互结束
  ensure worker:
      if worker_cid == "": cid = create_cmd(...); worker_cid = str(cid)
  ensure reports:
      if memory_read("$MEM.reports") 返回 Error:
          memory_make("$MEM", "reports", "dict")
  send_instruction(cid=worker_cid, content="JOB 1: <input> | 完成后 send_instruction(cid=0, content=RESULT 1 <单行结果>)")
  return "waiting"                    # 发送后主动休眠，不轮询

状态 处理结果（被 RESULT 唤醒）:
  record = memory_read("$MEM.board.latest")
  memory_write("$MEM.outputs.-1", record)
  goto 等待输入
```

要点：
- `create_cmd` 后父对话保持活跃、子对话休眠；所以 dispatcher 在创建后继续执行到投递步骤。
- `send_instruction` 只确认送达。dispatcher 发送后必须用「无工具调用的文本」结束交互，否则会一直占住活跃态。
- worker 的返回通道不是创建关系自动给的，而是 worker 自己调用 `send_instruction(cid=0, ...)`。

### 2.2 worker 状态机

```
状态 等待任务:
  收到 JOB <id>: <expression>
  memory_write("$MEM.tasks.expr", expression)
  result = create_sub(system_ref="$MEM.programs.calc",
                      user_ref="$MEM.tasks.expr",
                      para_ref="$MEM.para.simple")   # 工具响应即 calc 的最终输出
  memory_write("$MEM.reports.<id>", "JOB <id> <expression> = <result>")
  memory_write("$MEM.board.append", "JOB <id> <expression> = <result>")
  send_instruction(cid=0, content="RESULT <id> <expression> = <result>")
  return "done"
```

要点：
- `create_sub` 是紧耦合：worker 休眠，calc 立即执行；calc 无工具调用结束时，结果自动写入 worker 的 tool response 并唤醒 worker。
- worker 只做「外包 + 记录 + 回执」，不直接接触操作者，也不读 `$MEM.inputs`。
- 表达式限制为单行 Python 表达式；plm.simple 的沙箱会拒绝 import、双下划线等逃逸。

### 2.3 调度轨迹

1. cid0 活跃：create_cmd 后仍活跃；send_instruction 后 worker 进 ready 队尾。
2. cid0 无工具返回 → dormant；调度器选 cid1。
3. cid1 活跃：memory_write 后 create_sub 使 cid1 休眠、cid2 前插 ready 并立即执行。
4. cid2（Python）执行一次求值，无工具调用 → `_on_conversation_finished` 自动写回并唤醒 cid1（同一调用链）。
5. cid1 继续：写 board、send_instruction 唤醒 cid0；随后无工具返回 → dormant。
6. cid0 被唤醒：读 board、写 outputs；无工具返回 → dormant，运行结束。

## 3. 自定义设备 BoardDevice

文件 `devices/board.py`，继承 `MetaListDevice`。路径语义是显式的：

| 路径 | 语义 |
|---|---|
| `memory_read("$MEM.board.latest")` | 读最新一条记录；空板时报越界错误 |
| `memory_write("$MEM.board.append", record)` | 追加一条 str 记录 |
| `memory_read/write("$MEM.board.<数字索引>")` | 普通列表语义；`-1` 就是最后一条，读写都按索引操作，不追加 |

实现要点：

- `__getitem__("latest")`：返回队尾元素；
- `__setitem__("append", value)`：只接受 str，追加到队尾；
- `__getitem__/__setitem__` 对数字索引委托给 `MetaListDevice`；
- `set_value(value)`：str 追加，list 整体替换；
- `to_llm_string()`：返回 `list_len` 与最近记录摘要；
- `after_event("memory_write", remainder, result)`：返回 `board_len` / `latest`，供 Core 发布设备 after 事件。

`memory_write("$MEM.board.append", content)` 会被 `Memory.set_by_path` 路由到设备前缀 `board` 的 `__setitem__("append", content)`；`-1` 不再承担追加语义。

## 4. 消息协议

| 消息 | 格式 | 方向 |
|---|---|---|
| JOB | `JOB <id>: <单行表达式>` | dispatcher → worker |
| RESULT | `RESULT <id> <表达式> = <结果>` | worker → dispatcher |
| 完工记录 | `JOB <id> <表达式> = <结果>` | worker → `$MEM.board.append` |

## 5. 验证

- `tests/test_app_images.py::TestCourierOfficeImage`：
  1. 镜像严格加载，BoardDevice 挂载且 `$MEM.board` 不写入数据树；
  2. 设备 `.append` 追加、`.latest` 读取与数字索引（含 -1）普通列表语义；
  3. 脚本化 MockLMU 跑完整流程，断言 board / outputs / 调度终态。
- 手工：`python main.py images/apps/courier_office/courier_office.json`，输入 `3 + 5 * 2`、`exit`。

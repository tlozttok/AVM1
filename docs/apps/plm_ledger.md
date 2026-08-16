# AVM 应用：plm_ledger（账本服务）

**目标**: 验证 AVM 的 Python 对话执行器边界——一个 `ctrl.type="python"` 节点不一定是 plm.simple / plm.python，而可以是镜像外注册的任何 OpenAI 兼容服务器。本应用附带自定义 `LedgerPLM`，通过 `memory_write` 工具调用自行持久化账本，并借 `create_sub` 的紧耦合返回把结果交回 LLM 柜员。

**不验证**: 网络服务、PLM 与 LLM 的运行时热切换（Core 已规定按 `para.model` 在 `PLM_REGISTRY` 查找，本应用只使用该既有机制）。

---

## 1. 架构

```
操作者 --($MEM.inputs)--> teller(LLM, cid 0)
                            |
                            | memory_write($MEM.request, command)
                            | create_sub(system_ref=$MEM.programs.ledger,
                            |            user_ref=$MEM.request,
                            |            para_ref=$MEM.para.ledger)
                            v
                        LedgerPLM（Python 对话，model=plm.ledger）
                          system: {"path": "$MEM.ledger.state", "initial": {}}
                          user:   PUT a 1
                          |
                          | 第一步输出：content="", tool_calls=[memory_write($MEM.ledger.state, {"a":"1"})]
                          v
                        Core 执行 memory_write，返回 "Success set: ..."
                          |
                          | 第二步输入新增 tool 消息
                          v
                        LedgerPLM 输出："OK PUT a=1"
                          |
                          | 亚对话完成：结果自动写回 teller 的 tool response
                          v
teller --memory_write($MEM.outputs.-1)--> 操作者
```

组件：

| 组件 | 类型 | 职责 |
|---|---|---|
| `$MEM.programs.teller` | settingup | 入口柜员：读命令、写 request、create_sub、输出结果 |
| `$MEM.programs.ledger` | python | LedgerPLM 的系统配置（JSON 文本） |
| `$MEM.para.ledger` | para | `model="plm.ledger"`，选择自定义 PLM |
| `$MEM.para.llm` | para | teller 的 LLM 参数 |
| `$MEM.ledger.state` | str | 账本 JSON 快照 |
| `$MEM.request` | str | 传给 LedgerPLM 的单条命令 |
| `plm_ledger_server.py` | 宿主 Python | `LedgerPLM(PLMServer)` |
| `run_ledger.py` | 宿主 Python | 注册 PLM 后加载并运行镜像 |

## 2. LLM 式算法/控制流

### 2.1 teller

```
循环:
  command = memory_read("$MEM.inputs.-1")
  if command == "exit":
      return "ledger closed"                 # 无工具调用 => 交互结束
  memory_write("$MEM.request", command)
  reply = create_sub(system_ref="$MEM.programs.ledger",
                     user_ref="$MEM.request",
                     para_ref="$MEM.para.ledger")   # 工具响应即 LedgerPLM 最终输出
  memory_write("$MEM.outputs.-1", reply)
  goto 循环
```

teller 不解析账本语义，不检查 `$MEM.ledger.state`；它只是把命令原样外包并转述结果。`create_sub` 保证紧耦合：teller 休眠，LedgerPLM 完成后自动写回并唤醒。

### 2.2 LedgerPLM 状态机

命令文法（user 消息是单行文本）：

| 命令 | 行为 | 是否持久化 |
|---|---|---|
| `PUT <key> <value>` | 更新内存态，发 `memory_write` 工具调用，工具返回后答复 `OK PUT key=value` | 是 |
| `GET <key>` | 直接答复值或 `NONE` | 否 |
| `DEL <key>` | 删除，持久化，答复 `OK DEL key` | 是 |
| `LIST` | 答复 JSON 状态 | 否 |
| `RESET` | 清空，持久化，答复 `OK RESET` | 是 |
| `SUM` | 答复所有数值 value 之和 | 否 |

内部状态机：

```
system 消息: 解析 JSON 配置，设置 path 与 initial
user 消息:
  GET/LIST/SUM -> make_response(text)                    # 无 tool_calls，亚对话立即完成
  PUT/DEL/RESET -> 保存 pending_reply，返回 tool_calls=[memory_write]
                   （tool_call id 由实例计数器生成，保证同一对话内唯一）
tool 消息:
  若 call_id 与 pending 匹配:
      content 以 Error 开头 -> make_response("ERROR: " + content)
      否则                   -> make_response(pending_reply)
  无 pending -> make_response("ERROR: unexpected tool message")
```

关键契约（与真实 LLM 端点一致）：
- 输入是完整 OpenAI messages 列表，输出是 `{"content", "tool_calls"}`；
- 一次工具调用必有且仅有一次工具响应；`PLMServer._call_records` 由基类在输出带 tool_calls 时自动记录；
- 只有最后一条新增消息的返回被保留，保证确定性；
- LedgerPLM 不知道 Core 的存在，只通过工具名与参数和 AVM 交互。

### 2.3 启动器 run_ledger.py

```python
from avm.python_server import PLM_REGISTRY
from plm_ledger_server import LedgerPLM
from avm.image import load_image

PLM_REGISTRY["plm.ledger"] = LedgerPLM
core = load_image("plm_ledger.json")
core.run()
```

`Core._get_executor` 对 Python 对话按 `para.model` 查 `PLM_REGISTRY`；注册发生在 `load_image` 之前即可。镜像本身没有 PLM 声明段，所以这一步必须在宿主完成（记录为 issue，见 docs/Agents/plan-app-suite-2026-08-16.md）。

## 3. 持久化协议

- 账本状态始终是 `dict`；持久化时序列化为 JSON 字符串写入配置指定的 ref。
- 默认 ref 为 `$MEM.ledger.state`，可由 system 配置覆盖。
- 如果 Core 的 `memory_write` 返回 `Error`，LedgerPLM 把该文本作为 `ERROR:` 答复返回，不假装成功。

## 4. 验证

- `tests/test_app_images.py::TestPlmLedger`：
  1. `LedgerPLM` 服务器级测试：`PUT` 产生 memory_write 工具调用，补 tool 消息后得到 `OK PUT`；
  2. 镜像集成测试：注册 `plm.ledger` 后，脚本化 MockLMU 驱动 teller 完成 `PUT a 1` 往返，断言 `$MEM.ledger.state` 持久化 JSON 与 outputs；
  3. 未注册 `plm.ledger` 时运行 Python 对话会报 `ValueError`（保证不静默回退 LLM）。
- 手工：`python images/apps/plm_ledger/run_ledger.py`，依次输入 `PUT a 1`、`GET a`、`LIST`、`exit`。

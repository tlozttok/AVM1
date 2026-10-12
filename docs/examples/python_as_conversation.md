# Python 程序的对话式嵌入

一个 Python 程序可以无损表示为"对话 + 工具调用"的格式。
以下是三个例子，展示同一段 Python 代码如何在对话格式下执行。

---

## 例 1：简单计算

**Python 原始代码：**
```python
x = 2 + 3
y = x * 10
print(y)
```

**对话格式：**
```
system:   你是一个 Python 解释器。收到代码后执行并返回结果。
          允许的工具：
          - memory_write(ref, content)  把值存入内存
          - memory_read(ref)            读取内存值

user:     x = 2 + 3
          y = x * 10
          print(y)

assistant: (tool_calls: memory_write("$MEM.result", "50"))

user:     执行完成
assistant: 50
```

同样的对话，**换个 Executor**（Python REPL）直接得到 `50` 而不用调 LLM。

---

## 例 2：文件处理

**Python 原始代码：**
```python
import json
with open("config.json") as f:
    cfg = json.load(f)
result = f"model={cfg['model']}, temp={cfg['temp']}"
print(result)
```

**对话格式：**
```
system:   你是文件处理器。内存结构：
           $MEM.config  → 指向文件系统设备
           $MEM.output  → 结果输出设备

user:     import json
          cfg_str = memory_read("$MEM.config")
          cfg = json.loads(cfg_str)
          result = f"model={cfg['model']}, temp={cfg['temp']}"
          memory_write("$MEM.output", result)

assistant: (tool_calls:
            memory_read("$MEM.config")       → '{"model":"gpt-4","temp":0.7}'
            memory_write("$MEM.output", "model=gpt-4, temp=0.7"))
```

---

## 例 3：多进程并发 — 子对话

**Python 原始代码（多线程版）：**
```python
from concurrent.futures import ThreadPoolExecutor

def task(url):
    import urllib.request
    return urllib.request.urlopen(url).read()

urls = ["http://a.com", "http://b.com"]
with ThreadPoolExecutor() as pool:
    results = list(pool.map(task, urls))
print(results)
```

**对话格式（每个子任务是一个子对话）：**
```
system:   你是并发协调器。可用 create_cmd 建子对话、send_instruction 投递任务。
          子对话创建后休眠，投递指令才运行；创建关系不是返回通道，
          所以在指令里要求它们完成后用 send_instruction 把结果发回给你。

user:     处理这 3 个 URL：url_a, url_b, url_c
          每个 URL 启动一个子对话，等全部完成后返回结果列表。

assistant: (tool_calls:
            create_cmd(system_ref="$MEM.agent.fetch") -> cid 1
            send_instruction(cid=1, content="抓取 url_a；完成后 send_instruction(cid=0,
                content='RESULT 1 <内容>')")
            create_cmd(system_ref="$MEM.agent.fetch") -> cid 2
            send_instruction(cid=2, content="抓取 url_b；完成后 send_instruction(cid=0,
                content='RESULT 2 <内容>')")
            create_cmd(system_ref="$MEM.agent.fetch") -> cid 3
            send_instruction(cid=3, content="抓取 url_c；完成后 send_instruction(cid=0,
                content='RESULT 3 <内容>')"))

assistant: （本轮不再输出工具调用 → 协调器交互结束进入休眠，让出执行权）

--- 调度器依次运行 cid 1 / 2 / 3 ---

# 子对话 1（$MEM.agent.fetch + 收到的指令）
assistant: (tool_calls: memory_read("$MEM.network"))   → 触发网络设备
assistant: <url_a 的内容>
assistant: (tool_calls: send_instruction(cid=0, content="RESULT 1 <url_a 的内容>"))

# 其余两个子对话同理；每次投递把协调器放进就绪队列末尾

--- 协调器被唤醒三次：每次只拿到已到的结果，数够三条才输出 ---

assistant: 处理结果：[url_a_result, url_b_result, url_c_result]
```

`ThreadPoolExecutor` → 调度器的就绪/休眠队列
`pool.submit(task, url)` → `create_cmd` 建对话 + `send_instruction` 投递任务
`task(url)` → 子对话
`future.result()` → 子对话用 `send_instruction` 投递回来的消息（创建关系不提供返回通道，父要结果就得在指令里要）

---

## 关键点

| Python 概念 | 对话格式中的对应 |
|------------|----------------|
| 模块导入/全局变量 | system prompt（setup） |
| 函数调用 | 单次 `user → assistant` |
| 副作用（I/O, 写文件） | `memory_read` / `memory_write` |
| 多线程/并发 | `create_cmd` 建子对话 + `send_instruction` 投递 + 调度器就绪队列 |
| 返回值/结果 | assistant 文本 或 `memory_write` |
| 异常处理 | assistant 返回错误文本 + tool_calls 写入错误信息 |

**对 Executor 的要求**：不管背后是 GPT-4 还是 Python REPL 还是本地脚本引擎，只要遵守同一套 tool 协议（`memory_read`/`memory_write`/`create_cmd`/`create_sub`），就是合法的对话执行器。

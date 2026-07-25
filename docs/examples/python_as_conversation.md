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
system:   你是并发协调器。可以启动子对话来并行处理任务。
          子对话启动后进入"休眠"，完成后将结果写回你的 batch。

user:     处理这 3 个 URL：url_a, url_b, url_c
          每个 URL 启动一个子对话，等全部完成后返回结果列表。

assistant: (tool_calls:
            create_cmd(system_ref="$MEM.agent.fetch", user_ref="url_a")
            create_cmd(system_ref="$MEM.agent.fetch", user_ref="url_b")
            create_cmd(system_ref="$MEM.agent.fetch", user_ref="url_c"))

--- 挂起，切换到子对话执行 ---

# 子对话 1（$MEM.agent.fetch + url_a）
assistant: (tool_calls:
            memory_read("$MEM.url"))         → "url_a"
            memory_read("$MEM.network"))     → 触发网络设备
assistant: <html>result from url_a</html>

# 子对话 2 类似，返回 url_b 的结果
# 子对话 3 类似，返回 url_c 的结果

--- 全部完成，唤醒父对话 ---

assistant: 处理结果：[url_a_result, url_b_result, url_c_result]
```

`ThreadPoolExecutor` → 调度器 `dormant/ready` 队列
`task(url)` → 子对话
`pool.map()` → `create_cmd` 批量创建
`results` → 父对话被回调唤醒后从 `user_batch` 读取

---

## 关键点

| Python 概念 | 对话格式中的对应 |
|------------|----------------|
| 模块导入/全局变量 | system prompt（setup） |
| 函数调用 | 单次 `user → assistant` |
| 副作用（I/O, 写文件） | `memory_read` / `memory_write` |
| 多线程/并发 | `create_cmd` + 子对话调度 |
| 返回值/结果 | assistant 文本 或 `memory_write` |
| 异常处理 | assistant 返回错误文本 + tool_calls 写入错误信息 |

**对 Executor 的要求**：不管背后是 GPT-4 还是 Python REPL 还是本地脚本引擎，只要遵守同一套 tool 协议（`memory_read`/`memory_write`/`create_cmd`/`create_sub`），就是合法的对话执行器。

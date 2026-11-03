# AVM — 以 LLM 推理为原始计算单元的运行时

[![tests](https://github.com/tlozttok/AVM1/actions/workflows/tests.yml/badge.svg)](https://github.com/tlozttok/AVM1/actions/workflows/tests.yml)

AVM 是一个运行时：以 LLM 推理为最小可调度计算单元，程序以自然语言（提示词）或 Python 代码的形式存放在内存中，由 AVM 负责调度和执行。

AVM 不是 Agent 框架。它不预设编排模式，不提供业务提示词，不与操作系统、硬件或网络直接交互。它提供一套指令集、一个内存系统、一个调度器、一个设备接口，以及把三者组合成"系统镜像"的格式。AVM 自成体系，有自己的内存模型和计算调度模型。

术语以 [CONTEXT.md](CONTEXT.md) 为准。

[English](README_EN.md)

## 一个系统镜像长什么样

`images/demo.json` 全文 43 行。下面是它的骨架（省略了提示词正文）：

```json
{
  "meta": { "name": "demo", "version": 1,
            "description": "最小交互演示：读输入 → 处理 → 写输出，循环" },
  "mem": {
    "system": {
      "kind": "dict", "ctrl": { "type": "settingup" },
      "value": {
        "name": { "kind": "str", "value": "init" },
        "content": { "kind": "str", "value": "你是 AVM 中运行的 init 对话……" }
      }
    },
    "user": { "kind": "str", "value": "开始" },
    "model_params": { "kind": "dict", "ctrl": { "type": "para" },
                      "value": { "model": { "kind": "str", "value": "deepseek-flash" } } },
    "inputs": { "kind": "device" },
    "outputs": { "kind": "device" }
  },
  "para_ref": "$MEM.model_params",
  "devices": [
    { "path": "inputs",  "file": "devices/io.py", "class": "InputsListDevice",  "args": {} },
    { "path": "outputs", "file": "devices/io.py", "class": "OutputsListDevice", "args": {} }
  ],
  "init": { "name": "init", "system_ref": "$MEM.system", "user_ref": "$MEM.user" },
  "persist_to": "out/mem.json"
}
```

三样东西在一个文件里：内存树（`mem`）、设备挂载（`devices`）、启动哪个对话（`init`）。

值得注意的是 **程序的位置**：那段提示词不是主机代码里的常量，而是内存树上的一个 `str` 节点（`mem.system.value.content`）。运行期改写它，就是改写程序本身——这是"程序即数据"在格式上的落点，也是 AVM 与"框架 + 配置文件"的分界。

## 快速开始

```bash
pip install openai python-dotenv pytest httpx
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                 # 全部测试（MockLMU，不调真实 API，不需要 key）
python main.py                   # 启动入口（读 debug.json 指定的镜像）
python main.py <image.json>      # 加载并运行系统镜像
```

Windows 上如果控制台是英文 locale（cp1252），入口脚本打印的中文进度行会报 `UnicodeEncodeError`；设 `PYTHONIOENCODING=utf-8` 即可。

示例镜像见 `images/demo.json`、`images/calc_driver.json`；权限体系示例见 `images/security_demo.json`（内核态 init + 用户态工作对话 + 私有工作空间）与 `images/permission_acl_demo.json`（全 PLM 的镜像层 ACL 演示：允许/默认拒绝/写保护）；读本仓库自测的镜像见 `images/repo_tester.json`（对话经只读设备 `$MEM.AVMRepo` 读仓库跟踪文件，用法由设备自述给出），打包版见 `images/repo_tester_packed.json`（0 号是 `plm.python` 打包器，启动时按 `$MEM.file_list` 给出的清单读文档、把系统提示词组装好再建并启动测试对话，测试者不必再读这些文档；程序与清单在 `images/repo_tester_packed/`）。系统镜像的组成与要点见 [docs/runtime.md](docs/runtime.md) 的「系统镜像」一节。

## 状态

- 用例数约 960，全部走 MockLMU；CI 覆盖 ubuntu / windows × Python 3.12 / 3.13 四个组合。
- 尚未发布 release。接口仍在变动，镜像格式以 `docs/system-image.md` 为准。
- 许可证 AGPL-3.0：个人使用、修改、分发都可以；把它嵌进对外提供的服务需要同样开源。

## 文档索引

- [CONTEXT.md](CONTEXT.md) — 术语表（对话、子对话、亚对话、链接、名字、交互结束 / 个体结束等）
- [docs/runtime.md](docs/runtime.md) — 运行时详细介绍（计算模型、指令集、内存模型、对话间通信、Python 程序、系统镜像）
- [docs/programming-guide.md](docs/programming-guide.md) — 如何设计 AVM 程序
- [docs/philosophy.md](docs/philosophy.md) — 设计哲学
- [docs/system-image.md](docs/system-image.md) — 系统镜像格式规范
- [docs/examples/python_as_conversation.md](docs/examples/python_as_conversation.md) — 示例：Python 程序如何无损表示为"对话 + 工具调用"
- [docs/capabilities.md](docs/capabilities.md) — 能力模型与权限机制：根表/工作空间/工具面/事件策略/ACL 示例

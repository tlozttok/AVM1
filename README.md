# AVM — 以 LLM 推理为原始计算单元的运行时

[![tests](https://github.com/tlozttok/AVM1/actions/workflows/tests.yml/badge.svg)](https://github.com/tlozttok/AVM1/actions/workflows/tests.yml)

AVM 是一个运行时：以 LLM 推理为最小可调度计算单元，程序以自然语言（提示词）或 Python 代码的形式存放在内存中，由 AVM 负责调度和执行。

AVM 不是 Agent 框架。它不预设编排模式，不提供业务提示词，不与操作系统、硬件或网络直接交互。它提供一套指令集、一个内存系统、一个调度器、一个设备接口，以及把三者组合成"系统镜像"的格式。AVM 自成体系，有自己的内存模型和计算调度模型。

术语以 [CONTEXT.md](CONTEXT.md) 为准。

[English](README_EN.md)

## 一个系统镜像长什么样

仓库里的母板镜像是 [images/base/base.json](images/base/base.json)（说明见 [images/base/README.md](images/base/README.md)）。下面只是它的骨架（省略了大量节点与程序正文）：

```json
{
  "meta": { "name": "base", "version": 1,
            "description": "母板镜像：持久化三件套 + 装载设备 + OS 程序 + 空分区骨架",
            "instruction_budget": 100, "snapshot_every": 200,
            "guest_programs": { "boot": "$MEM.sys.programs.boot",
                                "shutdown": "$MEM.sys.programs.supervisor" } },
  "mem": {
    "sys": {
      "kind": "dict",
      "value": {
        "power": { "kind": "dict", "value": {
          "state": { "kind": "str", "value": "on" } } },
        "programs": { "kind": "dict", "value": {
          "boot": { "kind": "dict", "ctrl": { "type": "python" },
                    "value": { "content": { "kind": "str", "value": "def boot(): ..." } } } } }
      }
    },
    "loader": { "kind": "device" },
    "inputs": { "kind": "device" },
    "outputs": { "kind": "device" }
  },
  "para_ref": "$MEM.para.python",
  "devices": [
    { "path": "loader",  "file": "../packages/loader.py", "class": "LoaderDevice",      "args": {} },
    { "path": "inputs",  "file": "../devices/io.py",      "class": "InputsListDevice",  "args": {} },
    { "path": "outputs", "file": "../devices/io.py",      "class": "OutputsListDevice", "args": {} }
  ],
  "init": { "name": "init", "system_ref": "$MEM.sys.programs.boot",
            "user": "yield from init()", "para_ref": "$MEM.para.python" },
  "journal_to": "out/base.journal",
  "snapshot_to": "out/base.snapshot.json",
  "persist_to": "out/base_mem.json"
}
```

三样东西在一个文件里：内存树（`mem`）、设备挂载（`devices`）、启动哪个对话（`init`）。

值得注意的是 **程序的位置**：boot 程序的源码不是主机代码里的常量，而是内存树上的一个 `str` 节点（`mem.sys.programs.boot.value.content`）。运行期改写它，就是改写程序本身——这是"程序即数据"在格式上的落点，也是 AVM 与"框架 + 配置文件"的分界。

## 快速开始

```bash
pip install openai python-dotenv pytest httpx
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                 # 全部测试（MockLMU，不调真实 API，不需要 key）
python main.py                   # 启动入口（读 debug.json 指定的镜像）
python main.py <image.json>      # 加载并运行系统镜像
```

Windows 上如果控制台是英文 locale（cp1252），入口脚本打印的中文进度行会报 `UnicodeEncodeError`；设 `PYTHONIOENCODING=utf-8` 即可。

示例镜像保留两块：母板镜像 `images/base/base.json`（长期运行、持久化与 OS 程序，见 [images/base/README.md](images/base/README.md)）与程序包装载演示 `images/packages/packages_demo.json`（loader + 示例包，见 [images/packages/README.md](images/packages/README.md)）。系统镜像的组成与要点见 [docs/runtime.md](docs/runtime.md) 的「系统镜像」一节。

## 状态

- 用例数约 1141，全部走 MockLMU；CI 覆盖 ubuntu / windows × Python 3.12 / 3.13 四个组合。
- 尚未发布 release。接口仍在变动，镜像格式以 `docs/system-image.md` 为准。
- 许可证 AGPL-3.0：个人使用、修改、分发都可以；把它嵌进对外提供的服务需要同样开源。

## 文档索引

- [CONTEXT.md](CONTEXT.md) — 术语表（对话、子对话、亚对话、链接、名字、交互结束 / 个体结束等）
- [docs/runtime.md](docs/runtime.md) — 运行时详细介绍（计算模型、指令集、内存模型、对话间通信、Python 程序、系统镜像）
- [docs/programming-guide.md](docs/programming-guide.md) — 如何设计 AVM 程序
- [docs/philosophy.md](docs/philosophy.md) — 设计哲学
- [docs/system-image.md](docs/system-image.md) — 系统镜像格式规范
- [docs/examples/python_as_conversation.md](docs/examples/python_as_conversation.md) — 示例：Python 程序如何无损表示为"对话 + 工具调用"
- [docs/capabilities.md](docs/capabilities.md) — 能力模型与权限机制：根表与动态授予根/工具面/事件策略/Guard/ACL 示例
- [docs/porting-contract.md](docs/porting-contract.md) — 移植契约：把外部编排方案写成 AVM 程序包的交付约定

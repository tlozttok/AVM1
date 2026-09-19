# AVM — 以 LLM 推理为原始计算单元的运行时

AVM 是一个运行时：以 LLM 推理为最小可调度计算单元，程序以自然语言（提示词）或 Python 代码的形式存放在内存中，由 AVM 负责调度和执行。

AVM 不是 Agent 框架。它不预设编排模式，不提供业务提示词，不与操作系统、硬件或网络直接交互。它提供一套指令集、一个内存系统、一个调度器、一个设备接口，以及把三者组合成"系统镜像"的格式。AVM 自成体系，有自己的内存模型和计算调度模型。

术语以 [CONTEXT.md](CONTEXT.md) 为准。

[English](README_EN.md)

## 文档索引

- [CONTEXT.md](CONTEXT.md) — 术语表（对话、子对话、亚对话、链接、名字、交互结束 / 个体结束等）
- [docs/runtime.md](docs/runtime.md) — 运行时详细介绍（计算模型、指令集、内存模型、对话间通信、Python 程序、系统镜像、调试观测）
- [docs/programming-guide.md](docs/programming-guide.md) — 如何设计 AVM 程序
- [docs/philosophy.md](docs/philosophy.md) — 设计哲学
- [docs/system-image.md](docs/system-image.md) — 系统镜像格式规范
- [docs/apps](docs/apps) — AVM 应用套件（实际性测试镜像：传令站 / 金库守卫 / 账本服务）
- [docs/games](docs/games) — AVM 游戏（echo_mine 回声矿洞）

## 快速开始

```bash
pip install openai python-dotenv pytest
echo "OPENAI_API_KEY=sk-..." > .env

python -m pytest                          # 全部测试（MockLMU，不调真实 API）
python main.py                            # 从 debug.json 读取镜像配置并运行
python main.py <image.json> --frames      # 运行后打印监测帧摘要（stderr）
python main.py <image.json> --transcript api.txt  # API 调用全文落盘
```

调试参数写在 `debug.json`（`image` / `frames` / `transcript`），不需要改命令行（VSCode 固定运行 `python main.py`）。示例镜像见 `images/demo.json`、`images/calc_driver.json`；应用套件见 `images/apps/`（启动方式见 [docs/apps](docs/apps)）；AI 小游戏见 `images/games/echo_mine/`。系统镜像的组成与要点见 [docs/runtime.md](docs/runtime.md) 的「系统镜像」一节。


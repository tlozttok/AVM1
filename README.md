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
- [docs/Agents](docs/Agents) — 计划与设计决策留档
- [docs/record](docs/record) — 设计纪要（历史留档）

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

## 当前状态

已实现：

- 9 条指令（内存读写与创建、ctrl 元数据编辑、对话创建、投递、关闭、事件监听）；
- 内存树与持久化（写回文件）；
- 调度状态机：活跃/休眠/就绪，排队调度、核心中断（指令预算）、调用链预算继承；
- 简单投递（`send_instruction`：内容原样投递 + 唤醒 + "已投递"确认，无 ICC、无信封）；
- `close_conversation` 内核级关闭（个体结束：PLM 状态释放）；
- 系统镜像加载（显式节点、para、设备插件、init、persist_to、instruction_budget）与 `main.py` / `debug.json` 调试入口；
- Python 执行器（`plm.simple` / `plm.python`）与沙箱（消息直接作为代码/表达式执行，不解析信封）；
- 信息设备（conversations / scheduler / memory / monitor / listeners）；
- 事件模型 v1（Core 发布 before 请求事件、Memory/设备/Core 发布 after 事实事件、phase=before/after/all、设备 `authorize_listener` / `after_event`、Python 处理器、owner 校验、notify 内部投递、`$MEM.listeners`）；
- 监测器（帧 / 轨迹 / diff / find / 全文文件，含思维链记录与事件摘要）；
- 思维链内容回传（历史保留、总是回传、monitor 记录、不内部暴露）；
- 单对话检查点（`save_conversation` / `restore_conversation`，输入 `/save` 调试命令）；
- AVM 应用套件（courier_office / vault_guard / plm_ledger：镜像 + 自定义设备 + 镜像外自定义 PLM 服务器）；
- AVM 小游戏 echo_mine（GM 对话 + ghost 子对话 + 骰子/地图设备 + plm.python 结算 + HP 事件守卫）；
- 测试 301 个全部通过（MockLMU，不调真实 API）。

设计中（未实现）：

- 设备化：服务调用、注册表等改为设备访问（`xx.xx.post` / `xx.xx.response` 方向，见 docs/record/9.md）；
- 内核态 / 用户态分离；
- 链接设备；
- 事件条件（ECA 的 C）、事件持久化/重放、事件总线与定时器等剩余事件能力——"对话自决休眠 + 预期行为监听"的后续载体；
- 名字 ↔ cid 转换工具；
- 文件密码与访问回调；
- 错误处理机制细化（json_error / unknown_tool 的响应配对）；
- 阻塞设备的非阻塞替代（事件机制）。

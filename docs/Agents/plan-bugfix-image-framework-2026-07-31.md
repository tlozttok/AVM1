# 计划：修复调度/工具调用遗留问题 + JSON 系统镜像框架

**时间**: 2026-07-31
**状态**: bug 修复已实施（124 测试通过）；镜像框架待规范确认后实施
**背景**: 用户要求先修复现有代码的 bug 和遗留问题作为自测基线，随后提供"一个 JSON + 若干设备 Python 文件 = 标准 AVM 系统镜像"的加载框架与规范。
**起因**: 代码审查发现四类问题：
1. 调度状态机不干净：create_sub / call_service 的父/调用者同时进入就绪与休眠，唤醒后 dormant 标记不清除，状态互相重叠。
2. 工具调用错误被静默丢弃：工具参数 JSON 解析失败（json_error）与未知工具名没有任何反馈，LLM 不知道自己的调用失败了；`command` 是死代码分支。
3. ~~服务生命周期不完整：服务对话结束后仍留在 `_services` 注册表~~（已废弃此判断：用户纠正"对话结束不代表失效"，交互结束 ≠ 个体结束 ≠ 失效，服务保持注册可再次调用）。
4. 遗留注释：`MemoryCircularReferenceError` 写的是"$ 解引用成环"，与新确认的符号链接设计冲突。

**行动**:
1. 调度状态机：create_sub / call_service 不再把父/调用者插入就绪队列，改为休眠等待；对话完成时由 core 唤醒（`_resume`）并清理 dormant 标记；`_pick_next_active` 防御性清理状态重叠。与 CONTEXT.md 语义对齐（父休眠、事件唤醒）。
2. 工具调用错误面向运行者：json_error / unknown_tool 输出到 stderr（用户明确纠正：不静默是"给我"，不是"给 LLM"；最初实现为喂回 LLM 自纠，已按用户纠正废弃）；删除 `command` 死分支。
3. 服务注销：**已回滚**——用户纠正"对话结束不代表失效"（交互结束 ≠ 个体结束 ≠ 失效），服务保持注册、可再次调用。
4. 注释修正：`MemoryCircularReferenceError` → "符号链接成环"。
5. 测试：更新 TestCreateSub 的 2 处断言（父休眠而非就绪）；新增服务调度测试（call_service 唤醒 + 注册保持、transfer_service 不唤醒）、工具调用错误面向运行者测试 3 个。
6. 文档：CONTEXT.md 新增"对话结束"术语，区分交互结束与个体结束。
7. 新增 `return_result` 工具；call_service 改为工具返回（服务自然结束不再自动写回）；交互结束一律进入休眠；finished 概念化（仅记录被内核级关闭指令关闭的对话）。
8. 镜像框架实现（2026-08-01）：`avm/image.py`（显式节点解析、para 校验、设备插件加载、init 启动、persist_to）+ `python -m avm.image` 入口；`docs/system-image.md` 规范；测试 `tests/test_image.py` 14 个。
9. 类型层：MetaDict/MetaList 增加 `ctrl`（元数据二）；Conversation 增加 `name`。`Memory.save/load` 切换为显式节点格式（兼容旧 `__type`）；`Core.para_ref` 可配置；LMU 数值参数字符串→数值转换。
10. 调试入口（2026-08-01）：`main.py`（`python main.py <image.json>`，支持 `--frames` 帧摘要、`--transcript` API 全文落盘、`persist_to` 写回）+ 示例镜像 `images/demo.json` + `images/devices/io.py`；测试 `tests/test_main.py` 3 个。
11. 文件内配置调试路径（2026-08-01）：`main.py` 无参数时从 `debug.json` 读取 `image`/`frames`/`transcript`（`--debug-config` 可指定其他文件），VSCode 中固定运行 `python main.py` 即可，参数改文件不改命令行；测试 2 个。
12. 修复（2026-08-02）：`OutputsListDevice` 写入 `$MEM.outputs.-1` 不打印——docstring 承诺"并打印到屏幕"但实现缺失；补上 `print(value, flush=True)`（flush 保证逐步运行时立即可见），新增 2 个测试。

**影响**: 状态机三态（活跃/休眠/就绪）互斥；工具调用错误面向运行者（stderr）可见、对话继续；服务注册不受对话交互结束影响（可再次被调用），对话个体结束才失效。现有 create_cmd 语义不变（父休眠、不唤醒，符合设计）。监测器状态推导在新状态机下正确。

**改动标注**（每项改动按用户要求的四类标注）:
- create_sub / call_service 调度改动（父/调用者休眠等待、`_resume` 清理 dormant）：**按文档记录改动**——CONTEXT.md 语义为"父在子/服务完成后恢复"，现有代码创建时即把父/调用者放入就绪，与文档冲突，改动现有代码对齐文档。
- 工具调用错误处理（json_error / unknown_tool → stderr）：**按用户计划改动**——用户明确"不静默是给我，不是给 LLM"。原"喂回 LLM 自纠"实现已按用户纠正废弃。
- `command` 死分支删除：**判断性改动（认为现在的代码和文档记录冲突而改动现在的代码）**——文档指令集不含 `command`，判断其为遗留死代码而移除。
- 服务注销（已回滚）：**按用户计划改动**——用户明确"对话结束不代表失效"，原判断基于"结束即失效"的错误前提，已回滚；CONTEXT.md 已澄清"对话结束"的两种含义（交互结束 / 个体结束）。
- `MemoryCircularReferenceError` 注释：**按文档记录改动**——符号链接设计决策已确认，旧注释"$ 解引用成环"过时。
- 测试更新/新增：跟随其对应改动的标注。
- CONTEXT.md 修订与设计决策留档：**按用户计划改动**——用户确认的设计决策（链接、返回工具、事件注册、$ /& 废弃）。
- CONTEXT.md 新增"对话结束"术语（区分交互结束 / 个体结束）：**按用户计划改动**——用户要求明确该歧义。
- `return_result` 指令、call_service 工具返回、交互结束一律进入休眠、finished 概念化：**按用户计划改动**——用户明确指令与语义（"把call_service改成使用工具返回"、"finished 是概念态，记录被关闭的对话，关闭是内核级指令"）。
- `return_result` 工具调用后唤醒调用者（含 create_cmd 的父），无需回调注册：**按用户计划改动**——用户明确"消息已可用就不留在休眠队列"；create_cmd 父的等待位置确认为休眠队列（就绪队尾会变成 create_sub 的自动恢复语义）。返回必须由被调用方显式调用该工具完成。
- mem 显式节点格式（kind/meta/ctrl/value，str 也强制包装）、para 作为 `ctrl.type="para"` 节点 + 顶层 `para_ref`、`kind=device` 顶层标记（link 是设备不是数据类型，device value 结构未定）、不预置其他对话、设备必须继承 MemoryDevice、save/load 与镜像格式统一：**按用户计划改动**——用户逐点确认。
- `kind=device` 允许任意层级（点号路径如 `game.map.rooms` 与 devices 段 `path` 对应）：**按用户计划改动**——用户纠正"device 不是只允许在顶层"。
- `main.py` 调试入口与示例镜像：**按用户计划改动**——用户要求提供调试 main 和可修改的示例镜像。
- `debug.json` 文件内配置调试路径：**按用户计划改动**——用户明确不想在 VSCode 里反复改命令参数。
- `OutputsListDevice` 补上打印：**按文档记录改动**——认为现有代码与设备文档承诺（"写入即打印到屏幕"）冲突而改动现有代码。
- MetaDict/MetaList 增加 `ctrl`（元数据二）落地：**按文档记录改动**——CONTEXT.md 已定义双元数据（元数据二 `dict[str,str]` 给 Core）。
- LMU 数值参数字符串→数值转换：**判断性改动**——para 以 str 存储，数值型参数需还原为数值才能调用 API。

**待确认（镜像框架）**: JSON 镜像的 mem 段格式、init 段字段、设备插件约定、是否预置多对话/服务、是否持久化写回——见会话提问，确认后实施 `avm/image.py` + `avm/boot.py` + `docs/system-image.md`。

**待定设计（2026-08-02，用户明确暂不处理）**: 错误处理机制整体未定，不要轻举妄动。挂起的问题包括：`json_error` / `unknown_tool` 被丢弃后缺少工具响应配对（下一轮 API 会 400）；`memory_write` 传非 `$` 开头的 ref 时 `Memory.set` 抛 `ValueError` 未被捕获（run 终止）；错误信息只面向运行者 vs 满足 API 配对之间的取舍。

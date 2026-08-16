# 审计：AI 小游戏 echo_mine 完成度与证据

**时间**: 2026-08-16
**状态**: 审计完成，目标判定为已完成
**背景**: 用户在 game_test 分支提出目标：在 AVM 上做一个 AI 驱动小游戏，发挥想象、尽可能展现 AVM 独特能力，遇到问题可直接修改代码。
**起因**: 连续多轮交付后，按项目规则对目标做一次正式完成度审计，把证据、覆盖矩阵与遗留项固化为文档。

---

## 一、交付物

| 类别 | 文件 |
|---|---|
| 游戏镜像 | `images/games/echo_mine/echo_mine.json` |
| 设备 | `images/games/echo_mine/devices/dice.py`、`map.py`、`io.py` |
| 游戏说明 | `docs/games/echo_mine.md`、`docs/games/README.md` |
| 计划留档 | `docs/Agents/plan-game-echo-mine-2026-08-16.md` |
| 测试 | `tests/test_game_echo_mine.py` |

## 二、AVM 独特能力覆盖

| AVM 能力 | 游戏落点 | 测试证据 |
|---|---|---|
| 对话即进程 | guide 是 init 对话，ghost 是常驻子对话，跨回合保留历史 | 完整回合与 talk 测试 |
| create_cmd / send_instruction | 创建 ghost，投递 GHOST_ASK，ghost 回 GHOST_REPLY | `TestEchoMineFullTurn` |
| create_sub + plm.python | 搜索与移动由 Python 结算器完成，结果自动写回 | 完整胜利路线测试 |
| 共享内存世界 | `$MEM.world` 保存 hp/gold/room/status/turn | 全流程状态断言 |
| 虚拟设备 | `$MEM.dice.d6` 随机源；`$MEM.map` 实时地图 | Dice/Map 测试 |
| before 事件规则 | hp_guard 钳制 HP 到 0..3，非法值 replace | HP 守卫测试 |
| after 事件 + notify | hp_watch 通知 ghost，HP 归零触发 GHOST_ALERT 唤醒 GM | HP after 事件测试 |
| close_conversation | 胜利/失败/退出时关闭 ghost 与 GM，监听自动撤销 | 胜利分支与 after 测试 |
| main.py 调试入口 | 终端输入、persist_to、--transcript | main 入口与 subprocess 测试 |
| OpenAI 兼容端点 | 假 API 全链路跑通 look/quit 与完整胜利会话 | subprocess 测试 |
| 单对话检查点 | 终端 `/save game.conv.json` 写入 guide 检查点 | `/save` 测试 |

## 三、测试证据

- 游戏专项测试：12 个全部通过；
- 全量测试：303 个全部通过；
- 完整胜利路线：4 次 search（6/6/6/5）、2 次 move、exit，断言 won/vault/gold=10/turn=6；
- 完整失败路线：TRAP 使 hp 归零，after 事件唤醒 ghost 发 GHOST_ALERT，GM 关闭双方。

## 三·补、真实 API 冒烟（2026-08-16 实测）

使用 `.env` 中的真实 OpenAI 兼容端点（模型 `deepseek-v4-flash`）在 `/tmp` 副本运行游戏，不污染仓库：

1. `look / quit`：
   - 游戏正常初始化、注册监听、创建 ghost；
   - `look` 正确读房间描述与地图，turn 0 -> 1；
   - `quit` 正常关闭 ghost 与 guide；
   - `--transcript` 记录 10 个 FRAME。
2. `search / move north / look / talk 这里有什么 / quit`：
   - `search` 经 `$MEM.dice.d6` 与 plm.python 结算为 EMPTY；
   - `move north` 经 plm.python 返回 cavern，地图标记玩家位置更新；
   - `look` 正确读 cavern 描述；
   - `talk` 经 send_instruction 唤醒 ghost，ghost 读共享世界并回话；
   - 终态：status=playing，turn=3，room_id=cavern，last_event=MOVE entrance -> cavern。

结论：游戏在真实 LLM 端点上的实际行为与设计一致。

## 四、遗留项（不阻塞本目标）

1. `create_sub` 工具响应不返回亚对话 cid，短命结算对话只能休眠不能由 GM 精确关闭；运行时生命周期问题，另立计划处理。
2. `hp_watch` 处理器按本镜像约定把 ghost 固定为 cid=1（guide 初始化时第一个创建的就是 ghost）；若镜像提示词调整对话创建顺序，需同步处理器。
3. 自动化测试使用 OpenAI 兼容假 API；真实 API 冒烟已通过（见上），更长时间的人工游玩仍建议进行。

## 五、结论

目标已完成：echo_mine 是可启动、可交互、可胜利/失败、可观测、可保存检查点的 AI 驱动 AVM 小游戏，且未修改 `avm/` 运行时本体。

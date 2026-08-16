# 计划：AVM 小游戏 echo_mine（回声矿洞）—— 用游戏实测对话调度、虚拟设备与规则事件

**时间**: 2026-08-16
**状态**: 已实施（按用户在 game_test 分支上的目标直接落地；四部分计划供用户评分，修订意见回填本文件）
**背景**: AVM 已有应用套件（courier_office / vault_guard / plm_ledger），覆盖指令、事件与镜像外 PLM。用户提出新目标：在 AVM 上做一个 AI 驱动小游戏，允许发挥并直接修改代码，分支为 game_test。
**起因**: 小游戏比工具型应用更能同时压测 AVM 的独特组合：LLM 对话作为游戏进程、共享内存作为世界状态、虚拟设备作为骰子与地图、事件监听作为硬规则、Python 对话作为确定性结算器、子对话作为有独立记忆的 NPC。
**约束**: 优先不改 `avm/` 运行时；只有镜像格式确实无法承载时才改代码，并记录改动与影响。游戏必须可手工交互运行，也必须有 MockLMU 确定性集成测试。

---

## 一、目标与判定标准

1. 做一个可玩的回合制文字小游戏 `echo_mine`：
   - 玩家在终端通过 `$MEM.inputs` / `$MEM.outputs` 交互；
   - GM 是 init 对话，负责叙述、解析动作、更新世界、判断胜负；
   - NPC「矿洞回声」是 create_cmd 创建的子对话，拥有独立、跨回合保留的对话历史；
   - 确定性结算（搜索、移动）由 plm.python 亚对话完成；
   - 随机性由自定义 DiceDevice 以虚拟文件形式提供；
   - 地图由自定义 MapDevice 从共享内存实时渲染；
   - HP 钳制由 before `memory_write` 事件处理器强制，LLM 无法写穿边界。
2. 判定通过：
   - `python main.py images/games/echo_mine/echo_mine.json` 可启动并交互；
   - MockLMU 集成测试可完整跑通 搜索/移动/交谈/胜利 分支；
   - 全量测试不回退（当前 302 个，含新增游戏测试 11 个）。
3. 不通过即缺口：若游戏需要而 AVM 拒绝的能力，记录 issue；不在应用里用宿主 Python 绕过运行时边界。

## 二、游戏架构

```
player ── inputs/outputs ──> guide(GM, LLM, cid 0)
                              | 初始化: listen_event(hp_guard)
                              |          create_cmd(ghost) -> cid 1
                              |          create_sub(mechanics/plm.python) 每轮结算
                              v
                          $MEM.world（共享世界状态，owner=0）
                              |-- player.hp / gold / room_id
                              |-- rooms.entrance / cavern / vault
                              |-- status / turn / last_event
ghost(LLM, cid 1) <--send_instruction-- guide（交谈）
map(device)  读 $MEM.world 渲染 ASCII 地图
dice(device) 读子路径 d6 返回 1..6
```

规则事件：
- `before memory_write $MEM.world.player.hp`：handler 把内容钳到 0..3，非法内容直接 replace；
- `after memory_write $MEM.world.player.hp`：handler 通知 ghost（cid=1）；HP 归零时 ghost 发出 GHOST_ALERT 唤醒 GM；
- HP 是 Core 执行前被规则修正的，GM 的叙事不能绕过。

## 三、LLM 式算法/控制流

详见 `docs/games/echo_mine.md`。要点：
- guide 的提示词是状态机：初始化 → 读输入 → 选择动作分支 → 调用子对话/设备 → 写回世界 → 输出叙述 → 回到读输入；
- 计算不下沉到 LLM：结算代码存在 `$MEM.programs.mechanics`（python），guide 只写请求串并 create_sub；
- 随机性不下沉到 LLM：guide 读 `$MEM.dice.d6`，把它作为输入交给结算器；
- ghost 只通过 send_instruction 与 guide 通信，创建关系不构成自动返回通道；
- 游戏结束走 close_conversation（ghost 与 guide 自身），不残留活跃/就绪对话。

## 四、行动、影响与发现

行动：
1. 新增 `images/games/echo_mine/echo_mine.json` 与 `devices/dice.py`、`devices/map.py`、`devices/io.py`；
2. 新增 `docs/games/echo_mine.md`、`docs/games/README.md`；
3. 新增 `tests/test_game_echo_mine.py`（镜像加载、DiceDevice、MapDevice、HP 守卫、MockLMU 全流程）；
4. README 文档索引与测试计数同步。

影响：
- 不修改 `avm/` 运行时；新增文件独立；
- 游戏测试使用 MockLMU 与真实设备/Python 结算器，不调真实 LLM API。

发现的设计缺口：
1. `create_sub` 的工具响应不返回亚对话 cid，guide 无法在结算后关闭已休眠的 Python 亚对话；多回合游戏会累积短命亚对话。当前不影响可玩性，但属于运行时生命周期缺口，记录待后续计划。
2. DiceDevice 的宿主随机数绕过了 PLM 沙箱，这是设备接口的既定信任边界；镜像作者需自行保证设备不越权（与现有设备设计一致）。

**实施记录（2026-08-16）**:
- 新增游戏镜像、设备、文档、测试；全量 pytest 通过（302 个），测试计数见 README。

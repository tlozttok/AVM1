# echo_mine 真实 API 游玩记录

日期：2026-08-16
环境：真实 OpenAI 兼容端点，模型 `deepseek-v4-flash`；终端通过 `main.py` 交互。
命令序列：`search` → `move north` → `look` → `talk 这里有什么` → `quit`。

## 终端输出

```text
欢迎来到回声矿洞。命令：look / search / move <north|south|east|west> / talk <内容> / exit / quit。
echo_mine 地图:
[entrance]  *你在这里*
  north -> cavern
[cavern]
  south -> entrance
  east -> vault
[vault]
  west -> cavern

[guide（cid=0） 等待输入，/save 保存检查点]
你在矿洞中仔细搜索了一番，四周只有冰冷的岩壁与回荡的风声，一无所获（EMPTY）。
状态：HP 3 / Gold 0

[guide（cid=0） 等待输入，/save 保存检查点]
你向北走进 cavern：巨大的中央洞窟里到处是废弃的矿车。南面是入口，东面有一扇半掩的铁门，门后透出微弱的金光。
echo_mine 地图:
[entrance]
  north -> cavern
[cavern]  *你在这里*
  south -> entrance
  east -> vault
[vault]
  west -> cavern

[guide（cid=0） 等待输入，/save 保存检查点]
巨大的中央洞窟里到处是废弃的矿车。南面是入口，东面有一扇半掩的铁门，门后透出微弱的金光。
echo_mine 地图:
[entrance]
  north -> cavern
[cavern]  *你在这里*
  south -> entrance
  east -> vault
[vault]
  west -> cavern

[guide（cid=0） 等待输入，/save 保存检查点]
（回声）水滴在黑暗的洞壁上敲出回音，风从某条更深的裂缝里漏进来，像是在指引什么。

[guide（cid=0） 等待输入，/save 保存检查点]
```

## 终态

- `status = playing`
- `turn = 3`
- `room_id = cavern`
- `last_event = MOVE entrance -> cavern`
- HP/Gold：`3 / 0`

## 本次会话经过的 AVM 路径

| 玩家命令 | AVM 行为 |
|---|---|
| 初始化 | GM 注册 HP before/after 监听，create_cmd 创建 ghost，MapDevice 渲染地图 |
| `search` | GM 读 `$MEM.dice.d6`，把骰点交给 plm.python `resolve_search`；结果 EMPTY，世界状态更新 |
| `move north` | GM 把当前房间与方向交给 plm.python `next_room`；结果 cavern，MapDevice 实时显示玩家位置变化 |
| `look` | GM 读 `$MEM.world.rooms.cavern.desc` 与 `$MEM.map` |
| `talk` | GM `send_instruction` 给 ghost；ghost 读共享世界后回话并唤醒 GM |
| `quit` | GM `close_conversation` 关闭 ghost 与自身；持久化世界到 `out/mem.json` |

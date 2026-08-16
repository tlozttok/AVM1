# echo_mine 真实 API 失败路线记录

日期：2026-08-16
环境：真实 OpenAI 兼容端点，模型 `deepseek-v4-flash`。
方式：临时固定骰子（恒为 1），命令 `search` × 3。

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
你在周围翻找，触发了陷阱！受到 1 点伤害。
状态：HP 2/3，金币 0。

[guide（cid=0） 等待输入，/save 保存检查点]
你又触发了陷阱！受到 1 点伤害。
状态：HP 1/3，金币 0。

[guide（cid=0） 等待输入，/save 保存检查点]
你触发了最后的陷阱，HP 归零，眼前一黑……你失败了。
回声矿洞吞噬了你的身影。
GHOST_ALERT：回声正在消失，快离开矿洞
```

## 终态

- `status = lost`
- `hp = 0`
- `turn = 2`
- `room_id = entrance`

## 经过的 AVM 路径

| 步骤 | AVM 行为 |
|---|---|
| 搜索 1/2 | `$MEM.dice.d6` 返回 1；plm.python `resolve_search(1)` 返回 `TRAP 1`；HP before 守卫放行 2、1 |
| 搜索 3 | HP 写 0；GM 写 `status=lost`；HP after 事件 `notify` ghost |
| ghost 唤醒 | 收到 `HP_CHANGED 0`，`send_instruction` 回 `GHOST_ALERT` |
| GM 唤醒 | 输出警报，关闭 ghost 与自身；监听器自动撤销 |

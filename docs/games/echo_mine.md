# AVM 游戏：echo_mine（回声矿洞）

**目标**: 用一个可玩的 AI 文字冒险，展示 AVM 的几个独特组合：对话即进程、共享内存即世界、虚拟设备即骰子与地图、事件监听即硬规则、Python 对话即确定性结算器、子对话即有独立记忆的 NPC。

---

## 1. 玩法

玩家在终端输入命令，GM 对话主持游戏：

| 命令 | 效果 |
|---|---|
| `look` | 读取当前房间描述与 `$MEM.map` 渲染的地图 |
| `search` | 读 `$MEM.dice.d6`，把骰点交给 plm.python 结算器，按 `TRAP` / `EMPTY` / `GOLD` 更新世界 |
| `move <方向>` | 把当前房间与方向交给结算器，合法则更新 `$MEM.world.player.room_id` |
| `talk <内容>` | 把消息投递给常驻 ghost 子对话，ghost 回复后 GM 唤醒 |
| `exit` | 在 vault 且 gold >= 10 时胜利；`quit` 随时结束 |

胜利条件：到达 `vault` 并持有 10 gold。HP 上限 3，归零失败。

## 2. 架构

```
player -- $MEM.inputs / $MEM.outputs --> guide(GM, LLM, cid 0)
                                          | 初始化:
                                          |   listen_event(hp_guard, before, $MEM.world.player.hp)
                                          |   create_cmd(ghost) -> cid 1
                                          | 每轮结算:
                                          |   create_sub(mechanics, plm.python)
                                          v
                                      $MEM.world
                                        player.{hp,gold,room_id}
                                        rooms.{entrance,cavern,vault}
                                        status / turn / last_event
ghost(LLM, cid 1) <--send_instruction--> guide
map  device: 只读 $MEM.world，实时渲染地图
dice device: 读 $MEM.dice.d6 返回 1..6
```

## 3. LLM 式算法/控制流

### 3.1 guide 状态机

```
初始化(status=starting):
  listen_event(memory_write, $MEM.world.player.hp, hp_guard, before)
  ghost_cid = create_cmd($MEM.programs.ghost)
  status = playing
  outputs <- 开场白 + map

主循环:
  input = memory_read("$MEM.inputs.-1")
  quit        -> close ghost; close self
  look        -> 读 room.desc + map; turn += 1
  search      -> roll = memory_read("$MEM.dice.d6")
                 request = "print(resolve_search(roll))"
                 result = create_sub(mechanics, request)
                 TRAP n -> hp -= n（before 守卫钳制）; hp==0 -> status=lost
                 GOLD n -> gold += n
                 turn += 1; 输出叙述
  move dir    -> current = room_id
                 request = "print(next_room(current, dir))"
                 result = create_sub(mechanics, request)
                 BLOCKED -> 输出不可通行
                 room    -> room_id = result; 输出新房间与地图
                 turn += 1
  talk text   -> send_instruction(ghost_cid, "GHOST_ASK ...")
                 输出 waiting（无工具）-> 休眠
  exit        -> vault 且 gold>=10 -> status=won; close ghost; close self
  被 GHOST_REPLY 唤醒 -> 输出回复; 回主循环
```

关键控制流事实：
- `create_sub` 是紧耦合：guide 在结算时休眠，plm.python 输出写回 guide 的工具响应后 guide 被唤醒。
- `talk` 是松耦合：guide 只投递消息，然后主动结束交互休眠；ghost 完成后用 `send_instruction` 把回复投递回来，guide 才被唤醒。创建关系不提供默认返回通道。
- ghost 作为 create_cmd 子对话跨回合休眠，其消息历史保留，所以它能“记得”之前聊过。

### 3.2 ghost 状态机

```
收到 GHOST_ASK <text>:
  room_id = memory_read("$MEM.world.player.room_id")
  rooms   = memory_read("$MEM.world.rooms")
  reply   = 根据当前房间生成一句模糊提示
  send_instruction(cid=0, content="GHOST_REPLY <reply>")
  return done（无工具）-> 休眠
```

### 3.3 mechanics（plm.python）

系统消息定义纯函数，用户消息是一行 `print(<调用>)`：

```python
def resolve_search(roll):
    roll = int(roll)
    if roll == 1: return "TRAP 1"
    if roll in (2, 3, 4): return "EMPTY 0"
    if roll == 5: return "GOLD 1"
    return "GOLD 3"

def next_room(current, direction):
    links = {
        "entrance": {"north": "cavern"},
        "cavern": {"south": "entrance", "east": "vault"},
        "vault": {"west": "cavern"},
    }
    return links.get(current, {}).get(direction, "BLOCKED")
```

搜索和移动的算术/图查找都不由 LLM 完成；LLM 只写请求、读响应、写回状态。

## 4. 规则事件：HP 守卫

`$MEM.handlers.hp_guard` 是 Python 事件处理器，GM 在初始化时注册：

```python
def handle(event):
    content = event["payload"].get("content", "")
    try:
        hp = int(content)
    except Exception:
        return {"action": "replace", "response": "HP 写入无效：必须是整数"}
    if hp < 0:
        return {"payload": {"content": "0"}}
    if hp > 3:
        return {"payload": {"content": "3"}}
    return {}
```

- 这是 before `memory_write` 守卫：GM 可以写任意数值，Core 执行前由规则钳到 0..3。
- 非法内容被 `replace`，原写入不执行。
- `$MEM.world` 预置 `ctrl.owner="0"`，guide 才有权注册该监听。

## 5. 虚拟设备

### DiceDevice（$MEM.dice）

- `to_llm_string()`：告诉 GM 子路径用法；
- `resolve_path(["d6"])`：返回 1..6 的宿主随机数。
- 随机性是外部设备输入，不是 PLM 或 LLM 内部产生。

### MapDevice（$MEM.map）

- `attach_core(core)` 持有 Core 引用；
- `to_llm_string()`：读 `$MEM.world.rooms` 与 `$MEM.world.player.room_id`，实时生成文本地图并在当前房间标注 `*你在这里*`。
- 地图永远是共享内存的视图；GM 不需要在提示词里硬编码地图。

## 6. 验证

- `tests/test_game_echo_mine.py`：
  1. 镜像严格加载，地图/骰子设备挂载；
  2. HP 守卫钳制与 replace；
  3. DiceDevice 范围与 MapDevice 渲染；
  4. MockLMU 驱动完整回合：初始化、search（GOLD 结算）、move（cavern）、talk（ghost 回复），断言世界状态与调度终态。
- 手工：`python main.py images/games/echo_mine/echo_mine.json`。

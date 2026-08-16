# AVM 游戏

- [echo_mine](echo_mine.md)：回声矿洞，一个 AI 驱动的回合制文字冒险。
  - [真实 API 游玩记录](echo_mine-playthrough.md)
  - [完成度审计](../Agents/audit-game-echo-mine-2026-08-16.md)

## 启动

```bash
python main.py images/games/echo_mine/echo_mine.json
```

终端命令：`look`、`search`、`move <north|south|east|west>`、`talk <内容>`、`exit`、`quit`。

## 自动化测试

```bash
python -m pytest tests/test_game_echo_mine.py -q
```

"""echo_mine 小游戏集成测试。

覆盖：镜像加载、DiceDevice / MapDevice、事件 HP 守卫、MockLMU 完整回合
（初始化 -> search 结算 -> move 结算 -> 与 ghost 子对话交谈 -> 双方休眠）。
"""
import json
from pathlib import Path

import pytest

from avm.core import MemoryWriteInstruction
from avm.image import load_image

ROOT = Path(__file__).resolve().parents[1]
GAME_IMAGE = ROOT / "images" / "games" / "echo_mine" / "echo_mine.json"


def tool(call_id, name, **args):
    return {"call_id": call_id, "cmd_type": name, "args": args}


class ScriptedLMU:
    """按 cid 依次返回预设响应，并像 LMU.exec 一样提交对话历史。"""

    def __init__(self, scripts):
        self.scripts = scripts
        self.idx = {cid: 0 for cid in scripts}
        self.calls = []

    def exec(self, conversation, para):
        self.calls.append((conversation.cid, conversation.identity))
        cid = conversation.cid
        seq = self.scripts[cid]
        i = self.idx[cid]
        if i >= len(seq):
            raise RuntimeError(f"ScriptedLMU 对话 {cid} 第 {i} 次调用没有预设响应")
        self.idx[cid] += 1
        result, return_calls, _ = seq[i]

        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            conversation.append_user_message(user_content)

        tc_list = None
        if return_calls:
            tc_list = [
                {
                    "id": rc.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": rc.get("cmd_type"),
                        "arguments": json.dumps(rc.get("args", {}), ensure_ascii=False),
                    },
                }
                for rc in return_calls
                if rc.get("cmd_type") not in ("json_error", "unknown_tool")
            ] or None
        conversation.append_assistant_message(result or "", tool_calls=tc_list)
        return result, return_calls, conversation


def _state(core):
    return {
        "active": core._active_cid,
        "ready": list(core._ready_cids),
        "dormant": list(core._dormant_cids),
        "finished": list(core._finished),
    }


class TestEchoMineImage:
    def test_image_loads_world_programs_and_devices(self):
        core = load_image(str(GAME_IMAGE))

        conv = core.get_conversation(0)
        assert conv.name == "guide"
        assert conv.is_root

        assert core.unwrap("$MEM.world", for_llm=False).get_ctrl()["owner"] == "0"
        assert core.unwrap("$MEM.world.player.hp", for_llm=False) == "3"
        assert core.unwrap("$MEM.world.rooms.cavern.links", for_llm=False) == "south:entrance,east:vault"

        assert core.unwrap("$MEM.programs.guide", for_llm=False).get_ctrl()["type"] == "settingup"
        assert core.unwrap("$MEM.programs.ghost", for_llm=False).get_ctrl()["type"] == "settingup"
        assert core.unwrap("$MEM.programs.mechanics", for_llm=False).get_ctrl()["type"] == "python"
        assert core.unwrap("$MEM.handlers.hp_guard", for_llm=False).get_ctrl()["type"] == "python"

        assert "dice" in core.mem._devices
        assert "map" in core.mem._devices
        assert "dice" not in core.mem._data
        assert "map" not in core.mem._data

    def test_dice_device_returns_d6(self):
        core = load_image(str(GAME_IMAGE))
        dice = core.mem._devices["dice"]
        assert "d6" in core.unwrap("$MEM.dice")
        for _ in range(30):
            value = int(core.unwrap("$MEM.dice.d6"))
            assert 1 <= value <= 6

    def test_map_device_renders_player_position_from_memory(self):
        core = load_image(str(GAME_IMAGE))
        text = core.unwrap("$MEM.map")
        assert "[entrance]  *你在这里*" in text
        assert "[cavern]" in text and "[vault]" in text

        core.mem.set("$MEM.world.player.room_id", "cavern")
        text = core.unwrap("$MEM.map")
        assert "[cavern]  *你在这里*" in text
        assert "[entrance]  *你在这里*" not in text

    def test_hp_guard_clamps_and_replaces(self):
        core = load_image(str(GAME_IMAGE))
        core.events.register(
            0, "memory_write", "$MEM.world.player.hp", "$MEM.handlers.hp_guard", phase="before"
        )
        guide = core.get_conversation(0)

        MemoryWriteInstruction("tc1", 0, {"ref": "$MEM.world.player.hp", "content": "9"}).execute(core, guide)
        guide.user_batch.clear()
        assert core.unwrap("$MEM.world.player.hp", for_llm=False) == "3"

        MemoryWriteInstruction("tc2", 0, {"ref": "$MEM.world.player.hp", "content": "-4"}).execute(core, guide)
        guide.user_batch.clear()
        assert core.unwrap("$MEM.world.player.hp", for_llm=False) == "0"

        core.mem.set("$MEM.world.player.hp", "2")
        MemoryWriteInstruction("tc3", 0, {"ref": "$MEM.world.player.hp", "content": "not-a-number"}).execute(core, guide)
        assert core.unwrap("$MEM.world.player.hp", for_llm=False) == "2"
        responses = [r.content for r in guide.user_batch.tool_responses]
        assert any("HP 写入无效" in content for content in responses)


class TestEchoMineVictory:
    def test_victory_branch_closes_ghost_and_guide(self, capsys):
        core = load_image(str(GAME_IMAGE))
        core.mem.set("$MEM.world.player.room_id", "vault")
        core.mem.set("$MEM.world.player.gold", "10")
        core.lmu = ScriptedLMU({
            0: [
                ("", [tool("v1", "create_cmd", system_ref="$MEM.programs.ghost", para_ref="$MEM.para.llm")], None),
                ("", [
                    tool("v2", "memory_read", ref="$MEM.world.player.room_id"),
                    tool("v3", "memory_read", ref="$MEM.world.player.gold"),
                ], None),
                ("", [
                    tool("v4", "memory_write", ref="$MEM.world.status", content="won"),
                    tool("v5", "memory_write", ref="$MEM.outputs.-1", content="你带着 10 gold 离开了回声矿洞"),
                    tool("v6", "close_conversation", cid=1),
                    tool("v7", "close_conversation", cid=0),
                ], None),
            ],
        })

        core.run()
        capsys.readouterr()

        assert core.unwrap("$MEM.world.status", for_llm=False) == "won"
        assert core.mem._devices["outputs"]._data == ["你带着 10 gold 离开了回声矿洞"]
        st = _state(core)
        assert st["active"] is None
        assert st["ready"] == []
        assert set(st["finished"]) == {0, 1}


class TestEchoMineFullTurn:
    def test_scripted_game_turn_search_move_talk(self, capsys):
        core = load_image(str(GAME_IMAGE))
        core.lmu = ScriptedLMU({
            0: [
                # 初始化：注册 HP 守卫、创建 ghost、进入 playing、打印开场白
                ("", [
                    tool("s1", "listen_event", action="listen", event_type="memory_write",
                         target="$MEM.world.player.hp", handler_ref="$MEM.handlers.hp_guard", phase="before"),
                    tool("s2", "create_cmd", system_ref="$MEM.programs.ghost", para_ref="$MEM.para.llm"),
                    tool("s3", "memory_write", ref="$MEM.state.ghost_cid", content="1"),
                    tool("s4", "memory_write", ref="$MEM.world.status", content="playing"),
                    tool("s5", "memory_read", ref="$MEM.map"),
                    tool("s6", "memory_write", ref="$MEM.outputs.-1", content="欢迎来到回声矿洞"),
                ], None),
                # search：读骰子，把 resolve_search(6) 交给 plm.python
                ("", [
                    tool("p1", "memory_read", ref="$MEM.dice.d6"),
                    tool("p2", "memory_write", ref="$MEM.mechanics.request", content="print(resolve_search(6))"),
                    tool("p3", "create_sub", system_ref="$MEM.programs.mechanics",
                         user_ref="$MEM.mechanics.request", para_ref="$MEM.para.python"),
                ], None),
                # 结算 search=GOLD 3
                ("", [
                    tool("p4", "memory_read", ref="$MEM.world.player.gold"),
                    tool("p5", "memory_write", ref="$MEM.world.player.gold", content="3"),
                    tool("p6", "memory_write", ref="$MEM.world.last_event", content="search GOLD 3"),
                    tool("p7", "memory_read", ref="$MEM.world.turn"),
                    tool("p8", "memory_write", ref="$MEM.world.turn", content="1"),
                    tool("p9", "memory_write", ref="$MEM.outputs.-1", content="搜索：找到 3 gold"),
                ], None),
                # move north：交给 next_room
                ("", [
                    tool("m1", "memory_read", ref="$MEM.world.player.room_id"),
                    tool("m2", "memory_write", ref="$MEM.mechanics.request",
                         content="print(next_room('entrance', 'north'))"),
                    tool("m3", "create_sub", system_ref="$MEM.programs.mechanics",
                         user_ref="$MEM.mechanics.request", para_ref="$MEM.para.python"),
                ], None),
                # 应用 move=cavern
                ("", [
                    tool("m4", "memory_write", ref="$MEM.world.player.room_id", content="cavern"),
                    tool("m5", "memory_write", ref="$MEM.world.last_event", content="move cavern"),
                    tool("m6", "memory_read", ref="$MEM.world.turn"),
                    tool("m7", "memory_write", ref="$MEM.world.turn", content="2"),
                    tool("m8", "memory_read", ref="$MEM.map"),
                    tool("m9", "memory_write", ref="$MEM.outputs.-1", content="移动：到达 cavern"),
                ], None),
                # talk：投递给 ghost 后等待
                ("", [
                    tool("t1", "memory_read", ref="$MEM.state.ghost_cid"),
                    tool("t2", "send_instruction", cid=1,
                         content="GHOST_ASK 这里安全吗 | 完成后 send_instruction(cid=0, content=GHOST_REPLY <单行回复>)，然后不要调用工具直接结束"),
                ], None),
                ("waiting", [], None),
                # 被 ghost 唤醒
                ("", [tool("w1", "memory_write", ref="$MEM.outputs.-1", content="GHOST_REPLY 风里有金子的味道")], None),
                ("ready", [], None),
            ],
            1: [
                ("", [
                    tool("g1", "memory_read", ref="$MEM.world.player.room_id"),
                    tool("g2", "memory_read", ref="$MEM.world.rooms"),
                    tool("g3", "send_instruction", cid=0, content="GHOST_REPLY 风里有金子的味道"),
                ], None),
                ("done", [], None),
            ],
        })

        core.run()
        capsys.readouterr()

        # 世界状态
        assert core.unwrap("$MEM.world.player.hp", for_llm=False) == "3"
        assert core.unwrap("$MEM.world.player.gold", for_llm=False) == "3"
        assert core.unwrap("$MEM.world.player.room_id", for_llm=False) == "cavern"
        assert core.unwrap("$MEM.world.turn", for_llm=False) == "2"
        assert core.unwrap("$MEM.world.status", for_llm=False) == "playing"

        outputs = core.mem._devices["outputs"]._data
        assert outputs == [
            "欢迎来到回声矿洞",
            "搜索：找到 3 gold",
            "移动：到达 cavern",
            "GHOST_REPLY 风里有金子的味道",
        ]

        # create_sub 的结算文本进入 guide 历史
        guide = core.get_conversation(0)
        tool_contents = [m.content for m in guide.messages if m.role == "tool"]
        assert any("GOLD 3" in content for content in tool_contents)
        assert any(content.strip() == "cavern" for content in tool_contents)

        # ghost 是 create_cmd 子对话，有独立历史
        ghost = core.get_conversation(1)
        assert ghost.name == "ghost"
        assert any("GHOST_ASK" in m.content for m in ghost.messages if m.role == "user")

        # HP 守卫已注册
        assert "target=$MEM.world.player.hp" in core.unwrap("$MEM.listeners")

        # 终态：guide / ghost / 两个 mechanics 亚对话全部交互结束休眠，无 ready
        st = _state(core)
        assert st["active"] is None
        assert st["ready"] == []
        assert set(st["dormant"]) == {0, 1, 2, 3}
        assert st["finished"] == []

        # 调度轨迹：guide 7 次，ghost 2 次，guide 收尾 2 次
        assert [cid for cid, _ in core.lmu.calls] == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

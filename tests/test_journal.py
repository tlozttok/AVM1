"""回合级效果日志（avm/journal.py）测试。

验收（计划 A 批次）：日志重放一致、幂等、半截日志、投递/对话生命周期/监听器的
效果可重放；全部用脚本化执行器，不调真实 API。
"""

import json
import os

import pytest

from avm.journal import Journal, JournalError
from avm.snapshot import snapshot
from persist_helpers import (
    assert_same_state, load_core, out_path, run_turns, write_image,
)

# 三轮：写内存 → 建节点并写 → 无指令（对话休眠、run 结束）
THREE_TURNS = [
    ("", [{"call_id": "c1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v1"}}]),
    ("", [{"call_id": "c2", "cmd_type": "memory_make",
           "args": {"ref": "$MEM", "key": "made", "mem_type": "dict"}},
          {"call_id": "c3", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.made", "content": "x"}}]),
    ("done", []),
]


def _run(image_path, responses, turns=None):
    core = load_core(image_path, responses)
    core.attach_journal(core.journal_path)
    if turns is None:
        run_turns(core, len(responses))
    else:
        run_turns(core, turns)
    return core


def _replay(image_path):
    core = load_core(image_path)
    core.attach_journal(core.journal_path)
    applied = core.replay_journal()
    return core, applied


class TestJournalFile:
    def test_append_assigns_seq_and_reads(self, tmp_path):
        journal = Journal(str(tmp_path / "j.jsonl"))
        assert journal.append({"cid": 0}) == 1
        assert journal.append({"cid": 0}) == 2
        assert journal.last_seq() == 2
        assert [r["seq"] for r in journal.read()] == [1, 2]
        assert [r["seq"] for r in journal.read(after_seq=1)] == [2]

    def test_partial_last_line_is_ignored(self, tmp_path):
        path = str(tmp_path / "j.jsonl")
        journal = Journal(path)
        journal.append({"cid": 0})
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"seq": 2, "cid": 0')  # 半行
        # 重新打开：last_seq 停在上一条提交
        reopened = Journal(path)
        assert reopened.last_seq() == 1
        assert reopened.read() == [{"cid": 0, "seq": 1}]

    def test_truncate_keeps_prefix(self, tmp_path):
        journal = Journal(str(tmp_path / "j.jsonl"))
        for _ in range(3):
            journal.append({"cid": 0})
        journal.truncate(2)
        assert journal.last_seq() == 2
        assert [r["seq"] for r in journal.read()] == [1, 2]


class TestReplay:
    def test_replay_reproduces_end_state(self, tmp_path):
        image = write_image(tmp_path)
        mine = _run(image, THREE_TURNS)
        assert mine.mem["src"] == "v1"
        assert "made" in mine.mem

        replayed, applied = _replay(image)
        assert applied == 3
        assert replayed._journal_seq == 3
        assert_same_state(mine, replayed)

    def test_replay_is_idempotent(self, tmp_path):
        image = write_image(tmp_path)
        _run(image, THREE_TURNS)
        replayed, applied = _replay(image)
        before = snapshot(replayed)
        second = replayed.replay_journal()
        assert second == 0
        assert snapshot(replayed) == before

    def test_truncated_log_replays_previous_commit(self, tmp_path):
        image = write_image(tmp_path)
        _run(image, THREE_TURNS)
        journal_file = out_path(image, "run.journal")
        with open(journal_file, encoding="utf-8") as f:
            lines = f.readlines()
        with open(journal_file, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])

        expected = _run(image, THREE_TURNS[:2], turns=2)
        # 上面这次运行会往同一份日志追加两条新记录（seq 3/4），重放前先恢复文件
        with open(journal_file, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])

        replayed, applied = _replay(image)
        assert applied == 2
        assert replayed._journal_seq == 2
        assert_same_state(expected, replayed)

    def test_log_records_conversation_lifecycle_and_delivery(self, tmp_path):
        """create_cmd + send_instruction 的对话创建、投递、调度都能重放。"""
        responses = [
            ("", [{"call_id": "c1", "cmd_type": "create_cmd",
                   "args": {"system_ref": "$MEM.programs.main",
                            "para_ref": "$MEM.model_params"}},
                  {"call_id": "c2", "cmd_type": "send_instruction",
                   "args": {"cid": 1, "content": "子对话任务"}}]),
            ("parent done", []),
            ("child done", []),
        ]
        image = write_image(tmp_path)
        mine = _run(image, responses)
        assert set(mine._conv_by_cid) == {0, 1}
        child = mine._conv_by_cid[1]
        assert [m.content for m in child.messages][-1] == "child done"

        replayed, applied = _replay(image)
        assert applied == 3
        assert set(replayed._conv_by_cid) == {0, 1}
        assert_same_state(mine, replayed)

    def test_listener_register_and_cancel_replay(self, tmp_path):
        handler = {
            "kind": "dict",
            "ctrl": {"type": "handler"},
            "value": {
                "content": {"kind": "str",
                            "value": "def handle(event):\n    return {}\n"},
                "condition": {"kind": "str",
                              "value": json.dumps({"events": ["memory_write"],
                                                   "phase": ["before"],
                                                   "target": {"kind": "exact",
                                                              "ref": "$MEM.src"}})},
            },
        }
        mem = {
            "model_params": {
                "kind": "dict", "ctrl": {"type": "para"},
                "value": {"model": {"kind": "str", "value": "test-model"}},
            },
            "programs": {
                "kind": "dict", "value": {"main": {
                    "kind": "dict", "ctrl": {"type": "settingup"},
                    "value": {"content": {"kind": "str", "value": "sys"}},
                }},
            },
            "handlers": {"kind": "dict", "value": {"h": handler}},
            "src": {"kind": "str", "value": "hello"},
        }
        responses = [
            ("", [{"call_id": "c1", "cmd_type": "listen_event",
                   "args": {"action": "listen", "handler_ref": "$MEM.handlers.h"}}]),
            ("", [{"call_id": "c2", "cmd_type": "listen_event",
                   "args": {"action": "cancel", "listener_id": 1}}]),
            ("done", []),
        ]
        image = write_image(tmp_path, mem=mem)
        mine = _run(image, responses)
        assert mine.events.listeners == []

        replayed, applied = _replay(image)
        assert applied == 3
        assert replayed.events.listeners == []
        assert_same_state(mine, replayed)

    def test_no_journal_means_no_records(self, tmp_path):
        image = write_image(tmp_path, journal=False)
        core = load_core(image, THREE_TURNS)
        run_turns(core, 3)
        assert core.journal is None
        assert core._turn_effects == []
        assert not os.path.exists(out_path(image, "run.journal"))

    def test_detach_journal_stops_recording(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 1)
        assert core.journal.last_seq() == 1
        core.detach_journal()
        run_turns(core, 1)
        assert Journal(core.journal_path).last_seq() == 1


class TestApplyErrors:
    def test_missing_conversation_raises(self, tmp_path):
        from avm.journal import apply

        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 1)
        record = core.journal.read()[0]
        record["cid"] = 99
        with pytest.raises(JournalError, match="不存在"):
            apply(core, record)

    def test_unknown_effect_raises(self, tmp_path):
        from avm.journal import apply

        image = write_image(tmp_path)
        core = load_core(image, THREE_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 1)
        record = core.journal.read()[0]
        record["changes"] = [{"kind": "effect", "effect": {"kind": "no_such_effect"}}]
        with pytest.raises(JournalError, match="未知效果"):
            apply(core, record)


class TestSubConversationLifecycle:
    def test_create_sub_return_and_delete_replay(self, tmp_path):
        """亚对话创建 → 返回写回父批次 → 用完即删，整条链路可重放。"""
        from avm.types import ToolMessage

        responses = [
            ("", [{"call_id": "c1", "cmd_type": "create_sub",
                   "args": {"system_ref": "$MEM.programs.main",
                            "user_ref": "$MEM.src",
                            "para_ref": "$MEM.model_params"}}]),
            ("子对话结果", []),
            ("父对话收尾", []),
        ]
        image = write_image(tmp_path)
        mine = _run(image, responses)
        assert set(mine._conv_by_cid) == {0}
        tool_msgs = [m for m in mine._conv_by_cid[0].messages if isinstance(m, ToolMessage)]
        assert tool_msgs[-1].content == "子对话结果"

        replayed, applied = _replay(image)
        assert applied == 3
        assert set(replayed._conv_by_cid) == {0}
        assert_same_state(mine, replayed)

    def test_close_conversation_replay(self, tmp_path):
        responses = [
            ("", [{"call_id": "c1", "cmd_type": "create_cmd",
                   "args": {"system_ref": "$MEM.programs.main",
                            "para_ref": "$MEM.model_params"}}]),
            ("", [{"call_id": "c2", "cmd_type": "close_conversation",
                   "args": {"cid": 1}}]),
            ("done", []),
        ]
        image = write_image(tmp_path)
        mine = _run(image, responses)
        assert mine._conv_by_cid[1].cid in mine._finished
        replayed, applied = _replay(image)
        assert applied == 3
        assert 1 in replayed._finished
        assert_same_state(mine, replayed)


class TestImagePersistenceFields:
    def test_paths_resolve_relative_to_image_dir(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image)
        assert core.journal_path == out_path(image, "run.journal")
        assert core.snapshot_path == out_path(image, "run.snapshot.json")

    def test_bad_journal_to_is_rejected(self, tmp_path):
        from avm.image import ImageError, load_image
        from avm.core import Core
        import json as _json

        image = _json.loads(open(write_image(tmp_path), encoding="utf-8").read())
        image["journal_to"] = 3
        path = tmp_path / "bad.json"
        path.write_text(_json.dumps(image), encoding="utf-8")
        with pytest.raises(ImageError, match="journal_to"):
            load_image(Core(), str(path))

    def test_bad_snapshot_to_is_rejected(self, tmp_path):
        from avm.image import ImageError, load_image
        from avm.core import Core
        import json as _json

        image = _json.loads(open(write_image(tmp_path), encoding="utf-8").read())
        image["snapshot_to"] = ""
        path = tmp_path / "bad.json"
        path.write_text(_json.dumps(image), encoding="utf-8")
        with pytest.raises(ImageError, match="snapshot_to"):
            load_image(Core(), str(path))

    def test_restore_skips_init_and_conflicts_with_checkpoint_in(self, tmp_path):
        from avm.image import ImageError, load_image
        from avm.core import Core

        image = write_image(tmp_path)
        core = load_image(Core(), image, restore=True)
        assert core._conv_by_cid == {}
        with pytest.raises(ImageError, match="不能同时使用"):
            load_image(Core(), image, restore=True, checkpoint_in="cp.json")


class TestFailedTurnsAreNotCommitted:
    def test_api_error_retry_commits_only_successful_turn(self, tmp_path):
        from avm.exceptions import APIError
        from persist_helpers import ScriptedLMU

        class FlakyLMU(ScriptedLMU):
            def __init__(self, responses):
                super().__init__(responses)
                self.fail_next = True

            def exec(self, conversation, para):
                if self.fail_next:
                    self.fail_next = False
                    raise APIError("模拟外部失败")
                return super().exec(conversation, para)

        image = write_image(tmp_path)
        core = load_core(image)
        core.lmu = FlakyLMU([
            ("", [{"call_id": "c1", "cmd_type": "memory_write",
                   "args": {"ref": "$MEM.src", "content": "v1"}}]),
            ("done", []),
        ])
        core.api_error_policy = lambda error, attempt: "retry"
        core.attach_journal(core.journal_path)
        run_turns(core, 2)

        records = core.journal.read()
        assert [record["seq"] for record in records] == [1, 2]
        effects = [
            change["effect"] for change in records[0]["changes"]
            if change["kind"] == "effect"
        ]
        assert effects[0]["kind"] == "memory_set"

        replayed, applied = _replay(image)
        assert applied == 2
        assert_same_state(core, replayed)


class TestReservedConversationRecord:
    """保留对话（cid -1）的回合：按普通回合重放，对话随快照/日志恢复。"""

    def test_operator_turn_replays_into_reserved_conversation(self, tmp_path):
        from avm.console import OperatorChannel
        from avm.core import PANEL_CID

        image = write_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        channel = OperatorChannel()
        channel.submit_action({
            "tool": "memory_write",
            "arguments": {"ref": "$MEM.dst", "content": "hi"},
        })
        panel = core.bind_console_conversation(channel)
        assert panel.cid == PANEL_CID
        core._active_cid = panel.cid
        core.advance_conversation()

        records = core.journal.read()
        assert len(records) == 1
        assert records[0]["cid"] == PANEL_CID

        restored = load_core(image)
        restored.attach_journal(restored.journal_path)
        assert restored.replay_journal() == 1
        assert restored.mem["dst"] == "hi"
        conv = restored.get_conversation(PANEL_CID)
        assert conv is not None and restored.is_hidden_cid(PANEL_CID)

    def test_reserved_conversation_survives_snapshot_and_journal_tail(self, tmp_path):
        import json as _json

        from avm.console import OperatorChannel
        from avm.core import Core, PANEL_CID
        from avm.image import load_image

        image = write_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        channel = OperatorChannel()
        panel = core.bind_console_conversation(channel)
        assert panel.cid == PANEL_CID

        def act(ref, value):
            channel.submit_action({
                "tool": "memory_write",
                "arguments": {"ref": ref, "content": value},
            })
            core._active_cid = panel.cid
            core.advance_conversation()

        act("$MEM.dst", "in-snapshot")
        snap_path = tmp_path / "snap.json"
        snap_path.write_text(_json.dumps(snapshot(core), ensure_ascii=False), encoding="utf-8")
        act("$MEM.dst", "in-tail")

        restored = load_image(Core(), image, restore=True)
        restored.attach_journal(restored.journal_path)
        restored.restore_from(str(snap_path))
        assert restored.mem["dst"] == "in-snapshot"
        assert restored.replay_journal() == 1
        assert restored.mem["dst"] == "in-tail"
        conv = restored.get_conversation(PANEL_CID)
        assert conv is not None and restored.is_hidden_cid(PANEL_CID)


FOUR_TURNS = [
    ("", [{"call_id": "c1", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v1"}}]),
    ("", [{"call_id": "c2", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v2"}}]),
    ("", [{"call_id": "c3", "cmd_type": "memory_write",
           "args": {"ref": "$MEM.src", "content": "v3"}}]),
    ("done", []),
]


def _image_with_snapshot_every(tmp_path, every):
    image = write_image(tmp_path)
    with open(image, encoding="utf-8") as f:
        doc = json.load(f)
    doc.setdefault("meta", {})["snapshot_every"] = every
    with open(image, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    return image


class TestRecordShape:
    def test_records_are_ordered_changes(self, tmp_path):
        image = write_image(tmp_path)
        core = _run(image, THREE_TURNS)
        records = core.journal.read()
        assert records
        allowed_kinds = {"message", "effect", "batch", "scheduler"}
        scheduler_keys = {
            "kind", "active", "ready_add", "ready_remove", "dormant_add",
            "dormant_remove", "finished_add", "finished_remove",
            "activation_instructions", "next_cid",
        }
        for record in records:
            assert record["v"] == 2
            for legacy in ("history", "call", "batch", "scheduler", "effects"):
                assert legacy not in record
            assert isinstance(record["changes"], list)
            for change in record["changes"]:
                assert change["kind"] in allowed_kinds
                if change["kind"] == "scheduler":
                    assert set(change) <= scheduler_keys
            assert any(c["kind"] in ("message", "effect") for c in record["changes"])

    def test_old_format_record_is_rejected(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image)
        core.attach_journal(core.journal_path)
        core.journal.append({"cid": 0, "history": [], "effects": []})
        with pytest.raises(JournalError, match="旧格式"):
            core.replay_journal()


class TestPeriodicSnapshot:
    def test_snapshot_written_every_n_committed_turns(self, tmp_path):
        from avm.snapshot import read_snapshot

        image = _image_with_snapshot_every(tmp_path, 2)
        core = load_core(image, FOUR_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 4)
        doc = read_snapshot(core.snapshot_path)
        assert doc["seq"] == 4
        assert core.journal.last_seq() == 4

    def test_snapshot_plus_tail_replays_to_same_state(self, tmp_path):
        image = write_image(tmp_path)
        core = load_core(image, FOUR_TURNS)
        core.attach_journal(core.journal_path)
        run_turns(core, 2)
        snap_path = str(tmp_path / "snap.json")
        core.snapshot_to(snap_path)
        run_turns(core, 2)

        restored = load_core(image)
        restored.restore_from(snap_path)
        restored.attach_journal(restored.journal_path)
        assert restored.replay_journal() == 2
        assert_same_state(core, restored)
        assert restored._journal_seq == core._journal_seq

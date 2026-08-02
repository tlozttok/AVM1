"""变量监测器测试：帧序列、diff、trail、环形上限、全文文件、异常路径"""

import os

import pytest

from avm.core import Core
from avm.monitor import Monitor
from avm.types import MetaDict


class MockLMU:
    def __init__(self, responses=None):
        self.responses = list(responses) if responses else []
        self.call_index = 0
        self.last_call = None

    def _next(self, *args):
        if self.call_index >= len(self.responses):
            raise RuntimeError(f"MockLMU 第 {self.call_index} 次调用没有预设响应")
        resp = self.responses[self.call_index]
        self.call_index += 1
        if callable(resp):
            return resp(*args)
        return resp

    def exec(self, conversation, para):
        return self._next(conversation, para)


def make_core(responses=None):
    core = Core()
    if responses is not None:
        core.lmu = MockLMU(responses)
    return core


def _new_root(core):
    core.mem["model_params"] = MetaDict(data={"model": "test"})
    return core.start("sys", "hello")


def _with_last_call(core, result="ok", messages=None, tool_calls=None, error=None):
    core.lmu.last_call = {
        "model": "test",
        "messages": messages or [{"role": "user", "content": "full request text"}],
        "result": result,
        "tool_calls": tool_calls or [],
        "elapsed_ms": 12.5,
        "error": error,
    }


class TestFrameSequence:
    def test_run_records_baseline_and_advance_frames(self):
        core = make_core([("done", [], None)])
        _new_root(core)
        core.run()

        frames = core.monitor.frames()
        assert [f.seq for f in frames] == [0, 1]
        assert frames[0].sched["active_cid"] == 0   # 基线：启动后第一轮推进前
        assert frames[1].sched["active_cid"] is None  # 推进后对话完成

    def test_frame_after_advance_captures_new_batch(self):
        """一次对话推进后，某个对话多了一个输入 —— 帧的 batch 应可见"""
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.out", "content": "x"}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        core.advance_conversation()

        f = core.monitor.frame(1)
        assert f.conversations[0]["batch"]["tool_responses"] == ["Success set: $MEM.out"]
        assert f.sched["active_cid"] == 0  # 无 create，对话保持活跃

    def test_ring_buffer_limit(self):
        core = make_core()
        core.monitor = Monitor(max_frames=3)
        for _ in range(5):
            core.monitor.record_baseline(core)

        frames = core.monitor.frames()
        assert len(frames) == 3
        assert [f.seq for f in frames] == [2, 3, 4]

    def test_conversation_state_derivation(self):
        core = make_core([
            (None, [{"call_id": "c", "cmd_type": "create_cmd", "args": {"system_ref": "$MEM.s", "para_ref": "$MEM.p"}}], None),
        ])
        _new_root(core)
        core.mem["s"] = "x"
        core.mem["p"] = MetaDict(data={"model": "test"})
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        core.advance_conversation()

        f = core.monitor.frame(1)
        assert f.conversations[0]["state"] == "active"   # 父保持活跃
        assert f.conversations[1]["state"] == "dormant"  # 子对话休眠等待指令
        assert f.sched["dormant_cids"] == [1]


class TestQuery:
    def test_trail(self):
        core = make_core([("done", [], None)])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        assert core.monitor.trail("sched.active_cid") == [(0, 0), (1, None)]
        assert core.monitor.trail("conversations.0.msgs") == [(0, 2), (1, 2)]

    def test_diff_between_frames(self):
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.out", "content": "x"}}], None),
        ])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        d = core.monitor.diff(0, 1)
        assert ("conversations.0.batch.tool_responses", ([], ["Success set: $MEM.out"])) in d.items()

    def test_diff_missing_frame_raises(self):
        core = make_core()
        with pytest.raises(KeyError):
            core.monitor.diff(0, 99)

    def test_find(self):
        core = make_core([("done", [], None)])
        _new_root(core)
        core.run()

        finished = core.monitor.find(lambda f: f.sched["active_cid"] is None)
        assert len(finished) == 1
        assert finished[0].seq == 1

    def test_lmu_fields_from_last_call(self):
        core = make_core([("ok", [], None)])
        _with_last_call(core, result="answer", tool_calls=[{"cmd_type": "memory_read", "call_id": "tc1", "args": {"ref": "$MEM.x"}}])
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        f = core.monitor.frame(1)
        assert f.lmu["result"] == "answer"
        assert f.lmu["elapsed_ms"] == 12.5
        assert f.lmu["tool_calls"][0]["cmd_type"] == "memory_read"


class TestTruncation:
    def test_result_truncated(self):
        core = make_core([("x" * 500, [], None)])
        _with_last_call(core, result="x" * 500)
        core.monitor = Monitor(max_item_len=10)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        assert core.monitor.frame(1).lmu["result"] == "x" * 10 + "..."

    def test_batch_content_truncated(self):
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.out", "content": "y"}}], None),
        ])
        core.monitor = Monitor(max_item_len=8)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        resp = core.monitor.frame(1).conversations[0]["batch"]["tool_responses"]
        assert resp == ["Success " + "..."]  # "Success set: $MEM.out" 截断为 8 字符


class TestExceptionPath:
    def test_exception_propagates_and_records_error_frame(self):
        class BoomLMU(MockLMU):
            def exec(self, conversation, para):
                raise RuntimeError("API failed")

        core = make_core()
        core.lmu = BoomLMU()
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        with pytest.raises(RuntimeError):
            core.advance_conversation()

        f = core.monitor.frame(1)
        assert "API failed" in f.lmu["error"]
        assert f.sched["active_cid"] == 0  # 崩溃点状态保留

    def test_real_lmu_last_call_captured_on_api_error(self):
        """真实 LMU.exec：API 抛异常 → last_call 记录请求全文 + 异常传播"""
        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                raise ConnectionError("network down")

        class FakeChat:
            completions = FakeCompletions

        class FakeClient:
            chat = FakeChat

        from avm.core import LMU

        lmu = LMU()
        lmu._client = FakeClient
        core = make_core()
        core.lmu = lmu
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        with pytest.raises(ConnectionError):
            core.advance_conversation()

        assert lmu.last_call["error"] == "ConnectionError: network down"
        assert lmu.last_call["messages"][-1]["role"] == "user"
        assert core.monitor.frame(1).lmu["error"] == "ConnectionError: network down"


class TestTranscript:
    def test_transcript_sections_and_full_text(self, tmp_path):
        path = os.path.join(tmp_path, "t.txt")
        core = make_core([("ok", [], None)])
        _with_last_call(
            core,
            result="full response text",
            messages=[{"role": "user", "content": "full request text"}],
        )
        core.monitor = Monitor(transcript_path=path)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        text = open(path, encoding="utf-8").read()
        assert "===== FRAME 1 =====" in text
        assert "full request text" in text
        assert "full response text" in text

    def test_transcript_error_section(self, tmp_path):
        path = os.path.join(tmp_path, "t.txt")
        core = make_core([("ok", [], None)])
        _with_last_call(core, result=None, error="APIConnectionError: boom")
        core.monitor = Monitor(transcript_path=path)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)
        core.advance_conversation()

        text = open(path, encoding="utf-8").read()
        assert "--- error ---" in text
        assert "APIConnectionError: boom" in text

    def test_transcript_append_multiple_sections(self, tmp_path):
        path = os.path.join(tmp_path, "t.txt")
        core = make_core([
            (None, [{"call_id": "tc1", "cmd_type": "memory_write", "args": {"ref": "$MEM.a", "content": "1"}}], None),
            (None, [{"call_id": "tc2", "cmd_type": "memory_write", "args": {"ref": "$MEM.b", "content": "2"}}], None),
        ])
        core.monitor = Monitor(transcript_path=path)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.monitor.record_baseline(core)

        for i in range(2):
            _with_last_call(core, result=f"resp{i}")
            core.advance_conversation()

        text = open(path, encoding="utf-8").read()
        assert text.count("===== FRAME") == 2
        assert "===== FRAME 1 =====" in text
        assert "===== FRAME 2 =====" in text

    def test_no_transcript_without_last_call(self, tmp_path):
        path = os.path.join(tmp_path, "t.txt")
        core = make_core([("ok", [], None)])
        core.monitor = Monitor(transcript_path=path)
        _new_root(core)
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()

        assert not os.path.exists(path)

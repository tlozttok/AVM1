"""中断向量表测试

覆盖：向量注册、挂起去重、安全点派发的三种结果（夺取 / 空转 / 丢弃）、
被中断对话回到就绪队首、以及继续之后它先跑。
"""

import queue
import threading
import time

import pytest

from avm.core import Core


class RecordingChannel:
    """最小的操作员通道：plm.operator 只需要 wait_action 与 publish。"""

    def __init__(self):
        self._actions = queue.Queue()
        self.events = []

    def wait_action(self):
        return self._actions.get()

    def submit(self, action):
        self._actions.put(action)

    def publish(self, kind, data):
        self.events.append((kind, data))

    def kinds(self):
        return [k for k, _ in self.events]


class MockLMU:
    """假模型：按 cid 返回内容，可在每轮执行时挂钩子。"""

    def __init__(self, on_exec=None):
        self.on_exec = on_exec
        self.ran = []

    def exec(self, conversation, para):
        self.ran.append(conversation.cid)
        if self.on_exec is not None:
            self.on_exec(conversation)
        return (f"done-{conversation.cid}", [], None)


def make_core(on_exec=None):
    core = Core()
    core.lmu = MockLMU(on_exec)
    return core


def attach_panel(core):
    """建一条操作员对话并把操作员中断指向它。"""
    channel = RecordingChannel()
    panel = core.start_operator_conversation(channel)
    core.interrupts.register("operator", panel.cid)
    return panel, channel


def wait_for(predicate, timeout=5.0):
    # 单调钟：墙钟会被系统校时前后跳，用它算 deadline 会提前放弃等待
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestVectors:
    def test_register_and_read_back(self):
        core = Core()
        core.interrupts.register("operator", 7)
        assert core.interrupts.vectors == {"operator": 7}

    def test_vectors_is_a_copy(self):
        core = Core()
        core.interrupts.register("operator", 7)
        core.interrupts.vectors["operator"] = 9
        assert core.interrupts.vectors == {"operator": 7}

    def test_register_rejects_bad_arguments(self):
        core = Core()
        with pytest.raises(ValueError):
            core.interrupts.register("", 1)
        with pytest.raises(ValueError):
            core.interrupts.register("operator", "1")

    def test_unregister(self):
        core = Core()
        core.interrupts.register("operator", 7)
        core.interrupts.unregister("operator")
        assert core.interrupts.vectors == {}

    def test_register_replaces_previous_target(self):
        core = Core()
        core.interrupts.register("operator", 1)
        core.interrupts.register("operator", 2)
        assert core.interrupts.vectors == {"operator": 2}


class TestPending:
    def test_same_type_not_queued_twice(self):
        core = Core()
        core.interrupts.raise_("operator")
        core.interrupts.raise_("operator")
        assert core.interrupts.pending_count == 1
        assert core.interrupts.pending_types() == ["operator"]

    def test_different_types_queue_separately(self):
        core = Core()
        core.interrupts.raise_("a")
        core.interrupts.raise_("b")
        assert core.interrupts.pending_types() == ["a", "b"]

    def test_dispatch_clears_pending(self):
        core = Core()
        core.interrupts.raise_("operator")
        core.interrupts.dispatch()
        assert core.interrupts.pending_count == 0
        assert core.interrupts.pending_types() == []

    def test_raise_rejects_empty_type(self):
        core = Core()
        with pytest.raises(ValueError):
            core.interrupts.raise_("")

    def test_dispatch_is_thread_safe(self):
        """控制台在后端线程里挂起中断；派发在 Core 线程里做。"""
        core = Core()
        core.interrupts.register("operator", core.start_operator_conversation(
            RecordingChannel()).cid)
        threads = [threading.Thread(target=core.interrupts.raise_, args=("op%d" % i,))
                   for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert core.interrupts.pending_count == 16
        core.interrupts.dispatch()
        assert core.interrupts.pending_count == 0


class TestDispatch:
    def _three_way(self):
        """准备好 first 活跃、second 就绪、面板休眠的局面。"""
        core = make_core()
        panel, _ = attach_panel(core)
        first = core.start("sys", "one")
        second = core.start("sys", "two")
        core._pick_next_active()
        assert core._active_cid == first.cid
        assert panel.cid in core._dormant_cids
        return core, panel, first, second

    def test_switch_preempts_active_into_ready_front(self):
        core, panel, first, second = self._three_way()
        core.interrupts.raise_("operator")
        records = core.interrupts.dispatch()
        assert records == [{
            "type": "operator", "target": panel.cid, "action": "switch",
            "preempted": first.cid, "payload": None,
        }]
        assert core._active_cid == panel.cid
        assert core._ready_cids == [first.cid, second.cid]

    def test_switch_takes_target_out_of_waiting_queues(self):
        """目标若仍留在休眠队列里，本轮结束会被当成"自己让出"而立刻交出执行权。"""
        core, panel, first, second = self._three_way()
        core.interrupts.raise_("operator")
        core.interrupts.dispatch()
        assert panel.cid not in core._dormant_cids
        assert panel.cid not in core._ready_cids

    def test_switch_resets_instruction_budget_chain(self):
        """中断是调用链断开：与 _pick_next_active 取新对话时同样重新给满预算。"""
        core, panel, first, second = self._three_way()
        core._activation_instructions = 7
        core.interrupts.raise_("operator")
        core.interrupts.dispatch()
        assert core._activation_instructions == 0

    def test_noop_when_target_already_active(self):
        core, panel, first, second = self._three_way()
        core._active_cid = panel.cid
        core.interrupts.raise_("operator")
        records = core.interrupts.dispatch()
        assert records[0]["action"] == "noop"
        assert core._active_cid == panel.cid
        assert core._ready_cids == [second.cid]

    def test_dropped_when_no_vector(self, capsys):
        core, panel, first, second = self._three_way()
        core.interrupts.raise_("nobody")
        records = core.interrupts.dispatch()
        assert records[0]["action"] == "dropped"
        assert core._active_cid == first.cid
        assert "丢弃" in capsys.readouterr().err

    def test_dropped_when_target_closed(self, capsys):
        core, panel, first, second = self._three_way()
        core._finished.add(panel.cid)
        core.interrupts.raise_("operator")
        records = core.interrupts.dispatch()
        assert records[0]["action"] == "dropped"
        assert core._active_cid == first.cid
        assert core._ready_cids == [second.cid]
        assert "丢弃" in capsys.readouterr().err

    def test_last_dispatch_is_kept_for_inspection(self):
        core, panel, first, second = self._three_way()
        core.interrupts.raise_("operator")
        core.interrupts.dispatch()
        assert core.interrupts.last_dispatch[0]["target"] == panel.cid


class TestPauseAndResume:
    def test_pause_then_resume_returns_to_preempted_conversation(self):
        """暂停抢走执行权；继续之后被打断的对话先跑完，面板回到休眠。"""
        core = make_core()

        def on_exec(conv):
            if conv.cid == first.cid:
                core.interrupts.raise_("operator")

        panel, channel = attach_panel(core)
        first = core.start("sys", "one")
        second = core.start("sys", "two")
        core.lmu.on_exec = on_exec

        thread = threading.Thread(target=core.run, daemon=True)
        thread.start()
        try:
            assert wait_for(lambda: core._active_cid == panel.cid), "面板没有取得执行权"
            assert core._ready_cids == [second.cid]

            channel.submit({"end": True})
            thread.join(timeout=5)
            assert not thread.is_alive(), "Core.run 没有在面板让出后结束"
        finally:
            channel.submit({"end": True})  # 断言失败时也把执行器放出来

        assert panel.cid in core._dormant_cids
        assert core.lmu.ran == [first.cid, second.cid]
        assert core._active_cid is None

    def test_run_without_vectors_behaves_as_before(self):
        """中断表为空时，run 的循环体多一次无副作用的派发。"""
        core = make_core()
        first = core.start("sys", "one")
        second = core.start("sys", "two")
        core.run()
        assert core.lmu.ran == [first.cid, second.cid]
        assert core._active_cid is None
        assert sorted(core._dormant_cids) == sorted([first.cid, second.cid])

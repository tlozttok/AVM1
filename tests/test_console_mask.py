"""控制台的内存屏蔽与设备屏蔽测试

覆盖：detach/attach 往返保真（meta/ctrl/owner）、还原冲突不覆盖、设备卸载重挂保住
实例状态、设备调试台默认空、控制台直接调用设备的 run 且全程不产生事件、
输入设备按启动标志在终端与面板队列之间切换。
"""

import threading
import time

import pytest

from avm.console import Console
from avm.core import Core
from avm.exceptions import VMMemoryError
from avm.memory_device import InputsListDevice, MemoryDevice
from avm.types import MetaDict


class EchoDevice(MemoryDevice):
    """有自己内部状态的设备：用来验证卸载重挂不会重建实例。"""

    def __init__(self):
        self.calls = []
        self.state = {"n": 0}

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return f"echo 设备，已调用 {len(self.calls)} 次"

    def debug_actions(self):
        def bump(args):
            self.calls.append(args)
            self.state["n"] += 1
            return {"n": self.state["n"]}

        def boom(args):
            raise RuntimeError("设备自己炸了")

        return [
            {"name": "自增", "description": "把内部计数加一", "run": bump},
            {"name": "爆炸", "description": "用来验证错误被包住", "run": boom},
        ]


def make_core():
    core = Core()
    core.mem["tree"] = {
        "leaf": "hello",
        "nested": {"deep": MetaDict(data={"v": "x"}, metadata="深层节点",
                                    ctrl={"type": "note", "owner": "7"})},
        "items": ["a", "b", "c"],
    }
    core.mem.mount("echo", EchoDevice())
    return core


class TestMemoryMask:
    def test_detach_returns_the_node_and_removes_it(self):
        core = make_core()
        node = core.mem.detach("$MEM.tree.leaf")
        assert node == "hello"
        assert "leaf" not in core.mem["tree"]

    def test_attach_restores_everything(self):
        core = make_core()
        node = core.mem.detach("$MEM.tree.nested.deep")
        assert core.mem._locate("$MEM.tree.nested.deep")  # 只是确认路径还能解析
        assert "deep" not in core.mem["tree"]["nested"]

        core.mem.attach("$MEM.tree.nested.deep", node)
        restored = core.mem["tree"]["nested"]["deep"]
        assert restored is node, "原样放回，不重新包装"
        assert restored.get_metadata() == "深层节点"
        assert restored.get_ctrl() == {"type": "note", "owner": "7"}
        assert restored["v"] == "x"

    def test_attach_reports_conflict_instead_of_overwriting(self):
        core = make_core()
        node = core.mem.detach("$MEM.tree.leaf")
        core.mem["tree"]["leaf"] = "对话重建的"
        with pytest.raises(VMMemoryError, match="还原冲突"):
            core.mem.attach("$MEM.tree.leaf", node)
        assert core.mem["tree"]["leaf"] == "对话重建的", "冲突时不覆盖"

    def test_attach_reports_conflict_when_parent_is_gone(self):
        core = make_core()
        node = core.mem.detach("$MEM.tree.nested.deep")
        core.mem.detach("$MEM.tree.nested")
        with pytest.raises(VMMemoryError, match="nested"):
            core.mem.attach("$MEM.tree.nested.deep", node)

    def test_detach_missing_key(self):
        core = make_core()
        with pytest.raises(VMMemoryError):
            core.mem.detach("$MEM.tree.nope")

    def test_detach_root_refused(self):
        core = make_core()
        with pytest.raises(VMMemoryError, match="根节点"):
            core.mem.detach("$MEM")

    def test_detach_device_path_refused(self):
        core = make_core()
        with pytest.raises(VMMemoryError, match="设备"):
            core.mem.detach("$MEM.echo")
        with pytest.raises(VMMemoryError, match="设备"):
            core.mem.detach("$MEM.conversations.0")

    def test_list_element_round_trip(self):
        core = make_core()
        node = core.mem.detach("$MEM.tree.items.1")
        assert node == "b"
        assert list(core.mem["tree"]["items"]) == ["a", "c"]
        core.mem.attach("$MEM.tree.items.1", node)
        assert list(core.mem["tree"]["items"]) == ["a", "b", "c"]

    def test_list_index_out_of_range(self):
        core = make_core()
        with pytest.raises(VMMemoryError):
            core.mem.detach("$MEM.tree.items.9")

    def test_list_index_must_be_a_number(self):
        core = make_core()
        with pytest.raises(VMMemoryError):
            core.mem.detach("$MEM.tree.items.name")

    def test_masked_path_reads_as_key_not_found(self):
        core = make_core()
        core.mem.detach("$MEM.tree.leaf")
        with pytest.raises(VMMemoryError, match="不存在"):
            core.unwrap("$MEM.tree.leaf", for_llm=False)


class TestDeviceMask:
    def test_mask_device_keeps_the_instance(self):
        core = make_core()
        device = core.mem.mounted_devices()["echo"]
        core.mem.unmount("echo")
        assert "echo" not in core.mem.mounted_devices()
        assert device.state == {"n": 0}

    def test_unmask_restores_the_same_instance(self):
        core = make_core()
        device = core.mem.mounted_devices()["echo"]
        device.state["n"] = 5
        core.mem.unmount("echo")
        core.mem.mount("echo", device)
        assert core.mem.mounted_devices()["echo"] is device
        assert core.mem["echo"].state == {"n": 5}

    def test_mounted_devices_is_a_copy(self):
        core = make_core()
        snapshot = core.mem.mounted_devices()
        snapshot["fake"] = None
        assert "fake" not in core.mem.mounted_devices()

    def test_unmount_unknown_path(self):
        core = make_core()
        with pytest.raises(VMMemoryError):
            core.mem.unmount("nope")


class TestDeviceDebugActions:
    def test_default_is_empty(self):
        class Bare(MemoryDevice):
            def pretend_as_type(self):
                return "str"

            def to_llm_string(self):
                return "bare"

        assert Bare().debug_actions() == []

    def test_console_lists_actions_without_calling_them(self):
        core = make_core()
        console = Console(core)
        listing = console.devices()
        echo = [d for d in listing["devices"] if d["mount"] == "echo"][0]
        assert [a["name"] for a in echo["actions"]] == ["自增", "爆炸"]
        assert all(a["callable"] for a in echo["actions"])
        assert core.mem.mounted_devices()["echo"].calls == [], "声明期没有副作用"

    def test_console_invokes_run_directly(self):
        core = make_core()
        console = Console(core)
        record = console.invoke_device_action("echo", 0, {"reason": "冒烟"})
        assert record["error"] is None
        assert record["result"] == {"n": 1}
        assert core.mem.mounted_devices()["echo"].calls == [{"reason": "冒烟"}]
        assert console.devices()["action_log"][-1]["name"] == "自增"

    def test_device_action_produces_no_events(self):
        """设备调试是设备自己的事：不经核心、不进事件流。"""
        core = make_core()
        seen = []
        core.events.add_observer(seen.append)
        console = Console(core)
        console.invoke_device_action("echo", 0, {})
        assert seen == []
        assert core._turn_events == []

    def test_device_action_error_is_captured(self):
        core = make_core()
        console = Console(core)
        record = console.invoke_device_action("echo", 1, {})
        assert record["result"] is None
        assert "设备自己炸了" in record["error"]

    def test_unknown_device_or_action(self):
        core = make_core()
        console = Console(core)
        with pytest.raises(KeyError):
            console.invoke_device_action("nope", 0, {})
        with pytest.raises(KeyError):
            console.invoke_device_action("echo", 9, {})

    def test_masked_device_cannot_be_debugged(self):
        core = make_core()
        console = Console(core)
        console.mask_device("echo")
        with pytest.raises(KeyError, match="未挂载"):
            console.invoke_device_action("echo", 0, {})
        console.unmask_device("echo")
        assert console.invoke_device_action("echo", 0, {})["error"] is None


class TestInputsDevicePanelInput:
    """输入设备自己的调试台：面板上的输入框。"""

    def test_device_declares_an_input_box(self):
        actions = InputsListDevice().debug_actions()
        assert [a["name"] for a in actions] == ["投喂一行输入", "查询队列"]
        box = actions[0]
        assert box["parameters"]["properties"]["content"]["type"] == "string"
        assert box["parameters"]["required"] == ["content"]
        assert callable(box["run"])

    def test_declaring_the_box_has_no_side_effect(self):
        device = InputsListDevice(data=[])
        device.debug_actions()
        assert list(device) == []

    def test_panel_feeds_a_line_that_the_next_read_takes(self, monkeypatch):
        core = Core()
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        monkeypatch.setattr(device, "_terminal_prompt",
                            lambda: pytest.fail("有面板输入时不该碰终端"))
        console = Console(core)

        record = console.invoke_device_action("inputs", 0, {"content": "从面板来的"})
        assert record["error"] is None
        assert record["result"] == {"queued": 1}

        assert core.mem["inputs"][-1] == "从面板来的"
        assert list(device) == ["从面板来的"], "取走之后追加进设备列表"
        assert device._panel_inputs == [], "取走即出队"

    def test_terminal_is_the_fallback(self, monkeypatch):
        """不托管、队列为空 → 分派到终端，与改动前一致。"""
        core = Core()
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        calls = []

        def fake_terminal():
            calls.append(1)
            return "从终端来的"

        monkeypatch.setattr(device, "_terminal_prompt", fake_terminal)
        assert core.mem["inputs"][-1] == "从终端来的"
        assert calls == [1]
        assert device._console_input is False

    def test_queue_is_fifo(self, monkeypatch):
        core = Core()
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        monkeypatch.setattr(device, "_terminal_prompt",
                            lambda: pytest.fail("队列非空时不该碰终端"))
        console = Console(core)
        for line in ("一", "二"):
            console.invoke_device_action("inputs", 0, {"content": line})
        assert core.mem["inputs"][-1] == "一"
        assert core.mem["inputs"][-1] == "二"

    def _console_device(self):
        """托管设备：host_flags 在 attach_core 之前就位，与 load_image 的顺序一致。"""
        core = Core()
        core.host_flags = {"console": True}
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        device.attach_core(core)
        return core, device

    def test_console_flag_makes_the_device_wait_on_the_panel(self, monkeypatch):
        """托管时只等队列：正在等待的这一读也能被喂醒，不必先暂停再喂。"""
        core, device = self._console_device()
        monkeypatch.setattr(device, "_terminal_prompt",
                            lambda: pytest.fail("托管时不该碰终端"))
        console = Console(core)
        assert device._console_input is True
        got = []

        def reader():
            got.append(core.mem["inputs"][-1])

        thread = threading.Thread(target=reader)
        thread.start()
        for _ in range(300):  # 等到读真的等在队列上
            if device._pending_input:
                break
            time.sleep(0.01)
        assert device._pending_input, "读没有进入等待"

        console.invoke_device_action("inputs", 0, {"content": "等来的"})
        thread.join(timeout=5)
        assert not thread.is_alive(), "喂进来之后读没有返回"
        assert got == ["等来的"]
        assert list(device) == ["等来的"]

    def test_waiting_read_is_visible_on_the_panel(self, monkeypatch):
        """等待期间面板的「查询队列」显示正在等输入。"""
        core, device = self._console_device()
        monkeypatch.setattr(device, "_terminal_prompt",
                            lambda: pytest.fail("托管时不该碰终端"))
        console = Console(core)
        thread = threading.Thread(target=lambda: core.mem["inputs"][-1])
        thread.start()
        for _ in range(300):
            if device._pending_input:
                break
            time.sleep(0.01)

        assert console.invoke_device_action("inputs", 1, {})["result"] == {
            "queued": 0, "waiting": True}

        console.invoke_device_action("inputs", 0, {"content": "收工"})
        thread.join(timeout=5)

    def test_flag_is_snapshotted_at_attach(self):
        """标志是启动标志：attach_core 读一次，之后改 host_flags 不再影响设备。"""
        core, device = self._console_device()
        assert device._console_input is True
        core.host_flags["console"] = False
        assert device._console_input is True

    def test_failed_read_does_not_append_to_the_list(self, monkeypatch):
        """读失败时列表不多出元素（原实现在 finally 里追加，会把异常盖成 UnboundLocalError）。"""
        core = Core()
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)

        def boom():
            raise RuntimeError("读失败")

        monkeypatch.setattr(device, "_terminal_prompt", boom)
        with pytest.raises(RuntimeError, match="读失败"):
            core.mem["inputs"][-1]
        assert list(device) == []
        assert device._pending_input is False

    def test_input_box_produces_no_events(self):
        core = Core()
        core.mem.mount("inputs", InputsListDevice(data=[]))
        seen = []
        core.events.add_observer(seen.append)
        console = Console(core)
        console.invoke_device_action("inputs", 0, {"content": "x"})
        core.mem["inputs"][-1]
        assert seen == [], "设备自己的调试台不产生事件"
        assert core._turn_events == []

    def test_content_must_be_a_string(self):
        core = Core()
        core.mem.mount("inputs", InputsListDevice(data=[]))
        console = Console(core)
        record = console.invoke_device_action("inputs", 0, {})
        assert record["result"] is None
        assert "content 必须是字符串" in record["error"]
        assert core.mem.mounted_devices()["inputs"]._panel_inputs == []

    def test_query_action_reports_the_queue(self):
        core = Core()
        device = InputsListDevice(data=[])
        core.mem.mount("inputs", device)
        console = Console(core)
        assert console.invoke_device_action("inputs", 1, {})["result"] == {
            "queued": 0, "waiting": False}
        console.invoke_device_action("inputs", 0, {"content": "x"})
        assert console.invoke_device_action("inputs", 1, {})["result"] == {
            "queued": 1, "waiting": False}

    def test_input_box_shows_up_on_the_console(self):
        core = Core()
        core.mem.mount("inputs", InputsListDevice(data=[]))
        console = Console(core)
        entry = [d for d in console.devices()["devices"] if d["mount"] == "inputs"][0]
        assert [a["name"] for a in entry["actions"]] == ["投喂一行输入", "查询队列"]


class TestConsoleMaskLedger:
    def test_mask_memory_then_unmask(self):
        core = make_core()
        console = Console(core)
        console.mask_memory("$MEM.tree.leaf")
        assert console.state()["mask"]["memory"] == ["$MEM.tree.leaf"]
        with pytest.raises(VMMemoryError):
            core.unwrap("$MEM.tree.leaf", for_llm=False)

        console.unmask_memory("$MEM.tree.leaf")
        assert console.state()["mask"]["memory"] == []
        assert core.unwrap("$MEM.tree.leaf", for_llm=False) == "hello"

    def test_double_mask_refused(self):
        core = make_core()
        console = Console(core)
        console.mask_memory("$MEM.tree.leaf")
        with pytest.raises(ValueError, match="已经被屏蔽"):
            console.mask_memory("$MEM.tree.leaf")

    def test_unmask_conflict_keeps_the_ledger(self):
        """还原失败时账本不能丢，否则那条路径就永远回不来了。"""
        core = make_core()
        console = Console(core)
        console.mask_memory("$MEM.tree.leaf")
        core.mem["tree"]["leaf"] = "对话重建的"
        with pytest.raises(VMMemoryError, match="还原冲突"):
            console.unmask_memory("$MEM.tree.leaf")
        assert console.state()["mask"]["memory"] == ["$MEM.tree.leaf"]
        assert core.mem["tree"]["leaf"] == "对话重建的"

    def test_mask_failure_does_not_touch_the_ledger(self):
        core = make_core()
        console = Console(core)
        with pytest.raises(VMMemoryError):
            console.mask_memory("$MEM.tree.nope")
        assert console.state()["mask"]["memory"] == []

    def test_unmask_unknown_ref(self):
        core = make_core()
        console = Console(core)
        with pytest.raises(KeyError):
            console.unmask_memory("$MEM.tree.leaf")

    def test_device_mask_ledger(self):
        core = make_core()
        console = Console(core)
        console.mask_device("echo")
        assert console.state()["mask"]["devices"] == ["echo"]
        assert "echo" not in core.mem.mounted_devices()
        console.unmask_device("echo")
        assert console.state()["mask"]["devices"] == []
        assert "echo" in core.mem.mounted_devices()

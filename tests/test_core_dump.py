"""内核转储（core dump）的回归用例。

对应 plan-core-dump-2026-10-16.md 的验收一节。转储是诊断快照，不是恢复输入。
"""

import json
import os

from avm.boot import dump_on_death
from avm.core import Core
from avm.image import load_image
from avm.exceptions import KernelPanic
from avm.types import MetaDict


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _core():
    core = Core()
    conv = core.start("sys", "hello")
    core._active_cid = conv.cid
    return core, conv


class TestDumpCore:
    def test_panicked_core_leaves_a_readable_snapshot(self, tmp_path):
        core, conv = _core()
        core.mem["x"] = "y"
        path = str(tmp_path / "out.core.json")
        assert core.dump_core(path, error=KernelPanic("调度互斥被破坏")) == path
        doc = _read(path)
        assert doc["format"] == "avm.core_dump" and doc["version"] == 1
        assert doc["error"]["type"] == "KernelPanic"
        assert doc["error"]["panicked"] is True
        assert "调度互斥" in doc["error"]["message"]
        assert "不是恢复输入" in doc["note"]
        assert doc["scheduler"]["active"] == conv.cid
        assert "x" in doc["memory"]["value"]
        assert [d["mount"] for d in doc["devices"] if d["mount"] == "monitor"]
        brief = doc["conversations"][0]
        assert brief["cid"] == conv.cid and brief["state"] == "active"
        assert brief["messages"][0]["role"] == "system"

    def test_unhandled_defect_keeps_the_traceback(self, tmp_path):
        core, _conv = _core()
        path = str(tmp_path / "defect.core.json")
        try:
            raise RuntimeError("内核缺陷")
        except RuntimeError as e:
            assert dump_on_death(core, e, path) == path
        doc = _read(path)
        assert doc["error"]["type"] == "RuntimeError"
        assert doc["error"]["panicked"] is False
        assert "内核缺陷" in doc["error"]["traceback"]

    def test_disabled_by_flag_writes_nothing(self, tmp_path):
        core, _conv = _core()
        core.core_dump_enabled = False
        path = str(tmp_path / "none.json")
        assert dump_on_death(core, KernelPanic("x"), path) is None
        assert not os.path.exists(path)

    def test_messages_are_capped_with_marker(self, tmp_path):
        core, conv = _core()
        for i in range(500):
            conv.append_assistant_message(f"m{i}")
        path = str(tmp_path / "cap.core.json")
        core.dump_core(path, error=KernelPanic("x"), messages_per_conversation=200)
        brief = _read(path)["conversations"][0]
        assert brief["message_count"] == 502  # system + user + 500
        assert len(brief["messages"]) == 200
        assert brief["messages_truncated"] is True

    def test_frames_are_capped_with_marker(self, tmp_path):
        core, _conv = _core()
        core.monitor.record_baseline(core)
        for _ in range(5):
            core.monitor.record(core)
        path = str(tmp_path / "frames.core.json")
        core.dump_core(path, error=KernelPanic("x"), max_frames=2)
        doc = _read(path)
        assert len(doc["frames"]) == 2
        assert doc["truncated"]["frames"] is True

    def test_default_path_uses_image_context(self, tmp_path):
        core, _conv = _core()
        core.image_dir = str(tmp_path)
        core.image_name = "demo"
        assert core.core_dump_path() == os.path.join(str(tmp_path), "out", "demo.core.json")

    def test_unwritable_target_reports_and_returns_none(self, tmp_path, capsys):
        core, _conv = _core()
        target = str(tmp_path / "a-directory")
        os.makedirs(target)  # 把目录当文件写
        assert core.dump_core(target, error=KernelPanic("x")) is None
        assert "内核转储写入失败" in capsys.readouterr().err

    def test_memory_tree_keeps_non_string_leaves_as_text(self, tmp_path):
        core, _conv = _core()
        core.mem["blob"] = MetaDict(data={"nested": "x"})
        core.mem["plain_int"] = 7  # 历史遗留的非字符串叶子
        path = str(tmp_path / "odd.core.json")
        assert core.dump_core(path, error=KernelPanic("x")) == path
        assert _read(path)["memory"]["value"]["plain_int"]["value"] == "7"


class TestDeathPath:
    def test_kernel_panic_from_run_is_dumpable(self, tmp_path):
        """走 run() 的死亡路径：不变量被破坏 → KernelPanic → 宿主落一份转储。"""
        core, _conv = _core()
        core._ready_cids[:] = [999]  # 就绪队列里放一个不存在的 cid → 活跃 cid 不在对话表里
        path = str(tmp_path / "panic.core.json")
        try:
            core.run()
            raise AssertionError("应当抛 KernelPanic")
        except KernelPanic as e:
            dump_on_death(core, e, path)
        doc = _read(path)
        assert doc["error"]["panicked"] is True
        assert "不在对话表里" in doc["error"]["message"]
        assert len(doc["frames"]) == 2  # 基线 + 记录下来的错误帧


class TestImageFields:
    def _image(self, tmp_path, extra):
        doc = {
            "meta": {"version": 1},
            "mem": {
                "model_params": {"kind": "dict", "ctrl": {"type": "para"},
                                 "value": {"model": {"kind": "str", "value": "gpt-4"}}},
            },
        }
        doc.update(extra)
        path = tmp_path / "img.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return str(path)

    def test_core_dump_to_and_switch(self, tmp_path):
        core = load_image(Core(), self._image(tmp_path, {"core_dump_to": "out/my.core.json",
                                                         "core_dump": False}))
        assert core.core_dump_enabled is False
        assert core.core_dump_to == os.path.join(str(tmp_path), "out", "my.core.json")

    def test_core_dump_must_be_bool(self, tmp_path):
        from avm.image import ImageError
        import pytest

        with pytest.raises(ImageError):
            load_image(Core(), self._image(tmp_path, {"core_dump": "yes"}))

"""单对话检查点测试：无损往返、字节一致、截断、Python 拒绝、/save 调试命令"""

import json
import os

import pytest

from avm.core import Core, _build_conversation
from avm.types import (
    Conversation, SystemMessage, UserMessage, AssistantMessage, ToolMessage, MetaDict,
)
from avm.memory_device import InputsListDevice


def _root(core):
    core.mem["model_params"] = MetaDict(data={"model": "test"})
    conv = core.start("sys", "hello")
    conv.name = "init"
    return conv


class TestConversationCheckpoint:
    def test_roundtrip_byte_identical(self):
        """to_checkpoint → from_checkpoint：全部字段无损，to_api_messages 逐字节一致"""
        conv = Conversation(
            messages=[
                SystemMessage(content="sys"),
                UserMessage(content="hello"),
                AssistantMessage(
                    content="",
                    tool_calls=[{
                        "id": "c1", "type": "function",
                        "function": {"name": "memory_read", "arguments": '{"ref": "$MEM.inputs.-1"}'},
                    }],
                    reasoning_content="think",
                ),
                ToolMessage(content="用户输入", tool_call_id="c1"),
                AssistantMessage(content="final", reasoning_content="think2"),
            ],
            cid=0,
            is_root=True,
            metadata={"k": "v"},
            service_desc={"name": "svc", "what": "x"},
            name="init",
        )
        conv.para_ref = "$MEM.model_params"

        restored = Conversation.from_checkpoint(conv.to_checkpoint())

        assert restored.name == "init"
        assert restored.para_ref == "$MEM.model_params"
        assert restored.metadata == {"k": "v"}
        assert restored.service_desc == {"name": "svc", "what": "x"}
        assert restored.messages == conv.messages
        assert restored.to_api_messages() == conv.to_api_messages()
        assert json.dumps(restored.to_api_messages(), ensure_ascii=False) == \
            json.dumps(conv.to_api_messages(), ensure_ascii=False)

    def test_reasoning_content_none_vs_empty_preserved(self):
        conv = Conversation(messages=[
            SystemMessage(content="s"),
            UserMessage(content="u"),
            AssistantMessage(content="a", reasoning_content=None),
            AssistantMessage(content="b", reasoning_content=""),
        ], cid=0)
        restored = Conversation.from_checkpoint(conv.to_checkpoint())
        assert restored.messages[-2].reasoning_content is None
        assert restored.messages[-1].reasoning_content == ""
        assert restored.to_api_messages() == conv.to_api_messages()

    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError, match="版本"):
            Conversation.from_checkpoint({"version": 99, "messages": []})


class TestCoreCheckpoint:
    def test_save_restore_assigns_new_cid(self, tmp_path):
        core = Core()
        root = _root(core)
        root.append_assistant_message("done")
        path = str(tmp_path / "cp.json")

        core.save_conversation(root.cid, path)
        assert os.path.isfile(path)

        restored = core.restore_conversation(path)
        assert restored.cid != root.cid
        assert restored.cid in core._dormant_cids
        assert restored.para_ref == root.para_ref
        assert restored.to_api_messages() == root.to_api_messages()

    def test_restore_schedule_ready(self, tmp_path):
        core = Core()
        root = _root(core)
        root.append_assistant_message("done")
        path = str(tmp_path / "cp.json")
        core.save_conversation(root.cid, path)

        restored = core.restore_conversation(path, schedule="ready")
        assert restored.cid in core._ready_cids

    def test_save_truncates_unmatched_tool_call(self, tmp_path):
        """历史末尾是未配对工具调用（等待输入响应）时，检查点截到最后一个完整交换"""
        core = Core()
        root = _root(core)
        root.append_assistant_message("", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "memory_read", "arguments": '{"ref": "$MEM.inputs.-1"}'},
        }])
        path = str(tmp_path / "cp.json")

        core.save_conversation(root.cid, path)
        cp = json.loads(open(path, encoding="utf-8").read())
        assert cp["messages"][-1]["role"] == "user"  # 末尾未配对工具调用已被截掉
        assert all("tool_calls" not in m for m in cp["messages"])

        restored = core.restore_conversation(path)
        assert restored.to_api_messages() == root.to_api_messages()[:-1]

    def test_python_checkpoint_rejected(self, tmp_path):
        core = Core()
        core.mem["prog"] = MetaDict(data={"content": ""}, ctrl={"type": "python"})
        core.mem["p"] = MetaDict(data={"model": "plm.simple"}, ctrl={"type": "para"})
        root = _root(core)
        py = _build_conversation(core, "$MEM.prog", "$MEM.p", parent=root, is_sub=False)

        with pytest.raises(ValueError, match="Python"):
            core.save_conversation(py.cid, str(tmp_path / "cp.json"))

        path = tmp_path / "py.json"
        path.write_text(json.dumps(py.to_checkpoint(), ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="Python"):
            core.restore_conversation(str(path))

    def test_save_nonexistent_cid_errors(self, tmp_path):
        core = Core()
        with pytest.raises(Exception):
            core.save_conversation(99, str(tmp_path / "cp.json"))


class TestSaveDebugCommand:
    """输入设备 /save 调试命令：拦截在输入前，对话感知不到"""

    class FakeCore:
        def __init__(self, tmp_path):
            self._active_cid = 7
            self.image_dir = str(tmp_path)
            self.image_name = "demo"
            self.saved = []

        def checkpoint_path(self, cid):
            return str(self.image_dir) + "/default.conv.json"

        def save_conversation(self, cid, path):
            self.saved.append((cid, path))

    def _device_with_inputs(self, tmp_path):
        core = self.FakeCore(tmp_path)
        dev = InputsListDevice()
        dev.attach_core(core)
        return dev, core

    def test_save_default_path_then_real_input(self, monkeypatch, tmp_path):
        dev, core = self._device_with_inputs(tmp_path)
        seq = iter(["/save", "real input"])
        monkeypatch.setattr("builtins.input", lambda: next(seq))

        assert dev[-1] == "real input"
        assert core.saved == [(7, str(tmp_path) + "/default.conv.json")]
        assert dev._data == ["real input"]

    def test_save_with_custom_path(self, monkeypatch, tmp_path):
        dev, core = self._device_with_inputs(tmp_path)
        seq = iter(["/save out/custom.json", "x"])
        monkeypatch.setattr("builtins.input", lambda: next(seq))

        assert dev[-1] == "x"
        assert core.saved == [(7, str(tmp_path) + "/out/custom.json")]

    def test_save_without_core_does_not_crash(self, monkeypatch, capsys):
        dev = InputsListDevice()
        seq = iter(["/save", "x"])
        monkeypatch.setattr("builtins.input", lambda: next(seq))

        assert dev[-1] == "x"
        assert "未接入 Core" in capsys.readouterr().out

"""作者定义的 API 失败处置：分类、内核暂停/跳过/重试、宿主恢复。

全部用例离线：执行器是脚本替身，APIError 由测试直接构造，不调真实 API。
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from avm.console import Console, serve
from avm.core import Core
from avm.exceptions import APIError
from avm.image import ImageError

from persist_helpers import load_core, write_image


def _api_error(category, message="外部调用失败", status_code=None, context_limit=None):
    detail = {
        "category": category,
        "message": message,
        "status_code": status_code,
        "context_limit": context_limit,
    }
    return APIError(message, category=category, status_code=status_code, detail=detail)


BALANCE = _api_error("insufficient_balance", "账户余额不足", 402)
CONTEXT = _api_error(
    "context_limit",
    "This model's maximum context length is 128000 tokens",
    400,
    128000,
)
OTHER = _api_error("other", "boom")


class ScriptedLMU:
    """按 cid 出错的执行器替身：None 表示成功提交一轮，异常对象表示抛出。"""

    def __init__(self, script=None):
        self.script = {cid: list(items) for cid, items in (script or {}).items()}
        self.n_calls = 0
        self.calls_by_cid = {}
        self.last_call = None

    def exec(self, conversation, para):
        self.n_calls += 1
        self.calls_by_cid[conversation.cid] = (
            self.calls_by_cid.get(conversation.cid, 0) + 1
        )
        queue = self.script.setdefault(conversation.cid, [])
        action = queue.pop(0) if queue else None
        if isinstance(action, BaseException):
            self.last_call = {"error": str(action)}
            raise action
        conversation.append_assistant_message("done")
        self.last_call = {"result": "done"}
        return "done", [], conversation


# ----------------------------------------------------------------------
# 内核：三类处置
# ----------------------------------------------------------------------

class TestInsufficientBalance:
    def test_pauses_without_retry_or_close(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [BALANCE]})
        conv = core.start("sys", "user")

        core.run()

        assert core.paused is True
        assert core.pause_category == "insufficient_balance"
        assert "余额" in core.pause_reason
        assert core.lmu.n_calls == 1, "余额不足不重试"
        assert [m.content for m in conv.messages] == ["sys", "user"], "失败回合不提交"
        assert conv.cid not in core._finished, "暂停不关闭对话"
        assert core._active_cid == conv.cid

    def test_clear_pause_allows_continue(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [BALANCE, None]})
        conv = core.start("sys", "user")
        core.run()
        assert core.paused is True

        core.clear_api_pause()
        assert core.paused is False
        assert core.pause_reason is None
        assert core.pause_category is None

        core.run()
        assert [m.content for m in conv.messages] == ["sys", "user", "done"]


class TestContextLimit:
    def test_skips_conversation_and_continues_others(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [CONTEXT], 1: [None]})
        first = core.start("sys", "one")
        second = core.start("sys", "two")

        core.run()

        assert core.paused is False
        assert core.lmu.n_calls == 2, "上下文超限不重试"
        assert first.cid not in core._finished, "跳过不是关闭：对话个体保留"
        assert first.cid not in core._ready_cids, "跳过不立刻重跑"
        assert first.cid in core._dormant_cids, "跳过回休眠，等软件回收/再次唤醒"
        assert [m.content for m in second.messages] == ["sys", "two", "done"]
        assert first.last_api_error["category"] == "context_limit"
        assert first.last_api_error["context_limit"] == 128000

    def test_device_context_carries_classification(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [CONTEXT]})
        conv = core.start("sys", "one")
        core.run()

        data = json.loads(
            core.unwrap(f"$MEM.conversations.{conv.cid}.context", for_llm=False)
        )
        assert data["messages"] == 2
        assert data["chars"] == 6
        assert data["tool_messages"] == 0
        assert data["context_limit"] == 128000
        assert data["last_error"]["category"] == "context_limit"
        assert data["last_error"]["status_code"] == 400
        assert data["last_error"]["at"]

    def test_skip_commits_no_journal_record(self, tmp_path):
        """跳过不产生世界状态变化：失败回合零提交，对话留在原地等软件处理。"""
        core = Core()
        core.lmu = ScriptedLMU({0: [CONTEXT]})
        core.start("sys", "one")
        core.attach_journal(str(tmp_path / "journal.jsonl"))

        core.run()

        assert core.journal.read() == []
        assert 0 in core._conv_by_cid


class TestOtherErrors:
    def test_retries_five_times_then_pauses(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [OTHER] * 20})
        conv = core.start("sys", "user")

        core.run()

        assert core.paused is True
        assert core.pause_category == "other"
        assert core.lmu.n_calls == 6, "默认重试 5 次，第 6 次失败后暂停"
        assert "6" in core.pause_reason
        assert conv.api_error_streak == 6
        assert [m.content for m in conv.messages] == ["sys", "user"]

    def test_retry_limit_is_configurable(self):
        core = Core()
        core.api_error_max_retries = 2
        core.lmu = ScriptedLMU({0: [OTHER] * 10})
        core.start("sys", "user")

        core.run()

        assert core.paused is True
        assert core.lmu.n_calls == 3, "重试 2 次，第 3 次失败后暂停"
        assert "2" in core.pause_reason

    def test_success_resets_streak_and_last_error(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [OTHER, OTHER, None]})
        conv = core.start("sys", "user")
        core._active_cid = core._ready_cids.pop(0)

        core.advance_conversation()

        assert core.paused is False
        assert conv.api_error_streak == 0
        assert conv.last_api_error is None
        assert [m.content for m in conv.messages] == ["sys", "user", "done"]

    def test_clear_pause_resets_streak(self):
        core = Core()
        core.lmu = ScriptedLMU({0: [OTHER] * 10})
        conv = core.start("sys", "user")
        core.run()
        assert conv.api_error_streak == 6

        core.clear_api_pause()
        assert conv.api_error_streak == 0

    def test_host_policy_still_overrides_builtin_policy(self):
        """装了宿主策略时保留旧语义：回调决定 retry/reload/exit，不被内置重试上限打断。"""
        core = Core()
        core.lmu = ScriptedLMU({0: [OTHER, OTHER, None]})
        core.start("sys", "user")
        seen = []
        core.api_error_policy = lambda error, attempt: (
            seen.append(attempt) or "retry"
        )

        core.run()

        assert seen == [1, 2]
        assert core.paused is False
        assert core.lmu.n_calls == 3


# ----------------------------------------------------------------------
# 镜像 meta.api_error.max_retries
# ----------------------------------------------------------------------

class TestImageRetryLimit:
    def _write_with_meta(self, tmp_path, api_error):
        image = write_image(tmp_path)
        doc = json.loads(Path(image).read_text(encoding="utf-8"))
        doc["meta"]["api_error"] = api_error
        Path(image).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return image

    def test_meta_overrides_max_retries(self, tmp_path):
        image = self._write_with_meta(tmp_path, {"max_retries": 2})
        core = load_core(image)
        assert core.api_error_max_retries == 2

        core.lmu = ScriptedLMU({0: [OTHER] * 10})
        core.run()
        assert core.paused is True
        assert core.lmu.n_calls == 3

    def test_invalid_max_retries_rejected(self, tmp_path):
        image = self._write_with_meta(tmp_path, {"max_retries": -1})
        with pytest.raises(ImageError, match="api_error.max_retries"):
            load_core(image)


# ----------------------------------------------------------------------
# 宿主：console 状态 / 恢复端点；main 非 tty
# ----------------------------------------------------------------------

def _request(base, method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            raw, code = response.read(), response.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    try:
        return code, json.loads(raw) if raw else None
    except ValueError:
        return code, raw.decode("utf-8", "replace")


class TestConsolePause:
    def _console(self, tmp_path, script):
        core = Core()
        core.lmu = ScriptedLMU(script)
        conv = core.start("sys", "user")
        console = Console(
            core, web_root=str(tmp_path / "no-dist"), mount_io_devices=False
        )
        server = serve(console, port=0)
        base = f"http://127.0.0.1:{server.server_port}"
        return core, console, conv, server, base

    def test_state_shows_pause_and_resume_endpoint_continues(self, tmp_path):
        core, console, conv, server, base = self._console(
            tmp_path, {0: [BALANCE, None]}
        )
        try:
            core.run()
            assert core.paused is True

            code, state = _request(base, "GET", "/api/state")
            assert code == 200
            assert state["paused"] is True
            assert state["pause_category"] == "insufficient_balance"
            assert "余额" in state["pause_reason"]

            code, body = _request(base, "POST", "/api/system/resume", {})
            assert code == 200
            assert body == {"status": "resumed", "paused": False}
            assert core.paused is False

            code, _ = _request(base, "POST", "/api/system/resume", {})
            assert code == 409

            # 恢复后主线程会重新进入 core.run()；这里直接推进活跃对话，避免
            # console 的 idle 向量把测试阻塞在面板对话上（那是宿主循环的行为）。
            core.advance_conversation()
            assert [m.content for m in conv.messages] == ["sys", "user", "done"]
        finally:
            server.shutdown()
            server.server_close()

    def test_shutdown_from_paused_clears_pause_and_wakes_host(self, tmp_path):
        core, console, _conv, server, _base = self._console(tmp_path, {0: [BALANCE]})
        try:
            core.run()
            console.lifecycle = "paused"

            response = console.request_shutdown()

            assert response["status"] == "stopping"
            assert core.paused is False
            assert console.lifecycle == "stopping"
            # 唤醒信号住在 Lifecycle 里：关机要从暂停里把主线程拽出来。
            assert console.lifecycle_state.api_resume_event.is_set()
        finally:
            server.shutdown()
            server.server_close()


class TestTerminalHost:
    def test_non_tty_exits_cleanly_after_pause(self, tmp_path, monkeypatch, capsys):
        core = Core()
        core.lmu = ScriptedLMU({0: [BALANCE]})
        core.start("sys", "user")
        import main as main_module

        monkeypatch.setattr(main_module, "boot", lambda *a, **k: core)

        status = main_module.main([str(tmp_path / "image.json")])

        err = capsys.readouterr().err
        assert status == 0, "非 tty 按退出处理，不当作中断"
        assert "API 暂停" in err
        assert "余额" in err
        assert "不是终端" in err

    def test_tty_enter_clears_pause_and_continues(self, tmp_path, monkeypatch, capsys):
        core = Core()
        core.lmu = ScriptedLMU({0: [BALANCE, None]})
        conv = core.start("sys", "user")
        import main as main_module

        monkeypatch.setattr(main_module, "boot", lambda *a, **k: core)

        class Tty:
            @staticmethod
            def isatty():
                return True

        monkeypatch.setattr(main_module.sys, "stdin", Tty())
        monkeypatch.setattr("builtins.input", lambda prompt="": "")

        status = main_module.main([str(tmp_path / "image.json")])

        assert status == 0
        assert [m.content for m in conv.messages] == ["sys", "user", "done"]
        assert "API 暂停" in capsys.readouterr().err

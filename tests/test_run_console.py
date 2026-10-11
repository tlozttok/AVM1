"""启动脚本 `scripts/run_console.py` 的两件事：清洗规则，以及在启动前拦住坏环境。

真起控制台的部分不在这里——那要镜像、要模型；这里只保证它不会带着一段 httpx 的栈崩掉。
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_console.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_console", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return load_script()


class TestCleanNoProxy:
    def test_drops_bracketed_ipv6(self, script):
        cleaned, dropped = script.clean_no_proxy("localhost,[::1],::1")
        assert cleaned == "localhost,::1"
        assert dropped == ["[::1]"]

    def test_keeps_everything_else(self, script):
        raw = "127.0.0.1,172.31.*,10.*,localhost,<local>,::1"
        cleaned, dropped = script.clean_no_proxy(raw)
        assert cleaned == raw
        assert dropped == []

    def test_strips_spaces_and_empty_items(self, script):
        cleaned, dropped = script.clean_no_proxy(" localhost , , [fe80::1] ")
        assert cleaned == "localhost"
        assert dropped == ["[fe80::1]"]

    def test_empty_string(self, script):
        assert script.clean_no_proxy("") == ("", [])


class TestCleanEnvironment:
    def test_cleans_both_cases_and_reports_once(self, script):
        env = {"NO_PROXY": "localhost,[::1]", "no_proxy": "localhost,[::1]"}
        dropped = script.clean_environment(env)
        assert dropped == ["[::1]"]
        assert env == {"NO_PROXY": "localhost", "no_proxy": "localhost"}

    def test_leaves_untouched_when_there_is_nothing_to_drop(self, script):
        env = {"NO_PROXY": "localhost,::1"}
        before = dict(env)
        assert script.clean_environment(env) == []
        assert env == before

    def test_missing_variables_are_fine(self, script):
        env = {}
        assert script.clean_environment(env) == []
        assert env == {}


class TestMainGuard:
    def test_refuses_to_start_when_httpx_still_cannot_be_built(self, script, monkeypatch, capsys):
        monkeypatch.setattr(script, "clean_environment", lambda environ=None: [])
        monkeypatch.setattr(script, "client_buildable", lambda: "InvalidURL: Invalid port: ':1]'")
        assert script.main(["images/demo.json"]) == 1
        err = capsys.readouterr().err
        assert "建不了 httpx 客户端" in err
        assert "NO_PROXY" in err

    def test_passes_argv_through_to_the_console(self, script, monkeypatch):
        seen = {}
        monkeypatch.setattr(script, "clean_environment", lambda environ=None: [])
        monkeypatch.setattr(script, "client_buildable", lambda: None)

        import avm.console as console

        def fake_main(argv=None):
            seen["argv"] = argv
            return 0

        monkeypatch.setattr(console, "main", fake_main)
        assert script.main(["images/demo.json", "--port", "8701"]) == 0
        assert seen["argv"] == ["images/demo.json", "--port", "8701"]

    def test_script_is_runnable_from_anywhere(self, script):
        """脚本要能把仓库根塞进 sys.path，否则从别的目录跑就 import 不到 avm。"""
        assert str(REPO) in sys.path

    def test_real_environment_cleaning_makes_httpx_buildable(self, script, monkeypatch):
        """把本机这个坑原样搭出来：清洗后 httpx 客户端必须能建。"""
        monkeypatch.setenv("NO_PROXY", "localhost,[::1]")
        monkeypatch.setenv("no_proxy", "localhost,[::1]")
        script.clean_environment()
        assert os.environ["no_proxy"] == "localhost"
        assert script.client_buildable() is None

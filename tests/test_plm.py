"""沙箱与 PLM 测试：安全限制、plm.python（图灵完备）、SimplePLM 加固"""

import json

import pytest

from avm.sandbox import SecurityError, check_code, safe_globals, exec_safe
from avm.python_server import create_plm


class TestSandbox:
    def test_import_blocked(self):
        with pytest.raises(SecurityError):
            check_code("import os")

    def test_from_import_blocked(self):
        with pytest.raises(SecurityError):
            check_code("from os import path")

    def test_dunder_attribute_blocked(self):
        with pytest.raises(SecurityError):
            check_code("().__class__")

    def test_dangerous_call_blocked(self):
        with pytest.raises(SecurityError):
            check_code("open('/etc/passwd')")
        with pytest.raises(SecurityError):
            check_code("eval('1+1')")

    def test_normal_code_ok(self):
        ns = safe_globals()
        exec_safe("x = 1\ny = x + 1", ns)
        assert ns["y"] == 2


class TestPythonPLM:
    def _plm(self):
        return create_plm("plm.python")

    def test_setup_and_user_share_namespace(self):
        plm = self._plm()
        plm.handle_messages([{"role": "system", "content": "x = 40"}], {})
        assert plm.handle_messages([{"role": "user", "content": "print(x + 2)"}], {}) == {"content": "42\n"}

    def test_user_writes_persist(self):
        plm = self._plm()
        plm.handle_messages([{"role": "user", "content": "total = 0\nfor i in range(5): total += i"}], {})
        assert plm.handle_messages([{"role": "user", "content": "print(total)"}], {}) == {"content": "10\n"}

    def test_multiple_prints_concatenated(self):
        plm = self._plm()
        r = plm.handle_messages([{"role": "user", "content": "print('a')\nprint('b')\nprint('c')"}], {})
        assert r == {"content": "a\nb\nc\n"}

    def test_turing_complete_loop(self):
        plm = self._plm()
        code = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a\nprint(fib(10))"
        assert plm.handle_messages([{"role": "user", "content": code}], {}) == {"content": "55\n"}

    def test_envelope_returns_via_tool(self):
        plm = self._plm()
        env = json.dumps({"from": "init", "to": "py", "icc_id": "i1", "content": "print(6 * 7)"})
        r = plm.handle_messages([{"role": "user", "content": env}], {})
        assert r["content"] == "42\n"
        assert r["tool_calls"][0]["function"]["name"] == "return_result"
        assert json.loads(r["tool_calls"][0]["function"]["arguments"]) == {"content": "42\n", "icc_id": "i1"}

    def test_escape_attempt_returns_error(self):
        plm = self._plm()
        r = plm.handle_messages([{"role": "user", "content": "import os"}], {})
        assert "错误" in r["content"]
        assert "SecurityError" in r["content"]


class TestSimplePLMHardened:
    def test_calculator_escape_blocked(self):
        plm = create_plm("plm.simple")
        r = plm.handle_messages([{"role": "user", "content": "().__class__"}], {})
        assert "错误" in r["content"]
        assert "SecurityError" in r["content"]

    def test_calculator_still_works(self):
        plm = create_plm("plm.simple")
        assert plm.handle_messages([{"role": "user", "content": "1 + 2"}], {}) == {"content": "3"}

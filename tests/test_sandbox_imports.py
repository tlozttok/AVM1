"""沙箱导入机制：静态白名单检查、模块视图裁剪、运行期裁决与顺手加固。

对应计划 `docs/private/Agents/proposed/plan-sandbox-imports-2026-11-15.md` 的阶段 A/B。
"""

import pytest

from avm.modules import (
    BASE_MODULES,
    catalog_exports,
    GuestModule,
    sandbox_globals,
    shared_view,
)
from avm.python_server import create_plm
from avm.sandbox import SecurityError, check_code

ALLOWED = {"math", "json", "re", "itertools"}

EXPORTS = catalog_exports()


def _namespace(allowed=ALLOWED, cache=None):
    return sandbox_globals({}, allowed, cache)


def _run(code, allowed=ALLOWED, cache=None):
    namespace = _namespace(allowed, cache)
    exec(compile(code, "<guest>", "exec"), namespace)
    return namespace


class TestStaticImportCheck:
    def test_whitelisted_import_ok(self):
        check_code("import math", allow_import=True, imports=ALLOWED, exports=EXPORTS)

    def test_unknown_module_rejected(self):
        with pytest.raises(SecurityError, match="不在本对话的导入白名单"):
            check_code("import os", allow_import=True, imports=ALLOWED, exports=EXPORTS)

    def test_dotted_import_rejected(self):
        with pytest.raises(SecurityError, match="子模块"):
            check_code("import a.b", allow_import=True, imports=ALLOWED, exports=EXPORTS)

    def test_relative_import_rejected(self):
        with pytest.raises(SecurityError, match="相对导入"):
            check_code("from . import x", allow_import=True, imports=ALLOWED, exports=EXPORTS)

    def test_star_import_rejected(self):
        with pytest.raises(SecurityError, match=r"import \*"):
            check_code("from math import *", allow_import=True, imports=ALLOWED, exports=EXPORTS)

    def test_unexported_name_rejected(self):
        with pytest.raises(SecurityError, match="未导出"):
            check_code("from json import decoder", allow_import=True,
                       imports=ALLOWED, exports=EXPORTS)

    def test_dunder_import_name_rejected(self):
        with pytest.raises(SecurityError, match="双下划线"):
            check_code("from json import __dict__", allow_import=True,
                       imports=ALLOWED, exports=EXPORTS)

    def test_dunder_alias_rejected(self):
        with pytest.raises(SecurityError, match="双下划线别名"):
            check_code("import math as __m", allow_import=True,
                       imports=ALLOWED, exports=EXPORTS)

    def test_builtins_store_rejected(self):
        with pytest.raises(SecurityError, match="__builtins__"):
            check_code("__builtins__ = {}")

    def test_default_still_blocks_import(self):
        """事件处理器等未传 allow_import 的调用点保持禁 import。"""
        with pytest.raises(SecurityError, match="禁止 import"):
            check_code("import math")

    @pytest.mark.parametrize("code", [
        "x = ().gi_frame",
        "x = y.f_globals",
        "x = y.f_builtins",
        "x = e.tb_frame",
        "x = f.co_consts",
    ])
    def test_frame_introspection_attributes_rejected(self, code):
        with pytest.raises(SecurityError, match="内省属性"):
            check_code(code)


class TestRuntimeImport:
    def test_import_returns_trimmed_view(self):
        namespace = _run("import itertools\nv = list(itertools.islice(itertools.count(), 3))")
        assert namespace["v"] == [0, 1, 2]
        assert type(namespace["itertools"]) is GuestModule
        assert namespace["itertools"] is not __import__("itertools")

    def test_whitelist_enforced_at_runtime(self):
        namespace = _namespace()
        with pytest.raises(SecurityError, match="不在本对话的导入白名单"):
            exec("__builtins__['__import__']('os')", namespace)

    def test_view_trimming_and_read_only(self):
        namespace = _run("import re")
        view = namespace["re"]
        with pytest.raises(AttributeError):
            view.enum
        with pytest.raises(AttributeError):
            view.__dict__
        with pytest.raises(AttributeError):
            view.search = None

    def test_no_sys_modules_fallback(self):
        """未导出名不能经 IMPORT_FROM 的 sys.modules 回退拿到宿主真子模块。"""
        namespace = _namespace()
        with pytest.raises(ImportError):
            exec("from re import copyreg", namespace)

    def test_from_import_exported(self):
        namespace = _run("from json import loads\nv = loads('{\"a\": 1}')['a']")
        assert namespace["v"] == 1

    def test_baseline_names_injected(self):
        namespace = _namespace(BASE_MODULES)
        for name in BASE_MODULES:
            assert isinstance(namespace[name], GuestModule)

    def test_shared_view_used_by_event_handlers_is_trimmed(self):
        view = shared_view("json")
        assert type(view) is GuestModule
        with pytest.raises(AttributeError):
            view.decoder

    def test_per_instance_cache_isolation(self):
        cache_a, cache_b = {}, {}
        _run("import math", cache=cache_a)
        _run("import math", cache=cache_b)
        assert cache_a["math"] is not cache_b["math"]
        _run("import math", cache=cache_a)
        assert cache_a["math"] is cache_a["math"]


class TestEventHandlerNamespace:
    def _listener(self, code):
        return {"listener_id": 1, "handler_code": code}

    def test_handler_json_works(self):
        from avm.core import Core
        core = Core()
        handler = "def handle(event):\n    return {'content': json.dumps({'a': 1})}\n"
        out = core.events._run_listener(self._listener(handler), {})
        assert out == {"content": '{"a": 1}'}

    def test_handler_cannot_reach_json_submodule(self):
        from avm.core import Core
        from avm.events import EventHandlerError
        core = Core()
        handler = "def handle(event):\n    return {'content': str(json.decoder)}\n"
        with pytest.raises(EventHandlerError, match="AttributeError"):
            core.events._run_listener(self._listener(handler), {})


class TestPLMImportIntegration:
    def test_import_in_setup_and_user(self):
        plm = create_plm("plm.python")
        params = {"imports": sorted(ALLOWED)}
        plm.handle_messages([{"role": "system", "content": "import itertools"}], params)
        out = plm.handle_messages(
            [{"role": "user", "content": "print(next(itertools.count()))"}], params)
        assert out == {"content": "0\n"}

    def test_user_import_persists_in_namespace(self):
        plm = create_plm("plm.python")
        params = {"imports": sorted(ALLOWED)}
        plm.handle_messages([{"role": "user", "content": "import itertools\nx = 1"}], params)
        out = plm.handle_messages(
            [{"role": "user", "content": "print(next(itertools.islice(itertools.count(), 1)))"}],
            params)
        assert out == {"content": "0\n"}

    def test_import_denied_for_conversation(self):
        plm = create_plm("plm.python")
        params = {"imports": ["math", "json", "re"]}
        plm.handle_messages([{"role": "system", "content": "x = 1"}], params)
        out = plm.handle_messages([{"role": "user", "content": "import collections"}], params)
        assert "错误" in out["content"] and "SecurityError" in out["content"]

    def test_escape_import_path_blocked(self):
        plm = create_plm("plm.python")
        params = {"imports": ["math", "json", "re"]}
        plm.handle_messages([{"role": "system", "content": "x = 1"}], params)
        out = plm.handle_messages(
            [{"role": "user", "content": "import re\nprint(re.enum.sys.modules['os'])"}], params)
        assert "错误" in out["content"]

    def test_setup_direct_tool_call_rejected(self):
        """setup 函数体直接调用工具会拿到宿主生成器（帧内省逃逸的入口）。"""
        plm = create_plm("plm.python")
        with pytest.raises(SecurityError, match="yield from"):
            plm.handle_messages(
                [{"role": "system", "content": "def leak():\n    return memory_read('$MEM.x')\n"}], {})

    def test_user_generator_frame_introspection_blocked(self):
        plm = create_plm("plm.python")
        setup = "def leak():\n    yield from memory_read('$MEM.x')\n"
        plm.handle_messages([{"role": "system", "content": setup}], {})
        assert plm.handle_messages([{"role": "user", "content": "g = leak()"}], {}) == {"content": ""}
        out = plm.handle_messages([{"role": "user", "content": "print(g.gi_frame)"}], {})
        assert "错误" in out["content"] and "SecurityError" in out["content"]

    def test_command_direct_tool_call_rejected(self):
        plm = create_plm("plm.command")
        plm.handle_messages(
            [{"role": "system", "content": "def handle(cmd):\n    g = memory_read('$MEM.x')\n    return str(g)\n"}],
            {})
        out = plm.handle_messages([{"role": "user", "content": "go"}], {})
        assert "错误" in out["content"] and "SecurityError" in out["content"]

    def test_function_plm_import_whitelist(self):
        plm = create_plm("plm.function")
        params = {"imports": ["math", "json", "re", "itertools"]}
        plm.handle_messages(
            [{"role": "system", "content": "def handle(x):\n    import itertools\n    return {'v': next(itertools.count())}\n"}],
            params)
        assert plm.handle_messages([{"role": "user", "content": "{}"}], params) == {"content": '{"v": 0}'}

    def test_command_import_whitelist(self):
        plm = create_plm("plm.command")
        params = {"imports": ["math", "json", "re", "itertools"]}
        plm.handle_messages(
            [{"role": "system", "content": "def handle(cmd):\n    import itertools\n    return str(next(itertools.count()))\n"}],
            params)
        assert plm.handle_messages([{"role": "user", "content": "go"}], params) == {"content": "0"}

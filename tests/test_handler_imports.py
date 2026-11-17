"""事件处理器导入：按 owner 对话 caps 白名单，注册时抄快照。

对应计划 `implemented/plan-sandbox-imports-2026-11-15.md` 第 8 节 #6 的二期实现。
"""

import json

import pytest

from avm.core import Core
from avm.events import EventHandlerError
from avm.types import MetaDict, MetaList, StrNode


def _core():
    return Core()


def _handler_node(core, key, code, condition):
    node = MetaDict(
        data={"content": code, "condition": StrNode(json.dumps(condition))},
        ctrl={"type": "handler"},
    )
    core.mem[key] = node
    return f"$MEM.{key}"


def _exact_condition(event_type="memory_write", ref="$MEM.n", phase="before"):
    return {"events": [event_type], "phase": [phase], "target": {"kind": "exact", "ref": ref}}


IMPORT_IT = (
    "def handle(event):\n"
    "    import itertools\n"
    "    return {'response': str(next(itertools.count()))}\n"
)


class TestRegistrationSnapshot:
    def test_kernel_owner_gets_full_imports(self):
        core = _core()
        conv = core.start("sys", "user")
        ref = _handler_node(core, "h", IMPORT_IT, _exact_condition())
        listener_id = core.events.register_condition(
            conv.cid, ref, space=conv.caps.space(), kernel_mode=True)
        record = next(l for l in core.events.listeners if l["listener_id"] == listener_id)
        assert record["imports"] == sorted({"math", "json", "re", "itertools",
                                            "functools", "collections"})

    def test_owner_caps_imports_are_snapshotted(self):
        core = _core()
        conv = core.start("sys", "user")
        conv.caps = conv.caps.copy(imports=["itertools"])
        ref = _handler_node(core, "h", IMPORT_IT, _exact_condition())
        listener_id = core.events.register_condition(
            conv.cid, ref, space=conv.caps.space(), kernel_mode=True)
        record = next(l for l in core.events.listeners if l["listener_id"] == listener_id)
        assert record["imports"] == sorted({"math", "json", "re", "itertools"})
        assert core.events.targets.listener_summary(record)["imports"] == record["imports"]

    def test_legacy_register_also_snapshots(self):
        core = _core()
        conv = core.start("sys", "user")
        ref = _handler_node(core, "h", "def handle(event):\n    return {}\n", _exact_condition())
        listener_id = core.events.register(conv.cid, "memory_write", "$MEM.n", ref,
                                           phase="before", space=conv.caps.space(),
                                           kernel_mode=True)
        record = next(l for l in core.events.listeners if l["listener_id"] == listener_id)
        assert set(record["imports"]) >= {"math", "json", "re"}


class TestHandlerImportExecution:
    def _listener(self, code, imports):
        return {"listener_id": 1, "handler_code": code, "imports": imports}

    def test_granted_import_works(self):
        core = _core()
        out = core.events._run_listener(
            self._listener(IMPORT_IT, ["math", "json", "re", "itertools"]), {})
        assert out == {"response": "0"}

    def test_ungranted_import_fails_closed(self):
        core = _core()
        with pytest.raises(EventHandlerError, match="不在本对话的导入白名单"):
            core.events._run_listener(
                self._listener(IMPORT_IT, ["math", "json", "re"]), {})

    def test_baseline_import_works_without_grants(self):
        core = _core()
        code = "def handle(event):\n    import json\n    return {'response': json.dumps({'a': 1})}\n"
        assert core.events._run_listener(self._listener(code, ["math", "json", "re"]), {}) == {
            "response": '{"a": 1}'}

    def test_old_listener_record_without_imports_has_baseline_only(self):
        """旧快照/旧日志里的监听器没有 imports 字段：按基线恢复，不静默扩权。"""
        core = _core()
        old = {"listener_id": 1, "handler_code": IMPORT_IT}
        with pytest.raises(EventHandlerError, match="不在本对话的导入白名单"):
            core.events._run_listener(old, {})

    def test_dispatch_uses_imports(self):
        core = _core()
        conv = core.start("sys", "user")
        conv.caps = conv.caps.copy(imports=["itertools"])
        code = ("def handle(event):\n"
                "    import itertools\n"
                "    return {'action': 'replace', 'response': str(next(itertools.count()))}\n")
        ref = _handler_node(core, "h", code, _exact_condition())
        core.events.register_condition(conv.cid, ref, space=conv.caps.space(), kernel_mode=True)
        event = core.events.build_event(
            "memory_write", "before", conv=conv, target="$MEM.n",
            payload={"ref": "$MEM.n", "content": "1"},
            source={"kind": "core", "initiator_cid": conv.cid},
            target_ref=("memory", ("n",)),
        )
        out = core.events.dispatch(event)
        assert out["action"] == "replace"
        assert out["response"] == "Interrupted: 0"


class TestSnapshotCompat:
    def test_listener_imports_roundtrip_through_snapshot(self):
        from avm.events import listener_from_snapshot, listener_to_snapshot
        record = {"listener_id": 1, "owner_cid": 0, "imports": ["math", "json", "re", "itertools"]}
        restored = listener_from_snapshot(listener_to_snapshot(record))
        assert restored["imports"] == ["math", "json", "re", "itertools"]

    def test_old_snapshot_listener_without_imports_keeps_baseline(self):
        from avm.snapshot import restore, snapshot
        core = _core()
        conv = core.start("sys", "user")
        ref = _handler_node(core, "h", IMPORT_IT, _exact_condition())
        listener_id = core.events.register_condition(
            conv.cid, ref, space=conv.caps.space(), kernel_mode=True)
        doc = snapshot(core)
        record = next(l for l in doc["listeners"] if l["listener_id"] == listener_id)
        del record["imports"]

        restored = Core()
        restore(restored, doc)
        listener = next(l for l in restored.events.listeners
                        if l["listener_id"] == listener_id)
        assert "imports" not in listener
        with pytest.raises(EventHandlerError, match="不在本对话的导入白名单"):
            restored.events._run_listener(listener, {})

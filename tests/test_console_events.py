"""控制台的事件管线编辑测试

覆盖：enabled 屏蔽、update 就地改字段与重解析、事件观察者。

处理器契约：内存里的 python 节点被 exec 后要有可调用的 handle(event)，返回 dict。
这里的拦截型处理器返回 action=replace，于是写入不落地——用一个可观测的副作用
来判断处理器到底跑没跑。
"""

import pytest

from avm.core import Core, MemoryWriteInstruction
from avm.types import MetaDict


INTERCEPT = (
    "def handle(event):\n"
    "    return {'action': 'replace', 'response': '被拦截'}\n"
)
PASS_THROUGH = "def handle(event):\n    return {}\n"


def make_core(handler_code=INTERCEPT):
    core = Core()
    core.mem["h"] = MetaDict(data={"content": handler_code}, ctrl={"type": "python"})
    core.mem["own"] = MetaDict(data={}, ctrl={"owner": "1"})
    core.mem["own"]["child"] = ""
    return core


def add_listener(core, target="$MEM.own.child", phase="before"):
    return core.events.register(
        owner_cid=1, event_type="memory_write", target=target,
        handler_ref="$MEM.h", phase=phase,
    )


def write(core, content="v"):
    """绕过对话直接执行一条 memory_write 指令，触发事件管道。"""
    conv = core._conv_by_cid.get(0)
    if conv is None:
        conv = core.start("sys", "u")
    MemoryWriteInstruction("call-1", conv.cid, {"ref": "$MEM.own.child", "content": content}).execute(core, conv)


class TestEnabledFlag:
    def test_new_listener_is_enabled(self):
        core = make_core()
        listener_id = add_listener(core)
        assert core.events.listeners[0]["enabled"] is True
        assert f"enabled=True" in core.mem["listeners"].to_llm_string()
        assert listener_id == 1

    def test_enabled_listener_intercepts(self):
        core = make_core()
        add_listener(core)
        write(core, "v1")
        assert core.mem["own"]["child"] == ""

    def test_disabled_listener_does_not_fire(self):
        core = make_core()
        listener_id = add_listener(core)
        core.events.update(listener_id, enabled=False)
        write(core, "v1")
        assert core.mem["own"]["child"] == "v1"

    def test_disabled_listener_keeps_its_record(self):
        """暂时屏蔽要能恢复，所以记录必须留着。"""
        core = make_core()
        listener_id = add_listener(core)
        core.events.update(listener_id, enabled=False)
        assert len(core.events.listeners) == 1
        assert core.events.listeners[0]["listener_id"] == listener_id
        assert "enabled=False" in core.mem["listeners"].to_llm_string()

    def test_reenabling_restores_the_handler(self):
        core = make_core()
        listener_id = add_listener(core)
        core.events.update(listener_id, enabled=False)
        write(core, "v1")
        core.events.update(listener_id, enabled=True)
        write(core, "v2")
        assert core.mem["own"]["child"] == "v1"

    def test_enabled_must_be_boolean(self):
        core = make_core()
        listener_id = add_listener(core)
        with pytest.raises(ValueError):
            core.events.update(listener_id, enabled="no")


class TestUpdate:
    def test_update_changes_target(self):
        core = make_core()
        core.mem["other"] = MetaDict(data={"child": ""}, ctrl={"owner": "1"})
        listener_id = add_listener(core)
        # 改到自己的另一条路径上：原路径不再被拦截
        core.events.update(listener_id, target="$MEM.other.child")
        write(core, "v1")
        assert core.mem["own"]["child"] == "v1"

        MemoryWriteInstruction("call-2", 0, {"ref": "$MEM.other.child", "content": "w"}).execute(
            core, core._conv_by_cid[0])
        assert core.mem["other"]["child"] == ""

    def test_update_changes_phase(self):
        core = make_core()
        listener_id = add_listener(core, phase="before")
        core.events.update(listener_id, phase="after")
        write(core, "v1")
        assert core.mem["own"]["child"] == "v1", "after 阶段不能拦截"

    def test_update_rejects_unknown_fields(self):
        core = make_core()
        listener_id = add_listener(core)
        with pytest.raises(ValueError, match="不可改写的监听器字段"):
            core.events.update(listener_id, owner_cid=2)

    def test_update_does_not_touch_owner(self):
        core = make_core()
        listener_id = add_listener(core)
        core.events.update(listener_id, enabled=False)
        assert core.events.listeners[0]["owner_cid"] == 1

    def test_update_rechecks_target_permission(self):
        """改 target 要过与注册同一套权限：无主路径不接受。"""
        core = make_core()
        listener_id = add_listener(core)
        core.mem["free"] = "no owner"
        with pytest.raises(ValueError, match="无主路径"):
            core.events.update(listener_id, target="$MEM.free")

    def test_update_rejects_before_phase_for_after_only_event(self):
        core = make_core()
        listener_id = add_listener(core)
        with pytest.raises(ValueError):
            core.events.update(listener_id, event_type="close_conversation", phase="before")

    def test_update_reresolves_handler_code(self):
        """处理器代码是注册时抄下的副本；改内存节点不会自动生效，update 会重解析。"""
        core = make_core(INTERCEPT)
        listener_id = add_listener(core)
        write(core, "v1")
        assert core.mem["own"]["child"] == "", "拦截生效"

        # 只改内存节点：记录的副本没变，行为不变
        core.mem["h"]["content"] = PASS_THROUGH
        write(core, "v2")
        assert core.mem["own"]["child"] == "", "副本未刷新，仍然拦截"

        # update 重解析后新代码生效
        core.events.update(listener_id, target="$MEM.own.child")
        write(core, "v3")
        assert core.mem["own"]["child"] == "v3", "重解析后放行"

    def test_update_returns_the_new_record(self):
        core = make_core()
        listener_id = add_listener(core)
        record = core.events.update(listener_id, enabled=False)
        assert record["enabled"] is False
        assert record["listener_id"] == listener_id
        assert isinstance(record["handler_code"], str)

    def test_update_missing_listener(self):
        core = make_core()
        with pytest.raises(ValueError, match="不存在"):
            core.events.update(99, enabled=False)


class TestObservers:
    def test_observer_receives_the_full_event(self):
        core = make_core(handler_code=PASS_THROUGH)
        seen = []
        core.events.add_observer(seen.append)
        write(core, "v1")
        assert [e["phase"] for e in seen] == ["before", "after"]
        assert seen[0]["event_type"] == "memory_write"
        assert seen[0]["payload"] == {"ref": "$MEM.own.child", "content": "v1"}
        assert seen[0]["source"]["initiator_cid"] == 0
        assert seen[1]["source"] == {"kind": "memory"}

    def test_observer_sees_interception(self):
        core = make_core(handler_code=INTERCEPT)
        seen = []
        core.events.add_observer(seen.append)
        add_listener(core)
        write(core, "v1")
        before = [e for e in seen if e["phase"] == "before"][0]
        assert before["action"] == "replace"
        assert not any(e["phase"] == "after" for e in seen), "被拦截就没有 after 事实"

    def test_observer_failure_does_not_break_dispatch(self, capsys):
        core = make_core(handler_code=PASS_THROUGH)

        def broken(event):
            raise RuntimeError("观察者坏了")

        core.events.add_observer(broken)
        write(core, "v1")
        assert core.mem["own"]["child"] == "v1"
        assert "观察者坏了" in capsys.readouterr().err

    def test_observer_can_be_removed(self):
        core = make_core(handler_code=PASS_THROUGH)
        seen = []
        core.events.add_observer(seen.append)
        core.events.add_observer(seen.append)  # 同一个绑定方法不重复登记
        assert len(core.events.observers) == 1

        core.events.remove_observer(seen.append)
        assert core.events.observers == []
        core.events.remove_observer(seen.append)  # 重复移除无害

        write(core, "v1")
        assert seen == []

    def test_observers_are_not_shared_between_cores(self):
        first, second = Core(), Core()
        seen = []
        first.events.add_observer(seen.append)
        assert second.events.observers == []
        assert seen == []

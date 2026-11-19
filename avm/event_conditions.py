"""声明式监听条件：把 handler 节点上的 condition 数据解析成内部条件，并判断事件是否命中。

条件是一段写在 handler 节点里的 JSON（preset / events / phase / target / source）。
解析的产物是内部条件：事件名、阶段、canonical 目标（正常跟随与"删除视角"两套引用）、
来源过滤。之后每次派发都拿事件去比一遍。

为什么单独成模块：解析与匹配共用一个形状约定，散在事件系统里会与注册表、分发混在
一起看不出来；而它们只读内核（canonical_ref / get_conversation），不碰监听器表。

校验顺序也是行为的一部分——五个字段按固定顺序求值，同时多处非法时报哪一条是
可观测的，别重排。

依赖方向：event_conditions -> event_contracts / types / exceptions。不 import events。
"""

import json

from .event_contracts import EVENT_TYPES
from .exceptions import IllegalOperationError
from .types import MetaDict

CONDITION_KEYS = frozenset({"preset", "events", "phase", "target", "source"})
LIFECYCLE_EVENTS = ("create_cmd", "create_sub", "close_conversation", "listen_event")

# 用户态对话可用的命名条件子集（名称是机制常量；名单待作者细化，见 issue-2026-11-01）
# own_nodes.memory 依赖已删除的节点所有权 cid（owned_by_owner），不再提供。
USER_CONDITION_PRESETS = {
    "inbox": {
        "events": ["send_instruction"],
        "phase": ["before", "after"],
        "target": {"kind": "cid_owner"},
    },
    "self.lifecycle": {
        "events": list(LIFECYCLE_EVENTS),
        "phase": ["before", "after"],
        "target": {"kind": "self_lifecycle"},
    },
}


def condition_leaf(node):
    """读取 handler 节点的 condition 叶子（JSON 字符串）；缺省 None。"""
    if not isinstance(node, MetaDict):
        return None
    raw = node.get("condition")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise IllegalOperationError("handler 节点的 condition 必须是 str 数据叶子")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise IllegalOperationError(f"handler 节点的 condition 不是合法 JSON：{e}")


class EventConditions:
    """条件的解析与匹配。构造时只拿 core（读内存与对话表，不写）。"""

    def __init__(self, core):
        self.core = core

    def parse(self, node, owner_cid: int, owner_kernel: bool,
              space=None) -> dict:
        """把 handler 节点的 condition 数据解析成内部条件（canonical 目标）。

        五个字段各自一段校验，顺序就是下面字典的书写顺序 —— 字典字面量按插入顺序
        求值，同时有多处非法时报哪一条也属于可观测行为，别重排。
        """
        raw = condition_leaf(node)
        if raw is None:
            raise IllegalOperationError(
                "handler 节点没有 condition 数据；新式监听器必须在 handler 节点声明条件"
            )
        if not isinstance(raw, dict):
            raise IllegalOperationError("handler 节点的 condition 必须是 JSON 对象")
        unknown = sorted(set(raw) - CONDITION_KEYS)
        if unknown:
            raise IllegalOperationError(f"handler 节点 condition 含未知字段：{unknown}")
        raw = self._expand_preset(raw, owner_kernel)
        return {
            "preset": raw.get("preset"),
            "events": self._events(raw),
            "phases": self._phases(raw),
            "target": self._target(raw, owner_cid, space),
            "source": self._source(raw),
        }

    def _expand_preset(self, raw: dict, owner_kernel: bool) -> dict:
        """preset 展开成完整条件；用户态只能走 preset，不能自定义 condition。"""
        if "preset" not in raw:
            if owner_kernel:
                return raw
            # TODO(issue): owner 授权删除后用户态仍只能用 preset，无法给出 memory target；
            # Guard 的 caps/roots 策略因此只能观察到内核态注册的 memory 监听。是否放开
            # 用户自定义条件属于事件子集策略，留给作者决定（本计划未包含）。
            raise IllegalOperationError(
                "用户态对话只能使用命名条件子集（preset），不能自定义 condition"
            )
        preset = raw["preset"]
        if preset == "own_work.memory":
            raise IllegalOperationError(
                "条件 preset 'own_work.memory' 已移除：工作空间不再由内核分配。"
                "改用 self.lifecycle，或由软件动态加根后自定义条件"
            )
        if preset == "own_nodes.memory":
            raise IllegalOperationError(
                "条件 preset 'own_nodes.memory' 已移除：节点所有权 cid 不再写进 MEM。"
                "改用 self.lifecycle，或改用自定义 target（内核态）"
            )
        if preset not in USER_CONDITION_PRESETS:
            raise IllegalOperationError(
                f"未知的用户态条件子集 {preset!r}（可用：{sorted(USER_CONDITION_PRESETS)}）"
            )
        expanded = dict(USER_CONDITION_PRESETS[preset])
        expanded["target"] = dict(expanded.get("target") or {})
        expanded["preset"] = preset
        return expanded

    @staticmethod
    def _events(raw: dict) -> list:
        """events：缺省监听全部事件类型。"""
        events = raw.get("events")
        if events is None:
            events = list(EVENT_TYPES)
        if not isinstance(events, list) or not all(isinstance(x, str) for x in events):
            raise IllegalOperationError("condition.events 必须是事件类型字符串列表")
        unknown_events = [x for x in events if x not in EVENT_TYPES]
        if unknown_events:
            raise IllegalOperationError(f"condition.events 含未知事件类型：{unknown_events}")
        return list(events)

    @staticmethod
    def _phases(raw: dict) -> list:
        """phases：缺省 before+after，"all" 是它的别名。"""
        phases = raw.get("phase", ["before", "after"])
        if phases == "all":
            phases = ["before", "after"]
        if not isinstance(phases, list) or not all(x in ("before", "after") for x in phases):
            raise IllegalOperationError("condition.phase 必须是 before/after 列表")
        return list(phases)

    def _target(self, raw: dict, owner_cid: int, space=None):
        """target：换成 canonical 引用。exact/subtree 要"正常跟随"与"删除视角"两套引用，
        后者不跟随最终段的链接（删的是链接节点本身）。"""
        target = raw.get("target")
        if target is None:
            return None
        if not isinstance(target, dict):
            raise IllegalOperationError("condition.target 必须是对象")
        kind = target.get("kind")
        if kind == "owned_by_owner":
            raise IllegalOperationError(
                "condition.target.kind 'owned_by_owner' 已移除：节点所有权 cid 不再写进 MEM"
            )
        if kind not in ("exact", "subtree", "cid_owner", "self_lifecycle"):
            raise IllegalOperationError(f"condition.target.kind 不支持：{kind!r}")
        resolved = {"kind": kind}
        if kind in ("exact", "subtree"):
            ref = target.get("ref")
            if not isinstance(ref, str) or not ref.startswith("$"):
                raise IllegalOperationError("condition.target.ref 必须是 $ 引用")
            normal = self.core.canonical_ref(ref, space=space)
            delete = self.core.canonical_ref(ref, space=space, follow_final=False)
            if normal is None:
                raise IllegalOperationError(f"condition.target.ref 无法解析：{ref}")
            resolved["refs"] = {"normal": normal, "delete": delete}
        else:
            resolved["cid"] = owner_cid
        return resolved

    @staticmethod
    def _source(raw: dict):
        """source：按发起者的 mode/groups 过滤事件来源。"""
        source = raw.get("source")
        if source is None:
            return None
        if not isinstance(source, dict):
            raise IllegalOperationError("condition.source 必须是对象")
        unknown_source = sorted(set(source) - {"mode", "groups"})
        if unknown_source:
            if "cid" in unknown_source:
                raise IllegalOperationError(
                    "condition.source.cid 已移除：不能按来源 cid 过滤事件"
                )
            raise IllegalOperationError(
                f"condition.source 含未知字段：{unknown_source}（允许：mode/groups）"
            )
        resolved_source = {}
        for key in ("mode", "groups"):
            if key in source:
                value = source[key]
                if not isinstance(value, list):
                    raise IllegalOperationError(f"condition.source.{key} 必须是列表")
                resolved_source[key] = value
        return resolved_source

    def matches(self, listener: dict, event: dict) -> bool:
        condition = listener.get("condition")
        if condition is None:
            return False
        if event["event_type"] not in condition["events"]:
            return False
        if event["phase"] not in condition["phases"]:
            return False
        target = condition.get("target")
        if target is not None and not self._target_matches(target, event):
            return False
        if not self._source_matches(condition.get("source"), event):
            return False
        return True

    def _target_matches(self, target: dict, event: dict) -> bool:
        kind = target.get("kind")
        if kind in ("exact", "subtree"):
            refs = target.get("refs") or {}
            ref = refs.get("delete" if event.get("event_type") == "memory_delete" else "normal")
            if ref is None:
                return False
            ev = event.get("target_ref")
            if ev is None:
                return len(ref) >= 2 and event.get("target", "") == "$" + ref[0] + (
                    "." + ".".join(ref[1]) if ref[1] else "")
            if kind == "exact":
                return tuple(ev) == tuple(ref)
            if ref[0] != ev[0]:
                if not (ref[0] == "memory" and ev[0] == "memory"):
                    return False
            if ref[0] == "memory":
                return tuple(ev[1][:len(ref[1])]) == tuple(ref[1]) if len(ref[1]) <= len(ev[1]) else False
            return ref[1] == ev[1] and len(ref[2]) <= len(ev[2]) and tuple(ev[2][:len(ref[2])]) == tuple(ref[2])
        if kind == "cid_owner":
            ev = event.get("target_ref")
            return bool(ev) and ev[0] == "cid" and ev[1] == target.get("cid")
        if kind == "self_lifecycle":
            cid = target.get("cid")
            ev = event.get("target_ref")
            if ev and ev[0] == "cid" and ev[1] == cid:
                return True
            conv = self._actor_conv(event)
            return conv is not None and conv.cid == cid
        return False

    def _source_matches(self, source_filter, event: dict) -> bool:
        if not source_filter:
            return True
        conv = self._actor_conv(event)
        if conv is None:
            return False
        if "mode" in source_filter and conv.caps.mode not in source_filter["mode"]:
            return False
        if "groups" in source_filter and not (set(source_filter["groups"]) & set(conv.caps.groups)):
            return False
        return True

    def _actor_conv(self, event: dict):
        source = event.get("source") or {}
        cid = source.get("initiator_cid") or (event.get("payload") or {}).get("actor_cid")
        return self.core.get_conversation(cid) if isinstance(cid, int) else None

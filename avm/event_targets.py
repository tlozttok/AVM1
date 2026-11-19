"""事件目标的探针：把一个请求别名读成内核里的真实状态。

权限判定和事件载荷问的是同一组问题——这个目标指的是什么、在不在、是什么类型的
节点、它的祖先是谁、它的完整值长什么样。这些读法原来长在 EventSystem 里，于是
"判权"和"拼载荷"各自演进、容易漂移。现在收在一处。

这一层**只读**：它不调用任何写接口，改内存要走 Memory 的方法。唯一的例外是把
内核的只读结构（meta / ctrl）复制成普通 dict，供 JSON 序列化。

依赖方向：event_targets → event_contracts / memory / types。不 import events，
否则事件系统与探针会互为依赖。
"""

from typing import List, Optional

from .event_contracts import MAX_ANCESTORS_DEPTH, NORMAL_TIER
from .modules import BASE_MODULES
from .types import MetaDict, MetaList

# "没这个节点"的哨兵：不能用 None——`$MEM` 下真的可以有 None 值。
_MISSING = object()


class TargetProbe:
    """把目标别名读成内核状态。构造时只拿 core 与一张"当前监听器表"的读法。"""

    def __init__(self, core, listeners):
        self.core = core
        # listeners 必须是**取法**而不是列表本身：EventSystem 在批量注销时会整个
        # 重建这个列表（cancel_all_for_owner），持有旧列表就会读到过期快照。
        self.listeners = listeners

    # ------------------------------------------------------------------
    # 目标的规范化
    # ------------------------------------------------------------------

    def canonical_ref(self, target: str, space=None, follow_final: bool = True):
        """把请求别名转成内部 canonical target_ref；无法解析返回 None。

        格式：("memory", 段序列) / ("device", 挂载键, 剩余段) / ("cid", id) /
        ("listener", id) / ("*",)。canonical 只用于匹配与权限判定，不进 normal 载荷。
        """
        if not isinstance(target, str) or not target:
            return None
        if target == "*":
            return ("*",)
        if target.startswith("cid:"):
            try:
                return ("cid", int(target[4:]))
            except ValueError:
                return None
        if target.startswith("listener:"):
            try:
                return ("listener", int(target[9:]))
            except ValueError:
                return None
        if not target.startswith("$"):
            return None
        return self.core.canonical_ref(target, space=space, follow_final=follow_final)

    # ------------------------------------------------------------------
    # 对话
    # ------------------------------------------------------------------

    def conversation_snapshot(self, cid) -> Optional[dict]:
        """对话的只读摘要（含调度状态）；cid 不是已知对话时返回 None。"""
        if not isinstance(cid, int):
            return None
        conv = self.core.get_conversation(cid)
        if conv is None:
            return None
        if self.core.is_finished(cid):
            state = "finished"
        elif cid == self.core.active_conversation_id():
            state = "active"
        elif self.core.is_ready(cid):
            state = "ready"
        elif self.core.is_dormant(cid):
            state = "dormant"
        else:
            state = "unknown"
        return {
            "cid": cid,
            "identity": conv.identity,
            "name": getattr(conv, "name", None),
            "mode": conv.caps.mode,
            "groups": list(conv.caps.groups),
            "is_sub": conv.is_sub,
            "is_python": conv.is_python,
            "parent_cid": conv.parent.cid if conv.parent is not None else None,
            "state": state,
            "para_ref": conv.para_ref,
        }

    @staticmethod
    def actor_cid(event: dict):
        """事件的动作发起者：先看来源对话，再看载荷里的 actor_cid。"""
        source = event.get("source") or {}
        for cid in (source.get("initiator_cid"), (event.get("payload") or {}).get("actor_cid")):
            if isinstance(cid, int):
                return cid
        return None

    # ------------------------------------------------------------------
    # 节点读数
    # ------------------------------------------------------------------

    @staticmethod
    def node_kind(node) -> str:
        from .types import LinkNode
        if isinstance(node, MetaDict):
            return "dict"
        if isinstance(node, MetaList):
            return "list"
        if isinstance(node, LinkNode):
            return "link"
        if isinstance(node, str):
            return "str"
        return type(node).__name__

    @staticmethod
    def node_meta(node):
        if hasattr(node, "get_metadata"):
            return node.get_metadata()
        return None

    def node_ancestors(self, segments) -> List[dict]:
        """祖先链简单信息（type/meta/ctrl + depth）。

        normal 载荷不暴露真实路径；祖先从根到父排列，超过 MAX_ANCESTORS_DEPTH
        时只保留离目标最近的若干层（带 depth 便于判断截断）。
        """
        total = len(segments) - 1
        if total <= 0:
            return []
        first = max(1, total - MAX_ANCESTORS_DEPTH + 1)
        ancestors = []
        for depth in range(first, total + 1):
            prefix = list(segments[:depth])
            try:
                node = self.core.mem.resolve_path(["MEM"] + prefix, for_llm=False)
            except Exception:
                break
            ctrl = dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None
            ancestors.append({
                "depth": depth,
                "type": self.node_kind(node),
                "meta": self.node_meta(node),
                "ctrl": ctrl,
            })
        return ancestors

    @staticmethod
    def alias_parent(alias):
        """`$MEM.a.b` 的父别名 `$MEM.a`；不是别名或没有点时返回 None。"""
        if isinstance(alias, str) and alias.startswith("$") and "." in alias:
            return alias.rsplit(".", 1)[0]
        return None

    def resolve_node_no_follow(self, segments):
        """解析节点但不跟随最终段的链接（删链接时看链接节点本身）。"""
        segments = list(segments)
        if not segments:
            return self.core.mem.root
        try:
            parent = self.core.mem.resolve_path(["MEM"] + segments[:-1], for_llm=False)
        except Exception:
            return _MISSING
        key = segments[-1]
        try:
            if isinstance(parent, (MetaDict, dict)):
                return parent[key] if key in parent else _MISSING
        except Exception:
            return _MISSING
        return _MISSING

    def node_simple(self, segments, follow_final: bool = True):
        """一个节点的简单信息；节点不存在时返回 None。"""
        if follow_final:
            try:
                node = self.core.mem.resolve_path(["MEM"] + list(segments), for_llm=False)
            except Exception:
                return None
        else:
            node = self.resolve_node_no_follow(segments)
            if node is _MISSING:
                return None
        return {
            "type": self.node_kind(node),
            "meta": self.node_meta(node),
            "ctrl": dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None,
            "ancestors": self.node_ancestors(list(segments)),
        }

    def memory_context(self, segments, alias: str) -> dict:
        """目标不存在时的上下文：最近存在前缀 + 直接父节点（用于 make/追加裁决）。"""
        context = {}
        existing_index = None
        for i in range(len(segments) - 1, -1, -1):
            if self.node_simple(segments[:i]) is not None:
                existing_index = i
                break
            if i == 0:
                existing_index = 0
        if existing_index is not None:
            prefix = self.node_simple(segments[:existing_index]) or {}
            new_count = len(segments) - existing_index
            prefix_alias = None
            if isinstance(alias, str) and alias.startswith("$"):
                parts = alias[1:].split(".")
                keep = len(parts) - new_count
                if keep > 0:
                    prefix_alias = "$" + ".".join(parts[:keep])
                elif keep == 0:
                    prefix_alias = "$MEM"
            prefix.update({
                "alias": prefix_alias,
                "depth": existing_index,
                "ancestors": self.node_ancestors(segments[:existing_index]),
            })
            context["longest_existing_prefix"] = prefix
        parent_segments = list(segments[:-1])
        parent = self.node_simple(parent_segments) if parent_segments else None
        if parent is not None:
            parent["alias"] = self.alias_parent(alias)
        context["parent"] = parent
        return context

    # ------------------------------------------------------------------
    # 分层载荷：normal 只给"看得懂"，kernel 才给完整值
    # ------------------------------------------------------------------

    def target_info(self, target_ref, alias: str, follow_final: bool = True) -> dict:
        """normal 级目标信息：种类、别名、是否存在、类型、描述、ctrl、祖先链。"""
        info = {"kind": "unknown", "ref": alias, "exists": False, "type": None,
                "meta": None, "ctrl": None, "ancestors": []}
        if not target_ref:
            return info
        tag = target_ref[0]
        if tag == "memory":
            segments = list(target_ref[1])
            info.update({"kind": "memory"})
            if follow_final:
                try:
                    node = self.core.mem.resolve_path(["MEM"] + segments, for_llm=False)
                except Exception:
                    info.update(self.memory_context(segments, alias))
                    return info
            else:
                node = self.resolve_node_no_follow(segments)
                if node is _MISSING:
                    info.update(self.memory_context(segments, alias))
                    return info
            info["exists"] = True
            info["type"] = self.node_kind(node)
            info["meta"] = self.node_meta(node)
            info["ctrl"] = dict(node.get_ctrl()) if hasattr(node, "get_ctrl") and node.get_ctrl() else None
            info["ancestors"] = self.node_ancestors(segments)
            parent = self.node_simple(segments[:-1]) if segments else None
            if parent is not None:
                parent["alias"] = self.alias_parent(alias)
            info["parent"] = parent
            return info
        if tag == "device":
            mount_key, remainder = target_ref[1], list(target_ref[2])
            device = self.core.mem.devices.get(mount_key)
            info.update({"kind": "device", "mount": mount_key, "remainder": remainder})
            if device is not None:
                info["exists"] = True
                try:
                    info["device"] = device.to_llm_string()
                except Exception:
                    info["device"] = None
            return info
        if tag == "cid":
            cid = target_ref[1]
            conv = self.core.get_conversation(cid)
            info.update({"kind": "conversation", "type": "conversation",
                         "conversation": self.conversation_snapshot(cid)})
            info["exists"] = conv is not None
            return info
        if tag == "listener":
            lid = target_ref[1]
            record = next((x for x in self.listeners() if x["listener_id"] == lid), None)
            info.update({"kind": "listener", "type": "listener",
                         "listener": self.listener_summary(record) if record else None})
            info["exists"] = record is not None
            return info
        if tag == "*":
            info.update({"kind": "wildcard", "exists": True})
        return info

    @staticmethod
    def listener_summary(listener: dict) -> dict:
        """监听器记录摘要（不含 handler_code，避免把安全代码交给其他处理器）。"""
        return {
            "listener_id": listener["listener_id"],
            "owner_cid": listener.get("owner_cid"),
            "event_type": listener.get("event_type"),
            "target": listener.get("target"),
            "phase": listener.get("phase"),
            "priority": listener.get("priority", 0),
            "tier": listener.get("tier", NORMAL_TIER),
            "enabled": listener.get("enabled", True),
            "imports": sorted(listener.get("imports") or BASE_MODULES),
        }

    def target_full(self, target_ref, follow_final: bool = True):
        """kernel 级目标完整数据（JSON 安全结构）。"""
        if not target_ref or target_ref[0] != "memory":
            return None
        segments = list(target_ref[1])
        if follow_final:
            try:
                node = self.core.mem.resolve_path(["MEM"] + segments, for_llm=False)
            except Exception:
                return None
        else:
            node = self.resolve_node_no_follow(segments)
            if node is _MISSING:
                return None
        return self.json_safe_node(node)

    def json_safe_node(self, node):
        """把内存节点摊成 {kind, meta, ctrl, value}；链接只给目标，不展开。"""
        from .types import LinkNode, StrNode
        if isinstance(node, MetaDict):
            return {"kind": "dict", "meta": node.get_metadata(),
                    "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": {str(k): self.json_safe_node(v) for k, v in node.items()}}
        if isinstance(node, MetaList):
            return {"kind": "list", "meta": node.get_metadata(),
                    "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": [self.json_safe_node(v) for v in node]}
        if isinstance(node, LinkNode):
            return {"kind": "link", "target": node.target}
        if isinstance(node, StrNode):
            return {"kind": "str", "ctrl": dict(node.get_ctrl()) if node.get_ctrl() else None,
                    "value": str(node)}
        if isinstance(node, str):
            return {"kind": "str", "value": str(node)}
        return {"kind": type(node).__name__, "value": str(node)}

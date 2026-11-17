"""内存树的持久化格式：显式节点 `{kind, meta, ctrl, value}`。

镜像的 mem 段、persist_to 文件、重启快照、日志里的内存效果，用的是同一个格式——
所以它只能有一份实现。语义（怎么寻址、怎么挂设备、怎么写回）在 memory.py，
这里只管两件事：把节点摊成 JSON 安全的结构，以及把它装回节点树。

依赖方向：memory_codec -> types / exceptions。不 import memory——反过来的话
memory 就没法用它了（成环）。
"""

import json

from .exceptions import VMMemoryError
from .types import LinkNode, MetaDict, MetaList, StrNode


def serialize_node(value) -> dict:
    """单个内存节点的显式表示（与镜像 mem 段、persist_to、日志效果同格式）。"""
    if isinstance(value, MetaDict):
        node = {"kind": "dict", "value": {k: serialize_node(v) for k, v in value.items()}}
    elif isinstance(value, MetaList):
        node = {"kind": "list", "value": [serialize_node(v) for v in value]}
    elif isinstance(value, LinkNode):
        node = {"kind": "link", "target": value.target, "value": None}
    else:
        node = {"kind": "str", "value": str(value)}
        ctrl = value.get_ctrl() if isinstance(value, StrNode) else None
        if ctrl is not None:
            node["ctrl"] = ctrl
        return node
    meta = value.get_metadata() if hasattr(value, "get_metadata") else None
    ctrl = value.get_ctrl() if hasattr(value, "get_ctrl") else None
    if meta is not None:
        node["meta"] = meta
    if ctrl is not None:
        node["ctrl"] = ctrl
    return node


def deserialize_node(value):
    """解析单个显式节点（只接受 {kind, meta, ctrl, value/target} 格式）。

    裸值（字符串/数字）原样放行：data.json 与日志里存在直接写标量的老数据，
    拒绝它们等于让旧镜像打不开。
    """
    if isinstance(value, dict):
        if value.get("kind") in ("str", "dict", "list", "link"):
            kind = value["kind"]
            v = value.get("value")
            meta = value.get("meta")
            ctrl = value.get("ctrl")
            if kind == "link":
                target = value.get("target")
                if not isinstance(target, str) or not target:
                    raise VMMemoryError(f"link 节点的 target 必须是非空字符串：{target!r}")
                return LinkNode(target=target, metadata=meta, ctrl=ctrl)
            if kind == "str":
                return StrNode(v if isinstance(v, str) else str(v), ctrl=ctrl)
            if kind == "dict":
                return MetaDict(
                    data={k: deserialize_node(x) for k, x in (v or {}).items()},
                    metadata=meta, ctrl=ctrl,
                )
            if kind == "list":
                return MetaList(
                    data=[deserialize_node(x) for x in (v or [])],
                    metadata=meta, ctrl=ctrl,
                )
        raise VMMemoryError(
            f"内存节点格式无效（缺合法 kind）：{value!r}；旧 __type 格式不再支持，"
            "请用 reset 重建世界"
        )
    return value


def to_image_dict(memory) -> dict:
    """内存树的显式节点表示（与镜像 mem 段、persist_to、内核转储同格式）。

    不可 JSON 化的值降级为 `str(value)`（与既有行为一致）。取根走公开的 `root`
    ——序列化是"明知自己在操作整棵树"的场景，正是那个属性的用途。
    """
    return serialize_node(memory.root)


def save(memory, filepath: str) -> None:
    """将内存数据持久化到 JSON 文件（仅 _data，不含设备）。
    格式与系统镜像 mem 段一致：显式节点 {kind, meta, ctrl, value}。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_image_dict(memory), f, ensure_ascii=False, indent=2)

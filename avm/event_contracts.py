"""事件层的最小契约：事件类型名单、监听器层级名与载荷深度上限。

这两个东西事件系统的两条支路都要用——监听注册与分发（events.py）、目标探针
（event_targets.py）。放在最底层，是为了让探针不必反过来 import 事件系统：
依赖方向只有一条，才不会缠成循环导入。

改这里的值等于改所有监听器的默认行为，别顺手调。
"""

EVENT_TYPES = [
    "memory_read",
    "memory_write",
    "memory_make",
    "memory_delete",
    "edit_metadata",
    "send_instruction",
    "create_cmd",
    "create_sub",
    "close_conversation",
    "listen_event",
]

NORMAL_TIER = "normal"
KERNEL_TIER = "kernel"
LISTENER_TIERS = (NORMAL_TIER, KERNEL_TIER)

# normal 载荷祖先链的深度上限（只保留最近 MAX_ANCESTORS_DEPTH 个祖先）
MAX_ANCESTORS_DEPTH = 32

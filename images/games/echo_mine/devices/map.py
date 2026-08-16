"""echo_mine 地图设备：从共享内存实时渲染世界地图。

挂载到 $MEM.map。设备通过 attach_core 持有 Core 引用，只读
$MEM.world.rooms 与 $MEM.world.player.room_id，输出文本地图。
"""
from avm.memory_device import MemoryDevice


class MapDevice(MemoryDevice):
    def __init__(self):
        self._core = None

    def attach_core(self, core):
        self._core = core

    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        if self._core is None:
            return "map device: core not attached"
        try:
            rooms = self._core.unwrap("$MEM.world.rooms", for_llm=False)
            player_room = self._core.unwrap("$MEM.world.player.room_id", for_llm=False)
        except Exception as e:
            return f"map device error: {type(e).__name__}: {e}"

        lines = ["echo_mine 地图:"]
        for room_id, room in rooms.items():
            marker = "  *你在这里*" if room_id == player_room else ""
            lines.append(f"[{room_id}]{marker}")
            links = (room["links"] or "").strip()
            if not links:
                continue
            for item in links.split(","):
                item = item.strip()
                if not item or ":" not in item:
                    continue
                direction, target = item.split(":", 1)
                lines.append(f"  {direction.strip()} -> {target.strip()}")
        return "\n".join(lines)

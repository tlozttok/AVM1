"""echo_mine 骰子设备：以虚拟文件形式提供宿主侧随机性。

挂载到 $MEM.dice：
- 读 $MEM.dice     返回使用说明；
- 读 $MEM.dice.d6  返回 1..6 的随机数字符串。
"""
import random

from avm.exceptions import VMMemoryError
from avm.memory_device import MemoryDevice


class DiceDevice(MemoryDevice):
    def pretend_as_type(self):
        return "str"

    def to_llm_string(self):
        return "dice device: memory_read(\"$MEM.dice.d6\") 返回 1..6 的随机数字符串"

    def resolve_path(self, path):
        if path == ["d6"]:
            return str(random.SystemRandom().randint(1, 6))
        raise VMMemoryError("DiceDevice 只支持 $MEM.dice.d6")

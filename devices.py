"""AVM 设备 I/O 抽象"""
import logging

logger = logging.getLogger(__name__)


class Device:
    """设备抽象基类"""
    def read(self):
        """读取设备数据"""
        raise NotImplementedError

    def write(self, value):
        """写入数据到设备"""
        raise NotImplementedError


class UserInputDevice(Device):
    """用户输入设备示例"""
    def read(self):
        logger.info("[UserInputDevice] read")
        return input("User: ")

    def write(self, value):
        logger.info("[UserInputDevice] write: %r", value)
        print(f"AI: {value}")


class ConsoleOutputDevice(Device):
    """控制台输出设备示例"""
    def read(self):
        logger.info("[ConsoleOutputDevice] read")
        return ""

    def write(self, value):
        logger.info("[ConsoleOutputDevice] write: %r", value)
        print(value)

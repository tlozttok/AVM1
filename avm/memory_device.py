"""AVM 内存设备：可挂载到内存路径的虚拟设备"""
try:
    import readline  # noqa: F401  调试用：启用行编辑（历史/方向键/中文输入法组合支持）
except ImportError:
    readline = None  # Windows 无 readline，input() 仍可用
import os
import sys
from typing import Any, Optional, List


class MemoryDevice:
    """设备基类"""

    mount_path: Optional[str] = None  # 挂载键（不含 $MEM 前缀），由镜像加载器设置

    def attach_core(self, core) -> None:
        """调试用：可选钩子，把 Core 引用交给设备（设备需要访问运行时状态时覆写）。"""
        pass

    def authorize_listener(self, event_type: str, remainder: list, owner_cid: int) -> bool:
        """Core 注册监听时询问：是否允许该对话监听此设备内路径。

        默认拒绝。实现方不得在授权检查中调用 resolve_path / 触发设备副作用。
        """
        return False

    def after_event(self, event_type: str, remainder: list, result: dict) -> Optional[dict]:
        """设备操作成功后 Core 询问是否有事实事件。

        返回纯 payload dict；返回 None 表示无 after 事件。
        Core 会统一加 event_id / phase / source / target，并强制合并 actor_cid / ref。
        """
        return None

    def pretend_as_type(self) -> str:
        raise NotImplementedError

    def to_llm_string(self) -> str:
        raise NotImplementedError

    def resolve_path(self, path: list) -> Any:
        """设备子路径解析。默认不支持子路径。覆盖以实现符号链接等。"""
        from .exceptions import VMMemoryError
        raise VMMemoryError(f"{type(self).__name__} 不支持子路径访问")


class StringDevice(MemoryDevice):
    """假装是字符串的设备"""

    def __init__(self, value: str = ""):
        self._value: str = value

    def pretend_as_type(self) -> str:
        return "str"

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        if not isinstance(value, str):
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"StringDevice 只接受 str，got {type(value).__name__}")
        self._value = value

    def to_llm_string(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"StringDevice(value={self._value!r})"


class MetaListDevice(MemoryDevice):
    """假装是 MetaList 的设备"""

    def __init__(self, data=None, metadata=None):
        self._data: list = data if data is not None else []
        self._metadata: Optional[str] = metadata

    def pretend_as_type(self) -> str:
        return "MetaList"

    def __getitem__(self, index: int) -> Any:
        try:
            return self._data[index]
        except IndexError:
            from .exceptions import MemoryIndexOutOfRangeError
            raise MemoryIndexOutOfRangeError(f"索引越界：{index}，列表长度：{len(self._data)}") from None

    def __setitem__(self, index: int, value: Any) -> None:
        self._data[index] = value

    def append(self, value: Any) -> None:
        self._data.append(value)

    def get_metadata(self) -> Optional[str]:
        return self._metadata

    def set_metadata(self, metadata: str) -> None:
        self._metadata = metadata

    def set_value(self, value: list) -> None:
        """整体替换内部数据（用于 Memory.set_by_path 的设备写入）"""
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        if not isinstance(value, list):
            raise MemoryTypeError(f"MetaListDevice 只接受 list，got {type(value).__name__}")
        self._data = value

    def __contains__(self, value) -> bool:
        return value in self._data

    def to_llm_string(self) -> str:
        return f"list[len={len(self._data)},metadata={self._metadata!r}]"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f"MetaListDevice(data={self._data}, metadata={self._metadata})"


class InputsListDevice(MetaListDevice):
    """伪列表用户输入设备
    挂载到 $MEM.inputs，模拟一个不断增长的输入列表。
    - 读取 $MEM.inputs 时返回列表元数据
    - 读取 $MEM.inputs.-1 时阻塞请求用户输入，追加到列表后返回
    - 读取其他索引时，像正常列表一样返回已有元素
    """

    def __init__(self, data=None, metadata=None):
        super().__init__(data=data, metadata=metadata)
        self._pending_input = False
        self._core = None  # 调试用：Core 引用（/save 检查点命令需要），由镜像加载器注入
        self._encoding_warned = False

    def attach_core(self, core) -> None:
        """调试用：接收 Core 引用（检查点 /save 调试命令使用）"""
        self._core = core

    def _parse_index(self, index):
        if isinstance(index, str):
            try:
                return int(index)
            except ValueError:
                from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
                raise MemoryTypeError(f"InputsListDevice 索引必须是整数，got {index!r}")
        return index

    def __getitem__(self, index):
        index = self._parse_index(index)
        if index == -1:
            # 伪列表语义：读最后一个元素 = 请求新用户输入
            self._pending_input = True
            try:
                user_input = self._prompt_operator()
            except EOFError:
                user_input = ""
            finally:
                self._pending_input = False
                self._data.append(user_input)
            return user_input
        # 正常列表访问
        return super().__getitem__(index)

    def _prompt_operator(self) -> str:
        """调试用：读取操作者输入；行首 /save 触发单对话检查点保存并继续等待真实输入。

        对话感知不到 /save：命令只在设备层消费，返回给对话的永远是真实输入。
        显示等待输入的对话身份提示（避免黑屏裸等）；终端非 UTF-8 时警告一次
        （中文输入乱码通常源于终端编码）。
        """
        self._warn_encoding_once()
        while True:
            try:
                raw = input(self._prompt_text())
            except UnicodeDecodeError:
                print("[输入] 解码失败：请确认终端使用 UTF-8 编码", flush=True)
                continue
            if raw.startswith("/save"):
                self._save_checkpoint(raw)
                continue
            return raw

    def _prompt_text(self) -> str:
        """调试用：输入提示，标明正在等待哪个对话输入（及 /save 调试命令）"""
        who = "对话"
        if self._core is not None:
            cid = self._core._active_cid
            if cid is not None:
                convs = getattr(self._core, "_conv_by_cid", {})
                conv = convs.get(cid)
                who = f"{conv.identity}（cid={cid}）" if conv is not None else f"cid={cid}"
        return f"[{who} 等待输入，/save 保存检查点] "

    def _warn_encoding_once(self) -> None:
        """调试用：终端非 UTF-8 时警告一次（中文输入/显示乱码的常见原因）"""
        if self._encoding_warned:
            return
        self._encoding_warned = True
        enc = getattr(sys.stdout, "encoding", None) or ""
        if "utf" not in enc.lower():
            print(f"[输入] 警告：终端编码 {enc!r} 不是 UTF-8，中文输入可能乱码", flush=True)

    def _save_checkpoint(self, cmd: str) -> None:
        """调试用：保存当前活跃对话的检查点。缺省路径由 Core.checkpoint_path 决定；/save <path> 可指定。"""
        if self._core is None:
            print("[检查点] 设备未接入 Core，无法保存", flush=True)
            return
        cid = self._core._active_cid
        if cid is None:
            print("[检查点] 无活跃对话，无法保存", flush=True)
            return
        rest = cmd[len("/save"):].strip()
        if rest:
            base = getattr(self._core, "image_dir", None) or "."
            path = rest if os.path.isabs(rest) else os.path.join(base, rest)
        else:
            path = self._core.checkpoint_path(cid)
        try:
            self._core.save_conversation(cid, path)
        except Exception as e:
            print(f"[检查点] 保存失败: {e}", flush=True)
            return
        print(f"[检查点] 对话 {cid} 已保存到 {path}", flush=True)

    def resolve_path(self, path: list) -> Any:
        if len(path) == 1:
            return self.__getitem__(path[0])
        from .exceptions import VMMemoryError
        raise VMMemoryError(f"InputsListDevice 不支持多级子路径")

    def __setitem__(self, index, value):
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        raise MemoryTypeError("InputsListDevice 是只读的，不允许写入")

    def set_value(self, value):
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        raise MemoryTypeError("InputsListDevice 是只读的，不允许写入")

    def to_llm_string(self) -> str:
        if self._pending_input:
            return "[等待用户输入...]"
        if not self._data:
            return f"inputs[list_len=0,metadata={self._metadata!r}]"
        recent = self._data[-3:]
        return f"inputs[list_len={len(self._data)}, recent={recent!r},metadata={self._metadata!r}]"


class OutputsListDevice(MetaListDevice):
    """伪列表用户输出设备
    挂载到 $MEM.outputs，模拟一个输出列表。
    - 读取 $MEM.outputs 时返回列表元数据
    - 写入 $MEM.outputs.-1 时以追加方式写入，并打印到屏幕
    - 读取其他索引时，像正常列表一样返回已有元素
    - 不可通过索引修改已有元素，只支持追加（通过 -1 写入）
    """

    def _parse_index(self, index):
        if isinstance(index, str):
            try:
                return int(index)
            except ValueError:
                from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
                raise MemoryTypeError(f"OutputsListDevice 索引必须是整数，got {index!r}")
        return index

    def __getitem__(self, index):
        index = self._parse_index(index)
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        index = self._parse_index(index)
        if index == -1:
            self.append(value)
        else:
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError("OutputsListDevice 只支持追加写入（索引 -1），不允许修改已有元素")

    def append(self, value):
        if not isinstance(value, str):
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"OutputsListDevice 只接受 str，got {type(value).__name__}")
        self._data.append(value)
        print(value, flush=True)  # 写入即打印到屏幕（文档承诺的"并打印到屏幕"）

    def set_value(self, value):
        """支持直接写入单个字符串（追加）或列表（替换）"""
        if isinstance(value, list):
            self._data = [str(v) for v in value]
        elif isinstance(value, str):
            self.append(value)
        else:
            from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
            raise MemoryTypeError(f"OutputsListDevice 只接受 str 或 list，got {type(value).__name__}")

    def to_llm_string(self) -> str:
        return f"outputs[list_len={len(self._data)}, items={self._data!r},metadata={self._metadata!r}]"

    def resolve_path(self, path: list) -> Any:
        if len(path) == 1:
            return self.__getitem__(path[0])
        from .exceptions import VMMemoryError
        raise VMMemoryError(f"OutputsListDevice 不支持多级子路径")


class MetaDictDevice(MemoryDevice):
    """假装是 MetaDict 的设备"""

    def __init__(self, data=None, metadata=None):
        self._data: dict = data if data is not None else {}
        self._metadata: Optional[str] = metadata

    def pretend_as_type(self) -> str:
        return "MetaDict"

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_metadata(self) -> Optional[str]:
        return self._metadata

    def set_metadata(self, metadata: str) -> None:
        self._metadata = metadata

    def set_value(self, value: dict) -> None:
        """整体替换内部数据（用于 Memory.set_by_path 的设备写入）"""
        from .exceptions import MemoryTypeError, MemoryIndexOutOfRangeError
        if not isinstance(value, dict):
            raise MemoryTypeError(f"MetaDictDevice 只接受 dict，got {type(value).__name__}")
        self._data = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def to_llm_string(self) -> str:
        return f"dict[keys={list(self._data.keys())},metadata={self._metadata!r}]"

    def __repr__(self) -> str:
        return f"MetaDictDevice(data={self._data}, metadata={self._metadata})"

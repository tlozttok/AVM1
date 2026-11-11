"""AVM 类型系统：带元数据的集合类"""
import json
from typing import Dict, Optional

from .capabilities import Capability, caps_from_dict, caps_to_dict, default_kernel_caps

CHECKPOINT_VERSION = 2


def _tool_call_to_dict(tc) -> dict:
    """工具调用无损转 dict（兼容 ToolCall 对象与已转好的 dict，检查点序列化用）"""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return {
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {"name": fn.get("name"), "arguments": fn.get("arguments")},
        }
    fn = tc.function
    return {
        "id": tc.id,
        "type": getattr(tc, "type", "function"),
        "function": {"name": fn.name, "arguments": fn.arguments},
    }


def message_to_dict(msg: "Message") -> dict:
    """消息的无损 JSON 形状（检查点、重启快照、回合日志共用；字段顺序固定）。

    保留每条消息的全部字段：content 空串照写、tool_calls 原样、reasoning_content
    区分 None 与空串、tool_call_id 原样。恢复后发给 API 的前缀逐字节一致。
    """
    d = {"role": msg.role, "content": msg.content}
    if isinstance(msg, ToolMessage):
        d["tool_call_id"] = msg.tool_call_id
    if isinstance(msg, AssistantMessage):
        if msg.tool_calls:
            d["tool_calls"] = [_tool_call_to_dict(tc) for tc in msg.tool_calls]
        if msg.reasoning_content is not None:
            d["reasoning_content"] = msg.reasoning_content
    return d


def message_from_dict(d: dict) -> "Message":
    """message_to_dict 的逆操作（未知角色按普通 Message 保留）。"""
    role = d.get("role")
    content = d.get("content", "")
    if role == "assistant":
        return AssistantMessage(
            content=content,
            tool_calls=d.get("tool_calls"),
            reasoning_content=d.get("reasoning_content"),
        )
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=d.get("tool_call_id", ""))
    if role == "user":
        return UserMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return Message(content=content, role=role or "")


def batch_to_dict(batch: 'UserMessageBatch') -> dict:
    """待发批次的 JSON 形状（快照、回合日志共用）。"""
    return {
        "tool_responses": [
            {"content": r.content, "tool_call_id": r.tool_call_id}
            for r in batch.tool_responses
        ],
        "user_contents": list(batch.user_contents),
    }


def batch_from_dict(data) -> 'UserMessageBatch':
    """batch_to_dict 的逆操作。"""
    batch = UserMessageBatch()
    data = data or {}
    for item in data.get("tool_responses") or []:
        batch.add_tool_response(item.get("content", ""), item.get("tool_call_id", ""))
    for content in data.get("user_contents") or []:
        batch.add_user_content(content)
    return batch


class StrNode(str):
    """字符串叶子 + 控制数据（ctrl）。

    字符串只有控制数据，没有描述数据（描述属于容器）：`ctrl` 可存 `type` 等
    Core/内核态需要的结构化信息；`meta` 对字符串不适用。
    作为 str 的子类，比较、JSON 序列化、`isinstance(x, str)` 都保持原语义。
    """

    __slots__ = ("_ctrl",)

    def __new__(cls, value: str = "", ctrl=None):
        obj = super().__new__(cls, value)
        obj._ctrl = dict(ctrl) if ctrl else None
        return obj

    def get_ctrl(self):
        return self._ctrl

    def set_ctrl(self, ctrl) -> None:
        self._ctrl = ctrl

    def __reduce__(self):
        # pickle / copy 用：保留控制数据（否则 str 子类的属性会在复制中丢失）
        return (StrNode, (str(self), self._ctrl))

    def __copy__(self):
        return StrNode(str(self), dict(self._ctrl) if self._ctrl else None)

    def __deepcopy__(self, memo):
        import copy as _copy
        return StrNode(str(self), _copy.deepcopy(self._ctrl, memo))


class MetaList:
    """带元数据的列表，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None, ctrl=None):
        self._data: list = data if data is not None else []
        self._metadata: Optional[str] = metadata
        self._ctrl: Optional[Dict[str, str]] = ctrl

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def insert(self, index, value):
        self._data.insert(index, value)

    def append(self, value):
        self._data.append(value)

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def get_ctrl(self):
        """元数据二：给 Core/内核态的结构化控制信息（如 {"type": "para"}）"""
        return self._ctrl

    def set_ctrl(self, ctrl: Dict[str, str]):
        self._ctrl = ctrl

    def __contains__(self, key):
        return key in self._data

    def _elem_kind(self) -> str:
        """元素类型自描述：空列表默认 str（追加的默认类型），混合列表给出各类型。"""
        kinds = set()
        for value in self._data:
            if isinstance(value, (MetaDict, dict)):
                kinds.add("dict")
            elif isinstance(value, (MetaList, list)):
                kinds.add("list")
            elif isinstance(value, LinkNode):
                kinds.add("link")
            else:
                kinds.add("str")
        if not kinds:
            return "str"
        if len(kinds) == 1:
            return kinds.pop()
        return "mixed(" + ",".join(sorted(kinds)) + ")"

    def to_llm_string(self):
        """返回给 LLM 的字符串表示（含元素类型，便于判断追加是否同类）"""
        return f"list[len={len(self._data)},elem={self._elem_kind()},metadata={self._metadata!r}]"

    def to_list(self):
        """递归转换为普通 list（用于 API 参数等需要纯 JSON 结构的场景）"""
        result = []
        for v in self._data:
            if isinstance(v, MetaDict):
                result.append(v.to_dict())
            elif isinstance(v, MetaList):
                result.append(v.to_list())
            elif isinstance(v, LinkNode):
                result.append(v.to_placeholder())
            else:
                result.append(v)
        return result

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        if isinstance(other, MetaList):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented

    def __repr__(self):
        return f"MetaList(data={self._data}, metadata={self._metadata}, ctrl={self._ctrl})"


class MetaDict:
    """带元数据的字典，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None, ctrl=None):
        self._data: dict = data if data is not None else {}
        self._metadata: Optional[str] = metadata
        self._ctrl: Optional[Dict[str, str]] = ctrl

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)

    def copy(self):
        return MetaDict(data=self._data.copy(), metadata=self._metadata, ctrl=self._ctrl)

    def keys(self):
        return self._data.keys()
    
    def get(self, key, default=None):
        return self._data.get(key, default)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def get_ctrl(self):
        """元数据二：给 Core/内核态的结构化控制信息（如 {"type": "para"}）"""
        return self._ctrl

    def set_ctrl(self, ctrl: Dict[str, str]):
        self._ctrl = ctrl

    def __contains__(self, key):
        return key in self._data

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        return f"dict[keys={list(self._data.keys())},metadata={self._metadata!r}]"

    def to_dict(self):
        """递归转换为普通 dict（用于 API 参数等需要纯 JSON 结构的场景）"""
        result = {}
        for k, v in self._data.items():
            if isinstance(v, MetaDict):
                result[k] = v.to_dict()
            elif isinstance(v, MetaList):
                result[k] = v.to_list()
            elif isinstance(v, LinkNode):
                result[k] = v.to_placeholder()
            else:
                result[k] = v
        return result

    def __eq__(self, other):
        if isinstance(other, MetaDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def __repr__(self):
        return f"MetaDict(data={self._data}, metadata={self._metadata}, ctrl={self._ctrl})"


class LinkNode:
    """内存树链接节点：寻址遇到它时进入 target 指向的地址。

    target 以 `$` 开头 → 内存树地址（`$MEM.a.b`）；否则 → 设备唯一标识符。
    链接的读写与元数据编辑都进入目标；target 只在创建时写入，运行期不改写。
    跟随过程由 Memory 记访问集合，成环抛 MemoryCircularReferenceError。
    """

    def __init__(self, target: str, metadata=None, ctrl=None):
        self.target = target
        self._metadata: Optional[str] = metadata
        self._ctrl: Optional[Dict[str, str]] = ctrl

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def get_ctrl(self):
        """元数据二：给 Core/内核态的结构化控制信息"""
        return self._ctrl

    def set_ctrl(self, ctrl: Dict[str, str]):
        self._ctrl = ctrl

    def to_llm_string(self):
        """链接自身的 LLM 表示（寻址会进入目标，这个表示只在直接展示节点时用到）"""
        return f"link[target={self.target},metadata={self._metadata!r}]"

    def to_placeholder(self):
        """纯 JSON 结构里的占位（不跟随目标，避免只读转换产生寻址副作用）"""
        return {"$link": self.target}

    def __eq__(self, other):
        if isinstance(other, LinkNode):
            return self.target == other.target
        return NotImplemented

    def __repr__(self):
        return f"LinkNode(target={self.target!r}, metadata={self._metadata!r}, ctrl={self._ctrl!r})"

"""AVM 消息类型定义"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from openai.types.chat import ChatCompletionMessageToolCallUnion

type ToolCall=ChatCompletionMessageToolCallUnion

class Role(str, Enum):
    """消息角色"""
    SETTING = "system"
    CMD = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """消息基类"""
    content: str
    role: str = ""



@dataclass
class SystemMessage(Message):
    """系统消息"""
    role: str = "system"


@dataclass
class UserMessage(Message):
    """用户消息"""
    role: str = "user"


@dataclass
class AssistantMessage(Message):
    """助手消息"""
    role: str = "assistant"
    tool_calls: Optional[List[ToolCall]] = None
    reasoning_content: Optional[str] = None


@dataclass
class ToolMessage(Message):
    """工具响应消息"""
    role: str = "tool"
    tool_call_id: str = ""


def message_to_api_dict(msg: Message) -> dict:
    d: dict = {"role": msg.role, "content": msg.content}
    if isinstance(msg, AssistantMessage):
        # 用户决定：回传所有推理内容（越来越多的模型回传全部思维链）。
        # 总是带 reasoning_content 字段：有值带值，无值带空字符串。
        d["reasoning_content"] = msg.reasoning_content or ""
        if msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
    if isinstance(msg, ToolMessage):
        d["tool_call_id"] = msg.tool_call_id
    return d



class Conversation:
    """
    对话历史封装
    负责消息的验证、合并和转换，内嵌用户消息批次
    """
    messages: List[Message]
    user_batch: 'UserMessageBatch'
    cid: int
    is_sub:bool
    parent: Optional['Conversation'] = None
    is_root: bool = False
    metadata: Dict[str, str]
    name: Optional[str] = None
    para_ref: Optional[str] = None  # 本对话的 para 引用（缺省用 Core.para_ref）
    is_python: bool = False  # 程序类型：True = Python 对话（system_ref 指向 type="python" 节点），False = LLM 对话
    caps: 'Capability' = None  # 对话能力（模式/组/根表/工具集）；None 时用默认内核态全能力
    attrs: Dict[str, str] = None  # 自由元数据（镜像层权限策略用；不写内存树）

    def __init__(self, messages: List[Message] = None, cid: int = 0, is_sub:bool=False, parent: Optional['Conversation'] = None, is_root: bool = False, metadata: Dict[str, str] = None, name: Optional[str] = None, caps: Optional['Capability'] = None, attrs: Dict[str, str] = None):
        self.messages = messages or []
        self.user_batch = UserMessageBatch()
        self.cid = cid
        self.is_sub = is_sub
        self.parent = parent
        self.is_root = is_root
        self.metadata = metadata or {}
        self.name = name
        self.caps = caps if caps is not None else default_kernel_caps()
        self.attrs = dict(attrs or {})
        self.validate(require_last_assistant=False)

    @property
    def identity(self) -> str:
        """对话身份：有名字用名字；无名字（亚对话）用父身份 + 自身 cid。"""
        if self.name:
            return self.name
        if self.parent is not None:
            parent_name = self.parent.name or str(self.parent.cid)
            return f"{parent_name}#{self.cid}"
        return str(self.cid)
    
    

    def validate(self, require_last_assistant: bool = True) -> None:
        """
        验证对话格式
        :param require_last_assistant: 是否要求最后一条消息是 assistant
        :raises ValueError: 验证失败时抛出
        """
        if not self.messages:
            if require_last_assistant:
                raise ValueError("对话历史不能为空")
            return

        if require_last_assistant:
            last_msg = self.messages[-1]
            if last_msg.role != Role.ASSISTANT.value:
                raise ValueError(f"最后一条消息必须是 assistant，得到：{last_msg.role}")

    def merge_system_messages(self) -> List[SystemMessage]:
        """
        合并前 n 个连续的 system 消息为一个
        :returns: 合并后的消息列表
        """
        if not self.messages:
            return []

        result = []
        system_contents = []
        i = 0

        # 合并连续的 system 消息
        while i < len(self.messages) and self.messages[i].role == Role.SETTING.value:
            system_contents.append(self.messages[i].content)
            i += 1

        if system_contents:
            result.append(Message(
                role=Role.SETTING.value,
                content="\n\n".join(system_contents)
            ))

        # 添加剩余消息
        result.extend(self.messages[i:])
        return result

    def to_api_messages(self) -> List[dict]:
        """
        转换为 OpenAI API 格式的消息列表
        自动合并 system 消息并验证格式
        """
        self.validate(require_last_assistant=False)
        merged = self.merge_system_messages()
        result = [message_to_api_dict(msg) for msg in merged]
        return result

    def append_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append(UserMessage(role=Role.CMD.value, content=content))

    def append_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        """添加助手消息"""
        self.messages.append(AssistantMessage(
            role=Role.ASSISTANT.value,
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        ))

    def append_tool_message(self, content: str, tool_call_id: str) -> None:
        """添加工具响应消息"""
        self.messages.append(ToolMessage(
            role=Role.TOOL.value,
            content=content,
            tool_call_id=tool_call_id
        ))

    def get_last_messages(self, count: int = 1) -> List[Message]:
        """获取最后 n 条消息"""
        return self.messages[-count:] if self.messages else []

    def to_checkpoint(self) -> dict:
        """单对话检查点：无损、确定性序列化（调试用）。

        保留每条消息的全部字段（content 空串、tool_calls 原样、reasoning_content、
        tool_call_id），字段顺序固定，保证恢复后 to_api_messages() 与保存时逐字节一致
        （前缀缓存命中的前提）。
        """
        return {
            "version": CHECKPOINT_VERSION,
            "name": self.name,
            "para_ref": self.para_ref,
            "is_python": self.is_python,
            "is_sub": self.is_sub,
            "is_root": self.is_root,
            "metadata": dict(self.metadata or {}),
            "caps": caps_to_dict(self.caps),
            "attrs": dict(self.attrs or {}),
            "messages": [message_to_dict(m) for m in self.messages],
        }

    def to_snapshot(self) -> dict:
        """重启快照里的单对话形状：检查点 + cid/parent + 待发批次。

        与 to_checkpoint 的差别（计划 A5）：
        - 带 `cid` 与 `parent_cid`：重启恢复保留 cid，父子引用按 cid 重建；
        - 带 `batch`：待发批次是"结束态"的一部分（投递后目标还没跑的输入必须活下来）；
        - 不拒绝 Python 对话：Python 对话的跨轮状态按约定落在 `$MEM`，命名空间不序列化。
        """
        doc = self.to_checkpoint()
        doc["cid"] = self.cid
        doc["parent_cid"] = self.parent.cid if self.parent is not None else None
        doc["batch"] = batch_to_dict(self.user_batch)
        return doc

    @classmethod
    def from_checkpoint(cls, cp: dict) -> 'Conversation':
        """从检查点恢复对话（to_checkpoint 的逆操作；cid 由 Core 恢复时重新分配）。"""
        version = cp.get("version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"不支持的检查点版本 {version}（当前支持 {CHECKPOINT_VERSION}）")
        msgs = [message_from_dict(d) for d in cp.get("messages", [])]
        conv = cls(
            messages=msgs,
            is_sub=bool(cp.get("is_sub", False)),
            is_root=bool(cp.get("is_root", False)),
            metadata=cp.get("metadata") or {},
            name=cp.get("name"),
        )
        conv.para_ref = cp.get("para_ref")
        conv.is_python = bool(cp.get("is_python", False))
        caps = cp.get("caps")
        if caps is None:
            raise ValueError("检查点缺少 caps：旧版本检查点不再支持，请重新保存")
        conv.caps = caps_from_dict(caps)
        conv.attrs = dict(cp.get("attrs") or {})
        return conv

    @classmethod
    def from_snapshot(cls, doc: dict) -> 'Conversation':
        """从重启快照的单对话形状恢复（to_snapshot 的逆操作；cid/parent 由调用方接回）。

        parent 只按 parent_cid 记录，由 snapshot.restore 在全部对话建好之后统一接；
        这里 parent 保持 None。
        """
        conv = cls.from_checkpoint(doc)
        conv.cid = doc.get("cid", 0)
        batch = doc.get("batch") or {}
        conv.user_batch = batch_from_dict(batch)
        return conv

    @classmethod
    def from_any_list(cls, items: list) -> 'Conversation':
        msgs = []
        for item in items:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                role, content = item
                msgs.append(cls._msg_for(role, content, {}))
            elif isinstance(item, dict):
                msgs.append(cls._msg_for(
                    item.get("role", ""), item.get("content", ""), item))
        return cls(messages=msgs)

    @staticmethod
    def _msg_for(role: str, content: str, extra: dict) -> Message:
        if role == "system":
            return SystemMessage(content=content)
        if role == "user":
            return UserMessage(content=content)
        if role == "assistant":
            return AssistantMessage(content=content, tool_calls=extra.get("tool_calls"))
        if role == "tool":
            return ToolMessage(content=content, tool_call_id=extra.get("tool_call_id", ""))
        return SystemMessage(content=content)


@dataclass
class UserMessageBatch:
    """
    用户消息批量输入
    支持混合工具响应和普通用户内容
    """
    tool_responses: List[ToolMessage] = field(default_factory=list)
    user_contents: List[str] = field(default_factory=list)

    def add_tool_response(self, content: str, tool_call_id: str) -> None:
        """添加工具响应。

        内容与配对 id 都必须是字符串：非字符串以 TypeError 拒绝（不静默 str()），
        由指令边界翻成 Error 工具响应，避免活对象流进对话历史后在下一次 API 调用炸掉。
        """
        if not isinstance(content, str):
            raise TypeError(f"工具响应内容必须是字符串，得到 {type(content).__name__}")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise TypeError(f"工具响应的 tool_call_id 必须是非空字符串，得到 {tool_call_id!r}")
        self.tool_responses.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    def add_user_content(self, content: str) -> None:
        """添加用户内容"""
        self.user_contents.append(str(content))

    def clear(self) -> None:
        """清空所有内容"""
        self.tool_responses.clear()
        self.user_contents.clear()

    @classmethod
    def from_any_list(cls, items: List[Union[Tuple[str, str], str]]) -> 'UserMessageBatch':
        """从任意输入列表创建 UserMessageBatch"""
        batch = cls()
        for item in items:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                batch.add_tool_response(content=str(item[0]), tool_call_id=str(item[1]))
            else:
                batch.add_user_content(str(item))
        return batch

    def to_tool_messages(self) -> List[dict]:
        """转换为工具消息列表。"""
        out = []
        for resp in self.tool_responses:
            out.append({"role": "tool", "content": resp.content, "tool_call_id": resp.tool_call_id})
        return out

    def get_user_content(self) -> str:
        """获取合并后的用户内容"""
        return "\n\n".join(self.user_contents)



__all__ = [
    'MetaList',
    'MetaDict',
    'LinkNode',
    'Role',
    'ToolCall',
    'Message',
    'SystemMessage',
    'UserMessage',
    'AssistantMessage',
    'ToolMessage',
    'message_to_api_dict',
    'message_to_dict',
    'message_from_dict',
    'batch_to_dict',
    'batch_from_dict',
    'Conversation',
    'UserMessageBatch',
]

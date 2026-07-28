"""AVM 类型系统：带元数据的集合类"""


class MetaList:
    """带元数据的列表，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None):
        self._data: list = data if data is not None else []
        self._metadata: Optional[str] = metadata

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def append(self, value):
        self._data.append(value)

    def get_metadata(self):
        return self._metadata

    def set_metadata(self, metadata: str):
        self._metadata = metadata

    def __contains__(self, key):
        return key in self._data

    def to_llm_string(self):
        """返回给 LLM 的字符串表示"""
        return f"list[len={len(self._data)},metadata={self._metadata!r}]"

    def to_list(self):
        """递归转换为普通 list（用于 API 参数等需要纯 JSON 结构的场景）"""
        result = []
        for v in self._data:
            if isinstance(v, MetaDict):
                result.append(v.to_dict())
            elif isinstance(v, MetaList):
                result.append(v.to_list())
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
        return f"MetaList(data={self._data}, metadata={self._metadata})"


class MetaDict:
    """带元数据的字典，对 LLM 可展示自定义描述字符串"""
    def __init__(self, data=None, metadata=None):
        self._data: dict = data if data is not None else {}
        self._metadata: Optional[str] = metadata

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def setdefault(self, key, default=None):
        return self._data.setdefault(key, default)

    def copy(self):
        return MetaDict(data=self._data.copy(), metadata=self._metadata)

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
        return f"MetaDict(data={self._data}, metadata={self._metadata})"

"""AVM 消息类型定义"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
from openai.types.chat import ChatCompletionMessageToolCallUnion
import logging

logger = logging.getLogger(__name__)

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


@dataclass
class ToolMessage(Message):
    """工具响应消息"""
    role: str = "tool"
    tool_call_id: str = ""


def message_to_api_dict(msg: Message) -> dict:
    d: dict = {"role": msg.role, "content": msg.content}
    if isinstance(msg, AssistantMessage) and msg.tool_calls:
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
    service_desc: Optional[Dict[str, str]] = None  # {"name","what","needs","returns"}
    
    def __init__(self, messages: List[Message] = None, cid: int = 0, is_sub:bool=False, parent: Optional['Conversation'] = None, is_root: bool = False, metadata: Dict[str, str] = None, service_desc: Dict[str, str] = None):
        self.messages = messages or []
        self.user_batch = UserMessageBatch()
        self.cid = cid
        self.is_sub = is_sub
        self.parent = parent
        self.is_root = is_root
        self.metadata = metadata or {}
        self.service_desc = service_desc
        self.validate(require_last_assistant=False)
    
    

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
        logger.debug("[Conversation.to_api_messages] count=%d", len(result))
        return result

    def append_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append(UserMessage(role=Role.CMD.value, content=content))

    def append_assistant_message(self, content: str, tool_calls: Optional[List[dict]] = None) -> None:
        """添加助手消息"""
        self.messages.append(AssistantMessage(role=Role.ASSISTANT.value, content=content, tool_calls=tool_calls))

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
        """添加工具响应"""
        logger.debug("[UserMessageBatch.add_tool_response] id=%s content=%r", tool_call_id, content)
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
        """转换为工具消息列表"""
        return [
            {"role": "tool", "content": resp.content, "tool_call_id": resp.tool_call_id}
            for resp in self.tool_responses
        ]

    def get_user_content(self) -> str:
        """获取合并后的用户内容"""
        return "\n\n".join(self.user_contents)



__all__ = [
    'MetaList',
    'MetaDict',
    'Role',
    'ToolCall',
    'Message',
    'SystemMessage',
    'UserMessage',
    'AssistantMessage',
    'ToolMessage',
    'message_to_api_dict',
    'Conversation',
    'UserMessageBatch',
]

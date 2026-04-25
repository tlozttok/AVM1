"""AVM 消息类型定义"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    type: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolCallResponse:
    """工具调用响应"""
    content: str
    tool_call_id: str


@dataclass
class Message:
    """消息基类"""
    content: str
    role: str = ""

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


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

    def to_dict(self) -> dict:
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class ToolMessage(Message):
    """工具响应消息"""
    role: str = "tool"
    tool_call_id: str = ""

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "tool_call_id": self.tool_call_id}


# 类型别名
MessageDict = Dict[str, Any]
MessageTuple = Tuple[str, str]  # (role, content)
MessageInput = Union[MessageDict, MessageTuple, Message]


@dataclass
class ConversationMessage:
    """对话中的单条消息（内部表示）"""
    role: str
    content: str
    tool_calls: Optional[List[dict]] = None
    tool_call_id: str = ""

    @classmethod
    def from_any(cls, msg: MessageInput) -> 'ConversationMessage':
        """从任意输入类型创建 ConversationMessage"""
        if isinstance(msg, Message):
            return cls(role=msg.role, content=msg.content)
        elif isinstance(msg, dict):
            return cls(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id", "")
            )
        elif isinstance(msg, (list, tuple)):
            return cls(role=str(msg[0]), content=str(msg[1]))
        else:
            raise TypeError(f"不支持的消息类型：{type(msg)}")

    def to_dict(self) -> dict:
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class Conversation:
    """
    对话历史封装
    负责消息的验证、合并和转换
    """
    messages: List[ConversationMessage] = field(default_factory=list)

    @classmethod
    def from_any_list(cls, messages: List[MessageInput]) -> 'Conversation':
        """从任意输入列表创建 Conversation"""
        conv = cls()
        for msg in messages:
            conv.messages.append(ConversationMessage.from_any(msg))
        return conv

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

    def merge_system_messages(self) -> List[ConversationMessage]:
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
        while i < len(self.messages) and self.messages[i].role == Role.SYSTEM.value:
            system_contents.append(self.messages[i].content)
            i += 1

        if system_contents:
            result.append(ConversationMessage(
                role=Role.SYSTEM.value,
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
        result = [msg.to_dict() for msg in merged]
        logger.debug("[Conversation.to_api_messages] count=%d", len(result))
        return result

    def append_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append(ConversationMessage(role=Role.USER.value, content=content))

    def append_assistant_message(self, content: str, tool_calls: Optional[List[dict]] = None) -> None:
        """添加助手消息"""
        self.messages.append(ConversationMessage(role=Role.ASSISTANT.value, content=content, tool_calls=tool_calls))

    def append_tool_message(self, content: str, tool_call_id: str) -> None:
        """添加工具响应消息"""
        self.messages.append(ConversationMessage(
            role=Role.TOOL.value,
            content=content,
            tool_call_id=tool_call_id
        ))

    def get_last_messages(self, count: int = 1) -> List[ConversationMessage]:
        """获取最后 n 条消息"""
        return self.messages[-count:] if self.messages else []


@dataclass
class UserMessageBatch:
    """
    用户消息批量输入
    支持混合工具响应和普通用户内容
    """
    tool_responses: List[ToolCallResponse] = field(default_factory=list)
    user_contents: List[str] = field(default_factory=list)

    def add_tool_response(self, content: str, tool_call_id: str) -> None:
        """添加工具响应"""
        logger.debug("[UserMessageBatch.add_tool_response] id=%s content=%r", tool_call_id, content)
        self.tool_responses.append(ToolCallResponse(content=content, tool_call_id=tool_call_id))

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


# 类型别名
MessageHistory = List[MessageDict]

__all__ = [
    'Role',
    'ToolCall',
    'ToolCallResponse',
    'Message',
    'SystemMessage',
    'UserMessage',
    'AssistantMessage',
    'ToolMessage',
    'ConversationMessage',
    'Conversation',
    'UserMessageBatch',
    'MessageDict',
    'MessageTuple',
    'MessageInput',
    'MessageHistory',
]

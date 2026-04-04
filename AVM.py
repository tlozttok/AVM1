

from collections import defaultdict

from openai import OpenAI

class Msg:
    role=""
    def __init__(self,content):
        self.content=content

    def to_dict(self):
        return {"role":self.role,"content":self.content}

class UserMsg:
    role="user"
    def __init__(self,content):
        super().__init__(content)

class SystemMsg:
    role="system"
    def __init__(self,content):
        super().__init__(content)

class AssistantMsg:
    role="assistant"
    def __init__(self,content):
        super().__init__(content)


class LMU:

    client: OpenAI

    tools= {
        "type": "function",
        "function": {
            "name": "command",
            "description": "执行命令。如果提示词中没有命令格式，不要使用该工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "命令。如果提示词中没有命令格式，请忽略此字段",
                    }
                },
                "required": ["command"]
            }
        }
    },

    def exec_crt(self,system_prompt,user_prompt,para):
        """处理字符串输入的 create 模式
        system_prompt: 字符串，系统提示词
        user_prompt: 字符串，用户提示词
        para: 参数字典
        """
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=[self.tools] if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        result = message.content
        return_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    return_calls.append(f"exec_cmd {tool_call.function.arguments}")

        return result, return_calls

    def exec(self,last_msgs,user_msg,para):
        messages=[]
        # last_msgs 格式：[system...][user/assistant/tool...], 最后一个必须是 assistant
        # user_msg 是一个数组，包含 (content, tool_call_id) 元组和/或纯 content 字符串

        # 检查并合并连续的 system 消息
        if last_msgs:
            # 检查最后一个消息是否为 assistant
            if last_msgs[-1].get("role") != "assistant":
                raise ValueError("last_msgs 的最后一个消息必须是 assistant")

            # 合并前 n 个连续的 system 消息为一个
            system_contents = []
            i = 0
            while i < len(last_msgs) and last_msgs[i].get("role") == "system":
                system_contents.append(last_msgs[i].get("content", ""))
                i += 1

            if system_contents:
                messages.append({"role": "system", "content": "\n\n".join(system_contents)})

            # 剩余消息可以是 user/assistant/tool 混合，但 tool 必须跟在 assistant 后面
            rest = last_msgs[i:]
            messages.extend(rest)

        # 处理 user_msg 数组，将 tool 角色消息插入到 assistant 消息后，user 消息前
        tool_messages = []
        user_contents = []
        for item in user_msg:
            if isinstance(item, tuple) and len(item) == 2:
                content, tool_call_id = item
                tool_messages.append({"role": "tool", "content": content, "tool_call_id": tool_call_id})
            else:
                user_contents.append(str(item))

        # 将 tool 消息添加到 messages（在 assistant 消息后，user 消息前）
        messages.extend(tool_messages)

        # 添加最终的 user 消息
        if user_contents:
            messages.append({"role": "user", "content": "\n\n".join(user_contents)})

        response = self.client.chat.completions.create(
            model=para.get("model", "gpt-4"),
            messages=messages,
            tools=[self.tools] if para.get("use_tool", False) else None,
        )

        choice = response.choices[0]
        message = choice.message

        # 提取返回内容
        result = message.content
        return_calls = []

        # 如果有工具调用，解析并添加到 return_calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "command":
                    return_calls.append(f"exec_cmd {tool_call.function.arguments}")

        # 将本次对话添加到 last_msgs 用于后续调用
        last_msgs.append({"role": "user", "content": user_msg})
        last_msgs.append({"role": "assistant", "content": result or ""})

        return result, return_calls

    @staticmethod
    def build_last_msg(last_msgs,result,return_calls):
        # 构建用于后续调用的消息历史
        return last_msgs

MEM=defaultdict(list)
SYSTEM="system"
USER="user"
ASSISTANT="assistant"

class Core:
    command_stack=[]
    last_msg_reg=[]
    usr_tool_reg=[]

    def run(self):
        while self.command_stack:
            command=self.command_stack[-1]
            command=command.split()
            # 无前缀表示字面值，$表示变量，&表示地址
            if command[0]=="exec": #exec $last_msg $user_msg $para a/w &return_key id
                last_msg=command[1]
                last_msg=self.unwrap(last_msg) #寄存
                user_msg=command[2]
                user_msg=self.unwrap(user_msg) #如果是来源于寄存器的，则应该清空对应寄存器
                para=command[3]
                para=self.unwrap(para)
                mode=command[4]
                return_key=command[5]
                call_id=command[6]
                result,return_calls=LMU.exec(last_msg,user_msg,para)
                if result:
                    self.command_stack.pop()
                    self.last_msg_reg.pop(last_msg)
                    self.usr_tool_reg.pop(user_msg)
                    if mode=="a":
                        MEM[return_key].append(result,call_id) #如果不是寄存，则忽略id
                    if mode=="w":
                        MEM[return_key]=result 
                if return_calls:
                    self.command_stack.extend(return_calls[::-1])
                continue
            if command[0]=="create": #create $system_prompt $user_prompt $para a/w &return_key
                system_prompt=command[1]
                system_prompt=self.unwrap(system_prompt) #寄存
                user_prompt=command[2]
                user_prompt=self.unwrap(user_prompt)
                para=command[3]
                para=self.unwrap(para)
                mode=command[4]
                return_key=command[5]
                result,return_calls=LMU.exec_crt(system_prompt,user_prompt,para)
                if result:
                    if mode=="a":
                        MEM[return_key].append(result) #寄存
                    if mode=="w":
                        MEM[return_key]=result #寄存
                    self.command_stack.pop()
                if return_calls:
                    self.last_msg_reg.append([(SYSTEM,system_prompt),(USER,user_prompt)])
                    self.usr_tool_reg.append([])
                    this_command_context_id=len(self.last_msg_reg)
                    this_command_user_reg_id=len(self.usr_tool_reg)
                    self.command_stack[-1]=f"exec $last_msg_reg.{this_command_context_id} $user_tool_reg.{this_command_user_reg_id} ${para} {mode} &{return_key}"
                    self.command_stack.extend(return_calls[::-1])
                continue

    def unwrap(self,value):
        value=[value[0],*value[1:].split(".")]
        if value[1]=="last_msg_reg":
            assert value[0]=="$"
            return self.last_msg_reg[int(value[2])]
        elif value[1]=="usr_tool_reg":
            assert value[0]=="$"
            return self.usr_tool_reg[int(value[2])]
        else: 
            return self._mem_unwrap(value)
        
    def _mem_unwrap(self,value):
        if value[0]=="$":
            temp=MEM
            for i in range(len(value)-1):
                temp=temp[value[i+1]]
            if isinstance(temp,str):
                if temp.startswith("$"):
                    temp_value=[temp[0],*temp[1:].split(".")]
                    temp=self._mem_unwrap(temp_value)
            if not isinstance(temp,str):
                if isinstance(temp,list):
                    return temp[0] #期望约定：列表中第一个元素是列表元数据
                if isinstance(temp,dict):
                    return f"dict.keys:{temp.keys()}"
        if value[0]=="&":
            temp=MEM
            for i in range(len(value)-1):
                temp=temp[value[i+1]]
            if isinstance(temp,str):
                return temp
            if not isinstance(temp,str):
                if isinstance(temp,list):
                    return temp[0]
                if isinstance(temp,dict):
                    return f"dict.keys:{temp.keys()}"
                                    
                
"""plm.ledger：状态化键值账本 PLM 服务器。

一个 AVM Python 对话的 model 参数指向这里注册的 "plm.ledger"。
它遵守 OpenAI 兼容 chat 服务器契约：输入 messages，输出 content + tool_calls。

system 消息 = JSON 配置：
    {"path": "$MEM.ledger.state", "initial": {}}

user 消息 = 单行命令：
    PUT <key> <value> | GET <key> | DEL <key> | LIST | RESET | SUM

需要持久化的命令先返回 memory_write 工具调用，Core 执行后把工具响应
作为下一条 tool 消息送回；本服务器再返回最终答复。
"""
import json

from avm.python_server import PLMServer


class LedgerPLM(PLMServer):
    def __init__(self):
        super().__init__()
        self._path = "$MEM.ledger.state"
        self._state = {}
        self._pending_reply = None
        self._call_seq = 0

    # ------------------------------------------------------------------
    # 三类消息的固定行为
    # ------------------------------------------------------------------
    def _execute_system_message(self, content):
        """system = 配置：账本持久化路径与初始状态。"""
        text = (content or "").strip()
        if not text:
            return
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            cfg = {"path": text}
        if isinstance(cfg, dict):
            path = cfg.get("path")
            if isinstance(path, str) and path.startswith("$"):
                self._path = path
            initial = cfg.get("initial")
            if isinstance(initial, dict):
                self._state = dict(initial)
            elif isinstance(initial, str):
                try:
                    parsed = json.loads(initial)
                except json.JSONDecodeError:
                    parsed = {}
                if isinstance(parsed, dict):
                    self._state = parsed
        elif isinstance(cfg, str):
            self._path = cfg

    def _execute_user_message(self, content):
        """user = 单条命令。"""
        text = (content or "").strip()
        if not text:
            return self.make_response("ERROR: 空命令")
        parts = text.split(None, 2)
        op = parts[0].upper()

        if op == "PUT":
            if len(parts) != 3:
                return self.make_response("ERROR: PUT 需要 key 和 value")
            key, value = parts[1], parts[2]
            self._state[key] = value
            return self._persist(f"OK PUT {key}={value}")

        if op == "GET":
            if len(parts) != 2:
                return self.make_response("ERROR: GET 需要 key")
            value = self._state.get(parts[1])
            return self.make_response("NONE" if value is None else str(value))

        if op == "DEL":
            if len(parts) != 2:
                return self.make_response("ERROR: DEL 需要 key")
            key = parts[1]
            if key not in self._state:
                return self.make_response("NONE")
            del self._state[key]
            return self._persist(f"OK DEL {key}")

        if op == "LIST":
            return self.make_response(json.dumps(self._state, ensure_ascii=False, sort_keys=True))

        if op == "RESET":
            self._state = {}
            return self._persist("OK RESET")

        if op == "SUM":
            total = 0
            count = 0
            for value in self._state.values():
                try:
                    total += float(value)
                    count += 1
                except (TypeError, ValueError):
                    continue
            if count == 0:
                return self.make_response("NONE")
            return self.make_response(str(total))

        return self.make_response(f"ERROR: 未知命令 {op}")

    def _execute_tool_message(self, content, call_id):
        """tool = Core 对 memory_write 的响应。"""
        rec = self._call_records.get(call_id)
        if rec is None or rec.get("name") != "memory_write":
            return self.make_response("ERROR: 收到未知工具响应")
        reply = self._pending_reply
        self._pending_reply = None
        if reply is None:
            return self.make_response("ERROR: 没有待答复的账本操作")
        if (content or "").startswith("Error"):
            return self.make_response("ERROR: " + content)
        return self.make_response(reply)

    # ------------------------------------------------------------------
    def _persist(self, final_reply):
        """保存最终答复，并返回持久化账本的 memory_write 工具调用。"""
        self._pending_reply = final_reply
        self._call_seq += 1
        tool_call = {
            "id": f"ledger_tc_{self._call_seq}",
            "type": "function",
            "function": {
                "name": "memory_write",
                "arguments": json.dumps(
                    {
                        "ref": self._path,
                        "content": json.dumps(self._state, ensure_ascii=False, sort_keys=True),
                    },
                    ensure_ascii=False,
                ),
            },
        }
        return self.make_response("", tool_calls=[tool_call])

"""AVM Web Monitor — 纵向执行流图构建器

以程序执行步骤为节点，构建纵向时间线 + 子对话分支的可视化数据。
"""
import time
import threading


class MonitorState:
    """运行时状态收集器

    输出纵向执行流（flow），每个节点代表一个执行步骤：
    - start / llm_call / tool_call / input / subdialog
    - depth 表示分支层级（0=主轴，1=第一层子对话...）
    """

    def __init__(self, core):
        self.core = core
        self._events: list[dict] = []
        self._events_lock = threading.Lock()
        # 执行流节点
        self._flow: list[dict] = []
        # 当前正在执行的步骤（call_id -> flow node）
        self._active_steps: dict[str, dict] = {}
        # conversation 深度（call_id -> depth）
        self._conv_depths: dict[str, int] = {}
        # tool_call 深度（call_id -> depth）
        self._tool_call_depths: dict[str, int] = {}
        # 待关联的子 create：child_call_id -> parent_call_id
        self._pending_child_creates: dict[str, str] = {}
        # 当前是否有输入等待
        self._input_waiting: bool = False
        self._input_step: dict | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def attach(self):
        self.core.add_state_observer(self._on_event)

    def detach(self):
        self.core.remove_state_observer(self._on_event)

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _on_event(self, event_type: str, payload: dict):
        ts = time.time()
        with self._events_lock:
            self._events.append({"type": event_type, "timestamp": ts, "payload": payload})

        if event_type == "instruction_start":
            self._handle_instruction_start(payload, ts)
        elif event_type == "instruction_end":
            self._handle_instruction_end(payload, ts)
        elif event_type == "tool_calls_detected":
            self._handle_tool_calls_detected(payload, ts)
        elif event_type == "conversation_created":
            pass
        elif event_type == "conversation_updated":
            pass
        elif event_type == "conversation_completed":
            pass
        elif event_type == "run_finished":
            pass

    def on_input_wait(self):
        """由 WebInputDevice 调用：开始等待用户输入"""
        if self._input_waiting:
            return
        self._input_waiting = True
        ts = time.time()
        # 推断当前 depth（取最后一个 flow 节点的 depth，或 0）
        depth = self._flow[-1]["depth"] if self._flow else 0
        step = {
            "id": f"step-{len(self._flow)}",
            "type": "input",
            "label": "读取输入",
            "depth": depth,
            "started_at": ts,
            "status": "running",
        }
        self._flow.append(step)
        self._input_step = step
        self._push_snapshot()

    def on_input_received(self, text: str):
        """由 WebInputDevice 调用：用户输入已接收"""
        if not self._input_waiting or self._input_step is None:
            return
        self._input_waiting = False
        self._input_step["status"] = "done"
        self._input_step["finished_at"] = time.time()
        self._input_step["detail"] = text[:40]
        self._input_step = None
        self._push_snapshot()

    def _handle_instruction_start(self, payload: dict, ts: float):
        instr = payload.get("instruction", {})
        call_id = instr.get("call_id", "")
        instr_type = instr.get("type", "")
        utr = instr.get("utr_index", -1)

        depth = self._compute_depth(instr_type, call_id, utr)

        if instr_type == "create":
            if depth == 0:
                node_type = "start"
                label = f"开始 #{call_id}"
            else:
                node_type = "subdialog"
                label = f"子对话 #{call_id}"
            self._conv_depths[call_id] = depth
        elif instr_type == "exec":
            node_type = "llm_call"
            label = f"继续对话 #{call_id}"
        elif instr_type == "memory_read":
            node_type = "tool_call"
            label = f"读取 {instr.get('ref', '')}"
        elif instr_type == "memory_write":
            node_type = "tool_call"
            label = f"写入 {instr.get('ref', '')}"
        elif instr_type == "memory_make":
            node_type = "tool_call"
            label = f"创建 {instr.get('ref', '')}.{instr.get('key', '')}"
        else:
            node_type = "step"
            label = instr_type

        step = {
            "id": f"step-{len(self._flow)}",
            "type": node_type,
            "label": label,
            "depth": depth,
            "call_id": call_id,
            "instr_type": instr_type,
            "started_at": ts,
            "status": "running",
            "detail": "",
        }
        self._flow.append(step)
        if call_id:
            self._active_steps[call_id] = step
        self._push_snapshot()

    def _handle_instruction_end(self, payload: dict, ts: float):
        instr = payload.get("instruction", {})
        call_id = instr.get("call_id", "")
        return_type = payload.get("return_type", "EXIT")

        if call_id and call_id in self._active_steps:
            step = self._active_steps[call_id]
            step["status"] = "done"
            step["finished_at"] = ts
            step["return_type"] = return_type
            del self._active_steps[call_id]
            self._push_snapshot()

    def _handle_tool_calls_detected(self, payload: dict, ts: float):
        parent_call_id = payload.get("call_id", "")
        parent_depth = self._conv_depths.get(parent_call_id, 0)
        for tc in payload.get("tool_calls", []):
            tc_id = tc.get("call_id", "")
            if not tc_id:
                continue
            self._tool_call_depths[tc_id] = parent_depth + 1
            if tc.get("cmd_type") == "create":
                self._pending_child_creates[tc_id] = parent_call_id

    def _compute_depth(self, instr_type: str, call_id: str, utr: int) -> int:
        if instr_type == "create":
            if call_id in self._tool_call_depths:
                # create_cmd 产生的子对话
                return self._tool_call_depths[call_id]
            if utr == -1:
                return 0
            return 0
        elif instr_type == "exec":
            # exec 和对应的 create 同深度
            return self._conv_depths.get(call_id, 0)
        else:
            # 子指令深度 = 所属 conversation 深度 + 1
            return self._tool_call_depths.get(call_id, 0)

    def _push_snapshot(self):
        """主动推送快照到 SSE 客户端（非 Core 事件驱动时调用）"""
        try:
            snapshot = self.build_snapshot()
            # 由 server.py 中的 observer 负责推送
            # 这里只需要触发一次 Core 通知，让 observer 工作
            self.core._notify("__monitor_flush", {})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 快照构建
    # ------------------------------------------------------------------

    def build_snapshot(self) -> dict:
        with self._events_lock:
            flow = list(self._flow)

        return {
            "timestamp": time.time(),
            "command_stack": self._build_command_stack(),
            "flow": flow,
            "running": {k: v for k, v in self._active_steps.items()},
        }

    def _build_command_stack(self) -> list[dict]:
        stack = []
        for idx, instr in enumerate(self.core.command_stack):
            if isinstance(instr, str):
                parts = instr.strip().split()
                d = {
                    "idx": idx,
                    "type": parts[0] if parts else "unknown",
                    "call_id": parts[1] if len(parts) > 1 else "",
                    "utr": int(parts[2]) if len(parts) > 2 else -1,
                    "raw": instr,
                }
            else:
                d = self.core._instruction_to_dict(instr)
                d["idx"] = idx
            stack.append(d)
        return stack

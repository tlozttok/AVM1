"""AVM 应用套件集成测试。

三个应用镜像位于 images/apps/：
- courier_office：create_cmd / send_instruction / create_sub / plm.simple / 自定义设备
- vault_guard：listen_event / owner / before replace 与改写 / after notify
- plm_ledger：镜像外自定义 PLM 服务器 / 工具调用往返 / create_sub 自动返回

测试用脚本化 MockLMU 驱动 LLM 对话，Python 对话与设备使用真实实现；不调真实 LLM API。
"""
import json
from pathlib import Path

import pytest

from avm.exceptions import MemoryTypeError, MemoryKeyNotFoundError, VMMemoryError
from avm.core import Core
from avm.image import load_image
from avm.python_server import PLM_REGISTRY
from avm.types import LinkNode

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "images" / "apps"
COURIER_IMAGE = APPS / "courier_office" / "courier_office.json"
VAULT_IMAGE = APPS / "vault_guard" / "vault_guard.json"
LEDGER_IMAGE = APPS / "plm_ledger" / "plm_ledger.json"


def _seed(mem, ref, value, owner_cid=None):
    """测试夹具：先 make 建链再写（memory_write 不再建节点）。"""
    try:
        return mem.set(ref, value, owner_cid=owner_cid)
    except MemoryKeyNotFoundError:
        parent, _, key = ref.rpartition(".")
        mem.make(parent, key, "str", owner_cid=owner_cid)
        return mem.set(ref, value, owner_cid=owner_cid)


def tool(call_id, name, **args):
    return {"call_id": call_id, "cmd_type": name, "args": args}


class ScriptedLMU:
    """按 cid 依次返回预设 (result, return_calls, conversation) 的脚本化 LMU。

    与 LMU.exec 一样把待处理批次和新的 assistant 消息提交进历史；
    这样测试可以检查 tool 响应是否真正进入对话历史。
    """

    def __init__(self, scripts):
        self.scripts = scripts
        self.idx = {cid: 0 for cid in scripts}
        self.calls = []

    def exec(self, conversation, para):
        self.calls.append((conversation.cid, conversation.identity))
        cid = conversation.cid
        seq = self.scripts[cid]
        i = self.idx[cid]
        if i >= len(seq):
            raise RuntimeError(f"ScriptedLMU 对话 {cid} 第 {i} 次调用没有预设响应")
        self.idx[cid] += 1
        result, return_calls, _ = seq[i]

        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            conversation.append_user_message(user_content)

        tc_list = None
        if return_calls:
            tc_list = [
                {
                    "id": rc.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": rc.get("cmd_type"),
                        "arguments": json.dumps(rc.get("args", {}), ensure_ascii=False),
                    },
                }
                for rc in return_calls
                if rc.get("cmd_type") != "json_error"
            ] or None
        conversation.append_assistant_message(result or "", tool_calls=tc_list)
        return result, return_calls, conversation


def _state(core):
    return {
        "active": core._active_cid,
        "ready": list(core._ready_cids),
        "dormant": list(core._dormant_cids),
        "finished": list(core._finished),
    }


# ----------------------------------------------------------------------
# courier_office
# ----------------------------------------------------------------------

class TestCourierOfficeImage:
    def test_image_loads_and_mounts_board(self):
        core = load_image(Core(), str(COURIER_IMAGE))
        conv = core.get_conversation(0)
        assert conv.name == "dispatcher"
        assert conv.is_root

        # 设备挂载点物化为链接节点，设备表保留地址
        link = core.mem._data["board"]
        assert isinstance(link, LinkNode)
        assert link.target == "board"
        assert "board" in core.mem._devices
        assert core.unwrap("$MEM.board") == (
            "list[len=0,elem=str,metadata='courier_office 完工记录板（只追加）']"
        )

        # 程序节点类型
        assert core.unwrap("$MEM.programs.dispatcher", for_llm=False).get_ctrl()["type"] == "settingup"
        assert core.unwrap("$MEM.programs.calc", for_llm=False).get_ctrl()["type"] == "python"
        assert core.unwrap("$MEM.para.simple", for_llm=False).get_ctrl()["type"] == "para"

    def test_board_device_named_access_semantics(self):
        """设备按名称读写：append 写、latest 读；数字位置写入与负下标都不支持。"""
        core = load_image(Core(), str(COURIER_IMAGE))
        board = core.mem._devices["board"]

        # 追加走命名接口
        _seed(core.mem, "$MEM.board.append", "first")
        _seed(core.mem, "$MEM.board.append", "second")
        assert board._data == ["first", "second"]
        assert core.unwrap("$MEM.board.latest") == "second"

        # 数字索引只读
        assert core.unwrap("$MEM.board.0") == "first"

        # 数字位置写入不再可用（设备写只走命名接口）
        with pytest.raises(VMMemoryError):
            core.mem.set("$MEM.board.1", "x")
        # 负下标也不再是"最后一条"
        with pytest.raises(VMMemoryError):
            core.unwrap("$MEM.board.-1")
        assert board._data == ["first", "second"]

        assert board.after_event("memory_write", ["append"], {"content": "second"}) == {
            "board_len": 2,
            "latest": "second",
        }

        with pytest.raises(MemoryTypeError):
            board["append"] = 123

    def test_full_scripted_run(self, capsys):
        core = load_image(Core(), str(COURIER_IMAGE))
        core.lmu = ScriptedLMU({
            0: [
                ("", [
                    tool("a1", "create_cmd", system_ref="$MEM.programs.worker", para_ref="$MEM.para.llm"),
                    tool("a1b0", "memory_make", ref="$MEM.state", key="worker_cid", mem_type="str"),
                    tool("a1b", "memory_write", ref="$MEM.state.worker_cid", content="1"),
                    tool("a1c", "memory_read", ref="$MEM.reports"),
                ], None),
                ("", [
                    tool("a1d", "memory_make", ref="$MEM", key="reports", mem_type="dict"),
                    tool("a1e", "memory_make", ref="$MEM.reports", key="1", mem_type="str"),
                    tool("a2", "send_instruction", cid=1,
                         content="JOB 1: 1+2 | 完成后 send_instruction(cid=0, content=RESULT 1 1+2 = 3)"),
                ], None),
                ("waiting", [], None),
                ("", [
                    tool("a3", "memory_read", ref="$MEM.board.latest"),
                    tool("a4", "memory_write", ref="$MEM.outputs.out", content="JOB 1 1+2 = 3"),
                ], None),
                ("waiting", [], None),
            ],
            1: [
                ("", [
                    tool("b1m", "memory_make", ref="$MEM.tasks", key="expr", mem_type="str"),
                    tool("b1", "memory_write", ref="$MEM.tasks.expr", content="1+2"),
                    tool("b2", "create_sub", system_ref="$MEM.programs.calc",
                         user_ref="$MEM.tasks.expr", para_ref="$MEM.para.simple"),
                ], None),
                ("", [
                    tool("b3", "memory_write", ref="$MEM.reports.1", content="JOB 1 1+2 = 3"),
                    tool("b4", "memory_write", ref="$MEM.board.append", content="JOB 1 1+2 = 3"),
                    tool("b5", "send_instruction", cid=0, content="RESULT 1 1+2 = 3"),
                ], None),
                ("done", [], None),
            ],
        })

        core.run()
        capsys.readouterr()

        board = core.mem._devices["board"]
        outputs = core.mem._devices["outputs"]
        assert board._data == ["JOB 1 1+2 = 3"]
        assert outputs._data == ["JOB 1 1+2 = 3"]
        assert core.unwrap("$MEM.tasks.expr", for_llm=False) == "1+2"
        # 运行期 memory_make 创建的 reports 目录与 worker 的持久记录
        assert core.unwrap("$MEM.reports", for_llm=False).get_ctrl()["owner"] == "0"
        assert core.unwrap("$MEM.reports.1", for_llm=False) == "JOB 1 1+2 = 3"

        st = _state(core)
        assert st["active"] is None
        assert st["ready"] == []
        assert set(st["dormant"]) == {0, 1}  # dispatcher / worker 休眠；calc 亚对话用完即删
        assert st["finished"] == []

        # 调度轨迹：dispatcher 三次 + worker 三次 + dispatcher 收尾两次
        assert [cid for cid, _ in core.lmu.calls] == [0, 0, 0, 1, 1, 1, 0, 0]


    def test_close_conversation_ends_individual(self):
        core = load_image(Core(), str(COURIER_IMAGE))
        core.lmu = ScriptedLMU({
            0: [("", [tool("x1", "close_conversation", cid=0)], None)],
        })
        core.run()

        st = _state(core)
        assert st["active"] is None
        assert st["ready"] == []
        assert st["finished"] == [0]


# ----------------------------------------------------------------------
# vault_guard
# ----------------------------------------------------------------------

class TestVaultGuardImage:
    def _run_task(self, task):
        core = load_image(Core(), str(VAULT_IMAGE))
        core.lmu = ScriptedLMU({
            0: [
                ("", [
                    tool("g0", "edit_metadata", ref="$MEM.vault", type="set", key="policy", value="guard-redact"),
                    tool("g1", "listen_event", action="listen", event_type="memory_write",
                         target="$MEM.vault.deposit", handler_ref="$MEM.handlers.deposit_pipe", phase="all"),
                    tool("g2", "listen_event", action="listen", event_type="memory_write",
                         target="$MEM.vault.audit", handler_ref="$MEM.handlers.audit_tail", phase="after"),
                    tool("g3", "create_cmd", system_ref="$MEM.programs.worker", para_ref="$MEM.para.llm"),
                ], None),
                ("", [tool("g4", "send_instruction", cid=1,
                           content=f"TASK 1: {task} | 完成后 send_instruction(cid=0, content=RESULT 1 done)")], None),
                ("waiting", [], None),
                ("", [
                    tool("g5", "memory_read", ref="$MEM.vault.audit"),
                    tool("g6", "memory_read", ref="$MEM.vault.deposit"),
                    tool("g7", "memory_write", ref="$MEM.outputs.out", content="AUDIT: TASK 1 尝试写入 deposit"),
                    tool("g8", "memory_write", ref="$MEM.outputs.out",
                         content="FINAL: EMPTY" if task == "SECRET" else "FINAL: SAFE:" + task),
                ], None),
                ("done", [], None),
            ],
            1: [
                ("", [
                    tool("w1", "memory_write", ref="$MEM.vault.deposit", content=task),
                    tool("w2", "memory_write", ref="$MEM.vault.audit", content="TASK 1 尝试写入 deposit"),
                    tool("w3", "send_instruction", cid=0, content="RESULT 1 done"),
                ], None),
                ("done", [], None),
            ],
        })
        core.run()
        return core

    def test_image_loads_owner_and_handlers(self):
        core = load_image(Core(), str(VAULT_IMAGE))
        vault = core.unwrap("$MEM.vault", for_llm=False)
        assert vault.get_ctrl()["owner"] == "0"
        assert core.unwrap("$MEM.vault.deposit", for_llm=False) == "EMPTY"

        for ref in ("$MEM.handlers.deposit_pipe", "$MEM.handlers.audit_tail"):
            node = core.unwrap(ref, for_llm=False)
            assert node.get_ctrl()["type"] == "python"
            assert isinstance(node["content"], str) and "def handle(event)" in node["content"]

    def test_before_replace_blocks_secret_and_after_notify_wakes_guardian(self, capsys):
        core = self._run_task("SECRET")
        capsys.readouterr()

        # before 守卫短路：deposit 没有写入
        assert core.unwrap("$MEM.vault.deposit", for_llm=False) == "EMPTY"
        # 第二条写入不受影响
        assert core.unwrap("$MEM.vault.audit", for_llm=False) == "TASK 1 尝试写入 deposit"
        assert core.mem._devices["outputs"]._data == ["AUDIT: TASK 1 尝试写入 deposit", "FINAL: EMPTY"]

        # worker 收到的第一个工具响应是守卫的拦截文本
        worker = core.get_conversation(1)
        tool_messages = [m.content for m in worker.messages if m.role == "tool"]
        assert any("拦截" in content for content in tool_messages)

        # edit_metadata 在运行期写入的 ctrl 策略标记
        assert core.unwrap("$MEM.vault", for_llm=False).get_ctrl()["policy"] == "guard-redact"

        # 两个监听器都在，且归属 cid=0
        listeners = core.unwrap("$MEM.listeners")
        assert "target=$MEM.vault.deposit" in listeners
        assert "target=$MEM.vault.audit" in listeners

        st = _state(core)
        assert st["active"] is None
        assert set(st["dormant"]) == {0, 1}

    def test_before_rewrite_safe_content(self, capsys):
        core = self._run_task("hello")
        capsys.readouterr()

        assert core.unwrap("$MEM.vault.deposit", for_llm=False) == "SAFE:hello"
        assert core.mem._devices["outputs"]._data == [
            "AUDIT: TASK 1 尝试写入 deposit",
            "FINAL: SAFE:hello",
        ]


# ----------------------------------------------------------------------
# plm_ledger
# ----------------------------------------------------------------------

class TestPlmLedger:
    @pytest.fixture(autouse=True)
    def _ledger_registry(self):
        old = PLM_REGISTRY.get("plm.ledger")
        sys_path = str(APPS / "plm_ledger")
        import sys
        added = sys_path not in sys.path
        if added:
            sys.path.insert(0, sys_path)
        from plm_ledger_server import LedgerPLM
        PLM_REGISTRY["plm.ledger"] = LedgerPLM
        yield LedgerPLM
        if old is None:
            PLM_REGISTRY.pop("plm.ledger", None)
        else:
            PLM_REGISTRY["plm.ledger"] = old
        if added:
            sys.path.remove(sys_path)

    def test_image_loads(self):
        core = load_image(Core(), str(LEDGER_IMAGE))
        assert core.unwrap("$MEM.programs.ledger", for_llm=False).get_ctrl()["type"] == "python"
        assert core.unwrap("$MEM.para.ledger", for_llm=False)["model"] == "plm.ledger"

    def test_ledger_plm_put_uses_tool_call_and_replies(self, _ledger_registry):
        server = _ledger_registry()
        params = {"tools": []}
        messages = [
            {"role": "system", "content": json.dumps({"path": "$MEM.ledger.state", "initial": {}})},
            {"role": "user", "content": "PUT a 1"},
        ]
        first = server.handle_messages(messages, params)
        assert first["content"] == ""
        assert len(first["tool_calls"]) == 1
        tc = first["tool_calls"][0]
        assert tc["function"]["name"] == "memory_write"
        args = json.loads(tc["function"]["arguments"])
        assert args["ref"] == "$MEM.ledger.state"
        assert json.loads(args["content"]) == {"a": "1"}

        # PLMExecutor 每次构建新的 messages 列表；直接调用时要复制，避免 _prev_input 别名
        messages = list(messages)
        messages.append({"role": "assistant", "content": first["content"], "tool_calls": first["tool_calls"]})
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Success set: $MEM.ledger.state"})
        second = server.handle_messages(messages, params)
        assert second["content"] == "OK PUT a=1"

    def test_ledger_plm_commands(self, _ledger_registry):
        server = _ledger_registry()
        params = {"tools": []}
        messages = [
            {"role": "system", "content": json.dumps({"path": "$MEM.ledger.state", "initial": {"x": "2"}})},
            {"role": "user", "content": "GET x"},
        ]
        assert server.handle_messages(messages, params)["content"] == "2"

        server = _ledger_registry()
        messages = [
            {"role": "system", "content": json.dumps({"path": "$MEM.ledger.state", "initial": {"x": "2", "y": "3"}})},
            {"role": "user", "content": "SUM"},
        ]
        assert server.handle_messages(messages, params)["content"] == "5.0"

    def test_full_scripted_run(self, _ledger_registry, capsys):
        core = load_image(Core(), str(LEDGER_IMAGE))
        core.lmu = ScriptedLMU({
            0: [
                ("", [
                    tool("t1", "memory_write", ref="$MEM.request", content="PUT a 1"),
                    tool("t2", "create_sub", system_ref="$MEM.programs.ledger",
                         user_ref="$MEM.request", para_ref="$MEM.para.ledger"),
                ], None),
                ("", [tool("t3", "memory_write", ref="$MEM.outputs.out", content="OK PUT a=1")], None),
                ("done", [], None),
            ],
        })

        core.run()
        capsys.readouterr()

        assert json.loads(core.unwrap("$MEM.ledger.state", for_llm=False)) == {"a": "1"}
        assert core.mem._devices["outputs"]._data == ["OK PUT a=1"]

        # 亚对话用完即删：账本对话完成后不再保留，但返回结果进入了 father 的工具响应
        assert core.get_conversation(1) is None

        st = _state(core)
        assert st["active"] is None
        assert set(st["dormant"]) == {0}
        assert [cid for cid, _ in core.lmu.calls] == [0, 0, 0]

    def test_python_conversation_rejects_unregistered_plm(self, monkeypatch):
        """镜像已加载但宿主忘了注册：创建 Python 亚对话时就在输入边界报错，不静默回退 LLM。

        2026-10-16 起这条走模型目录的命名空间校验（create_sub 收 Error 工具响应），
        不再等到亚对话第一次推理。
        """
        import avm.core as core_module
        monkeypatch.delitem(core_module.PLM_REGISTRY, "plm.ledger", raising=False)
        core = load_image(Core(), str(LEDGER_IMAGE))
        core.lmu = ScriptedLMU({
            0: [
                ("", [
                    tool("t1", "memory_write", ref="$MEM.request", content="PUT a 1"),
                    tool("t2", "create_sub", system_ref="$MEM.programs.ledger",
                         user_ref="$MEM.request", para_ref="$MEM.para.ledger"),
                ], None),
            ],
        })
        core._active_cid = core._ready_cids.pop(0)
        core.advance_conversation()
        responses = [r.content for r in core._conv_by_cid[0].user_batch.tool_responses]
        assert any("不是 PLM 注册名" in r for r in responses), responses
        assert core.get_conversation(1) is None, "校验失败不得创建对话"

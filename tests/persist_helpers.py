"""持久运行（journal / snapshot / packages）测试共用夹具。

不调真实 API：LLM 对话用脚本化执行器（行为与真 LMU 一致：消费批次、提交历史、
返回 return_calls）；需要确定性时用 `plm.python` 的 Python 对话。
"""

import json

from avm.core import Core
from avm.image import load_image


class ScriptedLMU:
    """按脚本产生回合并像真 LMU 一样提交历史（不调 API）。

    响应条目是 (content, return_calls)；return_calls 形如
    [{"call_id": "c1", "cmd_type": "memory_write", "args": {...}}]。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.last_call = None
        self.calls = []

    def exec(self, conversation, para):
        if self.index >= len(self.responses):
            raise RuntimeError(f"ScriptedLMU 第 {self.index} 次调用没有预设响应")
        content, return_calls = self.responses[self.index]
        self.index += 1
        self.calls.append((content, return_calls))
        return_calls = list(return_calls or [])
        # 与 LMU.exec 的提交顺序一致：批次里的 tool 响应 → 用户内容 → 新的 assistant
        for resp in list(conversation.user_batch.tool_responses):
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            conversation.append_user_message(user_content)
        tool_calls = None
        if return_calls:
            tool_calls = [
                {
                    "id": rc["call_id"],
                    "type": "function",
                    "function": {
                        "name": rc["cmd_type"],
                        "arguments": json.dumps(rc["args"], ensure_ascii=False),
                    },
                }
                for rc in return_calls
            ]
        conversation.append_assistant_message(content or "", tool_calls=tool_calls)
        self.last_call = {
            "result": content,
            "tool_calls": return_calls,
            "messages": [],
        }
        return content, return_calls, conversation


def base_image_mem():
    """最小镜像 mem 段：para、settingup 程序、两个数据叶。"""
    return {
        "model_params": {
            "kind": "dict",
            "ctrl": {"type": "para"},
            "value": {"model": {"kind": "str", "value": "test-model"}},
        },
        "programs": {
            "kind": "dict",
            "value": {
                "main": {
                    "kind": "dict",
                    "ctrl": {"type": "settingup"},
                    "value": {"content": {"kind": "str", "value": "sys"}},
                },
            },
        },
        "src": {"kind": "str", "value": "hello"},
        "dst": {"kind": "str", "value": ""},
    }


def write_image(tmp_path, *, mem=None, devices=None, journal=True, snapshot=True,
                init=None, name="image.json"):
    """写一份最小镜像；journal_to / snapshot_to 默认落在 tmp_path/out/ 下。"""
    image = {
        "meta": {"version": 1},
        "para_ref": "$MEM.model_params",
        "mem": mem if mem is not None else base_image_mem(),
        "init": init if init is not None else {
            "name": "init",
            "system_ref": "$MEM.programs.main",
            "user": "start",
        },
    }
    if devices is not None:
        image["devices"] = devices
    if journal:
        image["journal_to"] = "out/run.journal"
    if snapshot:
        image["snapshot_to"] = "out/run.snapshot.json"
    path = tmp_path / name
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_core(image_path, responses=None):
    """装载镜像；给了 responses 就换成脚本化 LMU（不碰网络）。"""
    core = load_image(Core(), image_path)
    if responses is not None:
        core.lmu = ScriptedLMU(responses)
    return core


def run_turns(core, count):
    """手动推进 count 个回合（每个 advance = 一轮推理 + 该轮指令）。

    与 Core.run 的差别：不派发中断、不记监测基线；活跃对话保持时连续推进。
    """
    if core._active_cid is None:
        core._pick_next_active()
    for _ in range(count):
        if core._active_cid is None:
            break
        core.advance_conversation()


def run_core(core):
    if core._active_cid is None:
        core._pick_next_active()
    while core._active_cid is not None:
        core.advance_conversation()


def out_path(image_path, name):
    """镜像目录下的 out/ 落点（与 journal_to/snapshot_to 的相对路径一致）。"""
    import os

    return os.path.join(os.path.dirname(os.path.abspath(image_path)), "out", name)


def state_view(core):
    """参与"恢复是否一致"比较的字段（快照里除来源信息以外的全部段）。"""
    from avm.snapshot import snapshot

    doc = snapshot(core)
    return {
        key: doc[key]
        for key in ("scheduler", "next_cid", "memory", "security", "para_ref",
                    "conversations", "listeners", "devices", "device_specs")
    }


def assert_same_state(core_a, core_b, message=""):
    view_a = state_view(core_a)
    view_b = state_view(core_b)
    for key in view_a:
        assert view_a[key] == view_b[key], f"{message}字段 {key} 不一致"

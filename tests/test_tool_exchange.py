"""tool_exchange：最近一次工具调用与它已到响应的读取。

这个函数是 _pending_call_ids（调度判"还在等谁"）与 _check_pairing（安全点判"配
没配上"）的共同来源，两份以前各抄一遍。它一旦把重复响应去重，_check_pairing 就
再也发现不了"同一个 call id 收到两次响应"；一旦漏掉待发批次，父对话刚拿到亚对话
返回就会被判成历史不配对。这两条正是下面守着的东西。
"""

import pytest

from avm.core import Core
from avm.exceptions import KernelPanic
from avm.types import tool_exchange


def _conv():
    core = Core()
    return core.start("sys", "u")


def test_no_assistant_tool_calls_gives_empty():
    """历史里还没有工具调用：两边都是空，调用方各自判空即可。"""
    conv = _conv()
    assert tool_exchange(conv) == ([], [])


def test_wanted_keeps_declaration_order():
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    wanted, got = tool_exchange(conv)
    assert wanted == ["call_b", "call_a"]   # 按声明顺序，不排序
    assert got == []


def test_responses_from_history_and_batch_both_count():
    """响应可能已在历史里，也可能还在待发批次里（先入批次，下一轮才写历史）。"""
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    conv.append_tool_message("A", "call_a")
    conv.user_batch.add_tool_response("B", "call_b")
    wanted, got = tool_exchange(conv)
    assert wanted == ["call_a", "call_b"]
    assert got == ["call_a", "call_b"]


def test_only_the_last_tool_call_message_is_read():
    """更早那条带 tool_calls 的 assistant 消息已配对完毕，不再参与。"""
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"id": "old", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    conv.append_tool_message("旧结果", "old")
    conv.append_assistant_message("说完了")
    conv.append_assistant_message("", tool_calls=[
        {"id": "new", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    assert tool_exchange(conv) == (["new"], [])


def test_calls_without_id_are_skipped():
    """没有 id 的调用两边都配不上，不参与配对（否则永远判不齐）。"""
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    assert tool_exchange(conv) == (["call_a"], [])


def test_duplicate_response_is_kept_so_pairing_fails():
    """同一 id 收到两次响应：got 保留重复，_check_pairing 才判得出来。"""
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ])
    conv.append_tool_message("第一次", "call_a")
    conv.append_tool_message("第二次", "call_a")
    wanted, got = tool_exchange(conv)
    assert got == ["call_a", "call_a"]      # 去重会把它藏起来
    with pytest.raises(KernelPanic):
        Core._check_pairing(conv)
    # 只做包含判断的一方不受影响：call_a 已经有响应，不再算"在等"
    assert Core._pending_call_ids(conv) == []


def test_batch_response_makes_pending_empty():
    """父对话刚收到亚对话返回（响应在批次里、还没进历史）：不算仍在等。"""
    conv = _conv()
    conv.append_assistant_message("", tool_calls=[
        {"id": "call_sub", "type": "function", "function": {"name": "create_sub", "arguments": "{}"}},
    ])
    assert Core._pending_call_ids(conv) == ["call_sub"]
    conv.user_batch.add_tool_response("子结果", "call_sub")
    assert Core._pending_call_ids(conv) == []
    Core._check_pairing(conv)  # 不抛

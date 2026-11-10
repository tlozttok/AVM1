"""镜像层 ACL 示例：permission_acl_demo.json 的确定性端到端测试。

镜像全部由 PLM 组成（初始 Python 程序 + 用户态 Python 工作对话），不调用 LLM API。
初始程序注册 ACL 监听器、创建用户态工作对话、投递任务；工作对话尝试读/写受保护节点；
结果节点 after 监听器通知初始程序收尾，把审计报告写进 $MEM.report。
"""

import importlib.util
import json
import os

import pytest

from avm.core import Core
from avm.image import load_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = os.path.join(ROOT, "images", "permission_acl_demo.json")
BUILDER = os.path.join(ROOT, "images", "permission_acl_demo", "build_image.py")


def _builder():
    spec = importlib.util.spec_from_file_location("acl_demo_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_matches_builder():
    with open(IMAGE, encoding="utf-8") as f:
        on_disk = f.read()
    builder = _builder()
    assert builder.render(builder.build_image()) == on_disk, (
        "镜像与生成脚本不一致：重跑 python images/permission_acl_demo/build_image.py"
    )


def test_acl_demo_runs_to_completion_without_llm():
    core = load_image(Core(), IMAGE)
    core.run()

    report = str(core.mem["report"])
    assert "ACL 演示报告" in report
    # 允许的读
    assert "greeting=hello" in report
    # 拒绝的读（secret 对 worker 组 deny）
    assert "secret=Interrupted: ACL 拒绝读取" in report
    # 写保护（greeting 写 ACL 默认拒绝）
    assert "write=Interrupted: ACL 默认拒绝写入" in report
    # 库节点没有被改写
    assert str(core.mem["lib"]["greeting"]) == "hello"
    assert str(core.mem["lib"]["secret"]) == "top-secret"
    # 初始程序与工作对话都被关闭
    assert sorted(core._finished) == [0, 1]
    # 软件建的工作目录里留有工作对话的结果节点
    result = core.mem.resolve_path(["MEM", "workdirs", "w0", "result"], for_llm=False)
    assert "greeting=hello" in str(result)


def test_demo_image_declares_user_caps_and_acl_data():
    core = load_image(Core(), IMAGE)
    worker = core.mem.resolve_path(["MEM", "programs", "worker"], for_llm=False)
    caps_node = worker.get("caps")
    caps = {
        key: (str(caps_node.get(key)) if key == "mode"
              else [str(x) for x in caps_node.get(key)])
        for key in caps_node.keys()
    }
    assert "workspace" not in caps  # 内核不再分配工作空间
    assert caps["mode"] == "user"
    assert caps["roots"] == ["lib"]
    assert caps["groups"] == ["worker"]
    assert set(caps["tools"]) == {"memory_read", "memory_write", "memory_make"}
    greeting = core.mem.resolve_path(["MEM", "lib", "greeting"], for_llm=False)
    assert json.loads(greeting.get_ctrl()["acl.read"]) == {"allow": [{"group": "worker"}]}
    secret = core.mem.resolve_path(["MEM", "lib", "secret"], for_llm=False)
    assert json.loads(secret.get_ctrl()["acl.read"]) == {"deny": [{"group": "worker"}]}


def test_demo_finished_listeners_are_cleaned_up():
    core = load_image(Core(), IMAGE)
    core.run()
    assert core.events.listeners == []

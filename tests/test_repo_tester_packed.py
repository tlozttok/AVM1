"""打包版测试镜像：0 号是 plm.python 打包器，启动时读文档、组装提示词、建并启动测试对话。

不调真实 API：打包器是 Python 对话（工具调用不经过模型），测试对话装 MockLMU。
"""

import importlib.util
import json
import os

import pytest

from avm.core import Core
from avm.image import load_image
from avm.sandbox import check_code

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER_PATH = os.path.join(ROOT, "images", "repo_tester_packed", "build_image.py")


class MockLMU:
    """测试对话的假执行器：历史提交与真 LMU 一致，不产生工具调用。"""

    def __init__(self, text="收到，开始测试"):
        self.text = text
        self.calls = []

    def exec(self, conversation, para):
        self.calls.append(para)
        for resp in conversation.user_batch.tool_responses:
            conversation.append_tool_message(resp.content, resp.tool_call_id)
        user_content = conversation.user_batch.get_user_content()
        if user_content:
            conversation.append_user_message(user_content)
        conversation.append_assistant_message(self.text)
        return self.text, [], conversation


def _builder():
    spec = importlib.util.spec_from_file_location("repo_tester_packed_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def started(builder):
    """装载并跑完启动流程：0 号打包器把提示词组装好、建出测试对话后关闭自己。"""
    core = load_image(Core(), builder.IMAGE_PATH)
    lmu = MockLMU()
    core.lmu = lmu
    core.run()
    return core, lmu


def test_image_matches_builder(builder):
    with open(builder.IMAGE_PATH, "r", encoding="utf-8") as f:
        on_disk = f.read()
    assert builder.render(builder.build_image()) == on_disk, (
        "镜像与程序/清单不一致：重跑 python images/repo_tester_packed/build_image.py"
    )


def test_packer_program_passes_sandbox(builder):
    check_code(builder.build_image()["mem"]["packer"]["value"]["content"]["value"])


def test_user_message_is_file_list(builder):
    image = builder.build_image()
    assert image["init"] == {
        "name": "packer",
        "system_ref": "$MEM.packer",
        "user_ref": "$MEM.file_list",
    }
    assert image["mem"]["file_list"]["value"] == (
        'pack("CONTEXT.md", "README.md", "AGENTS.md", "docs/system-image.md", '
        '"docs/philosophy.md", "docs/programming-guide.md")'
    )
    assert image["mem"]["packer_params"]["value"]["model"]["value"] == "plm.python"


def test_startup_packs_docs_into_tester_prompt(builder, started):
    core, lmu = started
    content = core.mem["system"]["content"]

    assert content.startswith("你是这个系统中的一个对话")
    for rel in builder.FILES:
        assert f"### {rel}" in content
        with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as f:
            body = f.read()
        assert body.splitlines()[0].strip() in content
    assert "### docs/runtime.md" not in content
    assert "docs/runtime.md 不在打包里" in content

    # 清单节点：6 行
    assert core.mem["packed_docs"].count("\n") == len(builder.FILES) - 1
    assert "### README.md" in content


def test_packer_creates_and_starts_tester_then_closes(builder, started):
    core, lmu = started
    assert 0 in core._finished  # 打包器关闭了自己

    tester = next(c for c in core._conv_by_cid.values() if c.name == "repo_tester")
    assert tester.cid != 0
    assert tester.is_python is False
    assert tester.messages[0].content == core.mem["system"]["content"]
    assert tester.messages[1].content == core.mem["user"]
    assert tester.messages[-1].content == "收到，开始测试"
    assert len(lmu.calls) == 1

    report = core.mem["packer_report"]
    assert str(len(builder.FILES)) in report
    assert f"cid={tester.cid}" in report


def test_runtime_doc_still_reachable_through_device(builder, started):
    core, _ = started
    page = core.unwrap('$MEM.AVMRepo.read."docs/runtime.md".0', for_llm=True)
    assert "docs/runtime.md" in page


def test_prompt_has_no_page_headers(builder, started):
    """逐页读的页头/页尾不进提示词。"""
    core, _ = started
    content = core.mem["system"]["content"]
    assert 'read "CONTEXT.md" page=0' not in content
    assert "下一页:" not in content
    assert "（末页）" not in content


def test_image_json_is_compact(builder):
    """文档不再进镜像：镜像里只有程序与清单，文档启动时现读。"""
    with open(builder.IMAGE_PATH, "r", encoding="utf-8") as f:
        assert len(f.read().encode("utf-8")) < 20_000
    with open(os.path.join(ROOT, "CONTEXT.md"), "r", encoding="utf-8") as f:
        assert len(f.read().encode("utf-8")) > 10_000
    nodes = json.loads(open(builder.IMAGE_PATH, encoding="utf-8").read())["mem"]
    assert nodes["system"]["value"]["content"]["value"].startswith("（启动时由")

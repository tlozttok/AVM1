"""AVM 讨论社区地基的测试（服务 PLM + 镜像 + 面板路径）。

不调 LLM：所有对话都是 Python 对话（驱动用 `plm.python`，服务用 `plm.community`）。
面板路径的端到端验收在 `images/community/acceptance_panel.py`（起 HTTP 后端），
本文件覆盖命令语义、结构规则、派生量、合并/删除/标记与错误分支。
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMUNITY = ROOT / "images" / "community"
IMAGE_PATH = COMMUNITY / "community.json"
IO_DEVICE = str((ROOT / "images" / "devices" / "io.py").resolve())
PLM_SOURCE = str((COMMUNITY / "community_server.py").resolve())

sys.path.insert(0, str(COMMUNITY))

from avm.core import Core  # noqa: E402
from avm.image import load_image  # noqa: E402
from avm.python_server import PLM_REGISTRY  # noqa: E402


def _base_image() -> dict:
    with open(IMAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


DRIVER_CODE = '''
def run_requests():
    i = 0
    while True:
        req = memory_read("$MEM.test.requests." + str(i))
        if not isinstance(req, str) or req.startswith("Error:"):
            return
        if req.startswith("__CORRUPT_CARD__"):
            pid = req.split(" ")[1]
            ref = "$MEM.forum.posts." + pid + ".meta"
            meta = json.loads(memory_read(ref))
            meta["card"] = "BROKEN CARD"
            meta["stats"]["comments"] = 999
            memory_write(ref, json.dumps(meta, ensure_ascii=False))
        else:
            memory_write("$MEM.run.svc_request.panel", req)
            reply = create_sub(
                "$MEM.programs.svc_panel", "$MEM.run.svc_request.panel", "$MEM.para.svc"
            )
            memory_write("$MEM.test.results.-1", str(reply))
        i = i + 1
'''.strip()


def _str_node(value: str) -> dict:
    return {"kind": "str", "value": value}


def _make_test_image(tmp_path: Path, requests) -> Path:
    image = copy.deepcopy(_base_image())
    image.pop("persist_to", None)
    for device in image["devices"]:
        device["file"] = IO_DEVICE
    # 测试镜像写到 tmp_path，目录里没有 community_server.py；把 plms 的文件改成绝对路径
    for plm in image.get("plms") or []:
        plm["file"] = PLM_SOURCE
    mem = image["mem"]
    mem["test"] = {
        "kind": "dict",
        "value": {
            "requests": {
                "kind": "list",
                "value": [_str_node(req) for req in requests],
            },
            "results": {"kind": "list", "value": []},
        },
    }
    programs = mem["programs"]["value"]
    programs["test_driver"] = {
        "kind": "dict",
        "meta": "测试驱动：把 $MEM.test.requests 逐条交给 plm.community",
        "ctrl": {"type": "python"},
        "value": {"content": _str_node(DRIVER_CODE)},
    }
    image["init"] = {
        "name": "test_driver",
        "system_ref": "$MEM.programs.test_driver",
        "user": "run_requests()",
        "para_ref": "$MEM.para.python",
    }
    path = tmp_path / "community_test.json"
    path.write_text(json.dumps(image, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _results(core: Core) -> list[str]:
    values = core.unwrap("$MEM.test.results", for_llm=False)
    return [str(values[i]) for i in range(len(values))]


def run_requests(tmp_path: Path, requests) -> tuple[Core, list[str]]:
    image_path = _make_test_image(tmp_path, requests)
    core = load_image(Core(), str(image_path))
    core.run()
    return core, _results(core)


def _meta(core: Core, pid: str) -> dict:
    return json.loads(core.unwrap(f"$MEM.forum.posts.{pid}.meta", for_llm=False))


def _comment(core: Core, cid: str) -> dict:
    return json.loads(core.unwrap(f"$MEM.forum.comments.{cid}", for_llm=False))


def _forum_meta(core: Core) -> dict:
    return json.loads(core.unwrap("$MEM.forum.meta", for_llm=False))


# ----------------------------------------------------------------------
# 镜像与生成物
# ----------------------------------------------------------------------


class TestImage:
    def test_image_loads(self, tmp_path):
        image = _make_test_image(tmp_path, [])
        core = load_image(Core(), str(image))
        assert core.get_conversation(0).is_python
        forum_meta = _forum_meta(core)
        assert forum_meta["boards"] == ["design", "code", "meta", "usage"]
        assert forum_meta["seq"] == 3
        for pid in ("P000001", "P000002", "P000003"):
            assert _meta(core, pid)["pid"] == pid
        assert core.unwrap("$MEM.forum.boards.design.name", for_llm=False) == "设计"
        assert core.unwrap("$MEM.community.handbook", for_llm=False)
        assert {"inputs", "outputs"} <= set(core.mem.mounted_devices())

    def test_program_nodes_and_paras(self, tmp_path):
        image = _make_test_image(tmp_path, [])
        core = load_image(Core(), str(image))
        for ref in ("$MEM.programs.shell", "$MEM.programs.svc_panel",
                    "$MEM.programs.svc_public", "$MEM.programs.svc_admin"):
            node = core.unwrap(ref, for_llm=False)
            assert (node.get_ctrl() or {}).get("type") == "python"
            assert isinstance(node.get("content"), str) and node.get("content")
        assert core.unwrap("$MEM.para.svc", for_llm=False)["model"] == "plm.community"
        assert core.unwrap("$MEM.para.python", for_llm=False)["model"] == "plm.python"

    def test_build_image_in_sync(self):
        proc = subprocess.run(
            [sys.executable, str(COMMUNITY / "build_image.py"), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout

    def test_plm_declared_by_the_image_not_the_host(self):
        """标准入口能装进社区镜像：plm.community 由镜像 plms 段声明，宿主不注册。"""
        assert "plm.community" not in PLM_REGISTRY, "测试不该依赖宿主预注册"
        core = load_image(Core(), str(IMAGE_PATH))
        assert core.plms["plm.community"].__name__ == "CommunityPLM"
        names = [entry["name"] for entry in core.mem._data["model_catalog"]["plm"]]
        assert "plm.community" in names, "模型目录是 PLM 注册表的可见面"
        assert "plm.community" not in PLM_REGISTRY, "装载不该改写宿主注册表"

    def test_no_vote_fields_in_forum(self, tmp_path):
        image = _make_test_image(tmp_path, [])
        core = load_image(Core(), str(image))
        raw = json.dumps(core.mem.to_image_dict(), ensure_ascii=False)
        for banned in ('"vote"', '"like"', '"score"', '"upvote"', '"downvote"'):
            assert banned not in raw


# ----------------------------------------------------------------------
# 发帖、浏览、计数
# ----------------------------------------------------------------------


class TestPostAndBrowse:
    def test_new_then_board_and_post(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "BOARD design",
                "POST P000001",
                "AS D1\nNEW code question 新问题标题\n新问题的正文。",
                "BOARD code new 0",
                "POST P000004",
            ],
        )
        assert results[0].startswith("OK BOARD design")
        assert "为什么评论只允许一层" in results[0]
        assert results[1].startswith("OK POST P000001")
        assert "正文" in results[1]
        assert results[2].startswith("OK NEW P000004")
        assert "板=code" in results[2]
        assert "新问题标题" in results[3]
        assert results[4].startswith("OK POST P000004")
        assert "新问题的正文。" in results[4]
        assert "作者=D1" in results[4]
        meta = _meta(core, "P000004")
        assert meta["form"] == "question"
        assert meta["stats"]["comments"] == 0
        assert meta["card"].startswith("P000004 [问题] 新问题标题")
        assert _forum_meta(core)["seq"] == 4

    def test_comment_updates_counts_and_thread(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n评论一。",
                "AS D2\nCOMMENT P000001\n评论二。",
                "THREAD P000001",
                "POST P000001",
            ],
        )
        assert results[0].startswith("OK COMMENT C000004")
        assert "评1 人1" in results[0]
        assert results[1].startswith("OK COMMENT C000005")
        assert "评2 人2" in results[1]
        assert results[2].startswith("OK THREAD P000001")
        assert "C000004" in results[2] and "C000005" in results[2]
        meta = _meta(core, "P000001")
        assert meta["stats"]["comments"] == 2
        assert meta["stats"]["participants"] == 2
        assert "评2 人2" in meta["card"]
        assert _forum_meta(core)["seq"] == 5

    def test_inline_as_envelope(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1 COMMENT P000001\n单行信封写评论",
                "AS A1 DELETE P000002 单行信封删帖",
                "AS C1 BADCOMMAND 测试",
            ],
        )
        assert results[0].startswith("OK COMMENT C000004")
        assert "作者=D1" in results[0]
        assert results[1].startswith("OK DELETE P000002")
        assert "理由：单行信封删帖" in results[1]
        assert results[2].startswith("Error:") and "未知命令" in results[2]

    def test_new_rejects_bad_input(self, tmp_path):
        _core, results = run_requests(
            tmp_path,
            [
                "NEW unknownboard post 标题\n正文",
                "NEW design nope 标题\n正文",
                "NEW design post 标题\n",
                "AS bad!name\nBOARD design",
                "NOPE design",
            ],
        )
        assert results[0].startswith("Error:") and "未知板块" in results[0]
        assert results[1].startswith("Error:") and "form" in results[1]
        assert results[2].startswith("Error:") and "正文" in results[2]
        assert results[3].startswith("Error:") and "作者名" in results[3]
        assert results[4].startswith("Error:") and "未知命令" in results[4]


# ----------------------------------------------------------------------
# REPLY 外化与链式
# ----------------------------------------------------------------------


class TestReplyExternalization:
    def test_reply_root_comment_creates_branch(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n根评论。",
                "AS D2\nREPLY C000004\n第一次回复，应外化。",
                "AS D1\nREPLY C000004\n第二次回复，进同一个分叉帖。",
                "POST P000005",
                "THREAD P000005",
            ],
        )
        assert results[1].startswith("OK REPLY C000006")
        assert "已外化成分叉帖 P000005" in results[1]
        assert results[2].startswith("OK REPLY C000007")
        assert "已有分叉帖 P000005" in results[2]

        root_comment = _comment(core, "C000004")
        assert root_comment["promoted_to"] == "P000005"
        branch = _meta(core, "P000005")
        assert branch["form"] == "branch"
        assert branch["quote"] == {
            "kind": "comment",
            "id": "C000004",
            "post": "P000001",
            "excerpt": "根评论。",
        }
        assert branch["branch_of_comment"] == "C000004"
        assert branch["branch_of_post"] == "P000001"
        assert branch["comment_ids"] == ["C000006", "C000007"]
        assert branch["stats"]["comments"] == 2
        assert branch["stats"]["participants"] == 2
        root = _meta(core, "P000001")
        assert root["stats"]["branches"] == 1
        assert results[3].startswith("OK POST P000005")
        assert "引用：评论 C000004" in results[3]
        assert results[4].count("C00000") >= 2

    def test_reply_inside_branch_stays_flat(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n根评论。",
                "AS D2\nREPLY C000004\n外化一。",
                "AS D1\nREPLY C000006\n分叉帖内平层回复。",
            ],
        )
        assert "平层追加到 分叉帖 P000005" in results[2]
        branch = _meta(core, "P000005")
        assert branch["form"] == "branch"
        assert branch["comment_ids"] == ["C000006", "C000007"]
        assert "P000008" not in json.dumps(branch)
        # 分叉帖内评论的 reply_to 记录目标，但不产生新帖
        assert _comment(core, "C000007")["reply_to"] == "C000006"
        assert results[2].count("已外化") == 0

    def test_chain_upgrade_and_downgrade(self, tmp_path):
        requests = ["AS D1\nCOMMENT P000001\n根评论。"]
        requests.append("AS D2\nREPLY C000004\n一")
        requests.append("AS D1\nREPLY C000004\n二")
        requests.append("AS D2\nREPLY C000004\n三")
        requests.append("AS D1\nREPLY C000004\n四")
        requests.append("POST P000005")
        requests.append("AS D3\nREPLY C000004\n第三人")
        requests.append("POST P000005")
        core, results = run_requests(tmp_path, requests)
        assert "形态=对谈" in results[5]
        assert "对谈：D2 × D1" in results[5]
        branch = _meta(core, "P000005")
        assert branch["form"] == "branch"  # 第三人加入后退回
        assert "[分叉]" in results[6]
        assert "形态=分叉" in results[7]
        assert "分叉：回应 C000004" in results[7]

    def test_chain_hot_marking(self, tmp_path):
        requests = ["AS D1\nCOMMENT P000001\n根评论。"]
        for author in ("D2", "D1", "D2", "D1"):
            requests.append(f"AS {author}\nREPLY C000004\n{author} 发言")
        requests.append("BOARD design hot 0")
        core, results = run_requests(tmp_path, requests)
        stats = _meta(core, "P000005")["stats"]
        assert stats["heat"] >= 8
        assert stats["hot"] == "yes"
        # 热度排序把对谈帖排在第一
        assert any(line.startswith("P000005 [对谈]") for line in results[-1].splitlines())


# ----------------------------------------------------------------------
# 引用、合并、删除、标记
# ----------------------------------------------------------------------


class TestQuoteMergeDelete:
    def test_quote_post_and_comment(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n被引用的评论。",
                "AS D2\nQUOTE P000001 引用帖标题\n引用帖正文。",
                "AS D3\nQUOTE C000004 引用评论的帖\n另一个引用帖。",
            ],
        )
        assert results[1].startswith("OK QUOTE P000005")
        assert "引用 post P000001" in results[1]
        quote_post = _meta(core, "P000005")
        assert quote_post["form"] == "quote"
        assert quote_post["quote"]["kind"] == "post"
        assert quote_post["quote"]["id"] == "P000001"
        assert results[2].startswith("OK QUOTE P000006")
        assert "引用 comment C000004" in results[2]
        assert _meta(core, "P000006")["quote"]["kind"] == "comment"
        assert _meta(core, "P000001")["stats"]["quotes"] == 2

    def test_merge_moves_comments_and_alias(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000002\n源帖评论。",
                "AS D2\nCOMMENT P000001\n目标帖评论。",
                "AS A1\nMERGE P000001 P000002 重复问题合并",
                "THREAD P000001",
                "POST P000002",
                "BOARD code new 0",
            ],
        )
        assert results[2].startswith("OK MERGE P000001 ← P000002")
        assert "迁移评论 1 条" in results[2]
        target = _meta(core, "P000001")
        assert target["comment_ids"] == ["C000004", "C000005"]
        assert target["merged_from"] == ["P000002"]
        assert target["stats"]["comments"] == 2
        source = _meta(core, "P000002")
        assert source["status"] == "merged"
        assert source["merged_into"] == "P000001"
        assert source["comment_ids"] == []
        assert core.unwrap("$MEM.forum.aliases.P000002", for_llm=False) == "P000001"
        # 评论记录不动：仍可读、创建序号不变
        migrated = _comment(core, "C000004")
        assert migrated["content"] == "源帖评论。"
        assert migrated["created_seq"] == 4
        assert results[3].startswith("OK THREAD P000001")
        assert "C000004" in results[3] and "C000005" in results[3]
        assert results[5].startswith("OK BOARD code")
        assert "P000002" not in results[5]

    def test_merge_rejects_cross_group(self, tmp_path):
        _core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n根评论。",
                "AS D2\nREPLY C000004\n外化。",
                "AS A1\nMERGE P000005 P000001 跨组合并",
            ],
        )
        assert results[2].startswith("Error:") and "合并分组不一致" in results[2]

    def test_delete_comment_and_branch_source(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n根评论。",
                "AS D2\nCOMMENT P000001\n要删除的评论。",
                "AS A1\nDELETE C000005 无信息内容",
                "DELETE P000001 删除根帖",
            ],
        )
        assert results[2].startswith("OK DELETE C000005")
        assert _comment(core, "C000005")["status"] == "deleted"
        assert _meta(core, "P000001")["stats"]["comments"] == 1
        assert results[3].startswith("OK DELETE P000001")
        assert _meta(core, "P000001")["status"] == "deleted"

    def test_delete_branch_clears_promoted_to(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n根评论。",
                "AS D2\nREPLY C000004\n外化。",
                "AS A1\nDELETE P000005 冗余分叉",
            ],
        )
        assert results[2].startswith("OK DELETE P000005")
        assert _meta(core, "P000005")["status"] == "deleted"
        assert _comment(core, "C000004")["promoted_to"] == ""
        assert _meta(core, "P000001")["stats"]["branches"] == 0

    def test_tag_untag_and_solve(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "TAG P000001 调度",
                "TAG P000001 机制",
                "UNTAG P000001 机制",
                "AS A1\nSOLVE P000002 P000003 方案已记录",
                "POST P000002",
            ],
        )
        assert results[0].startswith("OK TAG P000001 #调度")
        assert results[1].startswith("OK TAG P000001 #机制")
        assert results[2].startswith("OK UNTAG P000001 #机制")
        assert _meta(core, "P000001")["tags"] == ["结构", "递归", "调度"]
        assert results[3].startswith("OK SOLVE P000002 ← P000003")
        assert _meta(core, "P000002")["status"] == "solved"
        assert _meta(core, "P000002")["solved_by"] == "P000003"
        assert "已解决：P000003" in results[4]
        assert "[问题·已解决]" in _meta(core, "P000002")["card"]

    def test_delete_requires_reason(self, tmp_path):
        _core, results = run_requests(tmp_path, ["AS A1\nDELETE P000001"])
        assert results[0].startswith("Error:") and "理由" in results[0]


# ----------------------------------------------------------------------
# 派生量维护与校验
# ----------------------------------------------------------------------


class TestMaintenance:
    def test_maint_repairs_corrupted_derived_fields(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D1\nCOMMENT P000001\n评论。",
                "__CORRUPT_CARD__ P000001",
                "MAINT",
                "CHECK",
                "POST P000001",
            ],
        )
        assert results[1].startswith("OK MAINT")
        assert "修复派生量 1 条" in results[1]
        meta = _meta(core, "P000001")
        assert "BROKEN" not in meta["card"]
        assert meta["stats"]["comments"] == 1
        assert results[2].startswith("OK CHECK 无违规")
        assert "统计：评论1 参与1" in results[3]

    def test_check_detects_corruption(self, tmp_path):
        _core, results = run_requests(
            tmp_path,
            [
                "__CORRUPT_CARD__ P000001",
                "CHECK",
            ],
        )
        assert results[0].startswith("CHECK 发现")
        assert "与重算不一致" in results[0]

    def test_quotes_follow_merge_alias(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D2\nQUOTE P000002 引用被合并的源帖\n正文一。",
                "AS A1\nMERGE P000001 P000002 重复问题合并",
                "MAINT",
                "POST P000004",
            ],
        )
        assert results[0].startswith("OK QUOTE P000004")
        assert results[2].startswith("OK MAINT")
        # 合并前指向源帖的引用，MAINT 后计数归到目标帖；墓碑帖不再计引用
        assert _meta(core, "P000001")["stats"]["quotes"] == 1
        assert _meta(core, "P000002")["stats"]["quotes"] == 0
        # 引用帖渲染时提示目标已并入
        assert "引用：帖子 P000002" in results[3]
        assert "（引用目标已并入 P000001）" in results[3]

        core2, results2 = run_requests(
            tmp_path,
            [
                "AS A1\nMERGE P000001 P000002 重复问题合并",
                "AS D3\nQUOTE P000002 引用旧 pid\n正文二。",
            ],
        )
        # 合并后引用旧 pid：QUOTE 沿别名解析，计数加到目标帖
        assert results2[1].startswith("OK QUOTE P000004")
        assert "引用 post P000001" in results2[1]
        assert _meta(core2, "P000001")["stats"]["quotes"] == 1

    def test_maint_recomputes_quotes_globally(self, tmp_path):
        core, results = run_requests(
            tmp_path,
            [
                "AS D2\nQUOTE P000001 引用一\n正文。",
                "AS D2\nQUOTE P000001 引用二\n正文。",
                "MAINT",
            ],
        )
        assert results[2].startswith("OK MAINT")
        assert _meta(core, "P000001")["stats"]["quotes"] == 2
        assert _forum_meta(core)["seq"] == 5


# ----------------------------------------------------------------------
# 面板路径
# ----------------------------------------------------------------------


class TestPanelAcceptance:
    """面板端到端脚本是独立可执行文件；这里只做冒烟（存在 + --help 可跑）。"""

    def test_acceptance_script_exists(self):
        script = COMMUNITY / "acceptance_panel.py"
        assert script.is_file()
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "面板验收" in proc.stdout

"""RepoDevice（AVMRepo 只读设备）测试：地址语法、引号还原、跟踪集合、只读与越界。

夹具用临时 git 仓库，不读本仓库当前状态；只有「缺省 root」一条用例落在本仓库上。
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from avm.core import Core, MemoryReadInstruction, MemoryWriteInstruction
from avm.exceptions import MemoryTypeError, VMMemoryError
from avm.image import load_image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_FILE = REPO_ROOT / "images" / "devices" / "avm_repo.py"
IMAGE_FILE = REPO_ROOT / "images" / "repo_tester.json"


def _git(cwd, *args):
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc


def _device_class():
    spec = importlib.util.spec_from_file_location("_test_avm_repo", DEVICE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RepoDevice


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("alpha\nneedle here\n", encoding="utf-8")
    (root / "pkg" / "sub" / "deep.txt").write_text("deep needle\n", encoding="utf-8")
    (root / "big.txt").write_text(
        "line 1\nline 2\nline 3 needle\nline 4\nline 5\n", encoding="utf-8"
    )
    (root / 'we"ird.txt').write_text("quoted name\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (root / "untracked.txt").write_text("not in git\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside needle\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    _git(root, "init", "-q")
    _git(root, "add", "pkg", "big.txt", 'we"ird.txt', "link.txt", ".gitignore")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    return root


@pytest.fixture
def device(repo):
    return _device_class()(root=str(repo), page_lines=2, hits_per_page=2)


def _core(device):
    core = Core()
    core.mem.mount("AVMRepo", device)
    device.mount_path = "AVMRepo"
    return core


class TestAddressGrammar:
    def test_root_returns_manual(self, device):
        text = _core(device).unwrap("$MEM.AVMRepo", for_llm=True)
        assert "跟踪文件=" in text and "HEAD=" in text
        assert '$MEM.AVMRepo.ls."<path>"' in text
        assert '$MEM.AVMRepo.read."<path>".<page>' in text
        assert '$MEM.AVMRepo.search."<expr>".<page>' in text
        assert "写入一律报错" in text

    def test_empty_remainder_returns_manual(self, device):
        assert "写入一律报错" in device.resolve_path([])

    def test_ls_bare_returns_help(self, device):
        out = _core(device).unwrap("$MEM.AVMRepo.ls", for_llm=True)
        assert out.startswith("ls：")
        assert '用法：$MEM.AVMRepo.ls."<path>"' in out

    def test_read_bare_returns_help(self, device):
        out = _core(device).unwrap("$MEM.AVMRepo.read", for_llm=True)
        assert out.startswith("read：")
        assert "page 缺省 0" in out

    def test_search_bare_returns_help(self, device):
        out = _core(device).unwrap("$MEM.AVMRepo.search", for_llm=True)
        assert out.startswith("search：")
        assert "大小写敏感" in out

    def test_ls_empty_path_lists_root(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.ls.""', for_llm=True)
        assert "d pkg" in out
        assert "f big.txt" in out
        assert "5 项" in out

    def test_ls_dot_path_is_root(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.ls."."', for_llm=True)
        assert out.startswith('ls ".": 5 项')

    def test_dot_leading_names(self, device):
        # 以点号开头的名字切出来是 ['"', 'gitignore"']，引号区间要照样接回去
        core = _core(device)
        assert core.unwrap('$MEM.AVMRepo.ls.".gitignore"', for_llm=True).startswith(
            'ls ".gitignore": 1 项'
        )
        assert "*.tmp" in core.unwrap('$MEM.AVMRepo.read.".gitignore".0', for_llm=True)
        assert ".gitignore:1:" in core.unwrap('$MEM.AVMRepo.search."*.tmp".0', for_llm=True)

    def test_ls_directory(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.ls."pkg"', for_llm=True)
        assert "d sub" in out and "f mod.py" in out
        assert "2 项" in out

    def test_ls_file_shows_one_entry(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.ls."big.txt"', for_llm=True)
        assert out.startswith('ls "big.txt": 1 项')
        assert "f big.txt" in out

    def test_read_pages(self, device):
        core = _core(device)
        first = core.unwrap('$MEM.AVMRepo.read."big.txt".0', for_llm=True)
        assert 'read "big.txt" page=0 lines=1-2/5' in first
        assert "line 1" in first and "line 2" in first
        assert '下一页: $MEM.AVMRepo.read."big.txt".1' in first
        second = core.unwrap('$MEM.AVMRepo.read."big.txt".1', for_llm=True)
        assert "line 3 needle" in second and "line 1\n" not in second
        last = core.unwrap('$MEM.AVMRepo.read."big.txt".2', for_llm=True)
        assert "lines=5-5/5" in last and "（末页）" in last

    def test_read_default_page_is_zero(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.read."pkg/mod.py"', for_llm=True)
        assert "page=0" in out and "alpha" in out and "（末页）" in out

    def test_page_out_of_range(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.read."big.txt".3', for_llm=True)

    def test_dotted_filename_roundtrip(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.read."pkg/mod.py".0', for_llm=True)
        assert "needle here" in out

    def test_escaped_quote_in_filename(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.read."we\\"ird.txt".0', for_llm=True)
        assert "quoted name" in out

    def test_unclosed_quote(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.ls."pkg', for_llm=True)

    def test_unquoted_path_rejected(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap("$MEM.AVMRepo.ls.pkg", for_llm=True)

    def test_unknown_verb(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.rm."big.txt"', for_llm=True)


class TestSearch:
    def test_pages(self, device):
        core = _core(device)
        first = core.unwrap('$MEM.AVMRepo.search."needle".0', for_llm=True)
        assert "hits=3" in first
        assert "big.txt:3:" in first and "pkg/mod.py:2:" in first
        assert '下一页: $MEM.AVMRepo.search."needle".1' in first
        second = core.unwrap('$MEM.AVMRepo.search."needle".1', for_llm=True)
        assert "pkg/sub/deep.txt:1:" in second and "（末页）" in second

    def test_no_hits(self, device):
        out = _core(device).unwrap('$MEM.AVMRepo.search."zzz".0', for_llm=True)
        assert "hits=0" in out and "（末页）" in out and "下一页" not in out

    def test_page_out_of_range(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.search."needle".2', for_llm=True)

    def test_empty_expression_rejected(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.search."".0', for_llm=True)

    def test_does_not_follow_symlink_out_of_root(self, device):
        # link.txt 指向仓库外的文件，其内容含 needle；搜索不跟着出去
        out = _core(device).unwrap('$MEM.AVMRepo.search."outside".0', for_llm=True)
        assert "hits=0" in out


class TestScope:
    def test_untracked_and_ignored_invisible(self, device):
        core = _core(device)
        root_ls = core.unwrap('$MEM.AVMRepo.ls.""', for_llm=True)
        assert "untracked.txt" not in root_ls and ".env" not in root_ls
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.AVMRepo.read."untracked.txt".0', for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.AVMRepo.read.".env".0', for_llm=True)

    def test_absolute_path_rejected(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.ls."/etc"', for_llm=True)

    def test_parent_segment_rejected(self, device):
        core = _core(device)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.AVMRepo.ls.".."', for_llm=True)
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.AVMRepo.read."pkg/../big.txt".0', for_llm=True)

    def test_symlink_escape_rejected(self, device):
        with pytest.raises(VMMemoryError):
            _core(device).unwrap('$MEM.AVMRepo.read."link.txt".0', for_llm=True)

    def test_paths_filter(self, repo):
        filtered = _device_class()(root=str(repo), paths=["pkg"])
        core = _core(filtered)
        root_ls = core.unwrap('$MEM.AVMRepo.ls.""', for_llm=True)
        assert "d pkg" in root_ls and "big.txt" not in root_ls
        with pytest.raises(VMMemoryError):
            core.unwrap('$MEM.AVMRepo.read."big.txt".0', for_llm=True)
        hits = core.unwrap('$MEM.AVMRepo.search."needle".0', for_llm=True)
        assert "hits=2" in hits and "big.txt" not in hits


class TestReadOnly:
    def test_set_raises_through_memory(self, device):
        core = _core(device)
        assert device.pretend_as_type() == "str"
        with pytest.raises(MemoryTypeError):
            core.mem.set("$MEM.AVMRepo", "value", owner_cid=0)
        with pytest.raises(MemoryTypeError):
            core.mem.set("$MEM.AVMRepo.x", "value", owner_cid=0)
        with pytest.raises(MemoryTypeError):
            core.mem.set('$MEM.AVMRepo.read."big.txt".0', "value", owner_cid=0)

    def test_memory_write_instruction_returns_error(self, device):
        core = _core(device)
        conv = core.start("sys", "user")
        for call_id, ref in (
            ("tc1", "$MEM.AVMRepo"),
            ("tc2", "$MEM.AVMRepo.x"),
            ("tc3", '$MEM.AVMRepo.read."big.txt".0'),
        ):
            MemoryWriteInstruction(call_id, conv.cid, {"ref": ref, "content": "v"}).execute(core, conv)
            assert conv.user_batch.tool_responses[-1].content.startswith("Error:")
        # 写失败不影响读
        assert "alpha" in core.unwrap('$MEM.AVMRepo.read."pkg/mod.py".0', for_llm=True)


class TestRepoImage:
    """出厂镜像 images/repo_tester.json：在用户指导下按 AVM 实际代码测整个系统；读文件接口从设备自述读出；镜像不规定步骤。"""

    def test_image_loads_device_and_terminal(self):
        core = load_image(Core(), str(IMAGE_FILE))
        conv = core.get_conversation(0)
        assert conv.name == "repo_tester" and conv.is_root
        assert {"AVMRepo", "inputs", "outputs"} <= set(core.mem._devices)
        assert "写入一律报错" in core.unwrap("$MEM.AVMRepo", for_llm=True)
        assert core.unwrap('$MEM.AVMRepo.ls."avm"', for_llm=True).startswith('ls "avm"')

    def test_entry_prompt_points_at_device_without_teaching_syntax(self):
        image = json.loads(IMAGE_FILE.read_text(encoding="utf-8"))
        system = image["mem"]["system"]["value"]["content"]["value"]
        user = image["mem"]["user"]["value"]
        prompt = system + "\n" + user
        # 身份与判据在 system 里：按实际代码测这个系统，测什么由用户指令决定
        assert "$MEM.AVMRepo" in prompt
        assert "实际代码" in prompt and "用户指令" in prompt
        # 镜像不规定步骤：步骤由运行时的用户指令给出，改起来不用动镜像
        assert "步骤" not in user and not re.search(r"\d\s*[.、]", user)
        assert "$MEM.inputs.-1" in user and "$MEM.outputs.-1" in user
        # 接口由设备自述给出：提示词里不写动作语法与分页规则
        for token in ("ls", 'read."', 'search."', "每页", "双引号", "页码"):
            assert token not in prompt

    def test_manual_reaches_conversation_through_memory_read(self):
        core = load_image(Core(), str(IMAGE_FILE))
        conv = core.get_conversation(0)
        MemoryReadInstruction("tc1", conv.cid, {"ref": "$MEM.AVMRepo"}).execute(core, conv)
        manual = conv.user_batch.tool_responses[-1].content
        assert '$MEM.AVMRepo.ls."<path>"' in manual
        MemoryReadInstruction("tc2", conv.cid, {"ref": '$MEM.AVMRepo.ls."avm"'}).execute(core, conv)
        assert conv.user_batch.tool_responses[-1].content.startswith('ls "avm"')
        MemoryWriteInstruction(
            "tc3", conv.cid, {"ref": "$MEM.outputs.-1", "content": "报到：开始测 AVMRepo"}
        ).execute(core, conv)
        assert core.mem._devices["outputs"]._data == ["报到：开始测 AVMRepo"]


class TestImage:
    def test_load_image_mounts_device(self, repo, tmp_path):
        image = {
            "meta": {"version": 1},
            "mem": {
                "AVMRepo": {"kind": "device"},
                "sys": {
                    "kind": "dict",
                    "ctrl": {"type": "settingup"},
                    "value": {"content": {"kind": "str", "value": "sys"}},
                },
                "user": {"kind": "str", "value": "开始"},
                "model_params": {
                    "kind": "dict",
                    "ctrl": {"type": "para"},
                    "value": {"model": {"kind": "str", "value": "deepseek-v4-flash"}},
                },
            },
            "devices": [
                {
                    "path": "AVMRepo",
                    "file": str(DEVICE_FILE),
                    "class": "RepoDevice",
                    "args": {"root": str(repo), "page_lines": 2},
                }
            ],
            "init": {"name": "init", "system_ref": "$MEM.sys", "user_ref": "$MEM.user"},
            "para_ref": "$MEM.model_params",
        }
        image_path = tmp_path / "image.json"
        image_path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
        core = load_image(Core(), str(image_path))
        assert "alpha" in core.unwrap('$MEM.AVMRepo.read."pkg/mod.py".0', for_llm=True)

    def test_non_repo_root_fails_at_load(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(VMMemoryError):
            _device_class()(root=str(plain))

    @pytest.mark.skipif(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
        ).returncode != 0,
        reason="本仓库不是 git 工作树",
    )
    def test_default_root_is_repo_root(self):
        device = _device_class()()
        assert "跟踪文件=" in device.to_llm_string()
        out = device.resolve_path(["ls", '""'])
        assert "d avm" in out and "f README.md" in out

    @pytest.mark.skipif(
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
        ).returncode != 0,
        reason="本仓库不是 git 工作树",
    )
    def test_manual_examples_work_on_real_repo(self):
        # 自述里的三个例子必须真的能跑通，否则测试对话会照着例子撞墙
        device = _device_class()()
        assert device.resolve_path(["ls", '"avm"']).startswith('ls "avm"')
        assert "class" in device.resolve_path(["read", '"avm/core.py"', "0"])
        # 只解析命中数：本测试自己的源码行也在跟踪集合里、也含有搜索词，子串判断会自己搜到自己
        first_line = device.resolve_path(["search", '"authorize_listener"', "0"]).splitlines()[0]
        assert int(first_line.split("hits=")[1]) > 0

"""AVMRepo 设备：把仓库的跟踪文件作为只读设备挂给对话。

地址语法（设备内部空间，与内存树的点号路径无关）：

    $MEM.AVMRepo                            设备说明书（根摘要 + 三个动作的语法）
    $MEM.AVMRepo.ls."<path>"                列目录；"" 与 "." 是仓库根
    $MEM.AVMRepo.read."<path>".<page>       按页读文件；page 缺省 0
    $MEM.AVMRepo.search."<expr>".<page>     在跟踪文件里找子串；page 缺省 0
    $MEM.AVMRepo.ls / .read / .search       不带参数读一个动作，得到该动作的用法说明

path 用双引号包裹，按 Linux 格式（/ 分隔），内部的双引号写 \\"，反斜杠写 \\\\。
引号是为了让带点号的文件名活下来：Core 把整条引用按点号切分（`avm/core.py` 的 `Core.unwrap`），
`"avm/core.py"` 到设备时已经是 ['"avm/core', 'py"'] 两段；设备把引号区间内的段用点号接回
原字符串，再做路径解析。

本文件不提供镜像。镜像里的声明形状：

    { "path": "AVMRepo", "file": "devices/avm_repo.py", "class": "RepoDevice", "args": {} }
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

from avm.exceptions import MemoryTypeError, VMMemoryError
from avm.memory_device import MemoryDevice


def _quote(text: str) -> str:
    """把路径或表达式包成设备语法的双引号形式（内部反斜杠与双引号转义）。"""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ends_with_unescaped_quote(text: str) -> bool:
    # 长度 1 的那个引号是开引号本身，还没闭合（以点号开头的路径会切出这种段）
    if len(text) < 2 or not text.endswith('"'):
        return False
    backslashes = 0
    index = len(text) - 2
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 0


def _unescape(text: str) -> str:
    out: List[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "\\" and index + 1 < len(text) and text[index + 1] in ('"', "\\"):
            out.append(text[index + 1])
            index += 2
            continue
        out.append(ch)
        index += 1
    return "".join(out)


def _take_quoted(segments: List[str]) -> Tuple[str, List[str]]:
    """还原引号区间：返回（去掉引号的字符串，引号之后的段）。

    Core 按点号切分引用，`"a/b.py"` 到设备时是 ['"a/b', 'py"']；区间内的段用点号接回。
    以点号开头的名字（`.env`、`.agents`）切出来是 ['"', 'agents"']，第一个段只有开引号，
    长度 1 不算闭合，因此照样接得回去。首个参数段没有引号、或引号未闭合，都抛 VMMemoryError。
    """
    if not segments or not segments[0].startswith('"'):
        raise VMMemoryError("AVMRepo 的路径参数必须用双引号包裹")
    buf = segments[0]
    index = 1
    while not _ends_with_unescaped_quote(buf):
        if index >= len(segments):
            raise VMMemoryError("AVMRepo 的路径参数引号未闭合")
        buf = buf + "." + segments[index]
        index += 1
    return _unescape(buf[1:-1]), segments[index:]


class RepoDevice(MemoryDevice):
    """仓库的只读设备。

    - 暴露集合是 git 跟踪集合（每次调用现取 `git ls-files`）；`paths` 给一组仓库相对前缀
      可以收窄，空表示全部。未跟踪与 .gitignore 忽略的文件不在地址空间内，`.env`、
      `docs/private/` 这类内容因此读不到。
    - 只读：三个写入落点都抛 `MemoryTypeError`——整路径写入走 `set_value`，子路径写入在
      `Memory.set_by_path` 里先走 `__getitem__`（`avm/memory.py`）。设备不发布事件，
      不参与监听授权（`authorize_listener` 保持基类的拒绝）。
    - 越界拒绝：绝对路径、含 `..` 段的路径、解析后落在仓库外的符号链接，都抛 `VMMemoryError`。
    - 装载期校验：root 存在且是 git 工作树，否则抛 `VMMemoryError`；镜像启动会以
      `[AVM] 镜像错误: ...` 退出。
    - 返回粒度由三个参数定：`page_lines`（read 每页行数）、`hits_per_page`（search 每页条数）、
      `max_line_chars`（search 命中行的截断长度）。
    - 设备自述：读设备本身（`$MEM.AVMRepo`）得到说明书（`to_llm_string()`）；读一个不带参数的
      动作（`$MEM.AVMRepo.ls` / `.read` / `.search`）得到该动作的用法说明。缺参数按"在找怎么用"
      处理，报错留给出错的参数。

    参数：
        root：仓库根，缺省取本设备文件的上两级目录（images/devices/ → 仓库根）。
        paths：仓库相对前缀列表，空表示全部跟踪文件。
    """

    def __init__(
        self,
        root: Optional[str] = None,
        paths: Optional[List[str]] = None,
        page_lines: int = 200,
        hits_per_page: int = 20,
        max_line_chars: int = 200,
    ):
        base = Path(__file__).resolve().parents[2] if root is None else Path(root)
        self._root = base.expanduser().resolve()
        if not self._root.is_dir():
            raise VMMemoryError(f"AVMRepo 根目录不存在：{self._root}")
        self._paths = tuple(str(p).strip("/") for p in (paths or []) if str(p).strip("/"))
        self._page_lines = max(1, int(page_lines))
        self._hits_per_page = max(1, int(hits_per_page))
        self._max_line_chars = max(1, int(max_line_chars))
        if self._git("rev-parse", "--show-toplevel").returncode != 0:
            raise VMMemoryError(f"AVMRepo 根目录不是 git 工作树：{self._root}")

    # ------------------------------------------------------------------
    # git 与跟踪集合
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> "subprocess.CompletedProcess":
        try:
            return subprocess.run(
                ["git", "-C", str(self._root), *args],
                capture_output=True,
                text=True,
                errors="replace",
            )
        except FileNotFoundError:
            raise VMMemoryError("AVMRepo 需要 git 可执行文件，宿主上找不到 git") from None

    def _tracked(self) -> List[str]:
        """当前跟踪文件（仓库相对路径，/ 分隔）。每次调用现取，运行中新跟踪的文件可见。"""
        proc = self._git("ls-files", "-z")
        if proc.returncode != 0:
            raise VMMemoryError(f"AVMRepo 读取跟踪集合失败：{proc.stderr.strip()}")
        files = [name for name in proc.stdout.split("\0") if name]
        if not self._paths:
            return files
        return [f for f in files if self._exposed(f)]

    def _exposed(self, rel: str) -> bool:
        return not self._paths or any(rel == p or rel.startswith(p + "/") for p in self._paths)

    # ------------------------------------------------------------------
    # 路径解析与内容读取
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Tuple[Path, str]:
        """把设备空间的 Linux 路径解析成（绝对路径，仓库相对路径）。越界与未暴露都报错。"""
        clean = path.strip()
        if clean.startswith("/"):
            raise VMMemoryError(f"AVMRepo 只接受仓库内相对路径：{path!r}")
        rel = ""
        if clean not in ("", "."):
            parts = [p for p in clean.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts):
                raise VMMemoryError(f"AVMRepo 路径不允许 .. 段：{path!r}")
            rel = "/".join(parts)
            if not self._exposed(rel):
                raise VMMemoryError(f"AVMRepo 未暴露该路径：{path!r}")
        target = (self._root / rel).resolve() if rel else self._root
        try:
            target.relative_to(self._root)
        except ValueError:
            raise VMMemoryError(f"AVMRepo 路径越出仓库：{path!r}") from None
        return target, rel

    def _size(self, path: Path) -> Optional[int]:
        if not self._inside(path):
            return None
        try:
            return path.stat().st_size
        except OSError:
            return None

    def _inside(self, path: Path) -> bool:
        """解析符号链接之后仍在仓库内。读内容与取大小都先过这一关。"""
        try:
            path.resolve().relative_to(self._root)
            return True
        except (OSError, ValueError):
            return False

    def _page(self, extra: List[str]) -> int:
        if not extra:
            return 0
        if len(extra) > 1:
            raise VMMemoryError("AVMRepo 的页码只能是单个整数")
        try:
            page = int(extra[0])
        except ValueError:
            raise VMMemoryError(f"AVMRepo 的页码必须是整数：{extra[0]!r}") from None
        if page < 0:
            raise VMMemoryError(f"AVMRepo 的页码不能为负：{page}")
        return page

    def _ls(self, segments: List[str]) -> str:
        if not segments:
            return self._help("ls")
        path, extra = _take_quoted(segments)
        if extra:
            raise VMMemoryError("AVMRepo.ls 只接受一个路径参数")
        target, rel = self._resolve(path)
        tracked = self._tracked()
        shown_path = path or "."
        if target.is_file():
            if rel not in tracked:
                raise VMMemoryError(f"AVMRepo 只暴露跟踪文件：{path!r}")
            return f"ls {_quote(shown_path)}: 1 项（跟踪集合）\nf {rel} {self._size(target)}B"
        if not target.is_dir():
            raise VMMemoryError(f"AVMRepo 路径不存在：{path!r}")
        prefix = rel + "/" if rel else ""
        dirs = set()
        files: List[str] = []
        for name in tracked:
            if prefix and not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if not rest:
                continue
            if "/" in rest:
                dirs.add(rest.split("/", 1)[0])
            else:
                files.append(rest)
        lines = [f"ls {_quote(shown_path)}: {len(dirs) + len(files)} 项（跟踪集合）"]
        for name in sorted(dirs):
            lines.append(f"d {name}")
        for name in sorted(files):
            lines.append(f"f {name} {self._size(self._root / prefix / name)}B")
        return "\n".join(lines)

    def _read(self, segments: List[str]) -> str:
        if not segments:
            return self._help("read")
        path, extra = _take_quoted(segments)
        page = self._page(extra)
        target, rel = self._resolve(path)
        if rel not in self._tracked():
            raise VMMemoryError(f"AVMRepo 只暴露跟踪文件：{path!r}")
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as e:
            raise VMMemoryError(f"AVMRepo 读文件失败：{path!r}: {e}")
        except UnicodeDecodeError:
            raise VMMemoryError(f"AVMRepo 不按行读非 UTF-8 文件：{path!r}") from None
        lines = text.splitlines()
        total_lines = len(lines)
        total_pages = max(1, (total_lines + self._page_lines - 1) // self._page_lines)
        if page >= total_pages:
            raise VMMemoryError(f"AVMRepo 页超出范围：{path!r} 共 {total_pages} 页")
        start = page * self._page_lines
        end = min(start + self._page_lines, total_lines)
        span = f"{start + 1}-{end}" if total_lines else "0"
        header = (
            f"read {_quote(path)} page={page} lines={span}/{total_lines} "
            f"bytes={self._size(target)}"
        )
        body = "\n".join(lines[start:end])
        tail = (
            "（末页）"
            if page + 1 >= total_pages
            else f"下一页: $MEM.AVMRepo.read.{_quote(path)}.{page + 1}"
        )
        return f"{header}\n{body}\n{tail}"

    def _search(self, segments: List[str]) -> str:
        if not segments:
            return self._help("search")
        expr, extra = _take_quoted(segments)
        page = self._page(extra)
        if expr == "":
            raise VMMemoryError("AVMRepo.search 的表达式不能为空")
        hits: List[Tuple[str, int, str]] = []
        for rel in self._tracked():
            target = self._root / rel
            if not self._inside(target):
                # 指向仓库外的符号链接：搜索不跟着出去
                continue
            try:
                text = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if expr in line:
                    shown = line.strip()
                    if len(shown) > self._max_line_chars:
                        shown = shown[: self._max_line_chars] + "…"
                    hits.append((rel, lineno, shown))
        total = len(hits)
        start = page * self._hits_per_page
        if total and start >= total:
            raise VMMemoryError(f"AVMRepo 页超出范围：共 {total} 条命中")
        window = hits[start:start + self._hits_per_page]
        lines = [f"search {_quote(expr)} page={page} hits={total}"]
        for rel, lineno, shown in window:
            lines.append(f"{rel}:{lineno}: {shown}")
        if not total:
            lines.append("（末页）")
            return "\n".join(lines)
        if start + self._hits_per_page < total:
            lines.append(f"下一页: $MEM.AVMRepo.search.{_quote(expr)}.{page + 1}")
        else:
            lines.append("（末页）")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # MemoryDevice 接口
    # ------------------------------------------------------------------

    def pretend_as_type(self) -> str:
        return "str"

    def to_llm_string(self) -> str:
        """设备说明书：读设备本身（`$MEM.AVMRepo`）看到的就是它。"""
        tracked = self._tracked()
        total_bytes = sum(self._size(self._root / rel) or 0 for rel in tracked)
        head = self._git("rev-parse", "--short", "HEAD").stdout.strip() or "无提交"
        dirty = "yes" if self._git("status", "--porcelain").stdout.strip() else "no"
        return (
            f"AVMRepo[root={self._root}, HEAD={head}, dirty={dirty}, "
            f"跟踪文件={len(tracked)}, 字节={total_bytes}]\n"
            "只读设备：暴露仓库的 git 跟踪集合，路径按 Linux 格式写在双引号里"
            '（内部双引号写 \\"），与内存树的点号路径无关。\n'
            '  $MEM.AVMRepo.ls."<path>"                列目录；"" 与 "." 是仓库根\n'
            f'  $MEM.AVMRepo.read."<path>".<page>       按页读文件，每页 {self._page_lines} 行，'
            "page 缺省 0\n"
            f'  $MEM.AVMRepo.search."<expr>".<page>     找子串，每页 {self._hits_per_page} 条，'
            "page 缺省 0\n"
            "读 $MEM.AVMRepo.<动作>（不带参数）得到该动作的用法说明；写入一律报错。"
        )

    def _help(self, verb: str) -> str:
        """动作的用法说明：缺少必需参数的读取返回它，不报错。

        参数齐全的动作照常执行（`read."<path>"` 的页码有缺省值，算齐全）。参数写错仍报错。
        """
        if verb == "ls":
            return (
                "ls：列目录（只看 git 跟踪集合）\n"
                '用法：$MEM.AVMRepo.ls."<path>"\n'
                '  "" 与 "." 是仓库根；例：$MEM.AVMRepo.ls."avm"\n'
                "返回：首行是条目数，随后每行 d <目录> 或 f <文件> <字节>B"
            )
        if verb == "read":
            return (
                "read：按页读文件\n"
                '用法：$MEM.AVMRepo.read."<path>".<page>\n'
                f"  page 缺省 0，每页 {self._page_lines} 行；"
                '例：$MEM.AVMRepo.read."avm/core.py".0\n'
                "返回：首行给出页范围与总行数，正文之后是下一页地址或「（末页）」"
            )
        if verb == "search":
            return (
                "search：在跟踪文件里找子串（大小写敏感）\n"
                '用法：$MEM.AVMRepo.search."<子串>".<page>\n'
                f"  page 缺省 0，每页 {self._hits_per_page} 条；"
                '例：$MEM.AVMRepo.search."authorize_listener".0\n'
                "返回：首行是命中总数，随后每行 路径:行号: 行内容，末行是下一页地址或「（末页）」"
            )
        raise VMMemoryError(
            f"AVMRepo 未知动词：{verb!r}（可用 ls / read / search；读 $MEM.AVMRepo 看说明书）"
        )

    def resolve_path(self, path: List[str]) -> Any:
        if not path:
            return self.to_llm_string()
        verb, rest = path[0], path[1:]
        if verb == "ls":
            return self._ls(rest)
        if verb == "read":
            return self._read(rest)
        if verb == "search":
            return self._search(rest)
        return self._help(verb)

    # 只读：Memory.set_by_path 对整路径走 set_value，对子路径先走 __getitem__ 再走
    # __setitem__（avm/memory.py）。三处都挡住，写入得到 Error 工具响应，内核不受影响。

    def set_value(self, value: Any) -> None:
        raise MemoryTypeError("AVMRepo 是只读设备，不允许写入")

    def __getitem__(self, key: str) -> Any:
        raise MemoryTypeError("AVMRepo 是只读设备，不允许写入")

    def __setitem__(self, key: str, value: Any) -> None:
        raise MemoryTypeError("AVMRepo 是只读设备，不允许写入")

"""PackageRepoDevice：把宿主上的一个包目录只读地接入 AVM。

地址语法（设备内部空间，与内存树的点号路径无关）：

    $MEM.pkgrepo                       设备说明书（装载状态 + 读接口语法）
    $MEM.pkgrepo.manifest              manifest.json 原文
    $MEM.pkgrepo.data                  data.json 原文
    $MEM.pkgrepo.installer             install.py 原文（包没有该文件时报错）
    $MEM.pkgrepo.files                 文件清单：每行 `<相对路径> <字节> <sha256>`
    $MEM.pkgrepo.read."<相对路径>"      读包内已列文件全文

写接口只在根地址：向 `$MEM.pkgrepo` 写入**一行相对路径**（相对本设备的 root）装载包；
空串或纯空白清除当前装载。相同路径再次写入会重新验证并替换。先验证后替换：写入失败
时上一次装载的包保持不变。

路径参数用双引号包裹，按 Linux 格式（/ 分隔），内部的双引号写 \\"，反斜杠写 \\\\。
引号是为了让带点号的文件名活下来：Core 把整条引用按点号切分（`avm/core.py` 的
`Core.unwrap`），`"devices/echo.py"` 到设备时已经是 ['"devices/echo', 'py"'] 两段；
设备把引号区间内的段用点号接回原字符串，再做查找。

包格式是 **OS 层约定，不是 VM 规则**；VM 不解释包、不校验包布局，本设备只按这个约定
验证：

    <包目录>/manifest.json    UTF-8 JSON 对象，至少含非空字符串 name、package_version；
                              其它字段原样保留
    <包目录>/data.json        UTF-8 JSON，显式节点格式且根节点 kind="dict"
    <包目录>/install.py       可选；存在时必须是可读 UTF-8 文本
    <包目录>/devices/*.py     可选；目录存在时，直接子级的 *.py 是设备代码文本

设备只把上述文件作为**文件**暴露：绝不 import、绝不执行 install.py 或 devices/*.py。
文件大小超过 `max_file_bytes` 的任何包文件都会让装载失败。装载成功后设备只缓存清单
（相对路径、大小、sha256）与文件文本，不写事件、不产生内存节点。

安全约束：

- 写接口只接受 root 内的相对路径：拒绝绝对路径、含 `..` 段的路径、解析后（含符号
  链接）越出 root 的路径；目标必须是可读目录。
- 包内文件解析后必须仍在包目录内（符号链接不得把宿主文件带进包）。
- 读接口只列/只读装载时已验证过的文件；子路径写入一律 MemoryTypeError。
- 错误信息不回显文件内容；路径类错误可以带上操作员写入或请求的路径。

本文件不提供镜像。镜像里的声明形状：

    {"path": "pkgrepo", "file": "devices/pkg_repo.py", "class": "PackageRepoDevice", "args": {}}
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from avm.exceptions import MemoryTypeError, VMMemoryError
from avm.memory import Memory
from avm.memory_device import MemoryDevice
from avm.types import MetaDict

_UNLOADED = "pkgrepo 尚未装载包：先向 $MEM.pkgrepo 写入一行相对路径"


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

    Core 按点号切分引用，`"devices/echo.py"` 到设备时是 ['"devices/echo', 'py"']；
    区间内的段用点号接回。首个参数段没有引号、或引号未闭合，都抛 VMMemoryError。
    """
    if not segments or not segments[0].startswith('"'):
        raise VMMemoryError("pkgrepo 的路径参数必须用双引号包裹")
    buf = segments[0]
    index = 1
    while not _ends_with_unescaped_quote(buf):
        if index >= len(segments):
            raise VMMemoryError("pkgrepo 的路径参数引号未闭合")
        buf = buf + "." + segments[index]
        index += 1
    return _unescape(buf[1:-1]), segments[index:]


class _LoadedPackage:
    """一次成功装载的只读快照；验证完成前不碰设备状态。"""

    def __init__(self, rel: str, directory: Path):
        self.rel = rel
        self.directory = directory
        self.manifest: Dict[str, Any] = {}
        self.manifest_text = ""
        self.data_text = ""
        self.install_text: Optional[str] = None
        self.order: List[str] = []
        self.entries: Dict[str, Tuple[int, str, str]] = {}  # 相对路径 → (字节, sha256, 文本)

    def add(self, rel: str, size: int, sha256: str, text: str) -> None:
        self.order.append(rel)
        self.entries[rel] = (size, sha256, text)

    def total_bytes(self) -> int:
        return sum(size for size, _sha, _text in self.entries.values())

    def files_text(self) -> str:
        return "\n".join(
            f"{rel} {self.entries[rel][0]} {self.entries[rel][1]}" for rel in self.order
        )


class PackageRepoDevice(MemoryDevice):
    """宿主包目录的只读接入设备（OS 层包约定，不提供镜像）。

    - **写接口（根地址）**：`set_value(value)` 接受一行相对 root 的目录路径；空串或纯
      空白清除装载。装载先验证路径安全性，再按包格式（manifest.json / data.json /
      install.py / devices/*.py）验证内容，最后**原子替换**已装载快照——失败时旧包不变。
    - **读接口（子路径）**：`.manifest` / `.data` / `.installer` / `.files` /
      `.read."<相对路径>"`；读设备本身得到说明书。未装载时子路径一律报错。
    - **只读**：根地址的写入是"装载命令"，不是数据写入；子路径写入在 `__getitem__` /
      `__setitem__` 抛 `MemoryTypeError`（`Memory.set_by_path` 先走这两个钩子）。
    - **不发事件、不产生节点**：装载只改设备内部缓存；设备不实现 `after_event` /
      `emit_device_event`，也不 import/执行包里的 Python。
    - **安全性**：拒绝绝对路径、`..` 段、解析后（含符号链接）越出 root 的路径；包内
      文件也不得通过符号链接越出包目录；任何文件超过 `max_file_bytes` 拒绝装载。
    - **装载期校验**：root 必须是存在的目录，否则抛 `VMMemoryError`；镜像启动会以
      `[AVM] 镜像错误: ...` 退出。

    参数：
        root：包仓库根目录，缺省取本设备文件的上两级目录（images/devices/ → 仓库根）。
        max_file_bytes：单文件读取上限，缺省 1000000 字节。
    """

    def __init__(self, root: Optional[str] = None, max_file_bytes: Any = 1000000):
        base = Path(__file__).resolve().parents[2] if root is None else root
        try:
            resolved = Path(base).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            raise VMMemoryError(f"pkgrepo 的 root 无效：{root!r}（{type(e).__name__}）") from None
        if not resolved.is_dir():
            raise VMMemoryError(f"pkgrepo 根目录不存在：{resolved}")
        self._root = resolved
        try:
            limit = int(max_file_bytes)
        except (TypeError, ValueError):
            raise VMMemoryError(
                f"pkgrepo 的 max_file_bytes 必须是正整数，得到 {max_file_bytes!r}"
            ) from None
        if limit <= 0:
            raise VMMemoryError(f"pkgrepo 的 max_file_bytes 必须是正整数，得到 {max_file_bytes!r}")
        self._max_file_bytes = limit
        self._pkg: Optional[_LoadedPackage] = None

    # ------------------------------------------------------------------
    # 写入 = 装载/替换/清除
    # ------------------------------------------------------------------

    def set_value(self, value: Any) -> None:
        """装载一行相对路径；空串/纯空白清除；失败时保持上一次装载。"""
        if not isinstance(value, str):
            raise VMMemoryError(
                f"pkgrepo 的写入必须是包目录路径字符串，得到 {type(value).__name__}"
            )
        text = value.strip()
        if text == "":
            self._pkg = None
            return None
        if "\n" in text or "\r" in text:
            raise VMMemoryError("pkgrepo 的写入必须是一行路径（不能含换行）")
        target, rel = self._resolve(text)
        pkg = self._load_package(target, rel)
        self._pkg = pkg  # 验证全部通过才替换：失败不影响旧包
        return None

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------

    def _resolve(self, raw: str) -> Tuple[Path, str]:
        """把写入路径解析成（绝对目录，root 相对路径）。越界/非目录/不可读都报错。"""
        shown = raw.strip()
        normalized = shown.replace("\\", "/")
        if Path(shown).is_absolute() or normalized.startswith("/"):
            raise VMMemoryError(f"pkgrepo 只接受 root 内相对路径：{shown!r}")
        parts = [p for p in normalized.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise VMMemoryError(f"pkgrepo 路径不允许 .. 段：{shown!r}")
        rel = "/".join(parts)
        try:
            target = (self._root / Path(*parts)).resolve() if parts else self._root
        except OSError as e:
            raise VMMemoryError(f"pkgrepo 无法解析路径 {shown!r}：{type(e).__name__}") from None
        try:
            target.relative_to(self._root)
        except ValueError:
            raise VMMemoryError(f"pkgrepo 路径解析后越出 root：{shown!r}") from None
        if not target.is_dir():
            if target.exists():
                raise VMMemoryError(f"pkgrepo 路径不是目录：{shown!r}")
            raise VMMemoryError(f"pkgrepo 包目录不存在：{shown!r}")
        if not os.access(target, os.R_OK | os.X_OK):
            raise VMMemoryError(f"pkgrepo 包目录不可读：{shown!r}")
        rel = target.relative_to(self._root).as_posix() or "."
        return target, rel

    # ------------------------------------------------------------------
    # 包格式验证（OS 层约定）
    # ------------------------------------------------------------------

    def _load_package(self, directory: Path, rel: str) -> _LoadedPackage:
        pkg = _LoadedPackage(rel, directory)

        manifest_path = self._package_file(directory, rel, "manifest.json", required=True)
        data_path = self._package_file(directory, rel, "data.json", required=True)
        manifest_data, manifest_text = self._read_package_file(manifest_path, rel, "manifest.json")
        data_data, data_text = self._read_package_file(data_path, rel, "data.json")
        manifest = self._parse_manifest(manifest_text, rel)
        self._validate_data(data_text, rel)
        pkg.manifest = manifest
        pkg.manifest_text = manifest_text
        pkg.data_text = data_text
        pkg.add("manifest.json", len(manifest_data), _sha256(manifest_data), manifest_text)
        pkg.add("data.json", len(data_data), _sha256(data_data), data_text)

        install_path = self._package_file(directory, rel, "install.py", required=False)
        if install_path is not None:
            install_data, install_text = self._read_package_file(install_path, rel, "install.py")
            pkg.install_text = install_text
            pkg.add("install.py", len(install_data), _sha256(install_data), install_text)

        devices_dir = self._devices_dir(directory, rel)
        if devices_dir is not None:
            try:
                names = sorted(child.name for child in devices_dir.iterdir()
                               if child.name.endswith(".py"))
            except OSError as e:
                raise VMMemoryError(
                    f"pkgrepo 无法列出包 {rel!r} 的 devices/：{type(e).__name__}"
                ) from None
            for name in names:
                child = devices_dir / name
                child_rel = f"devices/{name}"
                if not child.is_file():
                    raise VMMemoryError(f"pkgrepo 包文件不是普通文件：{child_rel!r}")
                target = self._inside_package(child, directory, rel, child_rel)
                data, text = self._read_package_file(target, rel, child_rel)
                pkg.add(child_rel, len(data), _sha256(data), text)
        return pkg

    def _package_file(self, directory: Path, rel: str, name: str,
                      *, required: bool) -> Optional[Path]:
        """取一个包文件；不存在按 required 决定返回 None 或报错，存在则做越界检查。"""
        path = directory / name
        if not path.exists() and not path.is_symlink():
            if required:
                raise VMMemoryError(f"pkgrepo 包 {rel!r} 缺少 {name}")
            return None
        if not path.is_file():
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 {name} 不是普通文件")
        return self._inside_package(path, directory, rel, name)

    def _devices_dir(self, directory: Path, rel: str) -> Optional[Path]:
        path = directory / "devices"
        if not path.exists() and not path.is_symlink():
            return None
        if not path.is_dir():
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 devices 不是目录")
        return self._inside_package(path, directory, rel, "devices")

    def _inside_package(self, path: Path, directory: Path, rel: str, label: str) -> Path:
        """解析符号链接后必须仍在包目录内（包目录已在 root 内）。"""
        try:
            target = path.resolve()
        except OSError as e:
            raise VMMemoryError(
                f"pkgrepo 无法解析包 {rel!r} 的 {label}：{type(e).__name__}"
            ) from None
        try:
            target.relative_to(directory)
        except ValueError:
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 {label} 解析后越出包目录"
            ) from None
        return target

    def _read_package_file(self, path: Path, rel: str, label: str) -> Tuple[bytes, str]:
        """读包文件（带大小上限与 UTF-8 校验）；错误只给类别，不回显内容。"""
        try:
            size = path.stat().st_size
        except OSError as e:
            raise VMMemoryError(
                f"pkgrepo 读取包 {rel!r} 的 {label} 失败：{type(e).__name__}"
            ) from None
        if size > self._max_file_bytes:
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 {label} 超过大小上限 "
                f"{self._max_file_bytes} 字节（{size} 字节）"
            )
        try:
            data = path.read_bytes()
        except OSError as e:
            raise VMMemoryError(
                f"pkgrepo 读取包 {rel!r} 的 {label} 失败：{type(e).__name__}"
            ) from None
        if len(data) > self._max_file_bytes:
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 {label} 超过大小上限 "
                f"{self._max_file_bytes} 字节（{len(data)} 字节）"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 {label} 不是 UTF-8 文本") from None
        return data, text

    @staticmethod
    def _parse_manifest(text: str, rel: str) -> Dict[str, Any]:
        try:
            manifest = json.loads(text)
        except (ValueError, RecursionError):
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 manifest.json 不是合法 JSON") from None
        if not isinstance(manifest, dict):
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 manifest.json 必须是 JSON 对象")
        for field in ("name", "package_version"):
            value = manifest.get(field)
            if not isinstance(value, str) or not value.strip():
                raise VMMemoryError(
                    f"pkgrepo 包 {rel!r} 的 manifest.json 缺少非空字符串 {field}"
                )
        return dict(manifest)

    @staticmethod
    def _validate_data(text: str, rel: str) -> None:
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            raise VMMemoryError(f"pkgrepo 包 {rel!r} 的 data.json 不是合法 JSON") from None
        if not isinstance(data, dict) or data.get("kind") != "dict":
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 data.json 根节点必须是显式 dict 节点（kind=\"dict\"）"
            )
        try:
            PackageRepoDevice._check_explicit_node(data)
            node = Memory.deserialize_node(data)
        except Exception as e:  # 节点树里任何非法形状都拒绝；异常文本可能带内容，不回显
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 data.json 不是合法的显式节点格式"
                f"（{type(e).__name__}）"
            ) from None
        if not isinstance(node, MetaDict):
            raise VMMemoryError(
                f"pkgrepo 包 {rel!r} 的 data.json 根节点必须是显式 dict 节点（kind=\"dict\"）"
            )

    @classmethod
    def _check_explicit_node(cls, value: Any, depth: int = 0) -> None:
        """显式节点格式的结构检查：每个值都必须是带合法 kind 的节点。

        `Memory.deserialize_node` 对裸值（字符串/数字）是宽容的；data.json 的约定是
        "整棵树都是显式节点"，所以这里再走一遍形状检查，避免把隐式数据当成包。
        """
        if depth > 64:
            raise VMMemoryError("节点嵌套超过 64 层")
        if not isinstance(value, dict):
            raise VMMemoryError("节点必须是对象")
        kind = value.get("kind")
        if kind == "str":
            if not isinstance(value.get("value"), str):
                raise VMMemoryError("str 节点的 value 必须是字符串")
            return
        if kind == "link":
            target = value.get("target")
            if not isinstance(target, str) or not target:
                raise VMMemoryError("link 节点的 target 必须是非空字符串")
            return
        if kind == "dict":
            children = value.get("value")
            if not isinstance(children, dict):
                raise VMMemoryError("dict 节点的 value 必须是对象")
            for child in children.values():
                cls._check_explicit_node(child, depth + 1)
            return
        if kind == "list":
            children = value.get("value")
            if not isinstance(children, list):
                raise VMMemoryError("list 节点的 value 必须是数组")
            for child in children:
                cls._check_explicit_node(child, depth + 1)
            return
        raise VMMemoryError(f"未知节点 kind：{kind!r}")

    # ------------------------------------------------------------------
    # 读接口
    # ------------------------------------------------------------------

    def _require_loaded(self) -> _LoadedPackage:
        if self._pkg is None:
            raise VMMemoryError(_UNLOADED)
        return self._pkg

    def _read(self, segments: List[str]) -> str:
        if not segments:
            raise VMMemoryError('pkgrepo.read 用法：$MEM.pkgrepo.read."<相对路径>"')
        path, extra = _take_quoted(segments)
        if extra:
            raise VMMemoryError("pkgrepo.read 只接受一个路径参数")
        rel = self._meaningful(path)
        pkg = self._require_loaded()
        if rel not in pkg.entries:
            raise VMMemoryError(f"pkgrepo 包内没有该文件：{path!r}")
        return pkg.entries[rel][2]

    @staticmethod
    def _meaningful(path: str) -> str:
        """读路径的轻量规范化：相对、无 ..；只用于查找已列文件。"""
        shown = path.strip()
        normalized = shown.replace("\\", "/")
        if Path(shown).is_absolute() or normalized.startswith("/"):
            raise VMMemoryError(f"pkgrepo.read 只接受包内相对路径：{shown!r}")
        parts = [p for p in normalized.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise VMMemoryError(f"pkgrepo.read 路径不允许 .. 段：{shown!r}")
        return "/".join(parts)

    # ------------------------------------------------------------------
    # MemoryDevice 接口
    # ------------------------------------------------------------------

    def pretend_as_type(self) -> str:
        return "str"

    def to_llm_string(self) -> str:
        """设备说明书：读设备本身（`$MEM.pkgrepo`）看到的就是它。"""
        pkg = self._pkg
        head = f"pkgrepo[root={self._root}, max_file_bytes={self._max_file_bytes}]"
        if pkg is None:
            state = "未装载包：向 $MEM.pkgrepo 写入一行相对路径装载；空串或纯空白清除。"
        else:
            state = (
                f"已装载包：path={pkg.rel}, name={pkg.manifest.get('name')}, "
                f"package_version={pkg.manifest.get('package_version')}, "
                f"文件={len(pkg.order)}, 字节={pkg.total_bytes()}"
            )
        return (
            f"{head}\n"
            f"{state}\n"
            "只读设备：包格式是 OS 约定（VM 不校验）；路径按 Linux 格式写在双引号里"
            '（内部双引号写 \\"），与内存树的点号路径无关。\n'
            "  $MEM.pkgrepo.manifest              manifest.json 原文\n"
            "  $MEM.pkgrepo.data                  data.json 原文\n"
            "  $MEM.pkgrepo.installer             install.py 原文（包没有该文件时报错）\n"
            '  $MEM.pkgrepo.files                 文件清单：每行 "<相对路径> <字节> <sha256>"\n'
            '  $MEM.pkgrepo.read."<相对路径>"      读包内已列文件全文\n'
            "装载：memory_write $MEM.pkgrepo \"<一行相对路径>\"（相对 root；"
            "绝对路径、.. 段、符号链接越界、非目录、不可读一律拒绝）。\n"
            "设备不 import/执行 install.py 或 devices/*.py，不发事件，不写内存节点；"
            "装载/安装是 OS 的事。"
        )

    def resolve_path(self, path: List[str]) -> Any:
        if not path:
            return self.to_llm_string()
        pkg = self._require_loaded()
        verb, rest = path[0], path[1:]
        if verb == "manifest":
            if rest:
                raise VMMemoryError("pkgrepo.manifest 不接受参数")
            return pkg.manifest_text
        if verb == "data":
            if rest:
                raise VMMemoryError("pkgrepo.data 不接受参数")
            return pkg.data_text
        if verb == "installer":
            if rest:
                raise VMMemoryError("pkgrepo.installer 不接受参数")
            if pkg.install_text is None:
                raise VMMemoryError(f"pkgrepo 包 {pkg.rel!r} 没有 install.py")
            return pkg.install_text
        if verb == "files":
            if rest:
                raise VMMemoryError("pkgrepo.files 不接受参数")
            return pkg.files_text()
        if verb == "read":
            return self._read(rest)
        raise VMMemoryError(
            f"pkgrepo 未知动作：{verb!r}（可用 .manifest / .data / .installer / .files / "
            '.read."<相对路径>"；读 $MEM.pkgrepo 看说明书）'
        )

    # 只读：Memory.set_by_path 对整路径走 set_value（装载命令），对子路径先走
    # __getitem__ 再走 __setitem__（avm/memory.py）。子路径两处都挡住。

    def __getitem__(self, key: str) -> Any:
        raise MemoryTypeError(
            "pkgrepo 是只读设备，子路径不允许写入（装载：向 $MEM.pkgrepo 写入一行相对路径）"
        )

    def __setitem__(self, key: str, value: Any) -> None:
        raise MemoryTypeError(
            "pkgrepo 是只读设备，子路径不允许写入（装载：向 $MEM.pkgrepo 写入一行相对路径）"
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

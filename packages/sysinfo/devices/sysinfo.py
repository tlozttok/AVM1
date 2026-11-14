"""SysInfo 设备：把宿主（运行 AVM 的 WSL/Ubuntu）的自举相关信息只读披露给对话。

地址语法（设备内部空间，与内存树的点号路径无关；参数用双引号包裹）：

    $MEM.sysinfo                      说明书 + 关键摘要（设备自述）
    $MEM.sysinfo.os                   平台身份（JSON）
    $MEM.sysinfo.python               宿主 Python 运行时（JSON）
    $MEM.sysinfo.resource             资源：CPU/负载/uptime/内存/关键路径容量（JSON）
    $MEM.sysinfo.proc                 进程与权限上下文（JSON）
    $MEM.sysinfo.env                  白名单环境变量（JSON 对象）
    $MEM.sysinfo.env."<NAME>"         单个环境变量（原始值字符串）
    $MEM.sysinfo.tools                PATH 里关键外部命令的有/无（JSON）
    $MEM.sysinfo.disk."<path>"        指定路径的容量与落点信息（JSON）
    $MEM.sysinfo.json                 以上全部合成一条 JSON（供程序解析）

自举需要哪些系统信息（本设备的取舍）：

  1. 平台身份：OS/发行版/内核/架构/WSL/容器/init 进程——决定能调用哪些宿主能力
     （apt、systemd、WSL 互操作）、路径与换行约定；
  2. 宿主 Python 运行时：AVM 本身是宿主 Python 进程；写设备、跑自测、重启 AVM 都要
     知道解释器在哪、什么版本、venv 还是系统解释器、需要的模块在不在；
  3. 资源：CPU 数与亲和数、负载、uptime、内存、关键路径容量——决定能否并行、快照与
     日志往哪写、够不够空间；
  4. 进程与权限上下文：pid/ppid、uid/gid/euid/egid、用户与组、cwd、umask、线程数、
     CapEff、父进程命令行、AVM 镜像目录——决定可写落点、能否提权、当初 AVM 是怎么被
     启动的（重启要复现同一条命令）；
  5. 环境变量（白名单）：PATH/HOME/LANG/TZ/VIRTUAL_ENV/WSL_* 决定子进程行为；
  6. 外部命令有/无：git/python3/pip/apt/tar 等——决定能否自己拉代码、装依赖。

不披露什么：宿主文件内容、目录列表、进程表、网络对端、墙钟时间、Python 包清单。
墙钟时间与包清单更适合各做成独立设备（时钟 / 环境清单），本设备只回答"有没有、在
哪里、多大、什么身份"。

环境变量按白名单读取，并强制屏蔽疑似机密的键（*KEY*/*TOKEN*/*SECRET*/*PASSWORD* 等）。

只读：整路径写入走 set_value，子路径写入在 Memory.set_by_path 里先走 __getitem__
（avm/memory.py），三处都抛 MemoryTypeError。不实现 after_event（设备不发事件），不
覆写 authorize_listener（保持基类拒绝监听），错误信息不回显写入内容。

本文件不提供镜像。镜像里的声明形状：

    {"path": "sysinfo", "file": "devices/sysinfo.py", "class": "SysInfoDevice",
     "args": {"probe_versions": ["git", "python3"]}}

参数：
    probe_versions      要跑 `--version` 的命令名列表；缺省空（只探测"有无"，不启动子进程）
    env_keys            额外纳入白名单的环境变量名
    env_block_prefixes  额外屏蔽的环境变量名片段（大小写不敏感）
    module_probe        find_spec 探测的模块名列表；缺省给一组 stdlib + 常用第三方
    timeout             版本探测的超时秒数（缺省 2）
    max_value_chars     单字段值的截断长度（缺省 4000）
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from avm.exceptions import MemoryTypeError, VMMemoryError
from avm.memory_device import MemoryDevice

DEVICE_NAME = "sysinfo"
DEVICE_VERSION = "1@2026-11-14"

# 白名单环境变量：决定子进程行为、路径与解释器的那些，不试图全覆盖。
_DEFAULT_ENV_KEYS = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "PWD", "OLDPWD",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TZ", "TMPDIR",
    "HOSTNAME", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE", "XDG_RUNTIME_DIR", "XDG_DATA_HOME",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "WSL_DISTRO_NAME", "WSL_INTEROP", "WSLENV",
)

# 名字里含这些片段的变量一律不披露（大小写不敏感）。
_SECRET_MARKERS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "COOKIE",
    "AUTH", "SESSION", "PRIVATE",
)

# 自举要看的外部命令：拉代码、装依赖、起进程、看状态各一组。
_DEFAULT_TOOLS = (
    "git", "python3", "pip3", "apt", "apt-get", "dpkg", "bash", "sh", "uname",
    "tar", "gzip", "curl", "wget", "gcc", "make", "systemctl", "ps", "sed",
    "awk", "jq", "sudo",
)

# 模块可用性：stdlib 里写设备常用的 + 常见第三方（find_spec 只查文件，不 import）。
_DEFAULT_MODULES = (
    "json", "sqlite3", "ctypes", "unittest", "argparse", "venv", "ssl", "email",
    "pytest", "yaml", "openai", "requests", "numpy", "pydantic",
)


# ----------------------------------------------------------------------
# 引用语法：与 images/devices/avm_repo.py 同形
#
# Core 按点号切分引用（avm/core.py 的 Core.unwrap），`"a/b.py"` 到设备时已经是
# ['"a/b', 'py"'] 两段，设备把引号区间内的段用点号接回原字符串。设备文件是独立模块
# （image._load_device_module 按文件装载），不能 import 另一个设备，所以这几个小工具
# 在这里复制一份。
# ----------------------------------------------------------------------

def _quote(text: str) -> str:
    """把路径或表达式包成设备语法的双引号形式（内部反斜杠与双引号转义）。"""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ends_with_unescaped_quote(text: str) -> bool:
    # 长度 1 的那个引号是开引号本身，还没闭合（以点号开头的名字会切出这种段）
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
    """还原引号区间：返回（去掉引号的字符串，引号之后的段）。"""
    if not segments or not segments[0].startswith('"'):
        raise VMMemoryError("sysinfo 的路径参数必须用双引号包裹")
    buf = segments[0]
    index = 1
    while not _ends_with_unescaped_quote(buf):
        if index >= len(segments):
            raise VMMemoryError("sysinfo 的路径参数引号未闭合")
        buf = buf + "." + segments[index]
        index += 1
    return _unescape(buf[1:-1]), segments[index:]


# ----------------------------------------------------------------------
# /proc 与容量采集（都是便宜读，不做重活、不启动子进程）
# ----------------------------------------------------------------------

def _read_text(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _cmdline(pid: int) -> List[str]:
    raw = _read_text("/proc/%d/cmdline" % pid, 4096)
    return [part for part in raw.split("\0") if part]


def _status_fields(pid: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in _read_text("/proc/%s/status" % pid, 16384).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _os_release() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in _read_text("/etc/os-release", 16384).splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _meminfo() -> Dict[str, int]:
    """选出的 /proc/meminfo 字段（单位 kB）。"""
    out: Dict[str, int] = {}
    for line in _read_text("/proc/meminfo", 65536).splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            out[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return out


def _kilobytes(value: Optional[int]) -> Optional[int]:
    return None if value is None else int(value) * 1024


def _human(count: Optional[int]) -> str:
    if not isinstance(count, int):
        return "?"
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(count)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return "%dB" % int(size)
            return "%.1f%s" % (size, unit)
        size /= 1024.0
    return "%dB" % count


def _mounts() -> List[Tuple[str, str, str]]:
    """(挂载点, 设备, 文件系统类型)，按挂载点长度升序。"""
    rows: List[Tuple[str, str, str]] = []
    for line in _read_text("/proc/mounts", 65536).splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount = (parts[1]
                 .replace("\\040", " ")
                 .replace("\\011", "\t")
                 .replace("\\134", "\\"))
        rows.append((mount, parts[0], parts[2]))
    rows.sort(key=lambda row: len(row[0]))
    return rows


def _mount_entry(path: str) -> Optional[Tuple[str, str, str]]:
    """路径所在挂载点：最长前缀匹配。"""
    best: Optional[Tuple[str, str, str]] = None
    for mount, device, fs_type in _mounts():
        if mount == path or path.startswith(mount.rstrip("/") + "/"):
            if best is None or len(mount) >= len(best[0]):
                best = (mount, device, fs_type)
    return best


def _statvfs(path: str) -> Dict[str, Any]:
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    free = st.f_bfree * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    used = total - free
    return {
        "total_bytes": total,
        "free_bytes": free,
        "avail_bytes": avail,
        "used_bytes": used,
        "used_percent": round(used * 100.0 / total, 1) if total else None,
        "inodes_total": st.f_files,
        "inodes_free": st.f_favail,
    }


def _isatty(stream: Any) -> Optional[bool]:
    try:
        return bool(stream.isatty())
    except Exception:
        return None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1, default=str)


# ----------------------------------------------------------------------
# 设备
# ----------------------------------------------------------------------

class SysInfoDevice(MemoryDevice):
    """宿主系统信息只读设备。各动作返回 JSON 字符串，根读返回说明书。"""

    def __init__(
        self,
        probe_versions: Optional[List[str]] = None,
        env_keys: Optional[List[str]] = None,
        env_block_prefixes: Optional[List[str]] = None,
        module_probe: Optional[List[str]] = None,
        timeout: int = 2,
        max_value_chars: int = 4000,
    ):
        self._probe_versions = [str(x).strip() for x in (probe_versions or [])
                                if str(x).strip()]
        self._env_keys = [str(x).strip() for x in (env_keys or []) if str(x).strip()]
        self._env_block = [str(x).strip().upper() for x in (env_block_prefixes or [])
                           if str(x).strip()]
        self._module_probe = [str(x).strip() for x in (module_probe or _DEFAULT_MODULES)
                              if str(x).strip()]
        self._timeout = max(1, int(timeout))
        self._max_value_chars = max(200, int(max_value_chars))
        self._core = None

    # ------------------------------------------------------------------
    # 钩子
    # ------------------------------------------------------------------

    def attach_core(self, core) -> None:
        self._core = core

    def pretend_as_type(self) -> str:
        return "str"

    def _image_dir(self) -> Optional[str]:
        core = self._core
        if core is None:
            return None
        path = getattr(core, "image_dir", None)
        return str(path) if path else None

    def _clip(self, text: Any) -> str:
        value = "" if text is None else str(text)
        if len(value) <= self._max_value_chars:
            return value
        return value[: self._max_value_chars] + "…（截断，原长 %d）" % len(value)

    # ------------------------------------------------------------------
    # 各节采集
    # ------------------------------------------------------------------

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def _os(self) -> Dict[str, Any]:
        release = _os_release()
        uname = platform.uname()
        banner = _read_text("/proc/version", 4096).strip()
        wsl_distro = os.environ.get("WSL_DISTRO_NAME") or ""
        wsl = bool(wsl_distro) or "microsoft" in banner.lower()
        cgroup = _read_text("/proc/1/cgroup", 16384)
        container = (os.path.exists("/.dockerenv")
                     or "docker" in cgroup or "lxc" in cgroup)
        return {
            "platform": sys.platform,
            "system": uname.system,
            "hostname": uname.node,
            "distro_id": release.get("ID"),
            "distro_version": release.get("VERSION_ID"),
            "distro_pretty": release.get("PRETTY_NAME"),
            "distro_codename": release.get("VERSION_CODENAME"),
            "kernel_release": uname.release,
            "kernel_version": uname.version,
            "kernel_banner": self._clip(banner),
            "machine": uname.machine,
            "processor": uname.processor,
            "libc": " ".join(part for part in platform.libc_ver() if part),
            "wsl": wsl,
            "wsl_distro": wsl_distro or None,
            "wsl_interop": os.environ.get("WSL_INTEROP"),
            "wsl_env": os.environ.get("WSLENV"),
            "container": container,
            "init_process": _read_text("/proc/1/comm", 256).strip(),
            "init_cmdline": _cmdline(1)[:8],
        }

    def _python(self) -> Dict[str, Any]:
        modules: Dict[str, bool] = {}
        for name in self._module_probe:
            modules[name] = self._module_available(name)
        avm_origin = None
        try:
            spec = importlib.util.find_spec("avm")
            if spec is not None:
                avm_origin = getattr(spec, "origin", None)
        except (ImportError, AttributeError, ValueError):
            avm_origin = None
        return {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_full": self._clip(sys.version.replace("\n", " ")),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_venv": sys.prefix != sys.base_prefix,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "frozen": bool(getattr(sys, "frozen", False)),
            "byteorder": sys.byteorder,
            "maxsize": sys.maxsize,
            "platform": sys.platform,
            "avm_package": avm_origin,
            "sys_path": [str(item) for item in sys.path[:16]],
            "modules": modules,
        }

    def _disk_targets(self) -> List[Tuple[str, str]]:
        """关键落点：根、家目录、cwd、AVM 镜像目录（去重，保序）。"""
        raw = [("root", "/"), ("home", os.path.expanduser("~")), ("cwd", os.getcwd())]
        image_dir = self._image_dir()
        if image_dir:
            raw.append(("image_dir", image_dir))
        seen = set()
        targets: List[Tuple[str, str]] = []
        for label, path in raw:
            key = os.path.abspath(os.path.expanduser(str(path)))
            if key in seen:
                continue
            seen.add(key)
            targets.append((label, key))
        return targets

    def _disk_info(self, path: str) -> Dict[str, Any]:
        target = os.path.abspath(os.path.expanduser(str(path)))
        info: Dict[str, Any] = {
            "path": target,
            "exists": os.path.exists(target),
            "is_dir": os.path.isdir(target),
            "is_file": os.path.isfile(target),
        }
        if not info["exists"]:
            return info
        entry = _mount_entry(target)
        if entry is not None:
            info["mount"], info["device"], info["fs_type"] = entry
        try:
            stats = _statvfs(target)
        except OSError as e:
            info["error"] = type(e).__name__
            return info
        info.update(stats)
        for key in ("total_bytes", "free_bytes", "avail_bytes", "used_bytes"):
            info[key.replace("_bytes", "_human")] = _human(stats.get(key))
        probe_dir = target if info["is_dir"] else (os.path.dirname(target) or "/")
        info["writable"] = os.access(probe_dir, os.W_OK)
        return info

    def _resource(self) -> Dict[str, Any]:
        mem = _meminfo()
        try:
            load: Optional[List[float]] = [round(x, 2) for x in os.getloadavg()]
        except (OSError, AttributeError):
            load = None
        uptime = None
        fields = _read_text("/proc/uptime", 256).split()
        if fields:
            try:
                uptime = round(float(fields[0]), 1)
            except ValueError:
                uptime = None
        affinity = None
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity = len(os.sched_getaffinity(0))
            except OSError:
                affinity = None
        disks: Dict[str, Any] = {}
        for label, path in self._disk_targets():
            disks[label] = self._disk_info(path)
        return {
            "cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "loadavg": load,
            "uptime_seconds": uptime,
            "memory": {
                "total": _human(_kilobytes(mem.get("MemTotal"))),
                "available": _human(_kilobytes(mem.get("MemAvailable"))),
                "free": _human(_kilobytes(mem.get("MemFree"))),
                "swap_total": _human(_kilobytes(mem.get("SwapTotal"))),
                "swap_free": _human(_kilobytes(mem.get("SwapFree"))),
            },
            "disks": disks,
        }

    def _proc(self) -> Dict[str, Any]:
        status = _status_fields("self")
        user = None
        try:
            import pwd
            user = pwd.getpwuid(os.geteuid()).pw_name
        except Exception:
            user = None
        groups: List[str] = []
        try:
            import grp
            groups = [grp.getgrgid(gid).gr_name for gid in os.getgroups()]
        except Exception:
            groups = []
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "uid": os.getuid(),
            "euid": os.geteuid(),
            "gid": os.getgid(),
            "egid": os.getegid(),
            "user": user,
            "groups": groups,
            "raw_gids": os.getgroups(),
            "cwd": os.getcwd(),
            "umask": status.get("Umask"),
            "threads": status.get("Threads"),
            "cap_effective": status.get("CapEff"),
            "no_new_privs": status.get("NoNewPrivs"),
            "seccomp": status.get("Seccomp"),
            "parent_cmdline": _cmdline(os.getppid())[:16],
            "avm_image_dir": self._image_dir(),
            "stdin_is_tty": _isatty(sys.stdin),
            "stdout_is_tty": _isatty(sys.stdout),
        }

    def _env_allowed(self, name: str) -> bool:
        upper = str(name).upper()
        for marker in _SECRET_MARKERS + tuple(self._env_block):
            if marker and marker in upper:
                return False
        return True

    def _env(self) -> Dict[str, str]:
        names: List[str] = []
        for name in list(_DEFAULT_ENV_KEYS) + list(self._env_keys):
            if name in os.environ and name not in names:
                names.append(name)
        values: Dict[str, str] = {}
        for name in sorted(names):
            if not self._env_allowed(name):
                continue
            values[name] = self._clip(os.environ[name])
        return values

    def _version_of(self, name: str) -> str:
        path = shutil.which(name)
        if not path:
            return "未找到"
        try:
            proc = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return "超时（>%ds）" % self._timeout
        except OSError as e:
            return "启动失败：%s" % type(e).__name__
        text = (proc.stdout or proc.stderr or "").strip()
        head = text.splitlines()[0] if text else ""
        head = self._clip(head)
        if proc.returncode != 0:
            return "退出码 %d：%s" % (proc.returncode, head)
        return head

    def _tools(self) -> Dict[str, Any]:
        found: Dict[str, str] = {}
        missing: List[str] = []
        for name in _DEFAULT_TOOLS:
            path = shutil.which(name)
            if path:
                found[name] = path
            else:
                missing.append(name)
        result: Dict[str, Any] = {
            "checked": len(_DEFAULT_TOOLS),
            "found": found,
            "missing": missing,
        }
        versions: Dict[str, str] = {}
        for name in self._probe_versions:
            versions[name] = self._version_of(name)
        if versions:
            result["versions"] = versions
        return result

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "device": DEVICE_NAME,
            "device_version": DEVICE_VERSION,
            "os": self._os(),
            "python": self._python(),
            "resource": self._resource(),
            "proc": self._proc(),
            "env": self._env(),
            "tools": self._tools(),
        }

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    @staticmethod
    def _no_args(verb: str, rest: List[str]) -> None:
        if rest:
            raise VMMemoryError("sysinfo.%s 不接受参数" % verb)

    def _env_action(self, rest: List[str]) -> str:
        if not rest:
            values = self._env()
            return _dump({
                "count": len(values),
                "values": values,
                "blocked_markers": list(_SECRET_MARKERS),
            })
        if rest[0].startswith('"'):
            name, extra = _take_quoted(rest)
            if extra:
                raise VMMemoryError("sysinfo.env 只接受一个变量名参数")
        else:
            name = rest[0]
            if len(rest) > 1:
                raise VMMemoryError("sysinfo.env 只接受一个变量名参数")
        name = name.strip()
        if not name:
            raise VMMemoryError("sysinfo.env 的变量名不能为空")
        if not self._env_allowed(name):
            raise VMMemoryError("sysinfo.env 拒绝披露疑似机密的变量：%s" % name)
        if name not in os.environ:
            raise VMMemoryError("sysinfo.env 该变量未设置：%s" % name)
        return self._clip(os.environ[name])

    def _disk_action(self, rest: List[str]) -> str:
        if not rest:
            return self._help("disk")
        path, extra = _take_quoted(rest)
        if extra:
            raise VMMemoryError("sysinfo.disk 只接受一个路径参数")
        return _dump(self._disk_info(path))

    def resolve_path(self, path: List[str]) -> Any:
        if not path:
            return self.to_llm_string()
        verb = path[0]
        rest = path[1:]
        if verb == "os":
            self._no_args(verb, rest)
            return _dump(self._os())
        if verb == "python":
            self._no_args(verb, rest)
            return _dump(self._python())
        if verb == "resource":
            self._no_args(verb, rest)
            return _dump(self._resource())
        if verb == "proc":
            self._no_args(verb, rest)
            return _dump(self._proc())
        if verb == "env":
            return self._env_action(rest)
        if verb == "tools":
            self._no_args(verb, rest)
            return _dump(self._tools())
        if verb == "disk":
            return self._disk_action(rest)
        if verb == "json":
            self._no_args(verb, rest)
            return _dump(self._snapshot())
        return self._help(verb)

    # ------------------------------------------------------------------
    # 自述与帮助
    # ------------------------------------------------------------------

    def to_llm_string(self) -> str:
        head = "sysinfo[%s]" % DEVICE_VERSION
        try:
            osi = self._os()
            pyi = self._python()
            res = self._resource()
            pr = self._proc()
            memory = res.get("memory") or {}
            root_disk = (res.get("disks") or {}).get("root") or {}
            flags = ""
            if osi.get("wsl"):
                flags += ", WSL"
                if osi.get("wsl_distro"):
                    flags += "(%s)" % osi.get("wsl_distro")
            if osi.get("container"):
                flags += ", container"
            head = (
                "sysinfo[%s]: %s%s, kernel=%s, %s, python=%s(%s), %s, "
                "user=%s(uid=%s/euid=%s), cwd=%s, cpu=%s, mem可用=%s, 磁盘可用=%s"
                % (
                    DEVICE_VERSION,
                    osi.get("distro_pretty") or osi.get("system") or "?",
                    flags,
                    osi.get("kernel_release"),
                    osi.get("machine"),
                    pyi.get("version"),
                    pyi.get("executable"),
                    "venv" if pyi.get("in_venv") else "system-python",
                    pr.get("user"),
                    pr.get("uid"),
                    pr.get("euid"),
                    pr.get("cwd"),
                    res.get("cpu_count"),
                    memory.get("available"),
                    root_disk.get("avail_human"),
                )
            )
        except Exception as e:
            head = "sysinfo[%s]（采集摘要失败：%s）" % (DEVICE_VERSION, type(e).__name__)
        lines = [
            head,
            "只读设备：披露运行 AVM 的宿主（WSL/Ubuntu）自举所需的系统信息。",
            '地址：参数用双引号包裹；各节返回 JSON，env."<NAME>" 返回原始值字符串。',
            "  $MEM.sysinfo                     本说明书 + 关键摘要",
            "  $MEM.sysinfo.os                  平台身份：发行版/内核/架构/WSL/容器/init",
            "  $MEM.sysinfo.python              宿主 Python：解释器/版本/venv/模块可用性",
            "  $MEM.sysinfo.resource            CPU/亲和数/负载/uptime/内存/关键路径容量",
            "  $MEM.sysinfo.proc                pid/uid/gid/cwd/umask/CapEff/父进程命令行/镜像目录",
            "  $MEM.sysinfo.env                 白名单环境变量（疑似机密键已屏蔽）",
            '  $MEM.sysinfo.env."<NAME>"        单个环境变量',
            "  $MEM.sysinfo.tools               PATH 里关键外部命令的有/无（可配版本探测）",
            '  $MEM.sysinfo.disk."<path>"       指定路径的容量、挂载点、文件系统与可写性',
            "  $MEM.sysinfo.json                上述全部合成一条 JSON",
            "写入一律报错；设备不发事件，也不接受监听注册。",
        ]
        return "\n".join(lines)

    def _help(self, verb: str) -> str:
        if verb == "disk":
            return (
                "disk：查指定路径的容量与落点\n"
                '用法：$MEM.sysinfo.disk."<path>"\n'
                '  例：$MEM.sysinfo.disk."/home"\n'
                "返回 JSON：path/exists/is_dir/is_file/mount/device/fs_type/"
                "total|free|avail|used(_bytes/_human)/used_percent/inodes_free/writable"
            )
        if verb == "env":
            return (
                "env：白名单环境变量\n"
                '用法：$MEM.sysinfo.env 或 $MEM.sysinfo.env."<NAME>"\n'
                "返回：前者是 JSON（count/values/blocked_markers）；后者是变量原始值。"
                "疑似机密的键、未设置的变量都返回 Error。"
            )
        known = ("os", "python", "resource", "proc", "env", "tools", "disk", "json")
        raise VMMemoryError(
            "sysinfo 未知动作：%r（可用：%s；读 $MEM.sysinfo 看说明书）"
            % (verb, ", ".join(known))
        )

    # ------------------------------------------------------------------
    # 只读：三处写入都挡住（整路径走 set_value，子路径先 __getitem__ 再 __setitem__）
    # ------------------------------------------------------------------

    def set_value(self, value: Any) -> None:
        raise MemoryTypeError("sysinfo 是只读设备，不允许写入")

    def __getitem__(self, key: str) -> Any:
        raise MemoryTypeError("sysinfo 是只读设备，不允许写入")

    def __setitem__(self, key: str, value: Any) -> None:
        raise MemoryTypeError("sysinfo 是只读设备，不允许写入")

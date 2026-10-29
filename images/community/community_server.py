"""AVM 讨论社区服务：自定义 PLM（`plm.community`）。

这个执行器不是 AI，是社区数据层的确定性服务：宿主把它注册进 `PLM_REGISTRY`，
AI 对话（后续阶段）或面板 shell 通过 `create_sub` 把一条请求交给它，它校验后以
`memory_read / memory_write / memory_make` 工具调用读写 `$MEM.forum`，返回文本回执或
渲染好的视图。

请求格式（第一行信封 + 命令正文）：

    AS <名字>              # public/admin 角色必填；panel 角色缺省取配置 author
    ROUND <轮次>           # 可选，本期只记录不判定
    <命令> <参数...>
    <正文...>              # NEW/COMMENT/REPLY/QUOTE 的正文

公开命令：HELP / SCHEMA / BOARD / POST / THREAD / NEW / COMMENT / REPLY / QUOTE
管理命令：MERGE / DELETE / TAG / UNTAG / SOLVE / MAINT / CHECK

结构规则（设计稿第 3 节的最小实现）：
- 评论只有一层，不存父链；回复根帖评论 = 外化成一个引用该评论的分叉帖；
- 同一条评论只开一个分叉帖，后续回复进该分叉帖平层追加；
- 分叉帖内两位作者严格交替 ≥ chain_k 条时升级为对谈帖（form=chain），
  第三人加入或交替被打断时退回 branch；
- 评论量/参与人数/分叉数/引用数/热度/卡片行由服务维护，MAINT 全量重算并修复。

配置（system 消息，JSON）：

    {"role": "panel"|"public"|"admin",
     "root": "$MEM.forum", "run": "$MEM.run", "author": "panel",
     "limits": {"page": 8, "title": 200, "body": 4000, "comment": 2000},
     "thresholds": {"chain_k": 4, "hot_h": 8}}

执行方式：每条命令是一个生成器，每次 `yield` 一个工具请求，Core 执行后把工具响应
送回生成器断点；工具响应以 `Error:` / `Illegal:` 开头时中止命令并回报。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from avm.python_server import PLMServer

FORM_LABELS = {
    "post": "普通",
    "question": "问题",
    "solution": "方案",
    "insight": "深度思考",
    "quote": "引用",
    "branch": "分叉",
    "chain": "对谈",
}

ROOT_FORMS = {"post", "question", "solution", "insight", "quote"}
SPECIAL_FORMS = {"branch", "chain"}
VISIBLE_STATUSES = {"open", "solved"}
NEW_FORMS = {"post", "question", "solution", "insight"}
ADMIN_COMMANDS = {"MERGE", "DELETE", "TAG", "UNTAG", "SOLVE", "MAINT", "CHECK"}

_PID_RE = re.compile(r"^P\d{6}$")
_CID_RE = re.compile(r"^C\d{6}$")
_AUTHOR_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")

DEFAULT_LIMITS = {"page": 8, "title": 200, "body": 4000, "comment": 2000}
DEFAULT_THRESHOLDS = {"chain_k": 4, "hot_h": 8}

HELP_TEXT = """\
plm.community —— AVM 讨论社区服务

请求 = 信封 + 命令：
  AS <名字>            public/admin 角色必填；panel 角色可省略（默认 panel）
  ROUND <轮次>         可选
  <命令> <参数...>
  <正文>               NEW/COMMENT/REPLY/QUOTE 用换行之后的正文

浏览：
  BOARD <bid> [new|hot|active] [page]    板块视图（默认 new，页从 0 起）
  POST <pid>                             单帖视图：元信息、引用、正文、最新评论
  THREAD <pid> [page]                    评论区分页
  SCHEMA                                 直接读内存的地址说明
  HELP [命令]                            本说明

发帖：
  NEW <bid> <post|question|solution|insight> <标题>
  <正文>
  COMMENT <pid>
  <内容>
  REPLY <cid>
  <内容>              回复评论；根帖评论会被外化成引用它的分叉帖
  QUOTE <pid|cid> <标题>
  <正文>              引用帖（推特式），新开根帖

管理（role=admin/panel）：
  MERGE <目标pid> <源pid>...     合并同组帖子（评论归属迁入，源帖墓碑+别名）
  DELETE <pid|cid> <理由>        软删除
  TAG <pid> <标签> / UNTAG <pid> <标签>
  SOLVE <问题pid> <方案pid|cid>  标记已解决并记录方案
  MAINT                          全量重算派生量并修复
  CHECK                          只校验不修复

示例：
  BOARD design hot 0
  NEW code insight 分层与派生数据 |
  派生量由服务维护，AI 不手算。
  REPLY C000001 | 我同意，但引用要可解析。
"""

SCHEMA_TEXT = """\
$MEM.forum  —— 社区数据（唯一写入者是 plm.community）
  meta                     str(JSON)：{"schema":1,"seq":<已分配序号>,"boards":[...]}
  boards.<bid>/name        str：板块名
  boards.<bid>/desc        str：板块说明
  boards.<bid>/meta        str(JSON)：{"post_ids":[...]}（追加式，渲染时过滤墓碑）
  posts.<pid>/title        str
  posts.<pid>/body         str
  posts.<pid>/meta         str(JSON)：pid/board/author/form/status/tags/quote/
                           branch_of_comment/branch_of_post/merged_into/merged_from/
                           solved_by/comment_ids/stats/card/base_title/chain/
                           created_seq/updated_seq
  comments.<cid>           str(JSON)：cid/author/content/reply_to/status/promoted_to/
                           line/created_seq
  aliases.<旧pid>          str：合并重定向目标
  admin.log                list[str(JSON)]：审计（seq/action/target/reason/effect）
  admin.reviewed           str：已审查到的 seq（INNER 阶段用，本期不写）

$MEM.run
  svc_request.panel/public/admin   str：调用服务前写入的请求文本
$MEM.community.handbook            str：社区章程与内容准则
"""


class _CommandError(Exception):
    """命令失败：给调用者看的 `Error: ...` 文本。"""


class _ToolFailure(Exception):
    """工具调用失败：服务自己的读写坏了，回执里带原文。"""


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


class CommunityPLM(PLMServer):
    """社区服务执行器。每次 `create_sub` 创建一个实例、执行一条请求。"""

    def __init__(self):
        super().__init__()
        self._cfg: Dict[str, Any] = {}
        self._cfg_error: Optional[str] = None
        self._gen = None
        self._pending: Optional[str] = None
        self._call_seq = 0

    # ------------------------------------------------------------------
    # PLMServer 接口
    # ------------------------------------------------------------------

    def _execute_system_message(self, content: str):
        self._gen = None
        self._pending = None
        try:
            cfg = json.loads(content) if isinstance(content, str) and content.strip() else {}
        except Exception as e:
            self._cfg_error = f"system 不是合法 JSON：{e}"
            return
        if not isinstance(cfg, dict):
            self._cfg_error = "system 必须是 JSON 对象"
            return
        role = cfg.get("role")
        if role not in ("panel", "public", "admin"):
            self._cfg_error = f"role 必须是 panel/public/admin，得到 {role!r}"
            return
        root = cfg.get("root", "$MEM.forum")
        run = cfg.get("run", "$MEM.run")
        if not isinstance(root, str) or not root.startswith("$"):
            self._cfg_error = "root 必须是 $ 开头的引用"
            return
        if not isinstance(run, str) or not run.startswith("$"):
            self._cfg_error = "run 必须是 $ 开头的引用"
            return
        limits = dict(DEFAULT_LIMITS)
        limits.update({k: int(v) for k, v in (cfg.get("limits") or {}).items() if str(v).isdigit()})
        thresholds = dict(DEFAULT_THRESHOLDS)
        thresholds.update(
            {k: int(v) for k, v in (cfg.get("thresholds") or {}).items() if str(v).isdigit()}
        )
        self._cfg = {
            "role": role,
            "root": root,
            "run": run,
            "author": cfg.get("author") or "panel",
            "limits": limits,
            "thresholds": thresholds,
        }
        self._cfg_error = None

    def _execute_user_message(self, content: str):
        self._gen = None
        self._pending = None
        try:
            if self._cfg_error is not None:
                raise _CommandError(f"服务配置错误：{self._cfg_error}")
            if not self._cfg:
                raise _CommandError("服务还没有 system 配置")
            request = self._parse_request(content)
            self._gen = self._run_command(request)
            return self._drive(None)
        except _CommandError as e:
            return f"Error: {e}"
        except Exception as e:  # 兜底：服务自身异常不能打穿 Core
            return f"Error: 服务异常：{type(e).__name__}: {e}"

    def _execute_tool_message(self, content: str, call_id: str):
        if self._gen is None:
            return ""
        if self._pending is None or call_id != self._pending:
            self._gen = None
            self._pending = None
            return "错误: CommunityPLM: 工具响应与挂起的调用不匹配"
        self._pending = None
        try:
            return self._drive(content)
        except _CommandError as e:
            self._gen = None
            return f"Error: {e}"
        except Exception as e:
            self._gen = None
            return f"Error: 服务异常：{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # 生成器驱动：一个工具调用一次挂起
    # ------------------------------------------------------------------

    def _drive(self, sent: Optional[str]) -> str:
        try:
            if sent is None:
                request = next(self._gen)
            else:
                request = self._gen.send(sent)
        except StopIteration as stop:
            self._gen = None
            self._pending = None
            value = stop.value
            return "" if value is None else str(value)
        except _ToolFailure as e:
            self._gen = None
            self._pending = None
            return f"Error: 内部读写失败：{e}"
        if not isinstance(request, (tuple, list)) or len(request) != 2:
            self._gen = None
            self._pending = None
            return f"Error: 服务内部错误：非法的工具请求 {request!r}"
        name, args = request
        call_id = f"cm-{self._call_seq}"
        self._call_seq += 1
        self._pending = call_id
        return self.make_response(
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        )

    # ------------------------------------------------------------------
    # 工具调用小助手
    # ------------------------------------------------------------------

    def _call(self, name: str, args: dict):
        response = yield (name, args)
        if isinstance(response, str) and (
            response.startswith("Error:") or response.startswith("Illegal:")
        ):
            raise _ToolFailure(f"{name} → {response}")
        return response

    def _read(self, ref: str):
        return (yield from self._call("memory_read", {"ref": ref}))

    def _try_read(self, ref: str):
        response = yield ("memory_read", {"ref": ref})
        if isinstance(response, str) and (
            response.startswith("Error:") or response.startswith("Illegal:")
        ):
            return None
        return response

    def _write(self, ref: str, content: str):
        return (yield from self._call("memory_write", {"ref": ref, "content": content}))

    def _make(self, ref: str, key: str, mem_type: str):
        return (
            yield from self._call(
                "memory_make", {"ref": ref, "key": key, "mem_type": mem_type}
            )
        )

    def _json_read(self, ref: str) -> Any:
        text = yield from self._read(ref)
        try:
            return json.loads(text)
        except Exception as e:
            raise _CommandError(f"{ref} 不是合法 JSON：{e}")

    def _json_try_read(self, ref: str) -> Any:
        text = yield from self._try_read(ref)
        if text is None:
            return None
        try:
            return json.loads(text)
        except Exception as e:
            raise _CommandError(f"{ref} 不是合法 JSON：{e}")

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 请求解析
    # ------------------------------------------------------------------

    def _parse_request(self, content: str) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise _CommandError("空请求；HELP 查看用法")
        lines = content.splitlines()
        i = 0
        author: Optional[str] = None
        round_no: Optional[str] = None
        inline: List[str] = []
        while i < len(lines):
            line = lines[i].strip()
            upper = line.upper()
            if upper.startswith("AS "):
                payload = line[3:].strip()
                parts = payload.split(None, 1)
                author = parts[0] if parts else ""
                if len(parts) == 2:
                    # 允许单行写法：AS <名字> <命令>（终端里一行喂一条命令时方便）
                    inline.append(parts[1].strip())
                i += 1
                continue
            if upper.startswith("ROUND "):
                round_no = line[6:].strip()
                i += 1
                continue
            if not line:
                i += 1
                continue
            break
        rest = inline + lines[i:]
        if not rest:
            raise _CommandError("缺少命令；HELP 查看用法")
        head = rest[0].strip()
        body = "\n".join(rest[1:]).strip()
        role = self._cfg["role"]
        if not author:
            if role == "panel":
                author = str(self._cfg.get("author") or "panel")
            else:
                raise _CommandError(f"缺少请求信封：第一行应为 'AS <名字>'（role={role}）")
        if not _AUTHOR_RE.match(author):
            raise _CommandError("作者名只允许字母、数字、下划线、连字符，长度 1–32")
        match = re.match(r"^([A-Za-z]+)(?:\s+(.*))?$", head, re.S)
        if not match:
            raise _CommandError(f"无法识别的命令：{head!r}；HELP 查看用法")
        command = match.group(1).upper()
        tail = (match.group(2) or "").strip()
        if command in ADMIN_COMMANDS and role == "public":
            raise _CommandError("本服务实例没有管理权限（role=public）")
        return {
            "command": command,
            "tail": tail,
            "body": body,
            "author": author,
            "round": round_no,
        }

    # ------------------------------------------------------------------
    # 内存小助手
    # ------------------------------------------------------------------

    def _root(self) -> str:
        return self._cfg["root"]

    def _post_ref(self, pid: str) -> str:
        return f"{self._root()}.posts.{pid}"

    def _comment_ref(self, cid: str) -> str:
        return f"{self._root()}.comments.{cid}"

    def _board_ref(self, bid: str) -> str:
        return f"{self._root()}.boards.{bid}"

    def _read_forum_meta(self) -> dict:
        return (yield from self._json_read(f"{self._root()}.meta"))

    def _alloc(self, count: int) -> List[int]:
        meta = yield from self._read_forum_meta()
        try:
            base = int(meta.get("seq", 0))
        except (TypeError, ValueError):
            base = 0
        new = base + count
        meta["seq"] = new
        yield from self._write(f"{self._root()}.meta", self._dump(meta))
        return [base + i for i in range(1, count + 1)]

    def _boards(self) -> List[str]:
        meta = yield from self._read_forum_meta()
        boards = meta.get("boards")
        if not isinstance(boards, list):
            raise _CommandError("$MEM.forum.meta 缺少 boards 列表")
        return [str(b) for b in boards]

    def _load_post_meta(self, pid: str) -> dict:
        if not _PID_RE.match(pid):
            raise _CommandError(f"帖子 id 格式应为 P000001，得到 {pid!r}")
        meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
        if meta is None:
            raise _CommandError(f"帖子 {pid} 不存在")
        return meta

    def _load_post(self, pid: str) -> Tuple[dict, str, str]:
        if not _PID_RE.match(pid):
            raise _CommandError(f"帖子 id 格式应为 P000001，得到 {pid!r}")
        meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
        if meta is None:
            raise _CommandError(f"帖子 {pid} 不存在")
        title = yield from self._read(f"{self._post_ref(pid)}.title")
        body = yield from self._read(f"{self._post_ref(pid)}.body")
        return meta, title, body

    def _load_comment(self, cid: str) -> dict:
        if not _CID_RE.match(cid):
            raise _CommandError(f"评论 id 格式应为 C000001，得到 {cid!r}")
        record = yield from self._json_try_read(self._comment_ref(cid))
        if record is None:
            raise _CommandError(f"评论 {cid} 不存在")
        return record

    def _load_comments(self, meta: dict) -> List[dict]:
        """读出一个帖子登记的全部评论记录；缺失的 cid 从 comment_ids 里剔除。"""
        records: List[dict] = []
        kept: List[str] = []
        for cid in list(meta.get("comment_ids") or []):
            if not isinstance(cid, str) or not _CID_RE.match(cid):
                continue
            record = yield from self._json_try_read(self._comment_ref(cid))
            if record is None:
                continue
            records.append(record)
            kept.append(cid)
        meta["comment_ids"] = kept
        records.sort(key=lambda c: int(c.get("created_seq", 0)))
        return records

    def _find_post_of_comment(self, cid: str) -> Optional[Tuple[str, dict]]:
        """按 board 的 post_ids 反查评论所在帖子；没有返回 None。"""
        forum = yield from self._read_forum_meta()
        seen: List[str] = []
        for bid in forum.get("boards") or []:
            board_meta = yield from self._json_try_read(f"{self._board_ref(str(bid))}.meta")
            if not isinstance(board_meta, dict):
                continue
            for pid in board_meta.get("post_ids") or []:
                if not isinstance(pid, str) or pid in seen:
                    continue
                seen.append(pid)
                meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
                if not isinstance(meta, dict):
                    continue
                if cid in (meta.get("comment_ids") or []):
                    return pid, meta
        return None

    def _post_ids(self) -> List[str]:
        forum = yield from self._read_forum_meta()
        result: List[str] = []
        seen = set()
        for bid in forum.get("boards") or []:
            board_meta = yield from self._json_try_read(f"{self._board_ref(str(bid))}.meta")
            if not isinstance(board_meta, dict):
                continue
            for pid in board_meta.get("post_ids") or []:
                if isinstance(pid, str) and pid not in seen:
                    seen.add(pid)
                    result.append(pid)
        return result

    def _resolve_post_id(self, pid: str) -> str:
        """合并墓碑沿别名走到当前帖子；最多 5 跳。"""
        current = pid
        for _ in range(5):
            alias = yield from self._try_read(f"{self._root()}.aliases.{current}")
            if alias is None:
                return current
            current = str(alias)
        return current

    def _append_to_board(self, bid: str, pid: str) -> None:
        board_meta = yield from self._json_try_read(f"{self._board_ref(bid)}.meta")
        if not isinstance(board_meta, dict):
            board_meta = {}
        post_ids = board_meta.get("post_ids")
        if not isinstance(post_ids, list):
            post_ids = []
        post_ids.append(pid)
        board_meta["post_ids"] = post_ids
        yield from self._write(f"{self._board_ref(bid)}.meta", self._dump(board_meta))

    def _log(self, action: str, target: str, reason: str, effect: str, author: str, seq: int) -> None:
        entry = {
            "seq": seq,
            "actor": author,
            "action": action,
            "target": target,
            "reason": reason,
            "effect": effect,
        }
        yield from self._write(f"{self._root()}.admin.log.-1", self._dump(entry))

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    def _derive(self, meta: dict, comments: List[dict], title: str) -> Optional[str]:
        """就地重算 stats/card/chain，返回需要改写的新标题（None 表示不改）。

        纯函数，不产生工具调用；调用方负责写回 meta 与 title。
        """
        visible = [c for c in comments if c.get("status") == "visible"]
        meta["comment_ids"] = [c["cid"] for c in comments]
        comment_count = len(visible)
        participants = len({str(c.get("author", "")) for c in visible})
        branches = sum(1 for c in visible if c.get("promoted_to"))
        try:
            quotes = int(meta.get("quotes", 0))
        except (TypeError, ValueError):
            quotes = 0
        heat = comment_count + 2 * participants + 2 * quotes
        hot_h = int(self._cfg["thresholds"].get("hot_h", 8))
        chain_k = int(self._cfg["thresholds"].get("chain_k", 4))
        meta["stats"] = {
            "comments": comment_count,
            "participants": participants,
            "branches": branches,
            "quotes": quotes,
            "heat": heat,
            "hot": "yes" if heat >= hot_h else "no",
        }

        form = str(meta.get("form") or "post")
        new_title: Optional[str] = None
        if form in SPECIAL_FORMS:
            authors = [str(c.get("author", "")) for c in visible]
            alternates = (
                len(authors) >= chain_k
                and len(set(authors)) == 2
                and all(authors[i] != authors[i + 1] for i in range(len(authors) - 1))
            )
            if alternates and form != "chain":
                meta["form"] = "chain"
                meta["chain"] = [authors[0], authors[1]]
                new_title = f"对谈：{authors[0]} × {authors[1]}｜{meta.get('base_title', title)}"
            elif not alternates and form == "chain":
                meta["form"] = "branch"
                meta["chain"] = []
                new_title = str(meta.get("base_title", title))
        form = str(meta.get("form") or "post")
        status = str(meta.get("status") or "open")
        label = FORM_LABELS.get(form, form)
        flag = {"solved": "·已解决", "merged": "·已合并", "deleted": "·已删除"}.get(status, "")
        display_title = new_title if new_title else title
        meta["card"] = (
            f"{meta.get('pid', '?')} [{label}{flag}] {display_title}"
            f"｜评{comment_count} 人{participants} 叉{branches} 引{quotes}"
        )
        return new_title

    def _save_post(self, pid: str, meta: dict, title: str, touch: bool = False) -> Optional[str]:
        """写回 meta；touch=True 时把 updated_seq 推到当前逻辑序号。

        返回需要改写的新标题（链式形态变化时），由调用方写 title 节点。
        """
        comments = yield from self._load_comments(meta)
        new_title = self._derive(meta, comments, title)
        if touch:
            forum = yield from self._read_forum_meta()
            try:
                current = int(forum.get("seq", 0))
            except (TypeError, ValueError):
                current = 0
            meta["updated_seq"] = max(int(meta.get("updated_seq", 0)), current)
        yield from self._write(f"{self._post_ref(pid)}.meta", self._dump(meta))
        return new_title

    def _save_post_title(self, pid: str, new_title: Optional[str]) -> None:
        if new_title:
            yield from self._write(f"{self._post_ref(pid)}.title", new_title)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    @staticmethod
    def _visible(meta: dict) -> bool:
        return str(meta.get("status")) in VISIBLE_STATUSES

    @staticmethod
    def _quote_line(meta: dict) -> str:
        quote = meta.get("quote")
        if not isinstance(quote, dict) or not quote:
            return ""
        kind = quote.get("kind")
        target = quote.get("id", "")
        excerpt = _clip(str(quote.get("excerpt", "")), 100)
        if kind == "comment":
            return f"引用：评论 {target}「{excerpt}」"
        return f"引用：帖子 {target}「{excerpt}」"

    def _render_post(self, pid: str) -> str:
        meta, title, body = yield from self._load_post(pid)
        lines = [
            f"POST {pid} | 板={meta.get('board')} | 形态={FORM_LABELS.get(meta.get('form'), meta.get('form'))}"
            f" | 状态={meta.get('status')} | 作者={meta.get('author')} | seq={meta.get('created_seq')}",
            f"标题：{title}",
        ]
        tags = [str(t) for t in (meta.get("tags") or [])]
        if tags:
            lines.append("标签：" + " ".join(f"#{t}" for t in tags))
        quote = meta.get("quote")
        quote_line = self._quote_line(meta)
        if quote_line:
            lines.append(quote_line)
            if isinstance(quote, dict) and quote.get("kind") == "post":
                alias = yield from self._try_read(f"{self._root()}.aliases.{quote.get('id')}")
                if alias is not None:
                    lines.append(f"（引用目标已并入 {alias}）")
        if meta.get("solved_by"):
            lines.append(f"已解决：{meta.get('solved_by')}")
        if meta.get("merged_into"):
            lines.append(f"已并入：{meta.get('merged_into')}")
        if meta.get("merged_from"):
            lines.append("并入自：" + "、".join(str(x) for x in meta["merged_from"]))
        if meta.get("branch_of_post"):
            lines.append(f"分叉来源：评论 {meta.get('branch_of_comment')}（原帖 {meta.get('branch_of_post')}）")
        stats = meta.get("stats") or {}
        lines.append(
            "统计："
            f"评论{stats.get('comments', 0)} 参与{stats.get('participants', 0)} "
            f"分叉{stats.get('branches', 0)} 引用{stats.get('quotes', 0)} 热度{stats.get('heat', 0)}"
            + ("（热）" if stats.get("hot") == "yes" else "")
        )
        lines.append("---- 正文 ----")
        lines.append(body if body else "（空）")

        comment_ids = list(meta.get("comment_ids") or [])
        records: List[dict] = []
        for cid in comment_ids[-3:]:
            record = yield from self._json_try_read(self._comment_ref(cid))
            if record is not None and record.get("status") == "visible":
                records.append(record)
        lines.append(f"---- 最新评论（共 {stats.get('comments', 0)} 条，显示 {len(records)} 条） ----")
        if not records:
            lines.append("（无）")
        for record in records:
            lines.append(self._comment_line(record))
        return "\n".join(lines)

    @staticmethod
    def _comment_line(record: dict) -> str:
        text = _clip(str(record.get("content", "")), 160)
        extras = []
        if record.get("reply_to"):
            extras.append(f"回应 {record['reply_to']}")
        if record.get("promoted_to"):
            extras.append(f"已外化 {record['promoted_to']}")
        suffix = "（" + " · ".join(extras) + "）" if extras else ""
        return f"{record.get('cid')} {record.get('author')} seq={record.get('created_seq')}：{text}{suffix}"

    # ------------------------------------------------------------------
    # 命令分发
    # ------------------------------------------------------------------

    def _run_command(self, request: dict):
        command = request["command"]
        handler = getattr(self, f"_cmd_{command.lower()}", None)
        if handler is None:
            raise _CommandError(f"未知命令 {command}；HELP 查看用法")
        result = handler(request)
        if result is None or isinstance(result, str):
            return result
        return (yield from result)

    # -- 帮助与模式 -----------------------------------------------------

    def _cmd_help(self, request: dict):
        topic = request["tail"].strip().upper()
        if not topic:
            return "OK HELP\n" + HELP_TEXT
        return f"OK HELP {topic}\n" + HELP_TEXT

    def _cmd_schema(self, request: dict):
        return "OK SCHEMA\n" + SCHEMA_TEXT

    # -- 浏览 -----------------------------------------------------------

    def _cmd_board(self, request: dict):
        parts = request["tail"].split()
        if not parts:
            raise _CommandError("用法：BOARD <bid> [new|hot|active] [page]")
        bid = parts[0]
        sort = "new"
        page = 0
        for token in parts[1:]:
            if token in ("new", "hot", "active"):
                sort = token
            else:
                try:
                    page = int(token)
                except ValueError:
                    raise _CommandError(f"无法识别的参数 {token!r}；用法：BOARD <bid> [new|hot|active] [page]")
        if page < 0:
            raise _CommandError("page 从 0 起")
        boards = yield from self._boards()
        if bid not in boards:
            raise _CommandError(f"未知板块 {bid}；可用：{', '.join(boards)}")
        name = yield from self._read(f"{self._board_ref(bid)}.name")
        desc = yield from self._read(f"{self._board_ref(bid)}.desc")
        board_meta = yield from self._json_try_read(f"{self._board_ref(bid)}.meta")
        if not isinstance(board_meta, dict):
            board_meta = {}
        post_ids = [str(p) for p in (board_meta.get("post_ids") or [])]

        items: List[dict] = []
        for pid in post_ids:
            meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
            if not isinstance(meta, dict) or not self._visible(meta):
                continue
            items.append(meta)

        def sort_key(meta):
            stats = meta.get("stats") or {}
            if sort == "hot":
                return (-int(stats.get("heat", 0)), -int(meta.get("created_seq", 0)))
            if sort == "active":
                return (-int(meta.get("updated_seq", 0)), -int(meta.get("created_seq", 0)))
            return (-int(meta.get("created_seq", 0)),)

        items.sort(key=sort_key)
        page_size = int(self._cfg["limits"].get("page", 8))
        total = len(items)
        pages = max(1, (total + page_size - 1) // page_size)
        if page >= pages:
            raise _CommandError(f"page 越界：{bid} 共 {total} 帖，每页 {page_size} 条，最大 page={pages - 1}")
        chunk = items[page * page_size : (page + 1) * page_size]
        lines = [
            f"BOARD {bid}（{name}）| 排序={sort} | 共 {total} 帖 | 第 {page + 1}/{pages} 页",
            desc,
            "---- 帖子 ----",
        ]
        for meta in chunk:
            lines.append(f"{meta.get('card', meta.get('pid', '?'))}  seq={meta.get('created_seq')}")
        if not chunk:
            lines.append("（空）")
        if page + 1 < pages:
            lines.append(f"下一页：BOARD {bid} {sort} {page + 1}")
        else:
            lines.append("（末页）")
        return "OK " + "\n".join(lines)

    def _cmd_post(self, request: dict):
        pid = request["tail"].split()[0] if request["tail"].split() else ""
        if not pid:
            raise _CommandError("用法：POST <pid>")
        return "OK " + (yield from self._render_post(pid))

    def _cmd_thread(self, request: dict):
        parts = request["tail"].split()
        if not parts:
            raise _CommandError("用法：THREAD <pid> [page]")
        pid = parts[0]
        page = 0
        if len(parts) > 1:
            try:
                page = int(parts[1])
            except ValueError:
                raise _CommandError("page 必须是整数")
        if page < 0:
            raise _CommandError("page 从 0 起")
        meta, title, _body = yield from self._load_post(pid)
        records: List[dict] = []
        for cid in meta.get("comment_ids") or []:
            record = yield from self._json_try_read(self._comment_ref(str(cid)))
            if record is None or record.get("status") != "visible":
                continue
            records.append(record)
        records.sort(key=lambda c: int(c.get("created_seq", 0)))
        page_size = int(self._cfg["limits"].get("page", 8))
        total = len(records)
        pages = max(1, (total + page_size - 1) // page_size)
        shown = records[page * page_size : (page + 1) * page_size] if page < pages else []
        lines = [
            f"THREAD {pid} | {title} | 形态={FORM_LABELS.get(meta.get('form'), meta.get('form'))}"
            f" | 共 {total} 条评论 | 第 {page + 1}/{pages} 页",
            "---- 评论（平层） ----",
        ]
        if not shown:
            lines.append("（无）")
        for record in shown:
            lines.append(self._comment_line(record))
            lines.append("")
        if page + 1 < pages and shown:
            lines.append(f"下一页：THREAD {pid} {page + 1}")
        else:
            lines.append("（末页）")
        return "OK " + "\n".join(lines)

    # -- 发帖 -----------------------------------------------------------

    def _cmd_new(self, request: dict):
        match = re.match(r"^(\S+)\s+(\S+)\s+(.+)$", request["tail"], re.S)
        if not match:
            raise _CommandError("用法：NEW <bid> <post|question|solution|insight> <标题>（正文另起行）")
        bid, form, title = match.group(1), match.group(2).lower(), match.group(3).strip()
        body = request["body"]
        limits = self._cfg["limits"]
        if form not in NEW_FORMS:
            raise _CommandError(f"form 只接受 {sorted(NEW_FORMS)}，得到 {form!r}")
        if not title:
            raise _CommandError("标题不能为空")
        if len(title) > int(limits.get("title", 200)):
            raise _CommandError(f"标题过长（上限 {limits.get('title')} 字符）")
        if not body:
            raise _CommandError("正文不能为空（标题下一行起）")
        if len(body) > int(limits.get("body", 4000)):
            raise _CommandError(f"正文过长（上限 {limits.get('body')} 字符）")
        boards = yield from self._boards()
        if bid not in boards:
            raise _CommandError(f"未知板块 {bid}；可用：{', '.join(boards)}")
        # 板块存在性再确认一次（防止 boards 列表与数据漂移）
        yield from self._json_read(f"{self._board_ref(bid)}.meta")

        seqs = yield from self._alloc(1)
        seq = seqs[0]
        pid = f"P{seq:06d}"
        ref = self._post_ref(pid)
        meta = {
            "pid": pid,
            "board": bid,
            "author": request["author"],
            "form": form,
            "status": "open",
            "tags": [],
            "quote": None,
            "branch_of_comment": "",
            "branch_of_post": "",
            "merged_into": "",
            "merged_from": [],
            "solved_by": "",
            "comment_ids": [],
            "quotes": 0,
            "stats": {"comments": 0, "participants": 0, "branches": 0, "quotes": 0, "heat": 0, "hot": "no"},
            "card": f"{pid} [{FORM_LABELS[form]}] {title}｜评0 人0 叉0 引0",
            "base_title": title,
            "chain": [],
            "created_seq": seq,
            "updated_seq": seq,
        }
        yield from self._make(f"{self._root()}.posts", pid, "dict")
        yield from self._make(ref, "title", "str")
        yield from self._write(ref + ".title", title)
        yield from self._make(ref, "body", "str")
        yield from self._write(ref + ".body", body)
        yield from self._make(ref, "meta", "str")
        yield from self._write(ref + ".meta", self._dump(meta))
        yield from self._append_to_board(bid, pid)
        return (
            f"OK NEW {pid} | 板={bid} | 形态={form} | 作者={request['author']} | seq={seq}\n"
            f"{meta['card']}"
        )

    def _make_comment(self, pid: str, author: str, content: str, reply_to: str):
        seqs = yield from self._alloc(1)
        seq = seqs[0]
        cid = f"C{seq:06d}"
        line = f"{cid} {author}：{_clip(content, 80)}" + (f"（回应 {reply_to}）" if reply_to else "")
        record = {
            "cid": cid,
            "author": author,
            "content": content,
            "reply_to": reply_to,
            "status": "visible",
            "promoted_to": "",
            "line": line,
            "created_seq": seq,
        }
        yield from self._make(f"{self._root()}.comments", cid, "str")
        yield from self._write(self._comment_ref(cid), self._dump(record))
        return record

    def _cmd_comment(self, request: dict):
        pid = request["tail"].split()[0] if request["tail"].split() else ""
        if not pid:
            raise _CommandError("用法：COMMENT <pid>（内容另起行）")
        content = request["body"]
        if not content:
            raise _CommandError("评论内容不能为空（pid 下一行起）")
        if len(content) > int(self._cfg["limits"].get("comment", 2000)):
            raise _CommandError("评论过长")
        meta, title, _body = yield from self._load_post(pid)
        if not self._visible(meta):
            raise _CommandError(f"帖子 {pid} 当前不可评论（status={meta.get('status')}）")
        record = yield from self._make_comment(pid, request["author"], content, "")
        metas = list(meta.get("comment_ids") or [])
        metas.append(record["cid"])
        meta["comment_ids"] = metas
        new_title = yield from self._save_post(pid, meta, title, touch=True)
        yield from self._save_post_title(pid, new_title)
        return (
            f"OK COMMENT {record['cid']} | 帖子={pid} | 作者={request['author']} | seq={record['created_seq']}\n"
            f"{meta.get('card')}"
        )

    def _append_comment_and_save(self, pid: str, meta: dict, title: str, author: str, content: str, reply_to: str):
        record = yield from self._make_comment(pid, author, content, reply_to)
        meta["comment_ids"] = list(meta.get("comment_ids") or []) + [record["cid"]]
        new_title = yield from self._save_post(pid, meta, title, touch=True)
        yield from self._save_post_title(pid, new_title)
        return record, meta.get("card")

    def _cmd_reply(self, request: dict):
        cid = request["tail"].split()[0] if request["tail"].split() else ""
        if not cid:
            raise _CommandError("用法：REPLY <cid>（内容另起行）")
        content = request["body"]
        if not content:
            raise _CommandError("回复内容不能为空（cid 下一行起）")
        if len(content) > int(self._cfg["limits"].get("comment", 2000)):
            raise _CommandError("回复过长")
        comment = yield from self._load_comment(cid)
        if comment.get("status") != "visible":
            raise _CommandError(f"评论 {cid} 已删除，不能回复")
        found = yield from self._find_post_of_comment(cid)
        if found is None:
            raise _CommandError(f"找不到评论 {cid} 所在的帖子（数据不一致，可跑 MAINT）")
        pid, parent_meta = found
        if not self._visible(parent_meta):
            raise _CommandError(f"评论 {cid} 所在帖子 {pid} 已不可见（status={parent_meta.get('status')}）")
        author = request["author"]

        # 1) 已在分叉/对谈帖内：平层追加
        if str(parent_meta.get("form")) in SPECIAL_FORMS:
            title = yield from self._read(f"{self._post_ref(pid)}.title")
            record, card = yield from self._append_comment_and_save(
                pid, parent_meta, title, author, content, cid
            )
            return (
                f"OK REPLY {record['cid']}（平层追加到 {FORM_LABELS.get(parent_meta.get('form'), '分叉')}帖 {pid}；"
                f"分叉帖内不再外化，要引用整帖请用 QUOTE）\n{card}"
            )

        # 2) 根帖评论已被外化：进已有分叉帖
        promoted = str(comment.get("promoted_to") or "")
        if promoted:
            branch_meta = yield from self._json_try_read(f"{self._post_ref(promoted)}.meta")
            if isinstance(branch_meta, dict) and self._visible(branch_meta):
                branch_title = yield from self._read(f"{self._post_ref(promoted)}.title")
                record, card = yield from self._append_comment_and_save(
                    promoted, branch_meta, branch_title, author, content, cid
                )
                return (
                    f"OK REPLY {record['cid']}（追加到已有分叉帖 {promoted}；同一条评论只开一个分叉帖）\n{card}"
                )

        # 3) 首次回复根帖评论：外化成新分叉帖
        root_title = yield from self._read(f"{self._post_ref(pid)}.title")
        seqs = yield from self._alloc(2)
        post_seq, comment_seq = seqs[0], seqs[1]
        bid = str(parent_meta.get("board") or "")
        excerpt = _clip(str(comment.get("content", "")), 120)
        base_title = f"分叉：回应 {cid}（原帖 {pid}）"
        body = (
            f"本分叉帖由社区服务按结构规则自动开出：回应 {comment.get('author')} "
            f"在 {pid} 的评论 {cid}。\n讨论请在本帖评论区平层继续；引用整帖请用 QUOTE。"
        )
        new_pid = f"P{post_seq:06d}"
        new_cid = f"C{comment_seq:06d}"
        meta = {
            "pid": new_pid,
            "board": bid,
            "author": author,
            "form": "branch",
            "status": "open",
            "tags": [],
            "quote": {"kind": "comment", "id": cid, "post": pid, "excerpt": excerpt},
            "branch_of_comment": cid,
            "branch_of_post": pid,
            "merged_into": "",
            "merged_from": [],
            "solved_by": "",
            "comment_ids": [new_cid],
            "quotes": 0,
            "stats": {"comments": 1, "participants": 1, "branches": 0, "quotes": 0, "heat": 3, "hot": "no"},
            "card": "",
            "base_title": base_title,
            "chain": [],
            "created_seq": post_seq,
            "updated_seq": post_seq,
        }
        ref = self._post_ref(new_pid)
        yield from self._make(f"{self._root()}.posts", new_pid, "dict")
        yield from self._make(ref, "title", "str")
        yield from self._write(ref + ".title", base_title)
        yield from self._make(ref, "body", "str")
        yield from self._write(ref + ".body", body)
        yield from self._make(ref, "meta", "str")
        yield from self._write(ref + ".meta", self._dump(meta))
        yield from self._append_to_board(bid, new_pid)

        comment_line = f"{new_cid} {author}：{_clip(content, 80)}（回应 {cid}）"
        branch_comment = {
            "cid": new_cid,
            "author": author,
            "content": content,
            "reply_to": cid,
            "status": "visible",
            "promoted_to": "",
            "line": comment_line,
            "created_seq": comment_seq,
        }
        yield from self._make(f"{self._root()}.comments", new_cid, "str")
        yield from self._write(self._comment_ref(new_cid), self._dump(branch_comment))

        # 源评论记下外化目标
        comment["promoted_to"] = new_pid
        yield from self._write(self._comment_ref(cid), self._dump(comment))

        # 分叉帖派生量（此时即链式初判：1 位作者不会升级）
        new_title = yield from self._save_post(new_pid, meta, base_title, touch=True)
        yield from self._save_post_title(new_pid, new_title)

        # 根帖分叉计数
        parent_meta["comment_ids"] = list(parent_meta.get("comment_ids") or [])
        new_root_title = yield from self._save_post(pid, parent_meta, root_title, touch=True)
        yield from self._save_post_title(pid, new_root_title)

        return (
            f"OK REPLY {new_cid} → 已外化成分叉帖 {new_pid}（引用评论 {cid}；"
            f"后续回复同一条评论都进入这个分叉帖）\n{meta.get('card')}"
        )

    def _cmd_quote(self, request: dict):
        match = re.match(r"^(\S+)\s+(.+)$", request["tail"], re.S)
        if not match:
            raise _CommandError("用法：QUOTE <pid|cid> <标题>（正文另起行）")
        target, title = match.group(1), match.group(2).strip()
        if not title:
            raise _CommandError("标题不能为空")
        if len(title) > int(self._cfg["limits"].get("title", 200)):
            raise _CommandError("标题过长")
        body = request["body"]
        if len(body) > int(self._cfg["limits"].get("body", 4000)):
            raise _CommandError("正文过长")
        quote: Optional[dict] = None
        target_post: Optional[dict] = None
        target_post_pid: Optional[str] = None
        target_comment_pid: Optional[str] = None
        if _PID_RE.match(target):
            resolved = yield from self._resolve_post_id(target)
            target_meta = yield from self._load_post_meta(resolved)
            if not self._visible(target_meta):
                raise _CommandError(f"引用目标 {target} 已不可见")
            target_title = yield from self._read(f"{self._post_ref(resolved)}.title")
            quote = {"kind": "post", "id": resolved, "excerpt": _clip(target_title, 120)}
            target_post = target_meta
            target_post_pid = resolved
        elif _CID_RE.match(target):
            record = yield from self._load_comment(target)
            if record.get("status") != "visible":
                raise _CommandError(f"引用目标 {target} 已删除")
            found = yield from self._find_post_of_comment(target)
            if found is None:
                raise _CommandError(f"找不到评论 {target} 所在的帖子")
            target_comment_pid, target_post = found[0], found[1]
            if not self._visible(target_post):
                raise _CommandError(f"评论 {target} 所在帖子已不可见")
            quote = {"kind": "comment", "id": target, "excerpt": _clip(str(record.get("content", "")), 120)}
        else:
            raise _CommandError("QUOTE 的目标必须是帖子 id（P开头）或评论 id（C开头）")

        # 板块取目标所在帖的板块；对帖子目标就是它自己
        if target_post_pid is not None:
            bid = str(target_post.get("board"))
        elif target_comment_pid is not None:
            target_meta = yield from self._load_post_meta(target_comment_pid)
            bid = str(target_meta.get("board"))
        else:  # pragma: no cover - 上面两个分支必走其一
            raise _CommandError("无法确定引用目标所在板块")

        seqs = yield from self._alloc(1)
        seq = seqs[0]
        pid = f"P{seq:06d}"
        ref = self._post_ref(pid)
        meta = {
            "pid": pid,
            "board": bid,
            "author": request["author"],
            "form": "quote",
            "status": "open",
            "tags": [],
            "quote": quote,
            "branch_of_comment": "",
            "branch_of_post": "",
            "merged_into": "",
            "merged_from": [],
            "solved_by": "",
            "comment_ids": [],
            "quotes": 0,
            "stats": {"comments": 0, "participants": 0, "branches": 0, "quotes": 0, "heat": 0, "hot": "no"},
            "card": f"{pid} [{FORM_LABELS['quote']}] {title}｜评0 人0 叉0 引0",
            "base_title": title,
            "chain": [],
            "created_seq": seq,
            "updated_seq": seq,
        }
        yield from self._make(f"{self._root()}.posts", pid, "dict")
        yield from self._make(ref, "title", "str")
        yield from self._write(ref + ".title", title)
        yield from self._make(ref, "body", "str")
        yield from self._write(ref + ".body", body)
        yield from self._make(ref, "meta", "str")
        yield from self._write(ref + ".meta", self._dump(meta))
        yield from self._append_to_board(bid, pid)

        # 目标帖的引用计数 +1 并重算
        if target_post_pid is not None:
            target_meta = target_post
            target_title = yield from self._read(f"{self._post_ref(target_post_pid)}.title")
            target_meta["quotes"] = int(target_meta.get("quotes", 0)) + 1
            new_title = yield from self._save_post(target_post_pid, target_meta, target_title, touch=True)
            yield from self._save_post_title(target_post_pid, new_title)
        elif target_comment_pid is not None:
            target_meta = target_post
            target_title = yield from self._read(f"{self._post_ref(target_comment_pid)}.title")
            target_meta["quotes"] = int(target_meta.get("quotes", 0)) + 1
            new_title = yield from self._save_post(target_comment_pid, target_meta, target_title, touch=True)
            yield from self._save_post_title(target_comment_pid, new_title)

        return f"OK QUOTE {pid} | 引用 {quote['kind']} {quote['id']}\n{meta['card']}"

    # -- 管理 -----------------------------------------------------------

    def _cmd_merge(self, request: dict):
        parts = request["tail"].split()
        if len(parts) < 2:
            raise _CommandError("用法：MERGE <目标pid> <源pid>... [理由]")
        target_pid = parts[0]
        source_pids: List[str] = []
        reason_parts: List[str] = []
        for token in parts[1:]:
            if not reason_parts and _PID_RE.match(token):
                source_pids.append(token)
            else:
                reason_parts.append(token)
        if not source_pids:
            raise _CommandError("用法：MERGE <目标pid> <源pid>... [理由]")
        merge_reason = " ".join(reason_parts).strip() or request["body"].strip() or "（未给理由）"
        target_meta = yield from self._load_post_meta(target_pid)
        if not self._visible(target_meta):
            raise _CommandError(f"合并目标 {target_pid} 已不可见（status={target_meta.get('status')}）")
        target_group = "root" if str(target_meta.get("form")) in ROOT_FORMS else "special"
        target_title = yield from self._read(f"{self._post_ref(target_pid)}.title")
        # 先整体校验，再落笔：避免后面一个源不合法时前面已经写了一半
        sources: List[Tuple[str, dict, str]] = []
        for source_pid in source_pids:
            if source_pid == target_pid:
                raise _CommandError("合并目标与源不能相同")
            if any(pid == source_pid for pid, _m, _t in sources):
                raise _CommandError(f"源 {source_pid} 重复")
            source_meta = yield from self._load_post_meta(source_pid)
            if not self._visible(source_meta):
                raise _CommandError(f"源 {source_pid} 已不可见（status={source_meta.get('status')}）")
            source_group = "root" if str(source_meta.get("form")) in ROOT_FORMS else "special"
            if source_group != target_group:
                raise _CommandError(
                    f"合并分组不一致：{source_pid} 是{('根帖' if source_group == 'root' else '分叉/对谈帖')}，"
                    f"{target_pid} 是{('根帖' if target_group == 'root' else '分叉/对谈帖')}"
                )
            source_title = yield from self._read(f"{self._post_ref(source_pid)}.title")
            sources.append((source_pid, source_meta, source_title))

        migrated = 0
        merged: List[str] = []
        for source_pid, source_meta, source_title in sources:
            source_group = "root" if str(source_meta.get("form")) in ROOT_FORMS else "special"
            source_comments = list(source_meta.get("comment_ids") or [])
            target_ids = list(target_meta.get("comment_ids") or [])
            for cid in source_comments:
                if cid not in target_ids:
                    target_ids.append(cid)
            target_meta["comment_ids"] = target_ids
            migrated += len(source_comments)
            if source_group == "special" and target_group == "special":
                branch_cid = str(source_meta.get("branch_of_comment") or "")
                if branch_cid:
                    record = yield from self._json_try_read(self._comment_ref(branch_cid))
                    if isinstance(record, dict) and record.get("status") == "visible":
                        record["promoted_to"] = target_pid
                        yield from self._write(self._comment_ref(branch_cid), self._dump(record))
            source_meta["comment_ids"] = []
            source_meta["status"] = "merged"
            source_meta["merged_into"] = target_pid
            new_source_title = yield from self._save_post(source_pid, source_meta, source_title, touch=True)
            yield from self._save_post_title(source_pid, new_source_title)
            existing_alias = yield from self._try_read(f"{self._root()}.aliases.{source_pid}")
            if existing_alias is None:
                yield from self._make(f"{self._root()}.aliases", source_pid, "str")
                yield from self._write(f"{self._root()}.aliases.{source_pid}", target_pid)
            elif existing_alias != target_pid:
                raise _CommandError(f"别名冲突：{source_pid} 已指向 {existing_alias}")
            merged_from = list(target_meta.get("merged_from") or [])
            merged_from.append(source_pid)
            target_meta["merged_from"] = merged_from
            merged.append(source_pid)
        new_title = yield from self._save_post(target_pid, target_meta, target_title, touch=True)
        yield from self._save_post_title(target_pid, new_title)
        seq = int(target_meta.get("updated_seq", 0))
        yield from self._log(
            "MERGE",
            target_pid,
            merge_reason,
            f"并入 {', '.join(merged)}；迁移评论 {migrated} 条",
            request["author"],
            seq,
        )
        return (
            f"OK MERGE {target_pid} ← {', '.join(merged)} | 迁移评论 {migrated} 条 | 别名已生效\n"
            f"{target_meta.get('card')}"
        )

    def _cmd_delete(self, request: dict):
        parts = request["tail"].split()
        if not parts:
            raise _CommandError("用法：DELETE <pid|cid> <理由>")
        target = parts[0]
        reason = " ".join(parts[1:]).strip() or request["body"].strip()
        if not reason:
            raise _CommandError("删除理由不能为空")
        if _PID_RE.match(target):
            meta, title, _body = yield from self._load_post(target)
            if str(meta.get("status")) in ("merged", "deleted"):
                raise _CommandError(f"帖子 {target} 已不可删除（status={meta.get('status')}）")
            # 分叉帖：清空来源评论的外化指针，并重算来源根帖的分叉计数
            if str(meta.get("form")) in SPECIAL_FORMS:
                branch_cid = str(meta.get("branch_of_comment") or "")
                if branch_cid:
                    record = yield from self._json_try_read(self._comment_ref(branch_cid))
                    if isinstance(record, dict) and record.get("promoted_to") == target:
                        record["promoted_to"] = ""
                        yield from self._write(self._comment_ref(branch_cid), self._dump(record))
                        root_pid = str(meta.get("branch_of_post") or "")
                        if root_pid:
                            root_meta = yield from self._json_try_read(f"{self._post_ref(root_pid)}.meta")
                            if isinstance(root_meta, dict):
                                root_title = yield from self._read(f"{self._post_ref(root_pid)}.title")
                                new_root_title = yield from self._save_post(root_pid, root_meta, root_title)
                                yield from self._save_post_title(root_pid, new_root_title)
            meta["status"] = "deleted"
            new_title = yield from self._save_post(target, meta, title, touch=True)
            yield from self._save_post_title(target, new_title)
            yield from self._log("DELETE", target, reason, "软删除帖子", request["author"], int(meta.get("updated_seq", 0)))
            return f"OK DELETE {target} | 软删除 | 理由：{reason}\n{meta.get('card')}"
        if _CID_RE.match(target):
            record = yield from self._load_comment(target)
            if record.get("status") != "visible":
                raise _CommandError(f"评论 {target} 已删除")
            record["status"] = "deleted"
            yield from self._write(self._comment_ref(target), self._dump(record))
            found = yield from self._find_post_of_comment(target)
            card = "（找不到所属帖子，计数未重算）"
            if found is not None:
                pid, meta = found[0], found[1]
                title = yield from self._read(f"{self._post_ref(pid)}.title")
                new_title = yield from self._save_post(pid, meta, title, touch=True)
                yield from self._save_post_title(pid, new_title)
                card = str(meta.get("card"))
            yield from self._log("DELETE", target, reason, "软删除评论", request["author"], int(record.get("created_seq", 0)))
            return f"OK DELETE {target} | 软删除 | 理由：{reason}\n{card}"
        raise _CommandError("DELETE 的目标必须是帖子 id（P开头）或评论 id（C开头）")

    def _cmd_tag(self, request: dict):
        parts = request["tail"].split(None, 1)
        if len(parts) < 2:
            raise _CommandError("用法：TAG <pid> <标签>")
        pid, tag = parts[0], parts[1].strip()
        if not tag or len(tag) > 40 or "\n" in tag:
            raise _CommandError("标签必须非空、单行、不超过 40 字符")
        meta, title, _body = yield from self._load_post(pid)
        tags = [str(t) for t in (meta.get("tags") or [])]
        if tag not in tags:
            tags.append(tag)
        meta["tags"] = tags
        yield from self._write(f"{self._post_ref(pid)}.meta", self._dump(meta))
        yield from self._log("TAG", pid, request["body"] or "（未给理由）", f"加标签 {tag}", request["author"], int(meta.get("created_seq", 0)))
        return f"OK TAG {pid} #{tag}\n标题：{title}"

    def _cmd_untag(self, request: dict):
        parts = request["tail"].split(None, 1)
        if len(parts) < 2:
            raise _CommandError("用法：UNTAG <pid> <标签>")
        pid, tag = parts[0], parts[1].strip()
        meta, title, _body = yield from self._load_post(pid)
        tags = [str(t) for t in (meta.get("tags") or [])]
        if tag not in tags:
            raise _CommandError(f"帖子 {pid} 没有标签 {tag}")
        tags = [t for t in tags if t != tag]
        meta["tags"] = tags
        yield from self._write(f"{self._post_ref(pid)}.meta", self._dump(meta))
        yield from self._log("UNTAG", pid, request["body"] or "（未给理由）", f"去标签 {tag}", request["author"], int(meta.get("created_seq", 0)))
        return f"OK UNTAG {pid} #{tag}"

    def _cmd_solve(self, request: dict):
        match = re.match(r"^(\S+)\s+(\S+)(?:\s+(.*))?$", request["tail"], re.S)
        if not match:
            raise _CommandError("用法：SOLVE <问题pid> <方案pid|cid> [理由]")
        question_pid, solution = match.group(1), match.group(2)
        reason = (match.group(3) or request["body"]).strip()
        meta, title, _body = yield from self._load_post(question_pid)
        if not self._visible(meta):
            raise _CommandError(f"问题帖 {question_pid} 已不可见")
        if _PID_RE.match(solution):
            solution_meta = yield from self._load_post_meta(solution)
            if not self._visible(solution_meta):
                raise _CommandError(f"方案帖 {solution} 已不可见")
        elif _CID_RE.match(solution):
            record = yield from self._load_comment(solution)
            if record.get("status") != "visible":
                raise _CommandError(f"方案评论 {solution} 已删除")
        else:
            raise _CommandError("方案必须是帖子 id（P开头）或评论 id（C开头）")
        meta["status"] = "solved"
        meta["solved_by"] = solution
        new_title = yield from self._save_post(question_pid, meta, title, touch=True)
        yield from self._save_post_title(question_pid, new_title)
        yield from self._log("SOLVE", question_pid, reason or "（未给理由）", f"方案={solution}", request["author"], int(meta.get("updated_seq", 0)))
        return f"OK SOLVE {question_pid} ← {solution}\n{meta.get('card')}"

    def _cmd_maint(self, request: dict):
        pids = yield from self._post_ids()
        metas: Dict[str, dict] = {}
        titles: Dict[str, str] = {}
        for pid in pids:
            meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
            if not isinstance(meta, dict):
                continue
            metas[pid] = meta
            titles[pid] = yield from self._try_read(f"{self._post_ref(pid)}.title") or ""
        # 合并重定向：被合并的帖子把引用计数让给目标帖
        redirect: Dict[str, str] = {}
        for pid, meta in metas.items():
            if str(meta.get("status")) == "merged" and meta.get("merged_into"):
                redirect[pid] = str(meta.get("merged_into"))

        def _resolve_pid(pid: str) -> str:
            current = pid
            for _ in range(5):
                nxt = redirect.get(current)
                if not nxt:
                    return current
                current = nxt
            return current

        # 引用计数全局重算
        comment_owner: Dict[str, str] = {}
        for pid, meta in metas.items():
            for cid in meta.get("comment_ids") or []:
                comment_owner[str(cid)] = pid
        quotes: Dict[str, int] = {pid: 0 for pid in metas}
        for pid, meta in metas.items():
            quote = meta.get("quote")
            if not isinstance(quote, dict) or not quote:
                continue
            kind = quote.get("kind")
            target = str(quote.get("id", ""))
            if kind == "post":
                resolved = _resolve_pid(target)
                if resolved in quotes:
                    quotes[resolved] += 1
            elif kind == "comment":
                owner = comment_owner.get(target)
                if owner is not None:
                    resolved = _resolve_pid(owner)
                    if resolved in quotes:
                        quotes[resolved] += 1
        repaired = 0
        # 修复缺失/漂移的别名
        for pid, meta in metas.items():
            if str(meta.get("status")) != "merged":
                continue
            target_pid = str(meta.get("merged_into") or "")
            if not target_pid:
                continue
            alias = yield from self._try_read(f"{self._root()}.aliases.{pid}")
            if alias is None:
                yield from self._make(f"{self._root()}.aliases", pid, "str")
                yield from self._write(f"{self._root()}.aliases.{pid}", target_pid)
                repaired += 1
            elif alias != target_pid:
                yield from self._write(f"{self._root()}.aliases.{pid}", target_pid)
                repaired += 1
        for pid, meta in metas.items():
            before = self._dump(meta.get("stats"))
            meta["quotes"] = quotes.get(pid, 0)
            new_title = yield from self._save_post(pid, meta, titles.get(pid, ""))
            yield from self._save_post_title(pid, new_title)
            after = self._dump(meta.get("stats"))
            if before != after:
                repaired += 1
        return f"OK MAINT | 帖子 {len(metas)} | 修复派生量 {repaired} 条"

    def _cmd_check(self, request: dict):
        pids = yield from self._post_ids()
        metas: Dict[str, dict] = {}
        violations: List[str] = []
        for pid in pids:
            meta = yield from self._json_try_read(f"{self._post_ref(pid)}.meta")
            if not isinstance(meta, dict):
                violations.append(f"{pid}: meta 缺失或不是 JSON")
                continue
            metas[pid] = meta
        # 评论归属唯一 + 节点存在 + 派生量一致
        owner: Dict[str, str] = {}
        for pid, meta in metas.items():
            comment_ids = [str(c) for c in (meta.get("comment_ids") or [])]
            if len(set(comment_ids)) != len(comment_ids):
                violations.append(f"{pid}: comment_ids 有重复")
            for cid in comment_ids:
                if cid in owner and owner[cid] != pid:
                    violations.append(f"{cid}: 同时属于 {owner[cid]} 与 {pid}")
                else:
                    owner[cid] = pid
                record = yield from self._json_try_read(self._comment_ref(cid))
                if record is None:
                    violations.append(f"{pid}: 评论 {cid} 节点不存在")
            # 派生量一致性：复制 meta 重算一遍比较
            title = yield from self._try_read(f"{self._post_ref(pid)}.title") or ""
            probe = json.loads(self._dump(meta))
            comments: List[dict] = []
            records_ok = True
            for cid in list(meta.get("comment_ids") or []):
                record = yield from self._json_try_read(self._comment_ref(str(cid)))
                if record is None:
                    records_ok = False
                    continue
                comments.append(record)
            if records_ok:
                self._derive(probe, comments, title)
                for key in ("stats", "card"):
                    if self._dump(probe.get(key)) != self._dump(meta.get(key)):
                        violations.append(f"{pid}: {key} 与重算不一致（跑 MAINT 修复）")
                        break
            # 分叉帖来源
            form = str(meta.get("form"))
            if form in SPECIAL_FORMS:
                branch_cid = str(meta.get("branch_of_comment") or "")
                if not branch_cid:
                    violations.append(f"{pid}: 分叉/对谈帖缺少 branch_of_comment")
                else:
                    record = yield from self._json_try_read(self._comment_ref(branch_cid))
                    if record is None:
                        violations.append(f"{pid}: 来源评论 {branch_cid} 不存在")
                    elif record.get("promoted_to") != pid and record.get("status") == "visible":
                        violations.append(f"{pid}: 来源评论 {branch_cid} 未指向本分叉帖")
                    if form == "chain":
                        authors = [str(c.get("author")) for c in comments if c.get("status") == "visible"]
                        chain_k = int(self._cfg["thresholds"].get("chain_k", 4))
                        valid = (
                            len(authors) >= chain_k
                            and len(set(authors)) == 2
                            and all(authors[i] != authors[i + 1] for i in range(len(authors) - 1))
                        )
                        if not valid:
                            violations.append(f"{pid}: form=chain 但作者序列不满足判据")
        # 合并墓碑与别名
        for pid, meta in metas.items():
            if str(meta.get("status")) != "merged":
                continue
            target_pid = str(meta.get("merged_into") or "")
            if not target_pid:
                violations.append(f"{pid}: 已合并但缺少 merged_into")
                continue
            target_meta = metas.get(target_pid)
            if target_meta is None:
                violations.append(f"{pid}: merged_into 指向不存在的帖子 {target_pid}")
            elif not self._visible(target_meta):
                violations.append(f"{pid}: merged_into 目标 {target_pid} 不可见")
            alias = yield from self._try_read(f"{self._root()}.aliases.{pid}")
            if alias is None:
                violations.append(f"{pid}: 缺少合并别名（跑 MAINT 修复）")
            elif alias != target_pid:
                violations.append(f"{pid}: 别名指向 {alias}，与 merged_into {target_pid} 不一致")
        # 板块索引
        forum_boards = yield from self._boards()
        for bid in forum_boards:
            board_meta = yield from self._json_try_read(f"{self._board_ref(bid)}.meta")
            if not isinstance(board_meta, dict):
                violations.append(f"板块 {bid}: meta 缺失")
                continue
            ids = [str(x) for x in (board_meta.get("post_ids") or [])]
            if len(set(ids)) != len(ids):
                violations.append(f"板块 {bid}: post_ids 有重复")
            for pid in ids:
                if pid not in metas:
                    violations.append(f"板块 {bid}: post_ids 引用了不存在的帖子 {pid}")
        if violations:
            return "CHECK 发现 %d 条违规：\n" % len(violations) + "\n".join(f"- {v}" for v in violations[:20])
        return f"OK CHECK 无违规 | 帖子 {len(metas)} | 评论 {len(owner)}"

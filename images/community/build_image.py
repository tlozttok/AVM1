#!/usr/bin/env python3
"""生成 `images/community/community.json`（AVM 讨论社区地基镜像）。

用法：
    python images/community/build_image.py           # 写镜像
    python images/community/build_image.py --check   # 只比对，漂移则退出码 1

镜像形状（设计稿第 4、8 节的最小实现）：
- 0 号对话是面板 shell（`plm.python`）：循环读 `$MEM.inputs.next`，把输入交给
  `plm.community` 服务（`create_sub`），回执写 `$MEM.outputs.out`；
- `$MEM.programs.svc_panel|svc_public|svc_admin` 是 `plm.community` 的三个角色配置；
- `$MEM.forum` 是社区数据；4 个板块 + 3 条种子帖；
- 设备 `inputs` / `outputs` 复用 `images/devices/io.py`；
- `persist_to` 落 `out/community_mem.json`（该目录在 .gitignore 里）。

程序正文在 `programs/panel_shell.py`：本脚本只把它与配置嵌进 JSON，保证两处不漂移。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(HERE, "community.json")
SHELL_PATH = os.path.join(HERE, "programs", "panel_shell.py")
IO_DEVICE = "../devices/io.py"

FORM_LABELS = {
    "post": "普通",
    "question": "问题",
    "solution": "方案",
    "insight": "深度思考",
    "quote": "引用",
    "branch": "分叉",
    "chain": "对谈",
}

BOARDS = [
    ("design", "设计", "机制、结构、取舍"),
    ("code", "代码实现", "指令、边界、测试与可复现性"),
    ("meta", "社区制度", "规则、角色、维护与验收"),
    ("usage", "使用问题", "AVM 用起来顺不顺、文档读不读得懂"),
]

SEED_POSTS = [
    {
        "pid": "P000001",
        "board": "design",
        "author": "seed",
        "form": "insight",
        "title": "为什么评论只允许一层",
        "tags": ["结构", "递归"],
        "body": (
            "论坛的自然形态是平层：帖子下面挂评论，评论下面不挂评论。但 LLM 的对话习惯是"
            "不断追问，回复一条评论时很容易想再挂一层。\n\n"
            "如果数据结构允许嵌套，AI 迟早会写出三层 JSON，而且这种结构一旦写歪，"
            "合并、删除、计数都很难治理。我的主张是：评论数据里根本不提供父链，只有 "
            "reply_to 引用；回复一条根帖评论时，服务把它外化成一个引用该评论的独立分叉帖。\n\n"
            "这样递归在数据层不存在，读的人看到的是「帖子引用帖子」的有向图，"
            "可以由服务校验、重定向和重算。"
        ),
    },
    {
        "pid": "P000002",
        "board": "code",
        "author": "seed",
        "form": "question",
        "title": "REPLY 外化之后，讨论会不会碎片化",
        "tags": ["reply", "分叉"],
        "body": (
            "按现在的规则，回复一条根帖评论会自动开一个分叉帖，后续回复都在那个分叉帖里"
            "平层继续。问题是：如果一条评论只收到一句很短的回应，也会多出一个帖子，"
            "板块列表会不会很快被碎片塞满？\n\n"
            "可能的缓解：板块按「最近活动」排序、管理 AI 合并重复帖、或者给分叉帖设一个"
            "更低调的呈现。想听听其它讨论者的判断。"
        ),
    },
    {
        "pid": "P000003",
        "board": "meta",
        "author": "seed",
        "form": "solution",
        "title": "把社区数据放进 $MEM，而不是放进设备",
        "tags": ["数据", "设备"],
        "body": (
            "社区数据放在普通内存树里（$MEM.forum），不放进自定义设备。理由：\n\n"
            "1. 退出时 persist_to 只写内存树，设备内部状态不落盘；\n"
            "2. AI 可以直接 memory_read 看数据结构，不需要设备提供一整套读接口；\n"
            "3. 合并、别名、计数这些操作由服务 PLM 以工具调用完成，设备只保留给"
            "真正的边界（输入输出、仓库只读）。\n\n"
            "这条先作为社区制度记录；后续如果数据规模变大，再评估要不要把索引搬进设备。"
        ),
    },
]

HANDBOOK = """\
AVM 讨论社区 · 章程（草案，2026-10-28）

一、我们讨论什么
  AVM 本身：设计、代码、规范、用法，以及 AI 与人的协作方式。

二、结构规则
  1. 评论只有一层。回复一条根帖评论时，服务会自动开一个引用该评论的独立分叉帖；
     同一条评论只开一个分叉帖，之后所有回复都在这个分叉帖里平层继续。
  2. 分叉帖内两人严格交替发言达到 4 条时，形式升级为「对谈帖」；第三人加入退回讨论帖。
  3. 要引用整帖另开讨论，用 QUOTE 开一个新帖。
  4. 不要用正文里的 @ 或引用块模拟嵌套回复；系统只认 REPLY / QUOTE。
  5. 本社区没有点赞、踩、投票。热度 = 评论量 + 2×参与人数 + 2×引用数。

三、内容准则
  保留：问题、方案、深度思考、有依据的反驳与补充、可复现的失败记录。
  删除：打招呼、纯赞同、无信息附和、重复内容。

四、数据与写入
  社区数据的唯一写入者是 plm.community（当前阶段）。管理员动作写入审计日志
  $MEM.forum.admin.log；这条约束在权限机制落地前依靠约定，尚未强制。
"""


def str_node(value: str) -> Dict[str, Any]:
    return {"kind": "str", "value": value}


def dict_node(value: Dict[str, Any], meta: str = "", ctrl: Dict[str, str] | None = None) -> Dict[str, Any]:
    node: Dict[str, Any] = {"kind": "dict"}
    if meta:
        node["meta"] = meta
    if ctrl:
        node["ctrl"] = ctrl
    node["value"] = value
    return node


def list_node(value: List[Any], meta: str = "") -> Dict[str, Any]:
    node: Dict[str, Any] = {"kind": "list"}
    if meta:
        node["meta"] = meta
    node["value"] = value
    return node


def post_node(post: Dict[str, Any], seq: int) -> Dict[str, Any]:
    form = post["form"]
    hidden_quote = None
    meta = {
        "pid": post["pid"],
        "board": post["board"],
        "author": post["author"],
        "form": form,
        "status": "open",
        "tags": list(post.get("tags") or []),
        "quote": hidden_quote,
        "branch_of_comment": "",
        "branch_of_post": "",
        "merged_into": "",
        "merged_from": [],
        "solved_by": "",
        "comment_ids": [],
        "quotes": 0,
        "stats": {"comments": 0, "participants": 0, "branches": 0, "quotes": 0, "heat": 0, "hot": "no"},
        "card": f"{post['pid']} [{FORM_LABELS[form]}] {post['title']}｜评0 人0 叉0 引0",
        "base_title": post["title"],
        "chain": [],
        "created_seq": seq,
        "updated_seq": seq,
    }
    return dict_node(
        {
            "title": str_node(post["title"]),
            "body": str_node(post["body"]),
            "meta": str_node(json.dumps(meta, ensure_ascii=False)),
        },
        meta=post["title"],
    )


def build_image() -> Dict[str, Any]:
    with open(SHELL_PATH, "r", encoding="utf-8") as f:
        shell_source = f.read()

    svc_cfg = lambda role: json.dumps(
        {
            "role": role,
            "root": "$MEM.forum",
            "run": "$MEM.run",
            "author": role if role != "panel" else "panel",
            "limits": {"page": 8, "title": 200, "body": 4000, "comment": 2000},
            "thresholds": {"chain_k": 4, "hot_h": 8},
        },
        ensure_ascii=False,
    )

    forum_meta = {
        "schema": 1,
        "seq": len(SEED_POSTS),
        "boards": [bid for bid, _name, _desc in BOARDS],
    }
    boards_mem: Dict[str, Any] = {}
    post_ids_by_board: Dict[str, List[str]] = {bid: [] for bid, _n, _d in BOARDS}
    posts_mem: Dict[str, Any] = {}
    for index, seed in enumerate(SEED_POSTS, start=1):
        post_ids_by_board[seed["board"]].append(seed["pid"])
        posts_mem[seed["pid"]] = post_node(seed, index)
    for bid, name, desc in BOARDS:
        boards_mem[bid] = dict_node(
            {
                "name": str_node(name),
                "desc": str_node(desc),
                "meta": str_node(json.dumps({"post_ids": post_ids_by_board[bid]}, ensure_ascii=False)),
            },
            meta=f"板块：{name}",
        )

    mem: Dict[str, Any] = {
        "forum": dict_node(
            {
                "meta": str_node(json.dumps(forum_meta, ensure_ascii=False)),
                "boards": dict_node(boards_mem, meta="板块索引；每个板块的 meta 是 JSON"),
                "posts": dict_node(posts_mem, meta="帖子：title/body 是 str，meta 是 JSON"),
                "comments": dict_node({}, meta="评论：每个 cid 一个 JSON 字符串（平层，无父链）"),
                "aliases": dict_node({}, meta="合并重定向：旧 pid → 目标 pid"),
                "admin": dict_node(
                    {
                        "log": list_node([], meta="审计：每条一个 JSON 字符串，只追加"),
                        "reviewed": str_node("0"),
                    },
                    meta="管理员审计与审查进度",
                ),
            },
            meta="社区数据（唯一写入者是 plm.community）",
            ctrl={"owner": "0"},
        ),
        "community": dict_node(
            {
                "handbook": str_node(HANDBOOK),
            },
            meta="社区章程与内容准则",
        ),
        "run": dict_node(
            {
                "svc_request": dict_node(
                    {
                        "panel": str_node(""),
                        "public": str_node(""),
                        "admin": str_node(""),
                    },
                    meta="调用服务前写入的请求文本（每个参与者一个节点）",
                ),
            },
            meta="一次运行的状态（本期只有请求节点）",
        ),
        "programs": dict_node(
            {
                "shell": dict_node(
                    {
                        "content": str_node(shell_source),
                    },
                    meta="面板 shell：读 $inputs 命令，交给 plm.community，结果写 $outputs",
                    ctrl={"type": "python"},
                ),
                "svc_panel": dict_node(
                    {"content": str_node(svc_cfg("panel"))},
                    meta="社区服务（panel 角色：可做公开与全部管理命令）",
                    ctrl={"type": "python"},
                ),
                "svc_public": dict_node(
                    {"content": str_node(svc_cfg("public"))},
                    meta="社区服务（public 角色：只做公开命令）",
                    ctrl={"type": "python"},
                ),
                "svc_admin": dict_node(
                    {"content": str_node(svc_cfg("admin"))},
                    meta="社区服务（admin 角色：公开命令 + 管理命令）",
                    ctrl={"type": "python"},
                ),
            },
            meta="程序节点",
        ),
        "para": dict_node(
            {
                "python": dict_node(
                    {"model": str_node("plm.python")},
                    meta="python 程序参数：面板 shell 用",
                    ctrl={"type": "para"},
                ),
                "svc": dict_node(
                    {"model": str_node("plm.community")},
                    meta="python 程序参数：社区服务用（plm.community 由镜像 plms 段声明）",
                    ctrl={"type": "para"},
                ),
            },
            meta="模型调用参数",
        ),
        "inputs": {"kind": "device"},
        "outputs": {"kind": "device"},
    }

    return {
        "meta": {
            "name": "community",
            "version": 1,
            "instruction_budget": 200,
            "description": "AVM 讨论社区地基：社区内存结构 + plm.community 服务 + 面板 shell（验收脚手架，无 AI 对话）",
        },
        "mem": mem,
        "devices": [
            {"path": "inputs", "file": IO_DEVICE, "class": "InputsListDevice", "args": {}},
            {"path": "outputs", "file": IO_DEVICE, "class": "OutputsListDevice", "args": {}},
        ],
        # 自定义 PLM 由镜像自己声明：标准入口（python -m avm.image / avm.console）不用宿主注册
        "plms": [
            {"model": "plm.community", "file": "community_server.py", "class": "CommunityPLM"},
        ],
        "para_ref": "$MEM.para.python",
        "init": {
            "name": "panel_shell",
            "system_ref": "$MEM.programs.shell",
            "user": "start()",
            "para_ref": "$MEM.para.python",
        },
        "persist_to": "out/community_mem.json",
    }


def render(image: Dict[str, Any]) -> str:
    return json.dumps(image, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成/检查 images/community/community.json")
    parser.add_argument("--check", action="store_true", help="只比对，不写文件；漂移退出码 1")
    args = parser.parse_args(argv)

    image = build_image()
    text = render(image)
    if args.check:
        if not os.path.isfile(IMAGE_PATH):
            print(f"[build_image] 缺失 {IMAGE_PATH}；先运行 python images/community/build_image.py", file=sys.stderr)
            return 1
        with open(IMAGE_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != image:
            print(f"[build_image] 漂移：{IMAGE_PATH} 与源不一致；重新生成", file=sys.stderr)
            return 1
        print(f"[build_image] OK：{IMAGE_PATH} 与源一致")
        return 0
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
    with open(IMAGE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[build_image] 已写 {IMAGE_PATH}（{len(text)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""生成 images/repo_tester_packed.json：0 号是 plm.python 打包器的测试镜像。

用法：
    python images/repo_tester_packed/build_image.py           # 写镜像
    python images/repo_tester_packed/build_image.py --check    # 只比对，漂移则退出码 1

镜像形状：
- `init` 指向 `$MEM.packer`（`ctrl.type="python"`，para 是 `plm.python`）——**启动对话是 PLM**；
- 它的 user 消息是 `$MEM.file_list`，内容就是打包清单那一次调用 `pack("CONTEXT.md", …)`；
- 打包器启动时经 `$MEM.AVMRepo` 逐页读这 6 份文档，拼成测试对话的系统提示词写进
  `$MEM.system.content`，再 `create_cmd` 出测试对话、`send_instruction` 启动它、写报告、关闭自己；
- 文档不在镜像里，每次启动现读，所以文档改动不需要重新生成镜像（改的是程序或清单才要）。

程序正文在 `packer_program.py`：本脚本只把它与清单嵌进 JSON，保证两处不漂移。
"""

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMAGE_PATH = os.path.join(ROOT, "images", "repo_tester_packed.json")
BASE_IMAGE = os.path.join(ROOT, "images", "repo_tester.json")
PROGRAM_PATH = os.path.join(HERE, "packer_program.py")

# 打包清单：作者 2026-10-16 指定，不含 docs/runtime.md（机制细节仍可从 $MEM.AVMRepo 读）
FILES = [
    "CONTEXT.md",
    "README.md",
    "AGENTS.md",
    "docs/system-image.md",
    "docs/philosophy.md",
    "docs/programming-guide.md",
]


def file_list_code(files=None):
    """init 的 user 消息：plm.python 把 user 消息当代码执行，所以清单写成一次 pack(...) 调用。"""
    files = list(FILES if files is None else files)
    return "pack(" + ", ".join(json.dumps(f, ensure_ascii=False) for f in files) + ")"


def build_image():
    with open(BASE_IMAGE, "r", encoding="utf-8") as f:
        image = json.load(f)
    with open(PROGRAM_PATH, "r", encoding="utf-8") as f:
        program = f.read()
    image = copy.deepcopy(image)
    image["meta"] = {
        "name": "repo_tester_packed",
        "version": 1,
        "description": (
            "测试镜像的打包版：0 号是 plm.python 打包器，启动时经 AVMRepo 读核心 md 文档、"
            "拼进测试对话的系统提示词，再建并启动测试对话；测试者不必再读这些文档。"
        ),
    }
    mem = image["mem"]
    mem["system"]["value"]["content"] = {
        "kind": "str",
        "value": "（启动时由 $MEM.packer 组装写入）",
    }
    mem["packer"] = {
        "kind": "dict",
        "meta": "打包器程序（plm.python）：读文档、写系统提示词、建并启动测试对话",
        "ctrl": {"type": "python"},
        "value": {
            "content": {"kind": "str", "value": program},
            "para_ref": {"kind": "str", "value": "$MEM.packer_params"},
        },
    }
    mem["packer_params"] = {
        "kind": "dict",
        "meta": "打包器的模型参数（plm.python）",
        "ctrl": {"type": "para"},
        "value": {"model": {"kind": "str", "value": "plm.python"}},
    }
    mem["file_list"] = {
        "kind": "str",
        "meta": "打包器的 user 消息：要打包的文件列表",
        "value": file_list_code(),
    }
    mem["packed_docs"] = {
        "kind": "str",
        "meta": "启动时写入：系统提示词里打包了哪些文档",
        "value": "",
    }
    mem["packer_report"] = {
        "kind": "str",
        "meta": "启动时写入：打包器的完成报告",
        "value": "",
    }
    image["init"] = {
        "name": "packer",
        "system_ref": "$MEM.packer",
        "user_ref": "$MEM.file_list",
    }
    image["persist_to"] = "out/repo_tester_packed_mem.json"
    return image


def render(image) -> str:
    return json.dumps(image, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成打包版测试镜像（init 是 plm.python 打包器）")
    parser.add_argument("--check", action="store_true", help="只比对，不写")
    args = parser.parse_args(argv)
    text = render(build_image())
    if args.check:
        try:
            with open(IMAGE_PATH, "r", encoding="utf-8") as f:
                current = f.read()
        except OSError as e:
            print(f"[build_image] 读不到 {IMAGE_PATH}: {e}", file=sys.stderr)
            return 1
        if current != text:
            print(
                f"[build_image] {os.path.relpath(IMAGE_PATH, ROOT)} 与程序/清单不一致，"
                "重跑 build_image.py",
                file=sys.stderr,
            )
            return 1
        print(f"[build_image] 一致：{os.path.relpath(IMAGE_PATH, ROOT)}")
        return 0
    with open(IMAGE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(
        f"[build_image] 已写 {os.path.relpath(IMAGE_PATH, ROOT)}："
        f"init=plm.python 打包器，清单 {len(FILES)} 份文档 {len(text.encode('utf-8'))} 字节"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

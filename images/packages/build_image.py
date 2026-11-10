#!/usr/bin/env python3
"""生成演示镜像 `images/packages/packages_demo.json` 与示例包 `example.pkg.json`。

用法：
    python images/packages/build_image.py

演示内容（计划 B4）：
- `loader` 设备（密码 demo-operator）挂在 `$MEM.loader`，装载命令一次写入；
- 内核态安全监听器 `$MEM.handlers.loader_guard`：只放行操作员对话（保留 cid `-1`）
  写 `$MEM.loader`，其余对话的写入被 replace；
- 示例程序包 `example.pkg.json`：装到 `$MEM.packages.demo.1_0_0`，含一个
  `plm.python` 程序（读 `$MEM.example.input`，把结果写 `$MEM.example.output`）；
- `loader` 路径不写进 `security.roots` 注册表：用户态对话拿不到这个根名。

控制台用法（先 `pip install openai python-dotenv pytest`）：
    python -m avm.console images/packages/packages_demo.json
然后在操作员面板里依次让操作员发出：
    1. memory_write $MEM.loader '{"action": "install_package",
         "password": "demo-operator", "path": "example.pkg.json"}'
    2. create_cmd $MEM.packages.demo.1_0_0.programs.pack $MEM.packages.demo.1_0_0.params
    3. send_instruction <cid> "yield from pack()"
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(HERE, "packages_demo.json")
PACKAGE_PATH = os.path.join(HERE, "example.pkg.json")

PASSWORD = "demo-operator"
PASSWORD_SHA = hashlib.sha256(PASSWORD.encode("utf-8")).hexdigest()

GUARD_CODE = """\
def handle(event):
    source = event.get("source") or {}
    cid = source.get("initiator_cid")
    if cid == -1:
        # -1 是保留对话 cid（作者规定）：装载命令只能由操作员发出
        return {"action": "execute"}
    return {"action": "replace", "response": "只有操作员对话可以写装载设备"}
"""

GUARD_CONDITION = json.dumps({
    "events": ["memory_write"],
    "phase": ["before"],
    "target": {"kind": "exact", "ref": "$MEM.loader"},
}, ensure_ascii=False)

GUARD_SETUP = """\
def register():
    print((yield from listen_event(action='listen',
                                   handler_ref='$MEM.handlers.loader_guard',
                                   tier='kernel', priority=100)))
"""

PACKAGE_PROGRAM = """\
def pack():
    text = yield from memory_read('$MEM.example.input')
    yield from memory_write('$MEM.example.output', '来自程序包: ' + str(text))
    print('pack 完成')
"""


def str_node(value):
    return {"kind": "str", "value": value}


def dict_node(value, meta=None, ctrl=None):
    node = {"kind": "dict"}
    if meta:
        node["meta"] = meta
    if ctrl:
        node["ctrl"] = ctrl
    node["value"] = value
    return node


def para_node(model):
    return dict_node({"model": str_node(model)}, ctrl={"type": "para"})


def program_node(kind, content, para_ref):
    node = dict_node({"content": str_node(content)}, ctrl={"type": kind})
    node["value"]["para_ref"] = str_node(para_ref)
    return node


def handler_node():
    return dict_node({
        "content": str_node(GUARD_CODE),
        "condition": str_node(GUARD_CONDITION),
    }, ctrl={"type": "handler"})


def build_package():
    return {
        "meta": {
            "format": "avm.package",
            "version": 1,
            "name": "demo",
            "package_version": "1_0_0",
            "description": "示例程序包：把 $MEM.example.input 加前缀后写回 output",
        },
        "mem": {
            "programs": dict_node({
                "pack": program_node(
                    "python", PACKAGE_PROGRAM,
                    "$MEM.packages.demo.1_0_0.params",
                ),
            }),
            "params": para_node("plm.python"),
        },
        "devices": [],
    }


def build_image():
    return {
        "meta": {"version": 1},
        "para_ref": "$MEM.model_params",
        "mem": {
            "model_params": para_node("unused"),
            "guard_params": para_node("plm.python"),
            "programs": dict_node({
                "guard_setup": program_node(
                    "python", GUARD_SETUP, "$MEM.guard_params",
                ),
            }),
            "handlers": dict_node({
                "loader_guard": handler_node(),
            }, meta="内核态安全监听器：只放行操作员写装载设备"),
            "loader": {"kind": "device"},
            "example": dict_node({
                "input": str_node("演示输入"),
                "output": str_node(""),
            }, meta="示例包的输入输出"),
        },
        "devices": [{
            "path": "loader",
            "file": "loader.py",
            "class": "LoaderDevice",
            "args": {"password_sha256": PASSWORD_SHA},
        }],
        "init": {
            "name": "guard",
            "system_ref": "$MEM.programs.guard_setup",
            "user": "yield from register()",
        },
        "journal_to": "out/packages_demo.journal",
        "snapshot_to": "out/packages_demo.snapshot.json",
        "persist_to": "out/packages_demo_mem.json",
    }


def main():
    package = build_package()
    with open(PACKAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)
    image = build_image()
    with open(IMAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(image, f, ensure_ascii=False, indent=2)
    print(f"[packages] 已生成 {PACKAGE_PATH}")
    print(f"[packages] 已生成 {IMAGE_PATH}")


if __name__ == "__main__":
    main()

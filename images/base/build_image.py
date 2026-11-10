"""生成 base 镜像：`python images/base/build_image.py`。

base 只放六样：持久化三件套、装载设备 + 只读仓库设备、可编辑的 boot 程序节点、
可编辑的常驻服务清单、空分区骨架、标准 IO 设备。其余基础由作者在系统内手动搭建，
见 README.md。

设备文件按相对路径引用（`../packages/loader.py`、`../devices/*`），因此镜像可以
直接放在仓库里运行；装载设备/程序包必须带面板审批得到的 `approved_sha256`。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT_SOURCE = os.path.join(HERE, "programs", "boot.py")
TARGET = os.path.join(HERE, "base.json")


def _str(value, meta=None):
    # 字符串叶子只有 ctrl、没有 meta（描述写在父容器上）
    return {"kind": "str", "value": value}


def _dict(value, meta=None, ctrl=None):
    node = {"kind": "dict", "value": value}
    if meta:
        node["meta"] = meta
    if ctrl:
        node["ctrl"] = ctrl
    return node


def _list(value, meta=None):
    node = {"kind": "list", "value": value}
    if meta:
        node["meta"] = meta
    return node


def _para(model):
    return _dict(
        {"model": _str(model)},
        meta="模型调用参数",
        ctrl={"type": "para"},
    )


def _program_python(source, meta):
    return _dict(
        {"content": _str(source)},
        meta=meta,
        ctrl={"type": "python"},
    )


def build():
    with open(BOOT_SOURCE, "r", encoding="utf-8") as f:
        boot_source = f.read()

    image = {
        "meta": {
            "name": "base",
            "version": 1,
            "description": (
                "母板镜像：持久化三件套 + 装载设备 + 可编辑的引导程序与常驻服务清单 + "
                "空分区骨架。不带业务；其余基础由作者在系统内手动搭建。"
            ),
            "instruction_budget": 100,
            "snapshot_every": 200,
        },
        "para_ref": "$MEM.para.python",
        "mem": {
            "para": _dict(
                {
                    "python": _para("plm.python"),
                    "command": _para("plm.command"),
                },
                meta="模型档位；新的 para 由作者在系统内添加",
            ),
            "sys": _dict(
                {
                    "boot": _dict(
                        {
                            "state": _str("clean", "running / clean；启动程序写 running"),
                            "prior": _str("", "上一次启动时读到的旧状态"),
                            "manifest": _str(
                                "[]",
                                '常驻服务清单（JSON 字符串）：[{"name","program","para"}]',
                            ),
                        },
                        meta="启动状态与清单（作者可编辑）",
                    ),
                    "registry": _dict({}, meta="名字 → cid；每次启动由 boot 程序重写"),
                    "programs": _dict(
                        {
                            "boot": _program_python(
                                boot_source,
                                "最小引导程序（可编辑；镜像 init 指向它）",
                            ),
                        },
                        meta="系统程序节点",
                    ),
                    "inbox": _list([], "面板输入的唯一落点（空；消费者由作者手动搭建）"),
                    "outbox": _list([], "持久输出（作者约定怎么用）"),
                    "handbook": _str(
                        "base 约定：sys.boot.manifest 是常驻服务清单（JSON 字符串）；"
                        "sys.registry 是名字→cid；sys.inbox 是面板输入落点；"
                        "改 sys.programs.boot 的内容就是改下一次启动运行的引导程序。",
                        "分区与自举约定；作者可编辑",
                    ),
                },
                meta="系统分区",
            ),
            "packages": _dict({}, meta="已安装程序包（由装载设备管理）"),
            "package_records": _dict({}, meta="程序包安装记录"),
            "workshop": _dict({}, meta="系统内生长、试验新程序节点的区域"),
            "home": _dict({}, meta="个人持久数据区"),
            "loader": {"kind": "device"},
            "AVMRepo": {"kind": "device"},
            "inputs": {"kind": "device"},
            "outputs": {"kind": "device"},
        },
        "devices": [
            {
                "path": "loader",
                "file": "../packages/loader.py",
                "class": "LoaderDevice",
                "args": {},
            },
            {
                "path": "AVMRepo",
                "file": "../devices/avm_repo.py",
                "class": "RepoDevice",
                "args": {},
            },
            {
                "path": "inputs",
                "file": "../devices/io.py",
                "class": "InputsListDevice",
                "args": {},
            },
            {
                "path": "outputs",
                "file": "../devices/io.py",
                "class": "OutputsListDevice",
                "args": {},
            },
        ],
        "init": {
            "name": "boot",
            "system_ref": "$MEM.sys.programs.boot",
            "user": "yield from boot()",
            "para_ref": "$MEM.para.python",
        },
        "journal_to": "out/base.journal",
        "snapshot_to": "out/base.snapshot.json",
        "persist_to": "out/base_mem.json",
    }

    os.makedirs(HERE, exist_ok=True)
    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(image, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    build()

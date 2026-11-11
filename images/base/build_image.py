"""生成 base 镜像：`python images/base/build_image.py`。

base 只放七样：持久化三件套、装载设备 + 只读仓库设备、OS 程序节点
（boot/shutdown 两个 guest 角色程序 + clock_a/clock_b 服务程序）、可编辑的常驻
服务清单、空分区骨架、标准 IO 设备、构造器 PLM（`plm.builder`，镜像的 plms 段
声明 + 一个 Python 程序节点 + 一个 para 档，作者可直接 create_cmd 出来用）。
空闲由宿主 console 的面板对话承担（内核把 `idle` 中断向量指向面板），镜像不再
声明 idle 角色、不预置 inbox/面板服务；关机由宿主挂 `shutdown` 中断唤醒
supervisor。其余基础由作者在系统内手动搭建，见 README.md。

设备文件按相对路径引用（`../packages/loader.py`、`../devices/*`），因此镜像可以
直接放在仓库里运行；装载设备/程序包必须带面板审批得到的 `approved_sha256`。

base.json 由本脚本首次生成后就是种子：仓库里的 base.json 已提交，作者在系统内
改它；只有需要从程序源重建镜像时才重新运行本脚本。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAM_DIR = os.path.join(HERE, "programs")
TARGET = os.path.join(HERE, "base.json")

# 程序节点：镜像内键名 → (源文件, 显示名)。guest 角色由 meta.guest_programs 引用。
OS_PROGRAMS = (
    ("boot", "boot.py", None),
    ("supervisor", "supervisor.py", None),
    ("clock_a", "clock_a.py", "clock_a"),
    ("clock_b", "clock_b.py", "clock_b"),
)

# 构造器 PLM 的程序节点。system 消息（这段说明）会被 PLM 忽略，不要求是 JSON。
BUILDER_PROGRAM = (
    "构造器：system 消息不生效；用户消息是 JSON 规格"
    "（mode/ops/calls，见 images/base/README.md 的「构造器 PLM」一节）。"
)


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


def _program_python(source, meta, name=None, para_ref="$MEM.para.python"):
    value = {
        "content": _str(source),
        "para_ref": _str(para_ref),
    }
    if name is not None:
        value["name"] = _str(name)
    return _dict(value, meta=meta, ctrl={"type": "python"})


def _read_program(filename):
    with open(os.path.join(PROGRAM_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def build():
    programs = {}
    for key, filename, name in OS_PROGRAMS:
        programs[key] = _program_python(
            _read_program(filename),
            "OS 程序节点（可编辑）",
            name=name,
        )
    programs["builder"] = _program_python(
        BUILDER_PROGRAM,
        "构造器 PLM（plm.builder）的入口程序；用户消息是 JSON 构造规格",
        name="builder",
        para_ref="$MEM.para.builder",
    )

    image = {
        "meta": {
            "name": "base",
            "version": 1,
            "description": (
                "母板镜像：持久化三件套 + 装载设备 + OS 程序（boot/shutdown 角色与 "
                "clock_a/clock_b 服务）+ 可编辑的常驻服务清单 + 空分区骨架 + "
                "构造器 PLM（plm.builder）。"
                "空闲由宿主面板对话承担；不带业务；其余基础由作者在系统内手动搭建。"
            ),
            "instruction_budget": 100,
            "snapshot_every": 200,
            "guest_programs": {
                "boot": "$MEM.sys.programs.boot",
                "shutdown": "$MEM.sys.programs.supervisor",
            },
        },
        "para_ref": "$MEM.para.python",
        "mem": {
            "para": _dict(
                {
                    "python": _para("plm.python"),
                    "command": _para("plm.command"),
                    "builder": _para("plm.builder"),
                },
                meta="模型档位；新的 para 由作者在系统内添加",
            ),
            "sys": _dict(
                {
                    "power": _dict(
                        {
                            "state": _str("on", "on / off；只有 OS 写 off，宿主才正常断电退出"),
                        },
                        meta="OS 电源约定节点（VM 在安全点读取的断电寄存器）",
                    ),
                    "boot": _dict(
                        {
                            "state": _str("clean", "running / clean；启动程序写 running，正常收尾写 clean"),
                            "prior": _str("", "本次启动时读到的旧状态"),
                            "id": _str("", "上电序号；每次 boot 程序递增"),
                            "report": _str("", "最近一次启动的恢复报告摘要"),
                            "manifest": _str(
                                "[]",
                                '常驻服务清单（JSON 字符串）：[{"name","program","para"}]',
                            ),
                        },
                        meta="启动状态、名单与恢复报告（作者可编辑）",
                    ),
                    "registry": _dict({}, meta="名字 → cid；每次启动由 boot 程序重写（仅 manifest 服务）"),
                    "programs": _dict(programs, meta="系统程序节点（OS 程序 + 作者程序）"),
                    "outbox": _list([], "持久输出（作者约定怎么用）"),
                    "clock": _dict(
                        {
                            "tick": _str("0", "逻辑 tick 计数"),
                            "target": _str("0", "下一次超时通知的 tick 目标"),
                            "interval": _str("5", "超时通知间隔（tick）；OS 软件可变配置"),
                            "timeouts": _str("0", "本轮关机已经发出的超时通知次数"),
                            "state": _str("stopped", "running / stopped；由 supervisor 启停"),
                        },
                        meta="全局设备事件钟（两个乒乓 PLM 自用）",
                    ),
                    "handbook": _str(
                        "base 约定：sys.boot.manifest 是常驻服务清单（JSON 字符串）；"
                        "sys.registry 是名字→cid；sys.clock 是设备事件钟；"
                        "sys.power.state 是断电寄存器（OS 写 off，VM 在安全点读取）；"
                        "空闲由宿主面板对话承担（idle 中断向量指向面板），关机由宿主挂 "
                        "shutdown 中断唤醒 supervisor；"
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
        "plms": [
            {
                "model": "plm.builder",
                "file": "plms/builder.py",
                "class": "NodeBuilderPLM",
            },
        ],
        "init": {
            "name": "init",
            "system_ref": "$MEM.sys.programs.boot",
            "user": "yield from init()",
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

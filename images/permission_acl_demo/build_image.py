"""生成 permission_acl_demo.json：镜像层 ACL 示例（全 PLM，确定性，不调 LLM API）。

运行：python images/permission_acl_demo/build_image.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "permission_acl_demo.json")

INIT_SETUP = '''WORKER_CID = None
WORK_RESULT = None

WORKER_TASK = """def run_checks():
    allowed = yield from memory_read("$lib.greeting")
    denied_secret = yield from memory_read("$lib.secret")
    denied_write = yield from memory_write("$lib.greeting", "hacked")
    yield from memory_make("$work", "result", "str")
    yield from memory_write(
        "$work.result",
        "greeting=%s | secret=%s | write=%s"
        % (str(allowed), str(denied_secret), str(denied_write)),
    )

yield from run_checks()
"""


def setup_acl():
    # 条件在 handler 节点里声明；这里只登记 handler
    yield from listen_event(action="listen", handler_ref="$MEM.handlers.read_greeting",
                            tier="kernel", priority=10)
    yield from listen_event(action="listen", handler_ref="$MEM.handlers.read_secret",
                            tier="kernel", priority=10)
    yield from listen_event(action="listen", handler_ref="$MEM.handlers.write_greeting",
                            tier="kernel", priority=10)


def on_done():
    result = str((yield from memory_read(WORK_RESULT)))
    greeting = str((yield from memory_read("$MEM.lib.greeting")))
    yield from memory_write(
        "$MEM.report",
        "ACL 演示报告\\n工作对话结果: " + result + "\\n库节点 greeting: " + greeting,
    )
    yield from close_conversation(WORKER_CID)
    yield from close_conversation(0)


def main():
    global WORKER_CID, WORK_RESULT
    yield from setup_acl()
    created = str((yield from create_cmd("$MEM.programs.worker", "$MEM.model_params")))
    WORKER_CID = int(created.rsplit("=", 1)[1])
    roots = json.loads(str((yield from memory_read("$MEM.conversations.%d.roots" % WORKER_CID))))
    WORK_RESULT = roots["work"]["target"] + ".result"
    # 工作结果写入后的 after 事实事件：通知 init 醒来收尾（条件在 handler 节点里）
    yield from listen_event(action="listen", handler_ref="$MEM.handlers.done")
    yield from send_instruction(WORKER_CID, WORKER_TASK)
'''

ACL_READ_HANDLER = '''def handle(event):
    info = event.get("target_info") or {}
    ctrl = info.get("ctrl") or {}
    source = (event.get("source") or {}).get("conversation") or {}
    # 内核态监督对话越过 ACL（它本来就持有 MEM 根）
    if source.get("mode") == "kernel":
        return {}
    groups = source.get("groups") or []
    cid = source.get("cid")
    try:
        acl = json.loads(ctrl.get("acl.read") or "{}")
    except Exception:
        acl = {}

    def match(rule):
        if not isinstance(rule, dict):
            return False
        if rule.get("cid") == cid:
            return True
        return rule.get("group") in groups

    if any(match(rule) for rule in (acl.get("deny") or [])):
        return {"action": "replace", "response": "ACL 拒绝读取"}
    if any(match(rule) for rule in (acl.get("allow") or [])):
        return {"action": "allow"}
    return {"action": "replace", "response": "ACL 默认拒绝读取"}
'''

ACL_WRITE_HANDLER = '''def handle(event):
    info = event.get("target_info") or {}
    ctrl = info.get("ctrl") or {}
    source = (event.get("source") or {}).get("conversation") or {}
    if source.get("mode") == "kernel":
        return {}
    groups = source.get("groups") or []
    cid = source.get("cid")
    try:
        acl = json.loads(ctrl.get("acl.write") or "{}")
    except Exception:
        acl = {}

    def match(rule):
        if not isinstance(rule, dict):
            return False
        if rule.get("cid") == cid:
            return True
        return rule.get("group") in groups

    if any(match(rule) for rule in (acl.get("deny") or [])):
        return {"action": "replace", "response": "ACL 拒绝写入"}
    if any(match(rule) for rule in (acl.get("allow") or [])):
        return {"action": "allow"}
    return {"action": "replace", "response": "ACL 默认拒绝写入"}
'''

DONE_HANDLER = '''def handle(event):
    return {"notify": [{"cid": 0, "content": "yield from on_done()"}]}
'''

WORKER_SETUP = ""


def node_str(value, ctrl=None):
    node = {"kind": "str", "value": value}
    if ctrl:
        node["ctrl"] = ctrl
    return node


def node_dict(value, ctrl=None, meta=None):
    node = {"kind": "dict", "value": value}
    if ctrl:
        node["ctrl"] = ctrl
    if meta is not None:
        node["meta"] = meta
    return node


def node_caps(caps):
    """程序能力声明：value.caps 数据节点（不是 ctrl）。"""
    value = {}
    for key in ("mode", "workspace"):
        if key in caps:
            value[key] = node_str(str(caps[key]))
    for key in ("groups", "roots", "tools"):
        if key in caps:
            value[key] = {
                "kind": "list",
                "value": [node_str(str(x)) for x in caps[key]],
            }
    return node_dict(value)


def node_handler(code, condition):
    return node_dict({"content": node_str(code),
                      "condition": node_str(json.dumps(condition))},
                     ctrl={"type": "handler"})


def build_image():
    image = {
        "meta": {"version": 1},
        "mem": {
            "model_params": node_dict(
                {"model": node_str("plm.python")}, ctrl={"type": "para"}),
            "lib": node_dict(
                {
                    "greeting": node_str("hello", ctrl={
                        "acl.read": json.dumps({"allow": [{"group": "worker"}]}),
                        "acl.write": json.dumps({"allow": []}),
                    }),
                    "secret": node_str("top-secret", ctrl={
                        "acl.read": json.dumps({"deny": [{"group": "worker"}]}),
                    }),
                },
                ctrl={"owner": "0"},
                meta="ACL 示例库：greeting 允许 worker 组读，secret 拒绝 worker 组读，greeting 禁止写",
            ),
            "programs": node_dict({
                "init": node_dict({"content": node_str(INIT_SETUP)}, ctrl={"type": "python"}),
                "worker": node_dict(
                    {"content": node_str(WORKER_SETUP),
                     "caps": node_caps({
                         "mode": "user",
                         "groups": ["worker"],
                         "roots": ["lib"],
                         "tools": ["memory_read", "memory_write", "memory_make"],
                         "workspace": "work",
                     })},
                    ctrl={"type": "python"},
                    meta="用户态工作程序：只持有 $work 与只读库根 $lib",
                ),
            }),
            "handlers": node_dict({
                "read_greeting": node_handler(ACL_READ_HANDLER, {
                    "events": ["memory_read"],
                    "target": {"kind": "exact", "ref": "$MEM.lib.greeting"},
                }),
                "read_secret": node_handler(ACL_READ_HANDLER, {
                    "events": ["memory_read"],
                    "target": {"kind": "exact", "ref": "$MEM.lib.secret"},
                }),
                "write_greeting": node_handler(ACL_WRITE_HANDLER, {
                    "events": ["memory_write"],
                    "target": {"kind": "exact", "ref": "$MEM.lib.greeting"},
                }),
                "done": node_handler(DONE_HANDLER, {
                    "events": ["memory_write"], "phase": ["after"],
                    "target": {"kind": "subtree", "ref": "$MEM.workspaces"},
                }),
            }),
            "init_user": node_str("yield from main()"),
            "report": node_str(""),
        },
        "security": {
            "roots": {"lib": "$MEM.lib"},
            "workspace_region": "$MEM.workspaces",
        },
        "para_ref": "$MEM.model_params",
        "init": {
            "name": "acl-demo",
            "system_ref": "$MEM.programs.init",
            "user_ref": "$MEM.init_user",
            "para_ref": "$MEM.model_params",
        },
    }
    return image


def render(image):
    return json.dumps(image, ensure_ascii=False, indent=2) + "\n"


if __name__ == "__main__":
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(render(build_image()))
    print(f"[build_image] 已写 {OUT}")

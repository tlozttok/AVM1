"""sysinfo 包的 OS 层安装程序（plm.command）：读 pkgrepo、把数据搬进 $MEM、提交设备挂载申请。

设备代码的审批归面板：`install` 只写 `$MEM.device_loader.request.<mount>` 登记申请
（不挂载）；操作员在面板 `$MEM.device_loader` 设备上批准后，执行本程序的 `resume`
动作写 `$MEM.device_loader.resume.<mount>` 完成挂载。
"""

def _err(msg):
    return "错误: installer: " + str(msg)


def _is_error(text):
    return isinstance(text, str) and text.startswith(("Error:", "Illegal:"))


def _mount_name(spec):
    mount = spec.get("mount", "sysinfo")
    if not isinstance(mount, str) or not mount or "." in mount:
        return None
    return mount


def _install(spec):
    data_text = yield from memory_read("$MEM.pkgrepo.data")
    if _is_error(data_text):
        return _err("读 data.json 失败：" + str(data_text))
    try:
        node = json.loads(str(data_text))
    except Exception as e:
        return _err("data.json 不是 JSON：" + str(e))
    if not isinstance(node, dict) or node.get("kind") != "dict":
        return _err("data.json 必须是显式 dict 节点")
    value = node.get("value")
    if not isinstance(value, dict):
        return _err("data.json 的 value 必须是对象")

    mount = _mount_name(spec)
    if mount is None:
        return _err("mount 必须是单段设备名（不含点号）")

    parent = spec.get("dest_parent", "$MEM.workshop.installed")
    name = spec.get("dest_name", "sysinfo")
    if parent == "$MEM.workshop":
        response = yield from memory_make("$MEM.workshop", "installed", "dict")
        if _is_error(response):
            return _err("建 installed 分区失败：" + str(response))
    response = yield from memory_make(parent, name, "dict")
    if _is_error(response):
        return _err("建安装目标失败：" + str(response))
    dest = parent + "." + name
    written = []
    for key, child in value.items():
        if not isinstance(child, dict) or child.get("kind") != "str":
            return _err("data.json 子节点 " + str(key) + " 不是 str 叶子")
        response = yield from memory_make(dest, key, "str")
        if _is_error(response):
            return _err("建字段 " + str(key) + " 失败：" + str(response))
        response = yield from memory_write(dest + "." + key, child.get("value", ""))
        if _is_error(response):
            return _err("写字段 " + str(key) + " 失败：" + str(response))
        written.append(key)

    request = {
        "file": spec["device_file"],
        "class": spec.get("device_class", "SysInfoDevice"),
        "args": spec.get("device_args", {}),
    }
    response = yield from memory_write(
        "$MEM.device_loader.request." + mount,
        json.dumps(request, ensure_ascii=False),
    )
    if _is_error(response):
        return _err("提交装载申请失败：" + str(response))
    return json.dumps({
        "ok": True,
        "installed_at": dest,
        "fields": written,
        "mount": mount,
        "pending": True,
        "next": "操作员在面板 $MEM.device_loader 设备上批准后，执行本程序的 resume 动作完成挂载",
    }, ensure_ascii=False)


def _resume(spec):
    mount = _mount_name(spec)
    if mount is None:
        return _err("mount 必须是单段设备名（不含点号）")
    response = yield from memory_write("$MEM.device_loader.resume." + mount, "")
    if _is_error(response):
        return _err("完成挂载失败：" + str(response))
    return json.dumps({"ok": True, "mount": mount, "mounted": True},
                      ensure_ascii=False)


def handle(cmd):
    try:
        spec = json.loads(cmd)
    except Exception as e:
        return _err("命令不是 JSON：" + str(e))
    if not isinstance(spec, dict):
        return _err("命令必须是 JSON 对象")
    action = spec.get("action")
    if action == "help":
        return ("sysinfo 包安装程序：{\"action\": \"install\", \"device_file\": ..., "
                "\"mount\": ...} 提交挂载申请；操作员批准后执行 "
                "{\"action\": \"resume\", \"mount\": ...} 完成挂载。")
    if action == "install":
        return (yield from _install(spec))
    if action == "resume":
        return (yield from _resume(spec))
    return _err("未知 action： " + repr(action))

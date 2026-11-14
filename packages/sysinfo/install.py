"""sysinfo 包的 OS 层安装程序（plm.command）：读 pkgrepo、把数据搬进 $MEM、请求挂载设备代码。"""

def _err(msg):
    return "错误: installer: " + str(msg)


def _is_error(text):
    return isinstance(text, str) and text.startswith(("Error:", "Illegal:"))


def _file_sha(files_text, rel):
    for line in files_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == rel:
            return parts[2]
    return None


def _make_node(dest, node):
    kind = node.get("kind")
    if kind != "str":
        return _err("data.json 只支持 str 叶子（示例安装程序）")
    response = yield from memory_make(dest, "x", "str")  # 占位，不会执行到：见 _install
    return response


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

    files_text = yield from memory_read("$MEM.pkgrepo.files")
    if _is_error(files_text):
        return _err("读 files 失败：" + str(files_text))
    sha = _file_sha(str(files_text), "devices/sysinfo.py")
    if sha is None:
        return _err("包内没有 devices/sysinfo.py")

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
        "action": "load_device",
        "path": spec.get("mount", "sysinfo"),
        "file": spec["device_file"],
        "class": spec.get("device_class", "SysInfoDevice"),
        "args": spec.get("device_args", {}),
        "approved_sha256": sha,
        "approved_at": "2026-11-14T21:40:00+0800",
        "approved_via": "pkg-install-trial",
    }
    response = yield from memory_write("$MEM.device_loader", json.dumps(request, ensure_ascii=False))
    if _is_error(response):
        return _err("挂载设备失败：" + str(response))
    return json.dumps({
        "ok": True,
        "installed_at": dest,
        "fields": written,
        "mount": spec.get("mount", "sysinfo"),
        "device_sha256": sha,
    }, ensure_ascii=False)


def handle(cmd):
    try:
        spec = json.loads(cmd)
    except Exception as e:
        return _err("命令不是 JSON：" + str(e))
    if not isinstance(spec, dict):
        return _err("命令必须是 JSON 对象")
    action = spec.get("action")
    if action == "help":
        return "sysinfo 包安装程序：命令 {\"action\": \"install\", \"device_file\": ..., \"mount\": ...}"
    if action != "install":
        return _err("未知 action： " + repr(action))
    return (yield from _install(spec))

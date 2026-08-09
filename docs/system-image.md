# AVM 系统镜像规范（System Image Specification）

**版本**: 1
**日期**: 2026-08-01
**一句话**: 一个 JSON 文件 + 若干设备 Python 文件 = 一个标准 AVM 系统镜像。

启动方式：

```bash
python -m avm.image <image.json>
```

---

## 顶层结构

| 键 | 必填 | 说明 |
|---|---|---|
| `meta` | 否 | 元信息，`version` 必须为 1 |
| `mem` | 是 | 内存树（对象，顶层键 → 显式节点） |
| `para_ref` | 否 | para 节点路径，默认 `$MEM.model_params` |
| `devices` | 否 | 设备挂载列表 |
| `init` | 否 | 启动对话配置（缺省时不启动任何对话） |
| `persist_to` | 否 | 运行结束后内存写回路径（相对镜像文件所在目录） |

## mem 节点格式

每个节点都是显式对象，**无任何隐式转换**（str 也必须显式包装）：

| 字段 | 含义 |
|---|---|
| `kind` | `str` / `dict` / `list` / `device` |
| `meta` | 可选字符串——元数据一，给 LLM 看的一句话描述 |
| `ctrl` | 可选对象（`dict[str,str]`）——元数据二，给 Core 的结构化控制信息 |
| `value` | `str` → 字符串；`dict` → 子节点对象；`list` → 子节点数组；`device` → 预留（结构未定） |

规则：

- `kind` 未知即报错；`ctrl` 必须是对象。
- `kind=device` 可出现在**任意层级**，作为可读性标记；该节点从 mem 根到自身的点号路径（如 `game.map.rooms`）必须与 `devices` 段的 `path` 一一对应。device 节点不写入数据树，实际挂载由 `devices` 段完成。
- `ctrl.type` 目前定义的取值：`para`（模型调用参数）、`settingup`（LLM 提示词程序）、`python`（Python 程序）。

**程序节点（settingup / python）**：`create_cmd` / `create_sub` 的 `system_ref` 必须指向这类节点（运行时校验；指向 str 节点、无类型 dict 节点或其他类型会报错，不创建对话）。`settingup` 节点的 `value` 含 `name`（对话身份）与 `content`（提示词文本）；`python` 节点的 `value` 含 `content`（Python 代码）。两者均可带可选的 `signature`（预期输入/输出描述）。`para_ref` 指向该程序使用的模型参数节点。

运行期可用 `edit_metadata` 指令给节点补 ctrl（例如给运行时创建的 dict 节点设置 `type='settingup'` / `'python'`），配合 `memory_make` / `memory_write` 组合出可被 `create_cmd` 使用的程序节点——程序节点不一定只在镜像中预置。

`init` 的 `system_ref` 仍指向 str 节点（入口对话的字面量提示词，不属于程序节点）。

## para（模型调用参数）

- mem 中必须存在一个 `ctrl.type="para"` 的 dict 节点（默认路径 `$MEM.model_params`），否则加载报错。
- 其 `value` 是 LLM API 参数（`model`、`temperature`、`use_tool` 等），均以 str 存储；数值型参数（temperature、max_tokens、top_p 等）加载后自动转回数值。
- `para_ref` 顶层声明 para 路径，覆盖默认值。

## devices（设备插件）

数组，每项：

| 键 | 必填 | 说明 |
|---|---|---|
| `path` | 是 | 挂载键，访问方式 `$MEM.<path>` |
| `file` | 是 | Python 模块路径，相对镜像文件所在目录（也允许绝对路径） |
| `class` | 是 | 模块中的类名 |
| `args` | 否 | 传给构造函数的对象，默认 `{}` |

设备类**必须继承** `avm.memory_device.MemoryDevice`，否则报错。接口约定：

- `to_llm_string() -> str`：LLM 读取该设备时看到的表示（必实现）。
- `resolve_path(path: list)`：子路径访问（可选覆盖，默认抛错）。
- `set_value(value)`：写入（可选）。

参考 `avm/memory_device.py` 中的现有设备（`InputsListDevice`、`OutputsListDevice` 等）。

## init（启动对话）

- `name`：0 号对话名，默认 `"init"`。
- `system` / `system_ref`：二选一。`system` 是字面量；`system_ref` 是 `$` 开头的内存引用，必须指向 str 节点。
- `user` / `user_ref`：同上。

只启动 0 号对话；**不预置其他对话**（服务由对话运行期自己 `register_service`）。

## 启动流程

1. 解析 mem（严格校验，任何格式错误抛 `ImageError`）；
2. 按 `para_ref` 校验 para 节点（`ctrl.type == "para"`）并配置 Core；
3. 加载设备插件文件（校验继承关系）并挂载；
4. 启动 init 对话（cid 0，`name`）；
5. `core.run()`；
6. 若声明 `persist_to`，在 `finally` 中写回内存（异常也写）。

错误不吞：格式错误输出到 stderr，CLI 退出码 1。

## 示例

```json
{
  "meta": { "name": "demo", "version": 1 },
  "mem": {
    "system": { "kind": "str", "value": "你是 init 对话。执行以下循环：……" },
    "user": { "kind": "str", "value": "开始" },
    "model_params": {
      "kind": "dict",
      "meta": "LLM API 调用参数",
      "ctrl": { "type": "para" },
      "value": {
        "model": { "kind": "str", "value": "gpt-4o-mini" },
        "temperature": { "kind": "str", "value": "0.7" },
        "use_tool": { "kind": "str", "value": "auto" }
      }
    },
    "inputs": { "kind": "device" },
    "outputs": { "kind": "device" }
  },
  "para_ref": "$MEM.model_params",
  "devices": [
    { "path": "inputs", "file": "devices/io.py", "class": "InputsListDevice", "args": {} },
    { "path": "outputs", "file": "devices/io.py", "class": "OutputsListDevice", "args": {} }
  ],
  "init": {
    "name": "init",
    "system_ref": "$MEM.system",
    "user_ref": "$MEM.user"
  },
  "persist_to": "out/mem.json"
}
```

对应的设备文件 `devices/io.py`（两个类可共存于一个文件）：

```python
from avm.memory_device import InputsListDevice, OutputsListDevice
```

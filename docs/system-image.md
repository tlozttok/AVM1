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
| `meta` | 否 | 元信息：`version` 必须为 1；`instruction_budget` 可覆盖每轮指令预算（正整数，默认 50） |
| `mem` | 是 | 内存树（对象，顶层键 → 显式节点） |
| `para_ref` | 否 | para 节点路径，默认 `$MEM.model_params` |
| `devices` | 否 | 设备挂载列表 |
| `init` | 否 | 启动对话配置（缺省时不启动任何对话） |
| `persist_to` | 否 | 运行结束后内存写回路径（相对镜像文件所在目录） |

## mem 节点格式

每个节点都是显式对象，**无任何隐式转换**（str 也必须显式包装）：

| 字段 | 含义 |
|---|---|
| `kind` | `str` / `dict` / `list` / `device` / `link` |
| `meta` | 可选字符串——描述（description），给 LLM 看的一句话摘要；元数据之一 |
| `ctrl` | 可选对象（`dict[str,str]`）——控制数据，给 Core 的结构化控制信息；元数据之一 |
| `target` | `kind: "link"` 必填——`$MEM` 地址或设备标识符 |
| `value` | `str` → 字符串；`dict` → 子节点对象；`list` → 子节点数组；`device` → 不使用（`kind: "device"` 只作可读性标记，实际挂载由 `devices` 段完成）；`link` → 不使用 |

规则：

- `kind` 未知即报错；`ctrl` 必须是对象。
- `kind=device` 可出现在**任意层级**，作为可读性标记；该节点从 mem 根到自身的点号路径（如 `game.map.rooms`）必须在 `devices` 段的 `path` 里出现，否则加载报错。反向不要求：`devices` 段可以有 mem 里没有标记的挂载项，照样挂载。device 节点不写入数据树，实际挂载由 `devices` 段完成（挂载会在该路径物化一个链接节点，见[运行时](runtime.md)的挂载与链接节点一节）。
- `kind=link` 声明链接节点：`target` 以 `$` 开头是内存树地址（`$MEM.a.b`），否则是设备唯一标识符（`devices` 段条目的 `id`，缺省取 `path`）。寻址遇到链接即进入目标，读/写/`memory_make`/`edit_metadata` 都作用于目标；链接只由内核侧声明，运行期没有创建/删除链接的指令。
- `ctrl.type` 目前定义的取值：`para`（模型调用参数）、`settingup`（LLM 提示词程序）、`python`（Python 程序）。
- `ctrl.owner` 是 Core 维护的所有者标记（字符串 cid）。`memory_make` / `memory_write` 创建容器节点时会自动写入；镜像作者也可在 `ctrl` 中显式预置 `"owner": "0"` 来声明所有权。无主节点在事件模型 v1 中不可注册 before 内存监听。

```json
{
  "kind": "dict",
  "ctrl": { "type": "settingup", "owner": "0" },
  "value": { "content": { "kind": "str", "value": "..." } }
}
```

**程序节点（settingup / python）**：`create_cmd` / `create_sub` 的 `system_ref` 必须指向这类节点（运行时校验；指向 str 节点、无类型 dict 节点或其他类型会报错，不创建对话）。`settingup` 节点的 `value` 含 `name`（对话身份）与 `content`（提示词文本）；`python` 节点的 `value` 含 `content`（Python 代码）。两者均可带可选的 `signature`（预期输入/输出描述）与 `para_ref`（该程序使用的模型参数节点，`init` 指向它时按这个字段取 para；`create_cmd` / `create_sub` 的 para 由指令参数给出）。`content` 必须是字符串（缺失或类型不符会在创建对话时被拒）；`name` 可选，缺省时创建出的对话身份回退为「父身份#cid」，根入口没有名字时身份 = cid。

运行期可用 `edit_metadata` 指令给节点补 ctrl（例如给运行时创建的 dict 节点设置 `type='settingup'` / `'python'`），配合 `memory_make` / `memory_write` 组合出可被 `create_cmd` 使用的程序节点——程序节点不一定只在镜像中预置。

程序的 `para` 里 `model` 取 LLM 模型名，或 PLM 注册表（`PLM_REGISTRY`）中的名字（`plm.simple` / `plm.function` / `plm.machine` / `plm.command` / `plm.python`）。

两个命名空间由 `create_cmd` / `create_sub` 在输入边界交叉校验，依据是启动自带的 `$MEM.model_catalog`（结构见 [docs/runtime.md](runtime.md) 的"模型目录"一节）：`python` 程序的 `model` 必须是 `plm` 注册名；`settingup` 程序的 `model` 在上游清单取到时必须在 `llm.models` 里，取不到时只警告不拦。清单来自宿主启动时的一次 `GET /models`，失败不影响加载。

`init` 的入口也是程序节点（`settingup` 或 `python`），校验与 `create_cmd` 一致；`init.system` 字面量那条路不注册节点，只用于 LLM 入口。

## para（模型调用参数）

- mem 中必须存在一个 `ctrl.type="para"` 的 dict 节点（默认路径 `$MEM.model_params`），否则加载报错。
- 其 `value` 是 LLM API 参数（`model`、`temperature`、`use_tool` 等），均以 str 存储；数值型参数（temperature、max_tokens、top_p 等）在每次调用 API 前由 LMU 转回数值，镜像加载本身不做转换。
- `para_ref` 顶层声明 para 路径，覆盖默认值。

## devices（设备插件）

数组，每项：

| 键 | 必填 | 说明 |
|---|---|---|
| `path` | 是 | 挂载键，访问方式 `$MEM.<path>` |
| `file` | 否* | Python 模块路径，相对镜像文件所在目录（也允许绝对路径） |
| `class` | 是 | 模块中的类名（外层插件）或内建注册表类名（`builtin=true`） |
| `builtin` | 否 | `true` 时使用内建设备注册表 `BUILTIN_DEVICE_CLASSES`，不允许提供 `file` |
| `args` | 否 | 传给构造函数的对象，默认 `{}` |
| `id` | 否 | 设备唯一标识符，缺省取 `path`；链接节点的 `target` 不以 `$` 开头时按它解析 |

- 外层设备插件：`builtin` 缺省或 `false`，必须提供 `file` + `class`，`class` 必须继承 `avm.memory_device.MemoryDevice`。
- 内建设备：`builtin: true`，`class` 取内建注册表（当前包含 `LinkDevice`，保留兼容、不用于新设计），参数走 `args`。
- 挂载会在 `path` 物化一个链接节点，因此挂载点出现在 `$MEM` 根读里；`path` 不能与已有数据节点同名，父段不存在时自动建空 dict。

接口约定：

- `to_llm_string() -> str`：LLM 读取该设备时看到的表示（必实现）。
- `resolve_path(path: list)`：子路径访问（可选覆盖，默认抛错）。
- `set_value(value)`：写入（可选）。
- `authorize_listener(event_type, remainder, owner_cid) -> bool`：是否允许监听该设备内路径（保留兼容的 `LinkDevice` 会委托给真实目标）。
- `after_event(event_type, remainder, result) -> Optional[dict]`：操作成功后返回纯 payload 事实事件。
- **子路径按名称访问**：设备自己定义命名子键（如 `next`、`out`、`append`、`latest`）；数字下标只用于列表式设备的内容读取（非负）。设备没有"`-1` 是最后一条/追加位置"的语义（`-1` 只属于内存列表的追加位置），设备写入只走命名接口。

**设备自述（帮助说明）**：镜像自带的设备对两类读取给出用法说明。

- 读设备本身（`$MEM.<path>`）返回设备说明书：设备做什么、支持哪些动作及其语法、一个可照抄的例子。由 `to_llm_string()` 给出。
- 读一个缺少必需参数的动作（如 `$MEM.<path>.ls`）返回该动作的用法说明；参数齐全时按动作执行，参数写错仍报错。由 `resolve_path()` 收到残缺段时给出；带缺省值的参数（如页码）不算缺参数。

挂载点在 `$MEM` 根读里以键的形式可见（键就是链接节点），但读取经过链接不会提示跨越，链接自身也不返回描述：设备是这条边界的唯一信息来源；第一次接触设备的对话应当一次读取拿到完整接口。运行时不为此增加机制：读设备本身已经走 `to_llm_string()`，残缺的段已经交给 `resolve_path()`。

参考现有设备：Core 内建的 `avm/memory_device.py`（`InputsListDevice`、`OutputsListDevice`、`LinkDevice` 等），以及镜像层的例子 `images/devices/`（`io.py`、`avm_repo.py`）和 `images/apps/*/devices/`。

#### 链接节点声明示例

```json
{
  "mem": {
    "game": { "kind": "dict", "value": { "map": { "kind": "dict", "value": {
      "rooms": { "kind": "dict", "meta": "房间列表", "value": {} }
    }}}},
    "rooms": { "kind": "link", "target": "$MEM.game.map.rooms" }
  },
  "devices": []
}
```

#### LinkDevice 示例（保留兼容，不用于新设计）

```json
{
  "mem": {
    "game": { "kind": "dict", "value": { "map": { "kind": "dict", "value": {
      "rooms": { "kind": "dict", "meta": "房间列表", "value": {} }
    }}}},
    "rooms": { "kind": "device" }
  },
  "devices": [
    { "path": "rooms", "class": "LinkDevice", "builtin": true,
      "args": { "target": "$MEM.game.map.rooms" } }
  ]
}
```

## init（启动对话）

- `name`：0 号对话名，默认 `"init"`。
- `system` / `system_ref`：二选一。`system` 是字面量提示词（匿名 LLM 入口程序，不注册为节点）；`system_ref` 是 `$` 开头的内存引用，指向程序节点：`ctrl.type="settingup"` 是 LLM 入口（`value` 含 `content` 提示词文本，可选 `name`），`ctrl.type="python"` 是 PLM 入口（`value` 含 `content` 程序代码，0 号建为 Python 对话，与 `create_cmd` 指向 python 节点时同义）。指向 str 节点或无类型 dict 节点会报错（与 `create_cmd` / `create_sub` 的校验一致）。
- `para_ref`：可选，`$` 开头的内存引用，指定入口对话的 para 节点。缺省时取程序节点 `value.para_ref`，再缺省取镜像顶层 `para_ref`；PLM 入口在前两者都没有时报错（顶层 para 按约定是 LLM 参数）。
- `user` / `user_ref`：二选一。`user` 是字面量；`user_ref` 是 `$` 开头的内存引用，必须指向 str 节点（启动用户消息）。PLM 入口的首条用户消息就是这条（`plm.function` 的 JSON 输入、`plm.command` 的整行命令都从这里来）。

只启动 0 号对话；**不预置其他对话**（其他对话由对话运行期自己 `create_cmd` / `create_sub` 创建）。PLM 入口的典型用法：0 号程序先做确定性的准备工作（读设备、组装提示词、写内存），再 `create_cmd` 出真正跑推理的 LLM 对话并 `send_instruction` 启动它，自己 `close_conversation`。

## 启动流程

生命周期显式三段（`avm/boot.py` 是加载区）：**创建空核心**（`Core()`）→ **加载镜像**（`load_image(core, image_path)`，要求核心还没有对话，否则抛 `ImageError`）→ **启动核心**（`start_core(core)`：刷新模型目录、装 API 失败的运行者策略）。宿主通常只调 `boot(image_path)`，三段串在一起。

**加载镜像**依次做：

1. 解析 mem（严格校验：结构错误与类型滑档都抛 `ImageError`——`meta` 必须是对象、`para_ref` 必须是非空字符串、`devices` 条目必须是对象）；
2. 按 `para_ref` 校验 para 节点（`ctrl.type == "para"`）并配置 Core；
3. 加载设备插件文件（校验继承关系与挂载路径唯一）并挂载；
4. 安装启动自带的模型目录 `$MEM.model_catalog`（镜像或检查点里已有的字段不覆盖，只补缺失）；
5. 启动 init 对话（cid 0，`name`；LLM 入口或 PLM 入口）。`init` 缺失或为空时给一行 stderr 诊断，不启动任何对话。

**启动核心**依次做：拉一次上游 `GET /models` 填模型目录（失败只警告，不拦启动）；装 API 失败的运行者策略（终端三选项：更新环境 / 继续 / 退出）。

之后 `core.run()`；若声明 `persist_to`，在 `finally` 中写回内存（异常也写）。

错误不吞：格式错误输出到 stderr，CLI 退出码 1。

## 示例

```json
{
  "meta": { "name": "demo", "version": 1 },
  "mem": {
    "system": {
      "kind": "dict",
      "meta": "init 的 LLM 提示词程序",
      "ctrl": { "type": "settingup" },
      "value": {
        "name": { "kind": "str", "value": "init" },
        "content": { "kind": "str", "value": "你是 init 对话。执行以下循环：……" }
      }
    },
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

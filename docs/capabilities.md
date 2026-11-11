# AVM 能力模型与权限机制

本文是 AVM 权限体系的完整说明：运行器提供了什么机制、镜像层如何用它做权限策略、以及写一个安全的镜像程序需要知道什么。

术语以 [CONTEXT.md](../CONTEXT.md) 为准，运行时机制细节见 [docs/runtime.md](runtime.md)，镜像格式见 [docs/system-image.md](system-image.md)。

## 1. 两层边界：机制在 Core，策略在镜像层

AVM 的权限不放在 Core 的硬编码规则里，分两层：

| 层 | 提供什么 | 不做什么 |
| --- | --- | --- |
| Core（机制层） | 每对话根表与地址空间、能力数据（模式/组/根/工具集）、工具面裁剪、10 条指令的 before/after 事件、载荷分级、监听器 priority/tier/决策动作、canonical 目标、内核设备控制面 | 不判断"谁可以访问哪个节点"、不做节点所有权判定、不内置 ACL 语法、不预置保护监听器、不强制引导顺序 |
| 镜像层（策略层） | 初始 Python 程序建立内核态安全对话；安全对话注册监听器（含裁决监听注册的 Guard）、在节点 `ctrl` 里放权限数据、按事件载荷做 allow/replace；运行期用控制面调整组/根/监听器 | 不修改 Core 代码、不假定内置规则、不依赖设备授权接口拿调用者身份 |

这条边界对应设计稿的"内核只提供可查、可拦、可分级的工具调用事件"。镜像作者写策略；Core 保证策略拿得到信息、拦得住操作。

## 2. 对话能力（caps）

每个对话有 `Conversation.caps`（[avm/capabilities.py](../avm/capabilities.py)）：

| 字段 | 含义 |
| --- | --- |
| `mode` | `kernel`：持有真实 `MEM` 根、默认全部工具；`user`：只有自己的根表，看不到 `MEM`、设备路径简写与 `$conv`/`$batch` |
| `groups` | 组标签（字符串元组），进事件载荷 `source.conversation.groups`，供 ACL 规则使用 |
| `roots` | 根表：根名 → `$MEM` 之后的真实段序列；解析引用时第一段按根表匹配 |
| `tools` | 工具集：本对话可调用的指令名集合 |

宿主侧创建（`Core.start`、操作员对话、旧检查点）默认内核态全能力：`MEM` 根 + 全部 10 条工具。检查点保存 `caps` 与 `attrs`；旧检查点回退内核态全能力。

### 2.1 创建时委托：程序节点的 `value.caps`

程序节点（`ctrl.type="settingup"` 或 `"python"`）用**节点数据** `value.caps` 声明由它创建的对话的能力：`ctrl` 只描述类型与数据组织格式，能力是普通数据节点。

```json
{
  "kind": "dict",
  "ctrl": {"type": "settingup"},
  "value": {
    "content": {"kind": "str", "value": "..."},
    "caps": {
      "kind": "dict",
      "value": {
        "mode":      {"kind": "str",  "value": "user"},
        "groups":    {"kind": "list", "value": [{"kind": "str", "value": "worker"}]},
        "roots":     {"kind": "list", "value": [{"kind": "str", "value": "lib"}]},
        "tools":     {"kind": "list", "value": [{"kind": "str", "value": "memory_read"}]}
      }
    }
  }
}
```

`ctrl.caps` 不再支持：镜像装载与创建对话都会拒绝（`ImageError` / `Illegal:`）。

规则（`Core._derive_child_caps`）：

- `mode` 缺省继承父模式；用户态父对话只能创建用户态对话；亚对话（`create_sub`）不能跨模式；镜像 init 强制内核态。
- `tools` 缺省继承父工具集；显式声明必须 ⊆ 父工具集；未知指令名报错。
- `groups` 缺省继承父组；用户态父对话只能给子对话父已有的组（内核态父对话可以指定任意组，这是策略权限）。
- `roots` 是镜像 `security.roots` 注册表里的**名字列表**，只有内核态父对话能授予；同域创建（亚对话、用户态父建子对话）只能沿用父根表，不能新增根。
- `workspace` 已移除：程序声明它会报错。内核不分配工作目录，目录由软件用 `memory_make` 建、根由控制面动态加（见 3.3）。
- 没有 `caps` 的程序继承父能力（旧镜像行为不变）。

### 2.2 模式的实际差别

- 根表：内核态含 `MEM`；用户态没有 `MEM`，因此无法命名内核设备、`$conv`/`$batch` 伪根与设备路径简写。
- 工具面：由 `tools` 决定，与模式独立（模式不硬编码工具宇宙）。普通对话是否能有 `create_cmd`/`listen_event`/`close_conversation` 由镜像授予。
- 事件注册：内核态可监听任意内存/cid 目标、可注册通配生命周期监听、可声明 `tier=kernel`/`priority>0`；用户态只能用命名条件子集（`inbox` / `self.lifecycle`）。内存目标不做所有者判定，授权由软件层 Guard 监听器裁决（见 5.3）。

## 3. 地址空间：根表、根注册表、动态授予根

### 3.1 根表与显示纪律

- 引用第一段是根名（`MEM`、`work`、`lib`…），按对话根表精确匹配；未命中的根得到统一错误 `引用根不可用…`，不区分拼错与未授权。
- 根表里含 `MEM` 的对话保留旧途径（`$MEM....`、设备路径简写、`$conv`/`$batch`）。
- 根目标缺失时报"引用根无效"；链接目标缺失同样不打印真实目标。
- 所有错误响应与成功地址只写请求别名（如 `$work.list.3`），不泄露真实路径。内核转储/宿主控制台是例外（宿主特权面）。
- 事件匹配与所有权判定用 canonical 目标（跟随根别名、链接与 `LinkDevice` 透明挂载），处理器看到的目标仍是别名。

### 3.2 根注册表（镜像 `security` 段）

```json
"security": {
  "roots": {"lib": "$MEM.lib", "repo": "$MEM.AVMRepo"}
}
```

- `roots` 是可授予根的注册表；装载时校验名字规则、目标两两前缀无关、不落在内核控制面设备路径。
- `workspace_region` 已移除：出现即报 `ImageError`，内核不再分配工作空间。
- 该段在设备挂载之后安装；`roots` 目标不要求已存在（先授根后建节点合法，访问时报"根无效"）。运行期根注册表可经 `$MEM.roots` 整体替换或按根名单条写入。

### 3.3 动态授予根（控制面 `$MEM.conversations.<cid>.roots`）

- 用户态对话创建时只有 `caps.roots` 声明的根与只读 `caps` 自述根，没有默认目录。目录由软件用 `memory_make` 建。
- 授予/撤销走控制面写入：`{"<根名>": "$MEM.<路径>", ...}`，**整体替换该对话的非 core 根表**——省略某个根即撤销，不是合并。
- Core 自述根 `caps` 保留、不可覆盖；内核态对话的 `MEM` 根表不能通过控制面改写。
- 校验：根名规则、目标非空且不是 `$MEM` 根本身、不落在内核控制面设备路径（显式授予设备本身允许）、与其他存活对话根及根注册表两两前缀无关（相同目标允许共享）。
- 成功写入记日志效果 `conversation_roots`（cid + 完整非 core 根表），随日志重放；`reboot` 丢弃对话态效果，由 init 里的软件重新授予。
- 整棵子树删除与自动回收未实现：软件自己删节点/撤根。

## 4. 工具面：三处一致裁剪

对话的 `tools` 决定三处：

1. 模型看到的工具 schema（`_build_tools(conv)`）；
2. PLM 注入的工具函数（`plm.python`/`plm.command`/`plm.machine` 的命名空间与校验）；
3. Core 实际调度（`_process_return_calls` 对未授权指令返回 `Error: 工具 <名> 未授权给本对话`）。

宿主控制台 `/api/tools` 始终是全量。旧对话（无 caps）默认全部 10 条。

## 5. 事件机制：策略的抓手

### 5.1 事件与阶段

10 条指令都有 before 请求事件；成功的操作另有一条 after 事实事件。before 在参数校验之前分发，可被决策；after 只做审计与通知。

### 5.2 载荷

- `source.conversation`：发起者的 `cid/identity/name/mode/groups/is_sub/is_python/parent_cid/state/para_ref`。
- `target_info`：目标的 `kind/ref/exists/type/meta/ctrl/ancestors/parent`；目标不存在时（make/追加）给 `longest_existing_prefix`；ancestors 只给 type/meta/ctrl 与 depth（不含真实路径，深度上限 32）；device 目标给设备自述；conversation/listener 目标给对应快照/摘要。
- normal 级拿以上；kernel 级追加 `target_full`（目标完整 JSON 数据）与 `source_context`（来源对话最近 20 条消息，带 total/truncated）。
- 内部 `target_ref`（canonical 真实目标）只用于匹配与 canonical 比较，不给处理器。

### 5.3 监听器

- 处理器节点必须 `ctrl.type="handler"`，`content` 定义 `handle(event)` 返回 dict；注册时抄副本，改节点不自动生效。
- 监听器是 `ctrl.type="handler"` 节点：`content` 是代码，`condition` 是声明式条件数据叶子（events/phase/target/source），注册时复制；`listen_event` 只提供 handler_ref/priority/tier，条件由节点声明。
- 用户态对话只能用命名条件子集：`inbox`、`self.lifecycle`；内核态可以写完整条件（含 subtree/任意范围，用于全局访问表）。`target.kind="owned_by_owner"`、`source.cid` 与 preset `own_work.memory` / `own_nodes.memory` 已移除。
- `tier`：`normal`（任何对话可注册）或 `kernel`（只有内核态对话可注册；拿完整载荷）。
- `priority`：非负整数，降序执行；同优先级按注册顺序（listener_id）。`priority>0` 只有内核态可设置。
- 决策动作（before）：`execute` 继续、`replace` 拒绝/改写（原指令不执行，调用者收到 `Interrupted:`）、`allow` 放行短路（原指令执行；只有 kernel 级可返回）。首个 allow/replace 短路；管道结束仍无决定则执行原指令。
- 处理器可返回 `notify` 让 Core 内部投递并唤醒目标；禁止自我投递；用户态监听器只能通知注册者自己，内核态监听器可通知任意对话（策略层负责）。
- 监听注册权限：`listen_event` 的新式路径**不做目标授权**——Core 没有节点所有权数据，也不调用设备 `authorize_listener`；策略整个交给软件层的 Guard 监听器（`listen_event` 的 before 事件带解析后的条件，Guard 可 `replace` 或 `allow`；镜像没装 Guard 时注册不受限）。用户态能干什么由命名条件子集限制（preset 只表达自己相关的目标），内核态可监听任意目标。设备 `authorize_listener` 钩子只在宿主侧旧注册/更新 API 上调用（见讨论区）。

- `modify`/`cancel` 只允许注册者；modify 可改 handler_ref 与/或 priority，tier/target/event_type/phase 不可改。
- 用户态对话不能关闭内核态对话（`Illegal:`）；关闭用户态对话仍受工具集与事件策略约束。
- before 处理器失败 fail closed（原指令不执行、收到 `Error:`）；after 处理器失败 fail open（记录 stderr、跳过该处理器）。

### 5.4 Guard 监听器

Guard 不是 Core 组件，而是镜像层的一个内核态监听器，用来把"谁可以注册监听"变成策略：

```python
# $MEM.guard：ctrl.type="handler"，condition={"events":["listen_event"],"phase":["before"]}
# 以 tier=kernel、priority>0 注册（register_condition / listen_event）。
def handle(event):
    condition = (event.get("payload") or {}).get("condition") or {}
    target = condition.get("target") or {}
    source = (event.get("source") or {}).get("conversation") or {}
    if target.get("kind") in ("exact", "subtree") and source.get("mode") == "user":
        return {"action": "replace", "response": "Guard: 用户态禁止监听内存路径"}
    return {}
```

- 放行返回 `{}`（不做决定，管道继续）；只有确实要短路后续监听器时才返回 `{"action": "allow"}`。
- 解析后的 `condition` 带 `events` / `phases` / `target`（`exact`/`subtree` 的 `refs` 已 canonical 化）/ `source`（`mode`/`groups`）。
- `listen_event action=listen` 的 before 事件目标就是条件目标的 canonical ref：`exact`/`subtree` 为内存/设备地址，`cid_owner`/`self_lifecycle` 为 `cid:<cid>`，无 target 为 `"*"`。
- `EventSystem.register` 是宿主/内部受信任入口，不经过 Guard；Guard 只管对话的 `listen_event` 指令。


## 6. 内核设备

| 类 | 设备 | 语义 |
| --- | --- | --- |
| 观测面（只读） | `$MEM.scheduler`、`$MEM.memory`、`$MEM.monitor` | 读运行时状态；写入拒绝 |
| 控制面（可写） | `$MEM.conversations`、`$MEM.listeners`、`$MEM.roots` | 读实时状态；写固定字段直接改 Core 对象 |
| 能力自述（只读） | `$MEM.caps` | 用户态经保留根 `caps` 读自己的能力摘要 |
| 镜像设备 | 镜像 `devices` 段挂载的设备 | 各自定义，与内核控制面分开 |

### 6.1 控制面

`$MEM.conversations`：

| 路径 | 写 | 语义 |
| --- | --- | --- |
| `<cid>.groups` | 开 | 整体替换组（JSON 字符串数组） |
| `<cid>.roots` | 开 | 整体替换非 core 根表（JSON 对象 `根名 -> "$MEM..."`）；Core 自述根 `caps` 保留并标 `core: true`；记 `conversation_roots` 效果 |
| `<cid>.meta` / `.meta.<key>` | 开 | 自由元数据进 `Conversation.attrs` |
| `<cid>.mode` / `.tools` | 只读 | 运行期改模式/工具集留在后续 |

控制面设备（`$MEM.conversations`、`$MEM.listeners`、`$MEM.roots`）与保留路径（`$MEM.caps` 等）不可作为普通根授予，镜像 `security.roots` 与运行期 `.roots` 写入都校验这一点；设备本身可以被显式授予。运行期根注册表可通过 `$MEM.roots` 修改（整体替换 JSON 或按根名单条写入），新根可被后续创建对话的 `caps.roots` 引用。用户态对话根表自动带只读保留根 `caps -> $MEM.caps.<cid>`，可读自己的 mode/groups/roots 名字/tools。

`.roots` 校验：根名规则、目标非空 `$MEM` 引用、内核态对话禁止改写根表、`caps` 自述根不可覆盖、不能落在内核控制面设备路径（显式授予设备本身允许）、与其他存活对话根目标和根注册表前缀无关（相同目标允许共享）。写入是整体替换，省略即撤销；成功写入记 `conversation_roots`。

`$MEM.listeners`：

| 路径 | 写 | 语义 |
| --- | --- | --- |
| `<id>.priority` | 开 | 非负整数，上限 10000；下一次派发按新顺序 |
| `<id>.enabled` | 开 | `true`/`false` 启停 |
| 其他字段 | 只读 | 改 handler_ref 用 `listen_event action=modify` |

控制面写入是普通 `memory_write`：before 事件 target 是设备路径，镜像层安全监听器可以 allow/replace；设备只做结构校验，授权由策略层做。控制面字段不进 `persist_to`，检查点/内核转储单独保存。

## 7. 写一个镜像层安全程序

### 7.1 引导顺序

Core 不预置保护监听器，也不强制顺序。镜像的初始程序应显式完成：

1. 建好权限数据节点（`ctrl.acl.*`、组标签、受保护节点）；
2. 建好处理器节点（`ctrl.type="handler"`）；
3. 注册保护监听器（内存读写、监听注册、对话创建/关闭、控制面写入）；
4. 再创建用户态对话、开放写入面。

顺序错误的后果是真实存在：先创建用户对话再注册保护，早期操作不受策略约束。这是镜像作者的责任。

### 7.2 ACL 数据约定

ACL 是策略层约定，Core 不认识 `acl.*`。推荐写在节点 `ctrl`（`dict[str,str]`，值为 JSON 字符串）：

```json
ctrl.acl.read  = {"allow": [{"group": "worker"}, {"cid": 3}], "deny": [{"cid": 9}]}
ctrl.acl.write = {"allow": [{"group": "worker"}], "deny": []}
```

规则项支持 `cid` 与 `group`；`deny` 与 `allow` 的取舍由处理器代码决定（推荐 deny 优先）；多条监听器规则冲突由 priority 顺序裁决。

### 7.3 处理器示例

```python
def handle(event):
    info = event.get("target_info") or {}
    ctrl = info.get("ctrl") or {}
    source = (event.get("source") or {}).get("conversation") or {}
    if source.get("mode") == "kernel":
        return {}  # 内核态监督对话越过 ACL
    groups = source.get("groups") or []
    cid = source.get("cid")
    acl = json.loads(ctrl.get("acl.read") or "{}")
    def match(rule):
        return rule.get("cid") == cid or rule.get("group") in groups
    if any(match(r) for r in (acl.get("deny") or [])):
        return {"action": "replace", "response": "ACL 拒绝读取"}
    if any(match(r) for r in (acl.get("allow") or [])):
        return {"action": "allow"}
    return {"action": "replace", "response": "ACL 默认拒绝读取"}
```

这段处理器由镜像作者在初始程序里注册；ACL 是镜像层策略，Core 只提供 before 事件与解析后的条件（见 7.1 与 7.4）。

### 7.4 自保护清单

策略层自己的配置也要被策略保护：

- 监听注册：`listen_event` 的 before 事件检查请求者模式、目标、tier/priority；
- 优先级：`priority>0` 与 tier=kernel 只有内核态可设置；安全对话占据高优先级；
- 处理器代码：注册时抄副本，之后改内存节点不影响已注册监听器；
- 权限数据：`ctrl.acl.*` 的 `memory_write`/`edit_metadata` before 由安全监听器保护；
- 对话元数据：控制面 `.groups`/`.roots`/`.meta` 写入会改变权限判定依据，安全监听器要保护（尤其 `.roots` 授予与撤销）；
- 控制面设备：只有内核态对话能命名；内核态之间按镜像层策略分工。

### 7.5 错误契约

| 前缀 | 含义 | 来源 |
| --- | --- | --- |
| `Error:` | 可判定失败（不存在/未授权/处理失败） | 指令参数校验、设备校验、before handler_error |
| `Illegal:` | 语法/契约违规 | 指令边界、控制面非法字段 |
| `Interrupted:` | 被监听器 `replace` 拦截 | before 决策 |

after 处理器失败不改变结果、只记录 `error_count`/`last_error` 与 stderr。

## 8. 当前边界（未实现）

- `mode` / `tools` 运行期写入（控制面只读）。
- 监听器 `.handler_ref` 控制面写入（用 `listen_event action=modify`）。
- 节点级内存访问权限（密码、访问回调等）未实现；设备 `authorize_access` 不做。
- 整棵子树删除、授予根的自动回收；事件持久化、条件触发、定时器。
- Guard 监听器没有内置实现：镜像作者要自己注册（Core 只提供 `listen_event` before 事件与解析后的条件）；`images/base` 目前没有 Guard，监听注册默认放行。
- 安全对话的自动初始化/监督（Core 不预置；镜像作者负责）。
- 多对话共享 ACL 数据的统一格式（当前是约定，不是 Core 校验的 schema）。

## 9. 参考

- [docs/runtime.md](runtime.md)：事件模型、设备、调度、检查点。
- [docs/system-image.md](system-image.md)：镜像 `security` 段与程序节点格式。
- [CONTEXT.md](../CONTEXT.md)：术语。
- 设计稿（开发记录，仓库内不跟踪）：`docs/private/Agents/implemented/design-permission-system-2026-10-27.md`、`docs/private/Agents/implemented/design-event-layer-permission-2026-10-28.md`。
- 计划与报告：`docs/private/Agents/implemented/plan-permission-phase{1,2,3,4}-*.md`、`report-permission-phase{1,2,3,4}-*.md`。

## 讨论区

- **设备目标的监听注册授权**：`listen_event` 的新式条件路径目前不调用设备 `authorize_listener`（该钩子只在宿主侧 `EventSystem.register/update` 上生效），所以内核态对话可以在未经设备同意时监听设备路径；用户态受 preset 限制表达不了设备目标。设备策略应该继续由设备钩子强制、还是统一交给 Guard 按 canonical target 裁决，需要作者决定。
- **用户态自定义条件的缺口**：节点所有权删除后，用户态仍只能用 preset，无法直接表达内存目标；因此按 caps/roots 做策略的 Guard 目前只能看到内核态注册的内存监听。是否放开用户态自定义条件是事件子集策略问题。

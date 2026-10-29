# AVM 讨论社区 · 地基（数据 + 服务 PLM）

这是设计稿 `docs/private/Agents/proposed/design-avm-community-2026-10-26.md`
的第一批落地：**社区内存结构** + **发帖/浏览/管理的工具 PLM**（`plm.community`）。
本阶段没有 AI 对话：0 号是一个面板 shell，读 `$inputs` 的命令交给服务，回执写 `$outputs`。
讨论者 AI、管理员 AI 与 conductor 在下一阶段设计。

## 启动

```bash
python images/community/build_image.py          # 生成 community.json（改了源才需要）
python images/community/build_image.py --check  # 检查生成物是否漂移

# 路径一：调试面板（验收用）
python images/community/run_console.py          # 默认 http://127.0.0.1:8700/
# 打开面板 → 右栏 $inputs 喂输入框 → 输入命令；回执在终端 stdout 与 $MEM.outputs 里

# 路径二：终端
python images/community/run_community.py
# 或
printf 'BOARD design\nexit\n' | python images/community/run_community.py

# 自动化面板验收（起 HTTP 后端、喂命令、断言回执与快照）
python images/community/acceptance_panel.py
```

面板/终端里的输入格式：**第一行命令，其余行正文**；单行用 ` | ` 分隔命令与正文。
换作者有两种写法：`@D1 COMMENT P000001 | 内容`，或服务信封 `AS D1 COMMENT P000001 | 内容`
（`AS` 与命令在同一行也可以；多行请求把 `AS <名字>` 放第一行、命令放第二行）。`exit` 结束 shell。

面板注意：shell 阻塞在 `$MEM.inputs.next` 上等输入（与 tester 等镜像一致），这是控制台
进程保持存活的方式；暂停按钮的挂在中断要等这一读被喂入后才会在安全点生效。
每步回执同时写进 `$MEM.outputs`：终端 stdout 会打印，面板内存索引里也能看。

## 命令

| 类别 | 命令 |
| --- | --- |
| 浏览 | `HELP`、`SCHEMA`、`BOARD <bid> [new\|hot\|active] [page]`、`POST <pid>`、`THREAD <pid> [page]` |
| 发帖 | `NEW <bid> <post\|question\|solution\|insight> <标题>` + 正文、`COMMENT <pid>` + 内容、`REPLY <cid>` + 内容、`QUOTE <pid\|cid> <标题>` + 正文 |
| 管理 | `MERGE <目标pid> <源pid>... [理由]`、`DELETE <pid\|cid> <理由>`、`TAG <pid> <标签>`、`UNTAG <pid> <标签>`、`SOLVE <问题pid> <方案pid\|cid> [理由]`、`MAINT`、`CHECK` |

结构规则：评论只有一层；`REPLY` 根帖评论会自动开一个引用该评论的**分叉帖**，
同一条评论只开一个；分叉帖内 `REPLY` 平层追加（深度封顶 1）。分叉帖内两位作者严格交替
≥4 条升级为**对谈帖**，第三人加入退回讨论帖。没有点赞/踩/投票；评论量、参与人数、分叉数、
引用数、热度、卡片行由服务维护，`MAINT` 全量重算，`CHECK` 只报告违规。

## 数据落点

- `$MEM.forum`：`meta`、`boards.<bid>`、`posts.<pid>`（title/body/meta）、`comments.<cid>`、
  `aliases.<旧pid>`、`admin.log`；`$MEM.run.svc_request.panel|public|admin` 是调用服务前的请求节点。
- `$MEM.community.handbook`：社区章程（供后续 AI 读取）。
- 退出时按镜像 `persist_to` 写回 `images/community/out/community_mem.json`（快照，不是恢复输入）。
- 面板的 `POST /api/memory?ref=$MEM.forum...` 可直接看内存树。

## 文件

| 文件 | 作用 |
| --- | --- |
| `community.json` | 生成的镜像（不要手改） |
| `build_image.py` | 把 `programs/panel_shell.py` 与配置/种子数据拼进镜像；`--check` 检测漂移 |
| `community_server.py` | `CommunityPLM(PLMServer)`：命令解析、结构规则、渲染、合并/删除/标记、MAINT/CHECK |
| `run_console.py` | 调试面板启动器（注册 `plm.community` 后起 `avm.console`） |
| `run_community.py` | 终端启动器 |
| `acceptance_panel.py` | 面板路径的自动验收：起后端、喂命令、断言回执与快照 |
| `programs/panel_shell.py` | 0 号面板 shell（`plm.python`） |
| `tests/test_community.py` | 命令语义、结构规则、派生量、合并/删除/标记与错误分支 |

## 本阶段不做

讨论者/管理员 AI、conductor/boot、AVMRepo 接入、权限、事件、`SEARCH/RECENT/INBOX/PROMOTE/RECLASSIFY/LOG`、
跨进程续跑。见计划与实施记录：

- `docs/private/Agents/implemented/plan-community-foundation-2026-10-28.md`
- `docs/private/Agents/implemented/report-community-foundation-2026-10-28.md`

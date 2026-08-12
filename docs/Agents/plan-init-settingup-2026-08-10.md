# 计划：镜像 init 的 system_ref 必须指向 settingup 程序节点

**时间**: 2026-08-10
**状态**: 已实施
**标注**: 按用户计划改动（用户指出镜像中设置初始对话时引用的还不是 settingup 节点）。此改动与 create_cmd/create_sub 的程序节点校验（2026-08-08 已实施）及 CONTEXT.md 的 Settingup 定义对齐。
**背景**: 2026-08-08 起 `create_cmd` / `create_sub` 的 `system_ref` 必须指向 `ctrl.type="settingup"`（LLM 程序）或 `"python"`（Python 程序）节点，但 init 仍走旧路径：`_start_init` 用 `core.unwrap(system_ref)`（for_llm=True），只要求最终是字符串——三个镜像（tester / demo / calc_driver）的 `$MEM.system` 都是 str 节点，与"对话由 Settingup 程序启动"的语义不一致。
**起因**: 用户指出 init 引用的还不是 settingup。
**行动**（提案，全部为"按用户计划改动"）:
1. **avm/image.py `_start_init`**：`system_ref` 必须指向 `ctrl.type="settingup"` 的 MetaDict 节点，取其 `content`（必须为字符串）作为提示词；str 节点、无类型 dict 节点报错（错误信息含 "settingup"）。`user_ref` 改用 `for_llm=False` 解引用并强制最终为字符串（str 节点），避免 MetaDict 被静默转成摘要。字面量 `system` 保留为匿名入口程序（不注册节点）。
2. **测试**（tests/test_image.py）：`_sample_image` 的 system 改为 settingup 节点并更新断言；新增 4 个错误用例——system_ref 指向 str 节点、无类型 dict、settingup 缺 content、user_ref 指向非 str 节点。
3. **镜像**（images/tester.json / demo.json / calc_driver.json）：`$MEM.system` 从 str 节点改为 `ctrl.type="settingup"` 的 dict 节点（value 含 name 与 content），提示词内容原样保留（tester 的提示词为用户近期修改过的版本，未改动文字）。
4. **文档**：docs/system-image.md init 一节改为"system_ref 必须指向 settingup 节点"，示例镜像同步更新；README.md / README_EN.md 系统镜像一节补注。
**影响**: init 与 create_cmd/create_sub 的程序节点语义统一；字面量 system 仍可用于最小镜像；str 节点作为 init 引用被拒绝（与 2026-08-08 的收紧方向一致）。
**实施记录（2026-08-10）**:
- avm/image.py、tests/test_image.py、images/tester.json、images/demo.json、images/calc_driver.json、docs/system-image.md、README.md、README_EN.md 已改；全量测试通过，三个镜像加载正常。

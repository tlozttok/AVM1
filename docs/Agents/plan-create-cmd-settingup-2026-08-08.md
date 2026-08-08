# 计划：create_cmd/create_sub 的 system_ref 必须指向程序节点（settingup / python）

**时间**: 2026-08-08
**状态**: 已实施
**标注**: 按用户计划改动（用户指出 create_cmd 遗留问题：它应该接收 settingup 类型的节点，就像创建 Python 对话时接收 python 类型的节点一样）。此改动同时与 CONTEXT.md 的 Settingup 定义（type="settingup"，create_cmd/create_sub 启动 Settingup 程序）对齐。
**背景**: `_build_conversation` 原本接受任何节点：str 节点直接当系统消息，MetaDict 只要有 content 就行，ctrl.type 只用于识别 python。tester 镜像的手册甚至写明"system_ref 可以是 dict 节点或直接指向 str 节点"，且镜像只预置了 python 程序、没有 settingup 程序。
**起因**: 用户发现镜像问题：create_cmd 应接收 settingup 类型的节点（LLM 程序），与 Python 对话接收 python 类型节点对等。接受任意节点导致：程序类型不显式、参考手册等无类型 dict 也可被误当作程序、str 节点绕过程序注册语义。
**行动**（提案）:
1. avm/core.py：`_build_conversation` 校验 system_ref 指向的节点必须是 `ctrl.type="settingup"`（LLM 程序）或 `"python"`（Python 程序）的 dict 节点；str 节点、无类型 dict 节点、其他类型一律报错。`create_cmd` / `create_sub` 的 execute 捕获校验错误，把 "Error: ..." 作为工具响应返回，不创建对话，本对话保持活跃可自纠（与 call_service 缺服务等指令错误的既有模式一致）。
2. 工具定义：`create_cmd` 的 description 与 system_ref 参数说明改为"程序节点（settingup 或 python）"。
3. 测试：更新所有以 str 节点喂 create_cmd/create_sub/_build_conversation 的用例为 settingup 程序节点；新增 create_cmd / create_sub 拒绝 str 节点、拒绝无类型 dict 节点的测试。
4. images/tester.json：手册第 4/5 条改写为"system_ref 必须指向程序节点"；预置一个 settingup 程序 `$MEM.programs.helper`（name/content）；概览的指令集数量 9 → 10；预置程序列表补 helper。
5. docs/system-image.md：`ctrl.type` 取值补 `settingup`、`python`；新增"程序节点（settingup / python）"说明与运行时校验规则；注明 init 的 system_ref 仍指向 str 节点（入口对话）。
6. README.md / README_EN.md：双元数据 ctrl 例子补 settingup；指令集表 create_cmd 行注明 system_ref 必须指向程序节点；"如何设计 AVM 程序"节补充 create_cmd/create_sub 的程序节点约束。
**影响**: create_cmd/create_sub 的入参语义收紧为显式程序节点；错误以工具响应返回（不崩溃、不创建半成品对话）；tester 镜像获得可创建 LLM 对话的预置程序。文档（CONTEXT.md 定义、system-image 规范、README）与代码对齐。
**实施记录（2026-08-08）**:
- avm/core.py、tests/test_avm.py、tests/test_monitor.py、images/tester.json、docs/system-image.md、README.md、README_EN.md 已改；184 个测试全部通过（原 181 + 新增 3 个拒绝非程序节点的测试）。

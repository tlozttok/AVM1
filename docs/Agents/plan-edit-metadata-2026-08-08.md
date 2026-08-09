# 计划：edit_metadata 指令（编辑节点的 ctrl 元数据）

**时间**: 2026-08-08
**状态**: 已实施
**标注**: 按用户计划改动（用户明确给出指令名、参数与语义：type=set/get/del，key/value 可选、set 时必需）。
**背景**: 镜像规范规定 ctrl 是 `dict[str,str]` 元数据二，但运行时没有编辑它的指令：`memory_make` 创建的节点没有 ctrl，程序节点（settingup/python）只能在镜像中预置。用户提出新指令 `edit_metadata`，专门编辑节点的 ctrl 元数据；JSON 图式难以表达"value 在 type=set 时必需"的条件约束，用户明确接受用"错误返回给 LLM 自纠"解决语法问题。
**起因**: 用户要求加入该指令。
**行动**（提案，全部为"按用户计划改动"）:
1. **avm/core.py**：新增 `EditMetadataInstruction`（工具名 `edit_metadata`，参数 `ref`、`type`(set/get/del)、`key`、`value`），注册进指令集。语义：
   - set：`ctrl[key] = value`（key/value 都必需；节点无 ctrl 时创建）；
   - get：带 key 返回 `ctrl[key]`，key 省略返回整个 ctrl 的 JSON（空则 `{}`）；
   - del：删除 `ctrl[key]`（key 必需）。
   目标必须是 MetaDict/MetaList 节点（str、设备、其他类型报错）；引用不存在报错。所有非法参数（type 非法、set 缺 value、get/del 缺 key、键不存在）以 `Error: ...` 工具响应返回，对话保持活跃可自纠——这正是用户指定的"错误返回给 LLM"模式。
2. **测试**（tests/test_avm.py）：新增 14 个测试——set 创建/保留 ctrl、set 缺 value、get 按 key/整体/空 ctrl、get 缺键、del、del 缺键、type 非法、str 节点报错、引用不存在、MetaList 节点、完整推进执行后对话保持活跃。208 个测试全部通过（原 194 + 14）。
3. **tester 提示词**（images/tester.json）：工具列表加 `edit_metadata`；指令集 11 → 12；指令参考新增第 4 条（原有条目顺延）；"运行时无法创建带 ctrl 的节点"改为"运行时可用 edit_metadata 补 ctrl，配合 memory_make/memory_write 组合程序节点"。
4. **文档**：CONTEXT.md 内存指令加 `edit_metadata`；docs/system-image.md 程序节点一节注明运行期可补 ctrl；README.md / README_EN.md 指令表加行、指令数 11 → 12。
**影响**: 运行时获得编辑 ctrl 的能力——程序节点（settingup/python）可以在运行期组合生成，不再只能镜像预置；非法参数通过 Error 工具响应回到 LLM 自纠（与既有指令参数错误模式一致，不涉及挂起中的 json_error/unknown_tool 错误机制）。
**实施记录（2026-08-08）**:
- avm/core.py、tests/test_avm.py、images/tester.json、CONTEXT.md、docs/system-image.md、README.md、README_EN.md 已改；208 个测试全部通过。

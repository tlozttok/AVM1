# 计划：终端输入体验优化（中文输入）

**时间**: 2026-08-12
**状态**: 已实施
**标注**: 按用户计划改动（用户要求优化终端输入中文的体验）。
**背景**: `InputsListDevice` 用裸 `input()` 阻塞读取：终端黑屏裸等、无 readline（无历史/方向键/输入法组合支持）、终端编码非 UTF-8 时中文乱码且无提示。
**起因**: 用户反馈终端中文输入体验需要优化。
**行动**（提案，全部为"按用户计划改动"）:
1. **输入提示**（avm/memory_device.py）：`_prompt_operator` 改为 `input(self._prompt_text())`，提示标明等待输入的对话身份（如 `[tester（cid=0）等待输入，/save 保存检查点]`）——不再黑屏裸等；身份取自 Core 的活跃对话（设备已有 attach_core 注入的 Core 引用）。
2. **readline**：模块顶部 try/import readline（Linux/macOS 启用行编辑与历史；Windows 无此模块时静默降级）。
3. **编码警告**：首次读取输入时检查 `sys.stdout.encoding`，非 UTF-8 打印一次警告（中文乱码常见原因是终端编码）；`UnicodeDecodeError` 捕获后提示并继续等待。
4. **测试**：更新既有 /save 测试的 input 桩（现在带 prompt 参数）；新增提示语包含对话身份的测试。全量 228 通过（原 227 + 1）。
5. **文档**：README.md / README_EN.md 调试与观测一节补充终端输入说明。
**影响**: 操作者终端输入从"黑屏裸等"变为带身份提示的交互；readline 改善中文输入法组合体验；非 UTF-8 终端得到明确警告。对话侧无任何变化（输入内容仍原样交付，不破坏检查点的字节一致性）。
**实施记录（2026-08-12）**:
- avm/memory_device.py、tests/test_checkpoint.py、README.md、README_EN.md 已改；228 个测试全部通过。

# 计划：手册讲有类型节点 + 补全节点类型检查（para_ref）

**时间**: 2026-08-12
**状态**: 已实施
**标注**: 按用户计划改动（用户要求：手册讲一下有类型节点，并且补全节点的类型检查）。
**背景**: 上一轮 tester 运行中，AI 自建了"参数类"示例模板并把 create_cmd 的 para_ref 指向它（model=gpt-4），child 首轮执行触发 API 400。根因有二：① tester_manual 只讲了 system_ref 必须是程序节点，完全没讲 para_ref 与 para 节点类型，AI 只能猜；② 代码层面 create_cmd/create_sub 只校验 system_ref（settingup/python），不校验 para_ref（必须是 ctrl.type='para' 的 dict 节点），`_get_para` 也只在"是 MetaDict"时静默接受任意类型节点。
**起因**: 用户要求手册补充有类型节点说明，并补全节点的类型检查。
**行动**（提案，全部为"按用户计划改动"）:
1. **类型检查补全**（avm/core.py）：
   - `_build_conversation`：para_ref 指向的节点必须是 `ctrl.type="para"` 的 dict 节点，否则抛 VMMemoryError——create_cmd / create_sub 捕获后以 Error 工具响应返回，不创建对话（与 system_ref 的程序节点校验对齐）。
   - `_get_para`：节点存在但不是 para 类型时不再静默使用，改为 stderr 警告（"不是 ctrl.type='para' 的 dict 节点，回退默认参数"）+ 回退 gpt-4；路径不存在仍静默回退（裸 Core / 无镜像上下文的设计行为）。
2. **tester_manual**（images/tester.json）：
   - memory_model 新增"有类型节点"：settingup / python / para 三种 ctrl.type 的含义、节点要求（content/name/signature；content；model/temperature）、system_ref 与 para_ref 的对应关系、本镜像 API 只接受 deepseek-v4-pro / deepseek-v4-flash、无类型节点被拒绝、edit_metadata 可运行时设置类型。
   - instruction_ref 第 5/6 条（create_cmd / create_sub）：补 para_ref 必须指向 ctrl.type='para' 参数节点，system_ref 或 para_ref 不合法都报错且不创建对话。
   - overview 增加一行"有类型节点"概述。
3. **测试**：批量给 39 处测试参数节点补 `ctrl={"type": "para"}`（机械替换）；新增 3 个测试——create_cmd 拒绝非 para 节点、create_sub 拒绝非 para 节点、`_get_para` 对错误类型警告并回退。全量 231 通过（原 228 + 3）。
**影响**: AI 从手册能知道 para_ref 必须指向 para 节点及 model 可选值；create_cmd/create_sub 在创建时即拒绝错误类型的 para_ref（错误不再推迟到 API 层）；`_get_para` 对运行时被破坏的 para 节点给出可见警告。
**实施记录（2026-08-12）**:
- avm/core.py、tests/test_avm.py、tests/test_monitor.py、tests/test_checkpoint.py、tests/test_main.py、images/tester.json 已改；231 个测试全部通过，tester 镜像加载正常。

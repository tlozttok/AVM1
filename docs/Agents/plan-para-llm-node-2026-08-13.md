# 计划：tester 参数库补上正常 LLM 模型参数（$MEM.para.llm）

**时间**: 2026-08-13
**状态**: 已实施
**标注**: 按用户计划改动（用户要求"把正常的模型参数也加进去"）。
**背景**: 上一轮讨论确认 `$MEM.para` 是预置参数目录，但只有 Python 执行器的两个参数节点（simple=plm.simple、python=plm.python），没有 LLM 对话用的正常 API 参数节点。上一轮运行时 AI 因此自建 para 节点并把 model 写成 gpt-4，触发 API 400。
**起因**: 用户要求把正常模型参数也预置进参数库。
**行动**（提案，全部为"按用户计划改动"）:
1. images/tester.json：`$MEM.para` 新增 `llm` 节点（ctrl.type="para"，model="deepseek-v4-flash"），与 simple / python 并列；容器 meta 改为"预置模型参数库：创建对话时使用（llm=LLM API 参数，simple/python=PLM 类型参数）"。
2. tester_manual 三处：
   - overview 预置行改为"参数库在 $MEM.para：llm（LLM 对话，deepseek-v4-flash）、simple（plm.simple）、python（plm.python）"；
   - instruction_ref 行为事实新增"创建对话时 para_ref 优先复用预置参数节点（LLM 用 $MEM.para.llm，Python 用 simple/python），不要自建参数节点"；
   - memory_model 的 para 条目补充参数库说明。
3. 验证：tester 镜像加载正常，`$MEM.para.llm` 是合法 para 节点；全量 231 测试通过（无代码改动）。
**影响**: AI 创建对话时参数库三选一即可（llm / simple / python），不再需要自建 para 节点，降低模型名写错的风险。
**实施记录（2026-08-13）**:
- images/tester.json 已改；tester 镜像加载正常，231 个测试全部通过。

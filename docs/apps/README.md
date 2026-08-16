# AVM 应用套件

这三个应用是 AVM 运行时的「实际性测试」：每个应用都从系统镜像启动，由提示词定义 LLM 式算法/控制流，并覆盖至少一条运行时机制边界。它们不修改 `avm/`，只使用镜像格式允许的扩展点。

| 应用 | 一句话 | 主要覆盖 |
|---|---|---|
| [courier_office](courier_office.md) | LLM 调度员创建 LLM 工人，工人把表达式外包给 plm.simple 计算，结果记入自定义 BoardDevice 并显式送回 | create_cmd / send_instruction / create_sub / memory_make / close_conversation / plm.simple / 自定义设备 |
| [vault_guard](vault_guard.md) | 守卫为金库打标并注册 before 写入守卫与 after 审计通知，工人写入被拦截或改写，审计通过 notify 唤醒守卫 | listen_event / edit_metadata / owner / replace / 改写 / after notify |
| [plm_ledger](plm_ledger.md) | 柜员把单行命令外包给镜像外的自定义 LedgerPLM，账本通过 memory_write 工具调用自行持久化 | PLMServer 契约 / 工具调用往返 / create_sub 自动返回 |

## 目录

```
images/apps/
  courier_office/
    courier_office.json       # 系统镜像
    devices/board.py          # 自定义记录板设备（.append 追加 / .latest 读最新）
    devices/io.py             # 操作者输入输出设备
  vault_guard/
    vault_guard.json          # 系统镜像
    devices/io.py
  plm_ledger/
    plm_ledger.json           # 系统镜像
    plm_ledger_server.py      # 自定义 PLM 服务器（OpenAI 兼容）
    run_ledger.py             # 宿主启动器：注册 PLM 后运行镜像
    devices/io.py
```

## 启动

前两个应用与普通镜像一致：

```bash
python main.py images/apps/courier_office/courier_office.json
python main.py images/apps/vault_guard/vault_guard.json
```

plm_ledger 需要先注册镜像外的 PLM 服务器：

```bash
python images/apps/plm_ledger/run_ledger.py
```

终端输入 `exit` 结束。`python main.py <镜像> --frames` 可查看监测帧摘要。

## 自动化验证

真实 LLM 应用不进入普通 pytest；以下测试用脚本化 MockLMU + 真实 PLM/设备验证控制流与状态：

```bash
python -m pytest tests/test_app_images.py -q
```

详细设计见各应用文档。

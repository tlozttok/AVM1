# 程序包与装载设备演示镜像

这块演示计划 `plan-persistent-runtime-2026-11-07` 的批次 B：运行期装卸程序包/设备，
以及"快照 + 日志"恢复后包仍可用。

```bash
python images/packages/build_image.py        # 生成 packages_demo.json 与 example.pkg.json
python -m avm.console images/packages/packages_demo.json
```

镜像与包：

| 文件 | 内容 |
|---|---|
| `loader.py` | `LoaderDevice`：把"装包 / 卸包 / 热装设备"变成对 `$MEM.loader` 的一次 `memory_write` |
| `example.pkg.json` | 示例程序包：`$MEM.packages.demo.1_0_0`，含一个 `plm.python` 程序（`yield from` 工具调用：读 `$MEM.example.input`，把结果写 `$MEM.example.output`） |
| `build_image.py` | 生成上面两个 JSON（loader 无密码；装载/安装命令必须带审批哈希） |
| `packages_demo.json` | 内核态安全监听器只放行操作员对话（保留 cid `-1`）写 `$MEM.loader`，其他对话的写入被拦截 |

装载与安装要带操作员审批得到的 `approved_sha256`：设备文件或程序包文件的 sha256。

操作员在控制台里依次发出（都是 `memory_write`/`create_cmd`/`send_instruction`，面板可以手选）：

1. `memory_write $MEM.loader`
   `{"action": "install_package", "path": "example.pkg.json", "approved_sha256": "<包文件 sha256>"}`
2. `create_cmd $MEM.packages.demo.1_0_0.programs.pack $MEM.packages.demo.1_0_0.params`
3. `send_instruction <上一步的 cid> "yield from pack()"`
4. 用 `memory_read $MEM.example.output` 看结果

恢复：镜像声明了 `journal_to` / `snapshot_to` / `persist_to`。正常退出后

```bash
python main.py images/packages/packages_demo.json --restore images/packages/out/packages_demo.snapshot.json
```

会恢复整机状态并按日志重放尾部——之前装的包、包里的设备和 `current` 指针都还在，可以直接
再 `create_cmd` 用包内程序。`images/packages/out/` 是运行落点（已 gitignore），删掉它就从
全新镜像重新开始。

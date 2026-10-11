import { expect, test } from '@playwright/test'

/**
 * 四个场景，共用一个 mock console（worker=1，串行）：
 * 观测、工具选择、三步交互、喂输入。后端契约变了这里会先红。
 */
test.describe.configure({ mode: 'serial' })

const CONSOLE_URL = process.env.AVM_CONSOLE_URL ?? 'http://127.0.0.1:8791'

test('观测：调度、对话详情、内存索引与设备信息', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByTestId('status-banner')).toContainText('运行中')
  await expect(page.getByTestId('event-log-connected')).toContainText('SSE 已连接')
  // 指令表只有后端一份：GET /api/tools，九条指令都得读到（读不到就是红）
  await expect(page.getByTestId('action-tools-source')).toContainText('GET /api/tools（9 条）')

  const list = page.getByTestId('conversation-list')
  await expect(list).toContainText('init')
  await expect(list).toContainText('面板')

  // 对话详情：正在跑的 init 对话，历史里有脚本执行器写的助手消息
  await page.getByTestId('conversation-item-0').click()
  await expect(page.getByTestId('conversation-detail')).toContainText('cid 0')
  await expect(page.getByTestId('conversation-detail')).toContainText('第 1 轮')

  // 内存索引：点节点看内容
  const tree = page.getByTestId('memory-tree')
  await expect(tree).toContainText('note')
  await tree.locator('[data-ref="$MEM.note"]').getByRole('button', { name: '看' }).click()
  await expect(page.getByTestId('memory-view')).toContainText('一条笔记')

  // 设备节点 → 设备信息（第一轮只读）
  await tree.locator('[data-ref="$MEM.inputs"]').getByRole('button', { name: '看' }).click()
  const device = page.getByTestId('device-view')
  await expect(device).toContainText('InputsListDevice')
  await expect(device).toContainText('投喂一行输入')

  // 读一个具体路径：ref 输入框
  await page.getByTestId('ref-input').fill('$MEM.note.v')
  await page.getByTestId('ref-read').click()
  await expect(page.getByTestId('memory-view')).toContainText('$MEM.note.v')
})

test('工具选择：九条全在下拉里，未知名可以手输且不被顶回', async ({ page }) => {
  await page.goto('/')

  // 九条指令加一项「其它（手输）」；少一条就是红的（曾经的 bug：datalist 只按已输入内容过滤）
  await expect(page.getByTestId('action-tool').locator('option')).toHaveText([
    'memory_read',
    'memory_write',
    'memory_make',
    'edit_metadata',
    'create_cmd',
    'create_sub',
    'send_instruction',
    'close_conversation',
    'listen_event',
    '其它（手输）',
  ])

  // 选中哪条，参数表单就跟着换
  await page.getByTestId('action-tool').selectOption('edit_metadata')
  await expect(page.getByTestId('action-field-type')).toBeVisible()

  // 手输未知名：这是后端有意允许的路径，名字必须留得住（曾经的 bug：effect 每敲一下就顶回第一项）
  await page.getByTestId('action-tool').selectOption({ label: '其它（手输）' })
  await page.getByTestId('action-tool-manual').fill('no_such_tool')
  await page.waitForTimeout(700)
  await expect(page.getByTestId('action-tool-manual')).toHaveValue('no_such_tool')

  // 切回已知的一条，手输框收起来
  await page.getByTestId('action-tool').selectOption('memory_read')
  await expect(page.getByTestId('action-tool-manual')).toHaveCount(0)
  await expect(page.getByTestId('action-field-ref')).toBeVisible()
})

test('三步交互：暂停 → 提交动作 → 继续', async ({ page }) => {
  await page.goto('/')

  await page.getByTestId('pause-button').click()
  await expect(page.getByTestId('status-banner')).toContainText('已暂停')

  await page.getByTestId('action-tool').selectOption('memory_write')
  await page.getByTestId('action-field-ref').fill('$MEM.note')
  await page.getByTestId('action-field-content').fill('来自面板的一次写入')
  await page.getByTestId('action-submit').click()
  await expect(page.getByTestId('action-result')).toContainText('已提交 memory_write')

  const log = page.getByTestId('event-log')
  // 工具调用与工具响应都由面板通道发出（init 的指令不产生 tool_call 帧）
  await expect(log.getByTestId('log-entry-tool_call').filter({ hasText: 'memory_write' }).first())
    .toBeVisible()
  await expect(log.getByTestId('log-entry-tool_result').filter({ hasText: 'Success set' }).first())
    .toBeVisible()
  await expect(log.getByTestId('log-entry-event').filter({ hasText: 'memory_write' }).first())
    .toBeVisible()

  await page.getByTestId('resume-button').click()
  await expect(page.getByTestId('status-banner')).toContainText('运行中')
})

test('喂输入：队列里的一行被下一次读取取走', async ({ page }) => {
  await page.goto('/')

  await page.getByTestId('input-feed-field').fill('你好，面板')
  await page.getByTestId('input-feed-send').click()
  await expect(page.getByTestId('input-feed-result')).toContainText('已排队')

  // init 的下一轮改读 $MEM.inputs.-1，事件流里出现这次读
  const log = page.getByTestId('event-log')
  await expect(log.getByTestId('log-entry-event').filter({ hasText: 'inputs' }).first())
    .toBeVisible()

  // 被读走的那一行留在设备列表里（$MEM.inputs.0 是普通下标读，不阻塞）。
  // 不查对话详情的 user_batch：它在 Core 每次 exec 后被清空（avm/core.py:1108），只在亚秒窗口内可见。
  await page.getByTestId('ref-input').fill('$MEM.inputs.0')
  await page.getByTestId('ref-read').click()
  await expect(page.getByTestId('memory-view')).toContainText('你好，面板')
})

test('构建产物由 console 自己托管：同一个面板从 8791 打开', async ({ page }) => {
  // 生产路径（计划 C9）：不经过 Vite，直接吃 web/dist。`npm run e2e` 会先构建。
  await page.goto(CONSOLE_URL)
  await expect(page.getByTestId('status-banner')).toContainText('运行中')
  await expect(page.getByTestId('action-tools-source')).toContainText('GET /api/tools')
  await expect(page.getByTestId('event-log-connected')).toContainText('SSE 已连接')
})

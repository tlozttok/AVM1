import { expect, test } from '@playwright/test'

/**
 * 操作面板 e2e：真实后端（`avm.console`）+ 脚本化 MockLMU。
 * 一个 worker 串行跑；每个用例从空闲态开始（init 跑完两轮后把执行权交给面板）。
 */
test.describe.configure({ mode: 'serial' })

const CONSOLE_URL = process.env.AVM_CONSOLE_URL ?? 'http://127.0.0.1:8791'

interface ConversationRow {
  cid: number
  name: string | null
  identity: string
}

async function shellCid(page: import('@playwright/test').Page): Promise<number> {
  const state = (await (await page.request.get('/api/state')).json()) as {
    conversations: ConversationRow[]
  }
  const shell = state.conversations.find((conv) => conv.name === 'shell' || conv.identity === 'shell')
  if (!shell) throw new Error('mock 里没有 shell 对话')
  return shell.cid
}

test('首屏：顶栏空闲、资源栏、命令台默认 shell、系统日志条、抽屉收起', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲（等待操作员）')
  await expect(page.getByTestId('topbar-active')).toHaveText('—')

  // 左栏：对话树与内存树都在
  await expect(page.getByTestId('conversation-list')).toContainText('init')
  await expect(page.getByTestId('conversation-list')).toContainText('shell')
  const tree = page.getByTestId('memory-tree')
  await expect(tree).toContainText('programs')
  await expect(tree.locator('[data-ref="$MEM.alias"]')).toContainText('→ $MEM.note')

  // 主入口：命令台默认目标是 shell，输入行就绪
  const cid = await shellCid(page)
  const deck = page.getByTestId('command-deck')
  await expect(deck).toBeVisible()
  await expect(deck).toContainText(`cid ${cid}`)
  await expect(deck).toContainText('操作员以 cid -1 身份发 send_instruction')
  await expect(page.getByTestId('command-input')).toBeEnabled()

  // 调试能力默认收起，底部系统日志条在（后台线程持续写 $MEM.outputs）
  await expect(page.getByTestId('debug-drawer-body')).toHaveCount(0)
  await expect(page.getByTestId('syslog-latest')).toContainText('mock 输出')
})

test('命令台：一行命令 → send_instruction + resume → shell 回复进历史', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')

  const seen: string[] = []
  const bodies: { path: string; body: string }[] = []
  page.on('request', (request) => {
    if (!request.url().includes('/api/')) return
    const path = new URL(request.url()).pathname
    seen.push(`${request.method()} ${path}`)
    if (request.postData() !== null) bodies.push({ path, body: request.postData() ?? '' })
  })

  await page.getByTestId('command-input').fill('help')
  await page.getByTestId('command-input').press('Enter')
  await expect(page.getByTestId('command-transcript')).toContainText('shell 收到：help')
  await expect(page.getByTestId('command-transcript')).toContainText('help')
  await expect(page.getByTestId('command-note')).toContainText('已投递')

  // 入口就是两个既有端点：投递 + 让出；不是新的输入机制
  expect(seen).toContain('POST /api/action')
  expect(seen).toContain('POST /api/resume')
  const send = bodies.find((item) => item.body.includes('send_instruction'))
  expect(send).toBeTruthy()
  const payload = JSON.parse(send!.body) as { tool: string; arguments: { content: string } }
  expect(payload.tool).toBe('send_instruction')
  expect(payload.arguments.content).toBe('help')

  // 发送后系统回到空闲，命令台继续可用
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')
  await expect(page.getByTestId('command-input')).toBeEnabled()
})

test('节点工作台：区域一节点类型编辑 + 区域二 ctrl.type 快捷修改/校验', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')

  const tree = page.getByTestId('memory-tree')
  await tree.locator('[data-ref="$MEM.programs"] .mem-twist').click()
  await tree.locator('[data-ref="$MEM.programs.shell"]').click()

  const workbench = page.getByTestId('node-workbench')
  await expect(workbench).toContainText('$MEM.programs.shell')
  await expect(page.getByTestId('node-shape')).toContainText('dict')
  await expect(page.getByTestId('node-ctrl-type')).toContainText('ctrl.type=python')
  await expect(page.getByTestId('node-region-shape')).toBeVisible()
  await expect(page.getByTestId('node-region-type')).toBeVisible()
  await expect(page.getByTestId('node-region-device')).toHaveCount(0)
  await expect(page.getByTestId('node-checks')).toContainText('para_ref')

  // 区域二：快捷修改写的是 content 子 str 节点
  await page.getByTestId('node-quick-content').fill("def handle(cmd):\n    return 'echo: ' + cmd\n")
  await page.getByTestId('node-quick-save-content').click()
  await expect(page.getByTestId('node-result')).toContainText('Success set')

  // 区域一：新建子节点（memory_make）
  await page.getByTestId('node-new-key').fill('hello')
  await page.getByTestId('node-make').click()
  await expect(page.getByTestId('node-result')).toContainText('Success created')

  // 区域二：ctrl 原样编辑（edit_metadata set）
  await page.getByTestId('node-ctrl-input-type').fill('python')
  await page.getByTestId('node-ctrl-set-type').click()
  await expect(page.getByTestId('node-result')).toContainText('ctrl.type')
})

test('handler 节点：区域二类型动作是监听（注册/查看）', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')

  const tree = page.getByTestId('memory-tree')
  await tree.locator('[data-ref="$MEM.handlers"] .mem-twist').click()
  await tree.locator('[data-ref="$MEM.handlers.guard"]').click()

  await expect(page.getByTestId('node-ctrl-type')).toContainText('ctrl.type=handler')
  await expect(page.getByTestId('node-region-type')).toContainText('监听')
  await page.getByTestId('node-register-listener').click()
  await expect(page.getByTestId('node-result')).toContainText('已注册')

  await page.getByTestId('node-view-listeners').click()
  await expect(page.getByTestId('debug-drawer-body')).toBeVisible()
  await expect(page.getByTestId('listener-editor')).toBeVisible()
})

test('设备节点：只有区域三，设备声明的调试动作可运行', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')

  await page.getByTestId('memory-tree').locator('[data-ref="$MEM.inputs"]').click()
  await expect(page.getByTestId('node-device')).toContainText('设备节点')
  await expect(page.getByTestId('node-region-device')).toBeVisible()
  await expect(page.getByTestId('node-region-type')).toHaveCount(0)

  await page.getByTestId('node-device-run-1').click()
  await expect(page.getByTestId('node-device-result-1')).toContainText('queued')
})

test('状态语义：空闲时暂停不可用；运行中暂停 → 操作员接管 → 继续', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲（等待操作员）')
  // 空闲 = 面板已持有执行权，暂停没有意义
  await expect(page.getByTestId('topbar-pause')).toBeDisabled()

  // 让 worker 跑起来（两个既有端点：投递 + 让出），系统进入运行中
  const state = (await (await page.request.get('/api/state')).json()) as {
    conversations: ConversationRow[]
  }
  const worker = state.conversations.find((conv) => conv.name === 'worker' || conv.identity === 'worker')
  expect(worker).toBeTruthy()
  await page.request.post('/api/action', {
    data: { tool: 'send_instruction', arguments: { cid: worker!.cid, content: 'go' } },
  })
  await page.request.post('/api/resume', { data: {} })
  await expect(page.getByTestId('topbar-status')).toContainText('运行中')
  await expect(page.getByTestId('topbar-active')).toContainText('worker')

  // 运行中点暂停：下一个安全点派发 operator 中断 → 操作员接管
  await page.getByTestId('topbar-pause').click()
  await expect(page.getByTestId('topbar-status')).toContainText('操作员接管（系统暂停）')
  await expect(page.getByTestId('topbar-pause')).toBeDisabled()
  await expect(page.getByTestId('topbar-resume')).toBeEnabled()

  // 继续 → 让出；worker 跑完或被关闭后回到空闲
  await page.getByTestId('topbar-resume').click()
  await expect(page.getByTestId('topbar-status')).toContainText(/运行中|空闲/)

  await page.request.post('/api/action', {
    data: { tool: 'close_conversation', arguments: { cid: worker!.cid } },
  })
  await page.request.post('/api/resume', { data: {} })
  await expect(page.getByTestId('topbar-status')).toContainText('空闲（等待操作员）')

  await page.getByTestId('topbar-shutdown').click()
  await expect(page.getByTestId('topbar-shutdown-confirm')).toBeVisible()
  await page.locator('.topbar-confirm').getByRole('button', { name: '取消' }).click()
  await expect(page.getByTestId('topbar-shutdown-confirm')).toHaveCount(0)
})

test('调试抽屉：原始动作、事件流、监听器编辑、系统信息', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('topbar-status')).toContainText('空闲')

  await page.getByTestId('debug-toggle').click()
  await expect(page.getByTestId('debug-drawer-body')).toBeVisible()
  await expect(page.getByTestId('drawer-pane-raw')).toBeVisible()

  await page.getByTestId('drawer-tab-events').click()
  await expect(page.getByTestId('event-log')).toBeVisible()

  await page.getByTestId('drawer-tab-listeners').click()
  const editor = page.getByTestId('listener-editor')
  // 前面的 handler 用例会新注册监听器，所以只断言列表非空、改第一条
  await expect(editor.getByTestId('listener-row').first()).toBeVisible()
  await editor.getByTestId('listener-edit').first().click()
  await editor.getByTestId('listener-field-enabled').uncheck()
  await editor.getByTestId('listener-save').click()
  await expect(editor.getByTestId('listener-result')).toContainText('已更新')
  await expect(editor.getByTestId('listener-row').first()).toContainText('停用')

  await page.getByTestId('drawer-tab-system').click()
  await expect(page.getByTestId('drawer-pane-system')).toContainText('生命周期')

  // 收起后再展开仍然可用
  await page.getByTestId('debug-toggle').click()
  await expect(page.getByTestId('debug-drawer-body')).toHaveCount(0)
  await page.getByTestId('debug-toggle').click()
  await expect(page.getByTestId('debug-drawer-body')).toBeVisible()
})

test('系统日志条：默认一行，展开后列输出、通知可过滤、可清屏', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('syslog-latest')).toContainText('mock 输出')

  await page.getByTestId('system-log').locator('.syslog-bar').click()
  await expect(page.getByTestId('syslog-body')).toBeVisible()
  await expect(page.getByTestId('syslog-row').first()).toBeVisible()
  await expect(page.getByTestId('syslog-notify-row').first()).toBeVisible()

  await page.getByTestId('syslog-notify-only').check()
  await expect(page.getByTestId('syslog-row')).toHaveCount(0)
  await expect(page.getByTestId('syslog-notify-row').first()).toBeVisible()
  await page.getByTestId('syslog-notify-only').uncheck()

  await page.getByTestId('syslog-clear').click()
  await expect(page.getByTestId('syslog-row')).toHaveCount(0)
})

test('构建产物由 console 自己托管：同一个面板从 8791 打开', async ({ page }) => {
  await page.goto(CONSOLE_URL)
  await expect(page.getByTestId('topbar-status')).toContainText(/空闲|运行中/)
  await expect(page.getByTestId('command-deck')).toBeVisible()
  const tree = page.getByTestId('memory-tree')
  await expect(tree.locator('[data-ref="$MEM.alias"]')).toContainText('→ $MEM.note')
  await expect(tree).toContainText('programs')
})

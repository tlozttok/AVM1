import { defineConfig } from '@playwright/test'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'

// 三个端口都可覆盖：默认值与本机其它调试进程不冲突。
const CONSOLE_PORT = Number(process.env.AVM_E2E_CONSOLE_PORT ?? 8791)
const WEB_PORT = Number(process.env.AVM_E2E_WEB_PORT ?? 5273)
const CONSOLE_URL = `http://127.0.0.1:${CONSOLE_PORT}`

// 本机 ~/.npm 与 ~/.cache/ms-playwright 只读：用盘上已有的 chromium，不下载浏览器。
// 换机器时设 AVM_CHROMIUM 指向本机 chrome 即可。
const cachedChromium =
  process.env.AVM_CHROMIUM ??
  path.join(homedir(), '.cache/ms-playwright/chromium-1243/chrome-linux64/chrome')
const executablePath = existsSync(cachedChromium) ? cachedChromium : undefined

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    headless: true,
    launchOptions: {
      executablePath,
      args: ['--no-sandbox', '--disable-dev-shm-usage'],
    },
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // 测试侧控制台：真后端 + 脚本化 MockLMU，不改 avm/
      command: `python3 ${path.join(import.meta.dirname, 'e2e/mock_console.py')} --port ${CONSOLE_PORT}`,
      url: `${CONSOLE_URL}/api/state`,
      reuseExistingServer: false,
      timeout: 60_000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // 前端开发服务器：/api 代理到上面的控制台
      command: `npm run dev -- --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}/`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { ...(process.env as Record<string, string>), AVM_CONSOLE_URL: CONSOLE_URL },
    },
  ],
})

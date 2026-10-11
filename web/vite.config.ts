import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发期把 /api 代理到控制台后端（默认 127.0.0.1:8700），于是前端不需要后端改动就能连真 console。
// 生产期页面由控制台自己托管，走同源相对路径，这里不参与。
const consoleUrl = process.env.AVM_CONSOLE_URL ?? 'http://127.0.0.1:8700'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // SSE 也走这条代理：Vite 的 http-proxy 按流转发，不做缓冲。
      '/api': { target: consoleUrl, changeOrigin: false },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})

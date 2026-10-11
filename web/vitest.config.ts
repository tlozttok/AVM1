import { defineConfig } from 'vitest/config'

// 单测只覆盖纯逻辑：SSE 帧 reducer 与 schema→表单映射。DOM 交互交给 e2e。
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})

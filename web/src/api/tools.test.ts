import { afterEach, describe, expect, it, vi } from 'vitest'
import { loadTools } from './tools'

function respond(status: number, body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    })),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('loadTools', () => {
  it('读到就用后端的指令表', async () => {
    const tools = [{ type: 'function', function: { name: 'memory_read', parameters: {} } }]
    respond(200, { tools })
    const table = await loadTools()
    expect(table.error).toBeNull()
    expect(table.tools).toEqual(tools)
  })

  it('端点 404 时报错，不退回任何内置表', async () => {
    respond(404, { error: '没有这个端点：/api/tools' })
    const table = await loadTools()
    expect(table.tools).toEqual([])
    expect(table.error).toContain('读不到 GET /api/tools')
  })

  it('空指令表也当读不到：后端在、表不在时别让表单假装有得选', async () => {
    respond(200, { tools: [] })
    const table = await loadTools()
    expect(table.tools).toEqual([])
    expect(table.error).toContain('空指令表')
  })

  it('五千错也走同一条报错路径', async () => {
    respond(500, { error: 'boom' })
    const table = await loadTools()
    expect(table.tools).toEqual([])
    expect(table.error).not.toBeNull()
  })
})

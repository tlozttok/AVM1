import { describe, expect, it } from 'vitest'
import type { LogEntry } from './frames'
import { toolReturns } from './operatorTrace'

function entry(id: number, kind: LogEntry['kind'], detail: unknown): LogEntry {
  return { id: String(id), kind, seq: id, time: 0, text: '', detail }
}

describe('toolReturns', () => {
  it('按 call_id 配对工具调用与返回值', () => {
    const rows = toolReturns([
      entry(1, 'tool_call', { call_id: 'op-1', tool: 'memory_read', arguments: { ref: '$MEM.a' } }),
      entry(2, 'tool_result', { call_id: 'op-1', content: 'hi' }),
      entry(3, 'tool_call', { call_id: 'op-2', tool: 'send_instruction', arguments: { cid: 5, content: 'help' } }),
    ])
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ callId: 'op-1', tool: 'memory_read', result: 'hi' })
    expect(rows[0].arguments).toContain('$MEM.a')
    expect(rows[1]).toMatchObject({ callId: 'op-2', tool: 'send_instruction', result: null })
  })

  it('只有返回值、调用帧不在缓冲区的行也保留', () => {
    const rows = toolReturns([entry(1, 'tool_result', { call_id: 'op-9', content: 'late' })])
    expect(rows).toEqual([
      { callId: 'op-9', tool: '（调用帧已滚出缓冲）', arguments: '', result: 'late' },
    ])
  })

  it('非工具帧不参与配对', () => {
    expect(toolReturns([entry(1, 'event', { call_id: 'x' }), entry(2, 'note', {})])).toEqual([])
  })
})

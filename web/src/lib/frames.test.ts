import { describe, expect, it } from 'vitest'
import type { SseFrame } from '../api/types'
import { formatEntry, initialFrameState, reduceFrame, type FrameState } from './frames'

function toolCall(seq: number): SseFrame {
  return {
    type: 'tool_call',
    entry: {
      seq,
      kind: 'tool_call',
      time: 1000 + seq,
      data: { call_id: `op-${seq}`, tool: 'memory_write', arguments: { ref: '$MEM.note' } },
    },
  }
}

function toolResult(seq: number): SseFrame {
  return {
    type: 'tool_result',
    entry: {
      seq,
      kind: 'tool_result',
      time: 1000 + seq,
      data: { call_id: `op-${seq}`, content: 'Success set: $MEM.note' },
    },
  }
}

describe('reduceFrame', () => {
  it('按 seq 去重：重连后服务端从头重放，老帧不再入日志', () => {
    let state: FrameState = initialFrameState
    state = reduceFrame(state, toolCall(1))
    state = reduceFrame(state, toolResult(2))
    const before = state.entries.length
    state = reduceFrame(state, toolCall(1))
    state = reduceFrame(state, toolResult(2))
    expect(state.entries).toHaveLength(before)
    expect(state.lastSeq).toBe(2)
  })

  it('跳号时插一条本地提示并累计 gaps', () => {
    let state: FrameState = initialFrameState
    state = reduceFrame(state, toolCall(1))
    state = reduceFrame(state, toolCall(4))
    expect(state.gaps).toBe(1)
    expect(state.entries[1].kind).toBe('note')
    expect(state.entries[1].text).toContain('seq 1 → 4')
    expect(state.lastSeq).toBe(4)
  })

  it('schedule 帧只更新 schedule，不进日志', () => {
    const schedule = {
      active_cid: 1,
      ready: [],
      dormant: [0],
      finished: [],
      instruction_budget: 50,
      activation_instructions: 0,
      panel_cid: 1,
    }
    const state = reduceFrame(initialFrameState, { type: 'schedule', data: schedule })
    expect(state.entries).toHaveLength(0)
    expect(state.schedule).toEqual(schedule)
  })

  it('超过上限时只留最后 limit 条', () => {
    let state: FrameState = { ...initialFrameState, limit: 3 }
    for (let seq = 1; seq <= 6; seq += 1) state = reduceFrame(state, toolCall(seq))
    expect(state.entries.map((e) => e.seq)).toEqual([4, 5, 6])
  })

  it('形状不对的帧被忽略', () => {
    const bad = { type: 'tool_call' } as unknown as SseFrame
    expect(reduceFrame(initialFrameState, bad)).toBe(initialFrameState)
  })
})

describe('formatEntry', () => {
  it('六种帧各给一行可读文本', () => {
    expect(formatEntry('tool_call', { tool: 'memory_read', arguments: { ref: '$MEM.a' } }))
      .toBe('工具调用 memory_read {"ref":"$MEM.a"}')
    expect(formatEntry('tool_result', { call_id: 'op-1', content: 'OK' })).toBe('工具响应 op-1 OK')
    expect(formatEntry('event', { event_type: 'memory_write', phase: 'after', target: '$MEM.note' }))
      .toBe('事件 memory_write after → $MEM.note')
    expect(formatEntry('user_message', { content: '继续' })).toBe('外部消息 继续')
    expect(formatEntry('action_error', { reason: '动作缺少字符串字段 tool' }))
      .toBe('动作错误 动作缺少字符串字段 tool')
    expect(formatEntry('note', null)).toBe('note')
  })

  it('长文本截断，不让一行日志吃掉整屏', () => {
    const text = formatEntry('tool_result', { call_id: 'op-1', content: 'x'.repeat(500) })
    expect(text.length).toBeLessThan(340)
    expect(text.endsWith('…')).toBe(true)
  })
})

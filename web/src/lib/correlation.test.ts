import { describe, expect, it } from 'vitest'
import type { SseFrame } from '../api/types'
import { copyTargets, extractLink, pairAction, startTrace } from './correlation'
import { initialFrameState, reduceFrame, type FrameState, type LogEntry } from './frames'

function frame(type: 'tool_call' | 'tool_result' | 'event', seq: number, data: unknown): SseFrame {
  return { type, entry: { seq, kind: type, time: 1000 + seq, data } }
}

function entries(...frames: SseFrame[]): LogEntry[] {
  let state: FrameState = initialFrameState
  for (const f of frames) state = reduceFrame(state, f)
  return state.entries
}

describe('extractLink', () => {
  it('before 事件：cid 在 source.initiator_cid', () => {
    const link = extractLink('event', {
      event_type: 'memory_write',
      phase: 'before',
      source: { kind: 'conversation', initiator_cid: 3 },
      target: '$MEM.note',
      payload: {},
    })
    expect(link).toEqual({ callId: null, cid: 3, ref: '$MEM.note', tool: 'memory_write' })
  })

  it('after 事件：source 没有 cid 时回退到 payload.actor_cid', () => {
    const link = extractLink('event', {
      event_type: 'memory_write',
      phase: 'after',
      source: { kind: 'memory' },
      target: '$MEM.note',
      payload: { actor_cid: 7 },
    })
    expect(link.cid).toBe(7)
    expect(link.ref).toBe('$MEM.note')
  })

  it('tool_call / tool_result 只以 call_id 为键，event_id 不混进来', () => {
    expect(extractLink('tool_call', { call_id: 'op-1', tool: 'memory_read', event_id: 9 })).toEqual({
      callId: 'op-1',
      cid: null,
      ref: null,
      tool: 'memory_read',
    })
    expect(extractLink('tool_result', { call_id: 'op-1', content: 'OK' }).callId).toBe('op-1')
  })

  it('target 只有以 $ 开头才认作地址；别名与设备标识符不当 ref', () => {
    expect(extractLink('event', { target: '$MEM.a.b' }).ref).toBe('$MEM.a.b')
    expect(extractLink('event', { target: 'note' }).ref).toBeNull()
    expect(extractLink('event', { target: 'dev-1' }).ref).toBeNull()
  })

  it('字段缺失、形状不对、其它帧类型都回落成空键，不抛错', () => {
    const empty = { callId: null, cid: null, ref: null, tool: null }
    expect(extractLink('event', null)).toEqual(empty)
    expect(extractLink('event', { source: null, payload: [], target: 12 })).toEqual(empty)
    expect(extractLink('tool_call', {})).toEqual(empty)
    expect(extractLink('user_message', { content: 'x' })).toEqual(empty)
    expect(extractLink('note', null)).toEqual(empty)
  })
})

describe('reduceFrame 带出 link', () => {
  it('建条目时附上关联键，detail 仍是原始 data', () => {
    const data = { call_id: 'op-2', tool: 'memory_write', arguments: { ref: '$MEM.note' } }
    const [entry] = entries(frame('tool_call', 1, data))
    expect(entry.link?.callId).toBe('op-2')
    expect(entry.detail).toBe(data)
  })
})

describe('copyTargets', () => {
  it('tool_call 同时给参数 JSON 与地址出口，event 给地址与 cid', () => {
    const [call] = entries(
      frame('tool_call', 1, { call_id: 'op-1', tool: 'memory_write', arguments: { ref: '$MEM.note' } }),
    )
    expect(copyTargets(call).map((t) => t.testId)).toEqual(['copy-arguments'])
    expect(copyTargets(call)[0].value).toBe('{\n  "ref": "$MEM.note"\n}')

    const [event] = entries(
      frame('event', 2, {
        event_type: 'memory_write',
        source: { initiator_cid: 4 },
        target: '$MEM.note',
      }),
    )
    expect(copyTargets(event).map((t) => t.testId)).toEqual(['copy-ref', 'copy-cid'])
    expect(copyTargets(event)[1].value).toBe('4')
  })
})

describe('pairAction', () => {
  const args = { ref: '$MEM.note', content: 'x' }

  it('按序找到同名 tool_call，再用 call_id 配到 tool_result', () => {
    const list = entries(
      frame('tool_call', 1, { call_id: 'op-1', tool: 'memory_write', arguments: args }),
      frame('event', 2, { event_type: 'memory_write', target: '$MEM.note' }),
      frame('tool_result', 3, { call_id: 'op-1', content: 'Success set: $MEM.note' }),
    )
    const trace = startTrace('memory_write', args, true, 0)
    const running = pairAction(trace, list.slice(0, 1))
    expect(running.phase).toBe('running')
    expect(running.callId).toBe('op-1')
    const done = pairAction(running, list)
    expect(done.phase).toBe('responded')
    expect(done.content).toBe('Success set: $MEM.note')
    expect(pairAction(done, list)).toBe(done)
  })

  it('只认提交时刻之后的调用；同名但更早的帧不算', () => {
    const list = entries(
      frame('tool_call', 1, { call_id: 'old', tool: 'memory_write', arguments: args }),
    )
    expect(pairAction(startTrace('memory_write', args, true, 1), list).phase).toBe('submitted')
  })

  it('参数相等者优先于「仅同名」', () => {
    const list = entries(
      frame('tool_call', 1, { call_id: 'other', tool: 'memory_write', arguments: { ref: '$MEM.a' } }),
      frame('tool_call', 2, { call_id: 'mine', tool: 'memory_write', arguments: args }),
    )
    expect(pairAction(startTrace('memory_write', args, true, 0), list).callId).toBe('mine')
  })

  it('未知工具名直接落「无响应」，不再参与配对', () => {
    const trace = startTrace('no_such_tool', {}, false, 0)
    expect(trace.phase).toBe('no_response')
    const list = entries(frame('tool_call', 1, { call_id: 'op-1', tool: 'no_such_tool', arguments: {} }))
    expect(pairAction(trace, list)).toBe(trace)
  })
})

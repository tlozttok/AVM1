import { describe, expect, it } from 'vitest'
import type { ConversationBrief, DispatchRecord, State } from '../api/types'
import { deriveSystemStatus } from './systemStatus'

function state(overrides: Partial<State> = {}): State {
  const base: State = {
    running: true,
    schedule: {
      active_cid: null,
      ready: [],
      dormant: [],
      finished: [],
      instruction_budget: 100,
      activation_instructions: 0,
      panel_cid: -1,
    },
    conversations: [],
    panel: { cid: -1, active: false, messages: 0, channel: { next_seq: 1, buffered: 0, dropped: 0, pending_actions: 0 } },
    interrupts: { vectors: {}, pending: [], last_dispatch: [] },
    mask: { memory: [], devices: [] },
    snapshots: 0,
    lifecycle: 'running',
    paused: false,
  }
  return { ...base, ...overrides }
}

function conv(cid: number, name: string): ConversationBrief {
  return {
    cid,
    identity: name,
    name,
    status: 'dormant',
    is_python: true,
    is_sub: false,
    is_root: true,
    parent: null,
    messages: 1,
  }
}

function dispatch(type: string, target: number): DispatchRecord {
  return { type, target, action: 'switch' }
}

describe('deriveSystemStatus', () => {
  it('连接中与已停止、关机中', () => {
    expect(deriveSystemStatus(null, []).phase).toBe('connecting')
    expect(deriveSystemStatus(state({ running: false, lifecycle: 'stopped' }), []).phase).toBe('stopped')
    expect(deriveSystemStatus(state({ lifecycle: 'stopping' }), []).phase).toBe('stopping')
  })

  it('API 暂停优先于调度状态', () => {
    const status = deriveSystemStatus(
      state({ paused: true, pause_category: '余额不足', schedule: { ...state().schedule, active_cid: 3 } }),
      [conv(3, 'clock_a')],
    )
    expect(status.phase).toBe('api-paused')
    expect(status.label).toContain('余额不足')
    expect(status.canOperate).toBe(false)
  })

  it('面板占执行权：operator 派发 = 操作员接管，idle 派发 = 空闲', () => {
    const paused = deriveSystemStatus(
      state({ schedule: { ...state().schedule, active_cid: -1 }, interrupts: { vectors: {}, pending: [], last_dispatch: [dispatch('operator', -1)] } }),
      [],
    )
    expect(paused.phase).toBe('operator')
    expect(paused.canOperate).toBe(true)

    const idle = deriveSystemStatus(
      state({ schedule: { ...state().schedule, active_cid: -1 }, interrupts: { vectors: {}, pending: [], last_dispatch: [dispatch('idle', -1)] } }),
      [],
    )
    expect(idle.phase).toBe('idle')
  })

  it('运行中只报 guest 活跃对话的身份', () => {
    const status = deriveSystemStatus(
      state({ schedule: { ...state().schedule, active_cid: 3 }, interrupts: { vectors: {}, pending: [], last_dispatch: [dispatch('idle', -1)] } }),
      [conv(3, 'clock_a')],
    )
    expect(status.phase).toBe('running')
    expect(status.label).toBe('运行中')
    expect(status.activeLabel).toBe('clock_a（cid 3）')
  })
})

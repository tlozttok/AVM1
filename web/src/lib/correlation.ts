import type { LogEntry, LogKind } from './frames'

/**
 * SSE 帧 → 关联键，以及「一次动作提交 → 工具响应」的按序配对。
 *
 * 两类帧的边界（不要混键）：
 * - `event` 是 Core 发布的 before / after 事实，编号是 `event_id`；
 * - `tool_call` / `tool_result` 是面板对话 `plm.operator` 自身的调用与响应，编号是 `call_id`。
 * 本模块只用 `call_id` 串调用与响应，不把 `event_id` 当成同一个键，也不跨类型推断因果。
 *
 * 后端 `POST /api/action` 的响应不带请求标识（见 avm/console.py），所以动作与调用之间
 * 只能按「提交时的 seq 之后、紧随的第一条同名 tool_call」做启发式关联。已知边界：
 * 连续快速提交、SSE 重连重放、seq 跳号时可能配错；界面标「按序关联」，不假装精确。
 */

export interface LogLink {
  callId: string | null
  cid: number | null
  ref: string | null
  tool: string | null
}

const EMPTY_LINK: LogLink = { callId: null, cid: null, ref: null, tool: null }

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return {}
  return value as Record<string, unknown>
}

function text(value: unknown): string | null {
  if (typeof value === 'string') return value === '' ? null : value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  return null
}

function number(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value)
    if (Number.isFinite(n)) return n
  }
  return null
}

/** 从一条日志条目的原始 data 里提取四个关联键；字段缺失一律回落成 null，不抛错。 */
export function extractLink(kind: LogKind, data: unknown): LogLink {
  const d = record(data)
  switch (kind) {
    case 'tool_call':
      return { ...EMPTY_LINK, callId: text(d.call_id), tool: text(d.tool) }
    case 'tool_result':
      return { ...EMPTY_LINK, callId: text(d.call_id) }
    case 'event': {
      // before 的 cid 在 source.initiator_cid；after 的 source 是 {"kind": "memory"} 等，
      // cid 落到 payload.actor_cid（avm/events.py:build_event、_actor_cid）。
      const source = record(d.source)
      const payload = record(d.payload)
      const target = text(d.target)
      return {
        ...EMPTY_LINK,
        cid: number(source.initiator_cid) ?? number(payload.actor_cid),
        ref: target !== null && target.startsWith('$') ? target : null,
        // 事件类型就是指令名：给「按工具名过滤」用，不参与 call_id 配对。
        tool: text(d.event_type),
      }
    }
    default:
      return EMPTY_LINK
  }
}

/** 一条事件行能带走哪些值：地址、参数 JSON、cid，按界面上的重要性排序。 */
export interface CopyTarget {
  label: string
  testId: string
  value: string
}

export function copyTargets(entry: LogEntry): CopyTarget[] {
  const d = record(entry.detail)
  const out: CopyTarget[] = []
  if (entry.link?.ref) {
    out.push({ label: '复制地址', testId: 'copy-ref', value: entry.link.ref })
  }
  if (entry.kind === 'tool_call' && d.arguments !== undefined) {
    try {
      out.push({
        label: '复制参数',
        testId: 'copy-arguments',
        value: JSON.stringify(d.arguments, null, 2),
      })
    } catch {
      // 循环引用等不可序列化的载荷：这条不给复制出口，其余出口照常可用
    }
  }
  if (entry.link?.cid !== null && entry.link?.cid !== undefined) {
    out.push({ label: '复制 cid', testId: 'copy-cid', value: String(entry.link.cid) })
  }
  return out
}

/** 动作表单提交成功后交给 ActionPanel 的原始数据，用来起一条 ActionTrace。 */
export interface SubmittedAction {
  tool: string
  args: Record<string, unknown>
  knownTool: boolean
  seqAfter: number
  /** 既有的 action-result 文案，原样保留。 */
  note: string
}

export type ActionPhase = 'submitted' | 'running' | 'responded' | 'no_response'

/** 一次动作提交的可追状态。knownTool=false 时 Core 不回工具响应，直接落「无响应」。 */
export interface ActionTrace {
  tool: string
  arguments: Record<string, unknown>
  knownTool: boolean
  /** 提交时刻的 frames.lastSeq；只在这个 seq 之后找 tool_call。 */
  seqAfter: number
  callId: string | null
  content: string | null
  phase: ActionPhase
}

export function startTrace(
  tool: string,
  args: Record<string, unknown>,
  knownTool: boolean,
  seqAfter: number,
): ActionTrace {
  return {
    tool,
    arguments: args,
    knownTool,
    seqAfter,
    callId: null,
    content: null,
    phase: knownTool ? 'submitted' : 'no_response',
  }
}

function sameArgs(expected: Record<string, unknown>, actual: unknown): boolean {
  try {
    return JSON.stringify(expected) === JSON.stringify(actual ?? {})
  } catch {
    return false
  }
}

/** 提交之后紧随的第一条同名 tool_call；参数完全相等者优先，否则退化为第一条。 */
function pickToolCall(trace: ActionTrace, entries: LogEntry[]): LogEntry | undefined {
  const candidates = entries.filter(
    (entry) =>
      entry.kind === 'tool_call' &&
      (entry.seq === null || entry.seq > trace.seqAfter) &&
      entry.link?.tool === trace.tool &&
      entry.link.callId !== null,
  )
  if (candidates.length === 0) return undefined
  return candidates.find((entry) => sameArgs(trace.arguments, record(entry.detail).arguments)) ?? candidates[0]
}

/**
 * 纯配对：已到终态就原样返回（引用不变，React 可据此跳过 setState）。
 * 只有出现新的 call_id 或新的 tool_result 时才产出新对象。
 */
export function pairAction(trace: ActionTrace, entries: LogEntry[]): ActionTrace {
  if (trace.phase === 'responded' || trace.phase === 'no_response') return trace
  const callId = trace.callId ?? pickToolCall(trace, entries)?.link?.callId ?? null
  if (callId === null) return trace
  const result = entries.find(
    (entry) => entry.kind === 'tool_result' && entry.link?.callId === callId,
  )
  if (result) {
    return {
      ...trace,
      callId,
      content: text(record(result.detail).content) ?? '',
      phase: 'responded',
    }
  }
  if (trace.callId === callId) return trace
  return { ...trace, callId, phase: 'running' }
}

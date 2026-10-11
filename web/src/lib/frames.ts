import type { Schedule, SseEntry, SseFrame } from '../api/types'

/**
 * 事件流的纯 reducer。
 *
 * 三条规则：
 * - **按 seq 去重**：`EventSource` 重连后服务端从缓冲区最早一条重推，seq 不大于已见最大值的帧丢掉；
 * - **跳号记账**：seq 出现缺口时插一条本地提示，并累计 `gaps`（环形缓冲丢帧时看得到）；
 * - **`schedule` 帧不入日志**：它只更新横幅与左栏，避免 0.2s 一次的调度变化把日志刷掉。
 */

export type LogKind =
  | 'tool_call'
  | 'tool_result'
  | 'event'
  | 'user_message'
  | 'action_error'
  | 'note'

export const LOG_KINDS: LogKind[] = [
  'tool_call',
  'tool_result',
  'event',
  'user_message',
  'action_error',
  'note',
]

export const LOG_KIND_LABEL: Record<LogKind, string> = {
  tool_call: '工具调用',
  tool_result: '工具响应',
  event: '事件',
  user_message: '外部消息',
  action_error: '动作错误',
  note: '本地提示',
}

export interface LogEntry {
  id: string
  kind: LogKind
  seq: number | null
  time: number
  text: string
  detail: unknown
}

export interface FrameState {
  entries: LogEntry[]
  /** 已见的最大 entry.seq；schedule 帧不影响它。 */
  lastSeq: number
  /** 跳号次数。 */
  gaps: number
  /** 日志保留条数上限。 */
  limit: number
  /** 最近一帧 schedule，供横幅即时反应；轮询仍是对账来源。 */
  schedule: Schedule | null
}

export const LOG_LIMIT = 800

export const initialFrameState: FrameState = {
  entries: [],
  lastSeq: 0,
  gaps: 0,
  limit: LOG_LIMIT,
  schedule: null,
}

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max)}…` : text
}

function str(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

function jsonish(value: unknown): string {
  try {
    return JSON.stringify(value ?? {})
  } catch {
    return String(value)
  }
}

/** 帧 → 一行可读文本；导出是给单测直接断言的。 */
export function formatEntry(kind: LogKind, data: unknown): string {
  const d = (data ?? {}) as Record<string, unknown>
  switch (kind) {
    case 'tool_call':
      return `工具调用 ${str(d.tool)} ${jsonish(d.arguments)}`
    case 'tool_result':
      return `工具响应 ${str(d.call_id)} ${truncate(str(d.content), 300)}`
    case 'event': {
      const bits = [str(d.event_type), str(d.phase), d.target ? `→ ${str(d.target)}` : '']
      return `事件 ${bits.filter(Boolean).join(' ')}`
    }
    case 'user_message':
      return `外部消息 ${truncate(str(d.content), 300)}`
    case 'action_error':
      return `动作错误 ${str(d.reason)}`
    default:
      return kind
  }
}

function trim(entries: LogEntry[], limit: number): LogEntry[] {
  return entries.length > limit ? entries.slice(entries.length - limit) : entries
}

let noteCounter = 0

/** 本地提示（跳号、解析失败、请求失败）：没有 seq，不参与去重。 */
export function noteFrame(state: FrameState, text: string): FrameState {
  noteCounter += 1
  const entry: LogEntry = {
    id: `note-${noteCounter}`,
    kind: 'note',
    seq: null,
    time: Date.now() / 1000,
    text,
    detail: null,
  }
  return { ...state, entries: trim([...state.entries, entry], state.limit) }
}

function entryFromFrame(frame: Exclude<SseFrame, { type: 'schedule' }>): LogEntry {
  const entry: SseEntry = frame.entry
  return {
    id: `${frame.type}-${entry.seq}`,
    kind: frame.type as LogKind,
    seq: entry.seq,
    time: entry.time,
    text: formatEntry(frame.type as LogKind, entry.data),
    detail: entry.data,
  }
}

export function reduceFrame(state: FrameState, frame: SseFrame): FrameState {
  if (frame.type === 'schedule') {
    return { ...state, schedule: frame.data }
  }
  const entry = frame.entry
  if (!entry || typeof entry.seq !== 'number') return state
  if (entry.seq <= state.lastSeq) return state // 重连重放
  const added: LogEntry[] = []
  let gaps = state.gaps
  if (state.lastSeq > 0 && entry.seq > state.lastSeq + 1) {
    gaps += 1
    added.push({
      id: `gap-${state.lastSeq}-${entry.seq}`,
      kind: 'note',
      seq: null,
      time: entry.time,
      text: `缓冲区跳号：seq ${state.lastSeq} → ${entry.seq}（中间 ${entry.seq - state.lastSeq - 1} 条未收到）`,
      detail: null,
    })
  }
  added.push(entryFromFrame(frame))
  return {
    ...state,
    entries: trim([...state.entries, ...added], state.limit),
    lastSeq: entry.seq,
    gaps,
  }
}

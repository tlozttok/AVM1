import type { LogEntry } from './frames'

/**
 * 面板对话（cid -1）收到的工具调用与返回值。
 *
 * 数据源是 `/api/events` 的操作员通道帧：`tool_call` 是面板对话发出的调用（带工具名与参数），
 * `tool_result` 是它收到的返回值，两者按 `call_id` 配对。通道缓冲区会随 SSE 连接重放，
 * 所以页面打开时能看到本次控制台启动以来的全部调用；`result === null` 表示还在等响应。
 */
export interface ToolReturnRow {
  callId: string
  tool: string
  arguments: string
  result: string | null
}

function record(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return {}
  return value as Record<string, unknown>
}

function text(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export function toolReturns(entries: LogEntry[]): ToolReturnRow[] {
  const calls = new Map<string, ToolReturnRow>()
  const results = new Map<string, string>()
  for (const entry of entries) {
    if (entry.kind === 'tool_call') {
      const detail = record(entry.detail)
      const callId = text(detail.call_id)
      if (callId === '') continue
      calls.set(callId, {
        callId,
        tool: text(detail.tool) || '（未知工具）',
        arguments: text(detail.arguments),
        result: null,
      })
    } else if (entry.kind === 'tool_result') {
      const detail = record(entry.detail)
      const callId = text(detail.call_id)
      if (callId === '') continue
      results.set(callId, text(detail.content))
    }
  }
  const rows = [...calls.values()]
  for (const row of rows) {
    if (results.has(row.callId)) row.result = results.get(row.callId) ?? ''
  }
  // 只收到返回值、调用帧已不在缓冲区里的（例如重连后截断）：补一行，避免漏看返回值。
  const seen = new Set(rows.map((row) => row.callId))
  for (const [callId, result] of results) {
    if (seen.has(callId)) continue
    rows.push({ callId, tool: '（调用帧已滚出缓冲）', arguments: '', result })
  }
  return rows
}

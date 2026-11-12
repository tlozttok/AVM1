import type { ConversationDetail, Message } from '../api/types'

/**
 * 命令台历史行：把对话历史压成「终端里看得到的东西」。
 *
 * - 用户消息 = 投递给这个对话的指令（`❯` 提示行）；
 * - 助手正文 = 命令输出（`Error:` / `Illegal:` / `错误:` 开头标成错误）；
 * - 工具调用与工具响应折叠成一行，原始详情在调试抽屉的对话视图里；
 * - system 消息（python 程序正文）不进主视图；
 * - 尚未执行的外部输入（user_batch）单列成「已投递、待执行」。
 */
export type TranscriptLineKind = 'cmd' | 'out' | 'error' | 'tools' | 'pending'

export interface TranscriptLine {
  kind: TranscriptLineKind
  text: string
  /** tools 行的调用名清单。 */
  tools?: string[]
}

export const TRANSCRIPT_LIMIT = 300

export function toolNames(toolCalls: unknown): string[] {
  if (!Array.isArray(toolCalls)) return []
  const names: string[] = []
  for (const call of toolCalls) {
    if (call === null || typeof call !== 'object') continue
    const fn = (call as { function?: unknown }).function
    if (fn === null || typeof fn !== 'object') continue
    const name = (fn as { name?: unknown }).name
    if (typeof name === 'string' && name !== '') names.push(name)
  }
  return names
}

export function isErrorMessage(text: string): boolean {
  return text.startsWith('Error:') || text.startsWith('Illegal:') || text.startsWith('错误:')
}

function linesOfMessage(message: Message): TranscriptLine[] {
  if (message.role === 'system') return []
  const content = message.content ?? ''
  if (message.role === 'user') {
    return content === '' ? [] : [{ kind: 'cmd', text: content }]
  }
  if (message.role === 'assistant') {
    const lines: TranscriptLine[] = []
    if (content !== '') {
      lines.push({ kind: isErrorMessage(content) ? 'error' : 'out', text: content })
    }
    const tools = toolNames(message.tool_calls)
    if (tools.length > 0) {
      lines.push({ kind: 'tools', text: `工具调用 ×${tools.length}（${tools.join('、')}）`, tools })
    }
    return lines
  }
  // tool 消息与未知角色：折叠，不在主视图展开
  return []
}

export function transcriptLines(
  detail: ConversationDetail | null,
  limit = TRANSCRIPT_LIMIT,
): { lines: TranscriptLine[]; hidden: number } {
  if (detail === null) return { lines: [], hidden: 0 }
  const lines: TranscriptLine[] = []
  for (const message of detail.history) {
    lines.push(...linesOfMessage(message))
  }
  for (const content of detail.user_batch.user_contents) {
    lines.push({ kind: 'pending', text: content })
  }
  const hidden = Math.max(0, lines.length - limit)
  return { lines: hidden > 0 ? lines.slice(hidden) : lines, hidden }
}

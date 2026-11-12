import { describe, expect, it } from 'vitest'
import type { ConversationDetail, Message } from '../api/types'
import { isErrorMessage, toolNames, transcriptLines } from './transcript'

function message(role: string, content: string | null, toolCalls: unknown = null): Message {
  return { role, content, tool_calls: toolCalls, tool_call_id: null }
}

function detail(history: Message[], pending: string[] = []): ConversationDetail {
  return {
    cid: 5,
    identity: 'shell',
    name: 'shell',
    status: 'dormant',
    is_python: true,
    is_sub: false,
    is_root: false,
    parent: null,
    messages: history.length,
    history,
    user_batch: { user_contents: pending, tool_responses: [] },
    metadata: {},
    para_ref: '$MEM.para.shell',
  }
}

describe('transcript', () => {
  it('system 不进主视图，用户/助手正文进', () => {
    const { lines } = transcriptLines(
      detail([
        message('system', '程序正文'),
        message('user', 'help'),
        message('assistant', '已注册命令（2）：help、builder'),
      ]),
    )
    expect(lines).toEqual([
      { kind: 'cmd', text: 'help' },
      { kind: 'out', text: '已注册命令（2）：help、builder' },
    ])
  })

  it('工具调用折叠成一行，工具响应不进主视图', () => {
    const { lines } = transcriptLines(
      detail([
        message('user', 'builder {}'),
        message('assistant', '', [
          { id: 'c1', type: 'function', function: { name: 'create_sub', arguments: '{}' } },
        ]),
        message('tool', '工具响应'),
        message('assistant', '{"ok": true}'),
      ]),
    )
    expect(lines.map((line) => line.kind)).toEqual(['cmd', 'tools', 'out'])
    expect(lines[1].tools).toEqual(['create_sub'])
  })

  it('错误正文标红，未执行输入单列待执行', () => {
    const { lines } = transcriptLines(
      detail([message('assistant', "错误: shell: 未知命令 'x'（用 help 查看）")], ['help']),
    )
    expect(lines[0]).toEqual({ kind: 'error', text: "错误: shell: 未知命令 'x'（用 help 查看）" })
    expect(lines[1]).toEqual({ kind: 'pending', text: 'help' })
  })

  it('超长历史只保留最近 N 行并报告隐藏数', () => {
    const history = Array.from({ length: 10 }, (_, index) => message('user', `cmd-${index}`))
    const { lines, hidden } = transcriptLines(detail(history), 4)
    expect(hidden).toBe(6)
    expect(lines.map((line) => line.text)).toEqual(['cmd-6', 'cmd-7', 'cmd-8', 'cmd-9'])
  })

  it('工具名与错误前缀', () => {
    expect(toolNames([{ function: { name: 'a' } }, { function: {} }, null])).toEqual(['a'])
    expect(toolNames('x')).toEqual([])
    expect(isErrorMessage('Error: x')).toBe(true)
    expect(isErrorMessage('Illegal: x')).toBe(true)
    expect(isErrorMessage('错误: x')).toBe(true)
    expect(isErrorMessage('Success')).toBe(false)
  })
})

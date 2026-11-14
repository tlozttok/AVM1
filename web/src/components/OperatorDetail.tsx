import { useMemo } from 'react'
import type { FrameState } from '../lib/frames'
import { toolReturns } from '../lib/operatorTrace'

interface Props {
  /** 面板对话 cid（-1）：说明这些调用的来源。 */
  cid: number
  frames: FrameState
}

const ARGS_PREVIEW = 160

function shorten(text: string): string {
  return text.length > ARGS_PREVIEW ? `${text.slice(0, ARGS_PREVIEW)}…` : text
}

/**
 * 面板对话（cid -1）收到的工具返回值。
 *
 * 操作员的每次动作在操作员通道上是一条 `tool_call`，随后的 `tool_result` 就是面板对话
 * 收到的返回值。这个页签把两者按 `call_id` 配起来按序显示——原始调用的回执都在这里。
 */
export default function OperatorDetail({ cid, frames }: Props) {
  const rows = useMemo(() => toolReturns(frames.entries), [frames.entries])
  const waiting = rows.filter((row) => row.result === null).length

  return (
    <div className="drawer-pane operator-detail" data-testid="operator-detail">
      <div className="drawer-note">
        面板对话 <code>cid {cid}</code>（操作员）：每次动作是一条工具调用（<code>tool_call</code>），
        对应的工具返回值（<code>tool_result</code>）按 <code>call_id</code> 列在下面。
        {` 共 ${rows.length} 条`}
        {waiting > 0 ? `，${waiting} 条等待响应` : ''}
      </div>
      {rows.length === 0 ? (
        <div className="drawer-note" data-testid="operator-empty">还没有收到过工具返回值。</div>
      ) : null}
      <div className="operator-returns">
        {rows.map((row) => (
          <div className="operator-return" key={row.callId} data-testid="operator-return">
            <div className="operator-return-head">
              <span className="mono">{row.callId}</span>
              <b>{row.tool}</b>
              {row.arguments !== '' ? <span className="muted small">{shorten(row.arguments)}</span> : null}
            </div>
            <pre className={`content${row.result === null ? ' operator-return-waiting' : ''}`} data-testid="operator-return-result">
              {row.result === null ? '等待工具响应…' : row.result}
            </pre>
          </div>
        ))}
      </div>
    </div>
  )
}

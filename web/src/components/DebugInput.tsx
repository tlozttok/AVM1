import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ConversationBrief } from '../api/types'

interface Props {
  conversations: ConversationBrief[]
  activeCid: number | null
}

/**
 * 调试抽屉里的宿主总线输入：只走 `POST /api/io/input`。
 * 不进日志、不产生事件、不重放；正式输入路径（UI 是设备）仍冻结，等作者定义。
 */
export default function DebugInput({ conversations, activeCid }: Props) {
  const [target, setTarget] = useState('')
  const [text, setText] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const send = async () => {
    if (text.trim() === '' || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const record = await apiSend<{ seq: number; target: string | null }>('/api/io/input', 'POST', {
        target: target === '' ? null : target,
        text,
      })
      setResult(`已发出 #${record.seq}${record.target ? ` → ${record.target}` : ' → 入口'}`)
      setText('')
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="drawer-block" data-testid="debug-input">
      <div className="drawer-block-title">宿主总线调试输入（/api/io/input）</div>
      <div className="drawer-note">发布到宿主侧 IO 总线：不进日志、不产生事件、不重放。正式输入路径（UI 是设备）仍冻结。</div>
      <div className="drawer-row">
        <label>
          目标
          <select data-testid="debug-input-target" value={target} onChange={(event) => setTarget(event.target.value)}>
            <option value="">入口（无目标）</option>
            {activeCid !== null ? <option value={String(activeCid)}>当前活跃 cid {activeCid}</option> : null}
            {conversations.map((conv) => (
              <option key={conv.cid} value={String(conv.cid)}>
                cid {conv.cid} · {conv.identity || conv.name || '对话'}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="drawer-row">
        <textarea
          data-testid="debug-input-text"
          rows={3}
          value={text}
          placeholder="要说的话…（Ctrl+Enter 或按钮发送）"
          onChange={(event) => setText(event.target.value)}
        />
        <button type="button" data-testid="debug-input-send" disabled={busy || text.trim() === ''} onClick={() => void send()}>
          {busy ? '发送中…' : '发送'}
        </button>
      </div>
      {result ? <div className="drawer-ok" data-testid="debug-input-result">{result}</div> : null}
      {error ? <div className="drawer-error" data-testid="debug-input-error">{error}</div> : null}
    </div>
  )
}

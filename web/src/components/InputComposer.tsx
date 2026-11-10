import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ConversationBrief, InputAccepted } from '../api/types'
import { buildInputBody, describeAccepted } from '../lib/inputPath'

interface Props {
  conversations: ConversationBrief[]
  activeCid: number | null
  onSent: () => void
}

/**
 * 操作面板的主区右半：正式输入。
 *
 * 走 `POST /api/input`：宿主把输入作为一条 JSON 字符串条目追加到
 * `$MEM.sys.inbox`（保留对话执行），随日志重放；202 表示已接收，不等待
 * 回合完成。目标为空表示无目标输入（入口对话可取）。
 * 调试通道 `/api/io/input` 在下方调试抽屉，不重放。
 */
export default function InputComposer({ conversations, activeCid, onSent }: Props) {
  const [target, setTarget] = useState('')
  const [text, setText] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const send = async () => {
    const body = buildInputBody(text, target)
    if (body === null || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const accepted = await apiSend<InputAccepted>('/api/input', 'POST', body)
      setResult(describeAccepted(accepted))
      setText('')
      onSent()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="pane input-composer" data-testid="input-composer">
      <div className="pane-title">
        输入
        <span className="muted small">正式输入：写 $MEM.sys.inbox，随日志重放</span>
      </div>
      <div className="pane-body composer-body">
        <label className="field">
          <span>目标</span>
          <select
            aria-label="输入目标"
            data-testid="input-target"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">入口（无目标）</option>
            {activeCid !== null ? (
              <option value={String(activeCid)}>当前活跃 cid {activeCid}</option>
            ) : null}
            {conversations
              .filter((conv) => conv.cid !== activeCid)
              .map((conv) => (
                <option key={conv.cid} value={String(conv.cid)}>
                  cid {conv.cid} · {conv.identity || conv.name || '对话'}
                </option>
              ))}
          </select>
        </label>
        <label className="field composer-field">
          <span>内容（Ctrl+Enter 发送，回车换行）</span>
          <textarea
            data-testid="input-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                void send()
              }
            }}
            placeholder="要说的话…"
          />
        </label>
        <div className="composer-actions">
          <button
            type="button"
            data-testid="input-send"
            disabled={busy || !text.trim()}
            onClick={() => void send()}
          >
            {busy ? '发送中…' : '发送'}
          </button>
          <button type="button" className="link" onClick={() => setText('')} disabled={!text}>
            清空
          </button>
        </div>
        {result ? (
          <div className="hint ok" data-testid="input-result">
            {result}
          </div>
        ) : null}
        {error ? (
          <div className="hint error" data-testid="input-error">
            {error}
          </div>
        ) : null}
      </div>
    </section>
  )
}

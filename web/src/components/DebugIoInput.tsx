import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { IoInputResponse } from '../api/types'

/**
 * 调试输入（`POST /api/io/input`）：直接发布到宿主侧 IO 总线。
 *
 * 不进日志、不产生事件、不重放——正式输入请用主区的「输入」框（写
 * `$MEM.sys.inbox`）。这里只为调试等待 `$MEM.inputs.next` 的读或验证总线本身。
 * 默认收起：面板右栏空间有限，调试通道不该挤掉事件流。
 */
export default function DebugIoInput() {
  const [target, setTarget] = useState('')
  const [text, setText] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const send = async () => {
    if (!text.trim() || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const record = await apiSend<IoInputResponse>('/api/io/input', 'POST', {
        target: target.trim() || null,
        text,
      })
      setResult(
        `已发出 #${record.seq}${record.target ? ` → ${record.target}` : ' → 入口'}` +
          '（调试通道，不重放）',
      )
      setText('')
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <details className="input-feed" data-testid="debug-io-input">
      <summary className="section-title" data-testid="debug-io-toggle">
        调试输入（/api/io/input，不重放）
      </summary>
      <div className="row">
        <input
          aria-label="调试输入目标"
          data-testid="debug-io-target"
          style={{ width: 120 }}
          value={target}
          placeholder="目标（留空=入口）"
          onChange={(e) => setTarget(e.target.value)}
        />
        <textarea
          data-testid="debug-io-text"
          rows={2}
          value={text}
          placeholder="只进宿主 IO 总线，不进日志"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault()
              void send()
            }
          }}
        />
        <button
          type="button"
          data-testid="debug-io-send"
          onClick={() => void send()}
          disabled={busy || !text.trim()}
        >
          发出
        </button>
      </div>
      <div className="hint muted small">
        调试通道：不进日志、不产生事件，面板重连/恢复后不重放；正式输入走上方「输入」。
      </div>
      {error ? (
        <div className="hint error" data-testid="debug-io-error">
          {error}
        </div>
      ) : null}
      {result ? (
        <div className="hint ok" data-testid="debug-io-result">
          {result}
        </div>
      ) : null}
    </details>
  )
}

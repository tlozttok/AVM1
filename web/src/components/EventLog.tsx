import { useEffect, useRef, useState } from 'react'
import {
  LOG_KINDS,
  LOG_KIND_LABEL,
  type FrameState,
  type LogKind,
} from '../lib/frames'

const KIND_CLASS: Record<LogKind, string> = {
  tool_call: 'kind-tool-call',
  tool_result: 'kind-tool-result',
  event: 'kind-event',
  user_message: 'kind-user',
  action_error: 'kind-error',
  note: 'kind-note',
}

function timeText(seconds: number): string {
  return new Date(seconds * 1000).toLocaleTimeString('zh-CN', { hour12: false })
}

interface Props {
  frames: FrameState
  connected: boolean
  onClear: () => void
}

/** 右栏下：事件流日志。schedule 帧不入日志，只更新横幅。 */
export default function EventLog({ frames, connected, onClear }: Props) {
  const [hidden, setHidden] = useState<Set<LogKind>>(new Set())
  const [follow, setFollow] = useState(true)
  const [openId, setOpenId] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (follow) endRef.current?.scrollIntoView({ block: 'end' })
  }, [frames.entries.length, follow])

  const counts = new Map<LogKind, number>()
  for (const entry of frames.entries) counts.set(entry.kind, (counts.get(entry.kind) ?? 0) + 1)
  const visible = frames.entries.filter((entry) => !hidden.has(entry.kind))

  const toggleKind = (kind: LogKind) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }

  return (
    <div className="event-log" data-testid="event-log">
      <div className="pane-title">
        事件流
        <span className={connected ? 'muted small' : 'muted small warn'} data-testid="event-log-connected">
          {connected ? 'SSE 已连接' : 'SSE 断开（浏览器自动重连）'}
        </span>
        <button type="button" className="link" onClick={onClear}>
          清屏
        </button>
      </div>
      <div className="filters">
        {LOG_KINDS.map((kind) => (
          <button
            type="button"
            key={kind}
            className={`chip ${hidden.has(kind) ? 'chip-off' : ''} ${KIND_CLASS[kind]}`}
            onClick={() => toggleKind(kind)}
            title="点一下隐藏/显示这一类"
          >
            {LOG_KIND_LABEL[kind]} {counts.get(kind) ?? 0}
          </button>
        ))}
        <label className="muted small">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> 跟随
        </label>
        {frames.gaps > 0 ? <span className="warn small">跳号 {frames.gaps} 次</span> : null}
      </div>
      <div className="scroll log-scroll">
        {visible.map((entry) => (
          <div
            key={entry.id}
            className={`log-entry ${KIND_CLASS[entry.kind]}`}
            onClick={() => setOpenId(openId === entry.id ? null : entry.id)}
            data-testid={`log-entry-${entry.kind}`}
          >
            <span className="muted small">{timeText(entry.time)}</span>
            {entry.seq !== null ? <span className="muted small">#{entry.seq}</span> : null}
            <span className="log-kind">{LOG_KIND_LABEL[entry.kind]}</span>
            <span className="log-text">{entry.text}</span>
            {openId === entry.id && entry.detail ? (
              <pre className="content muted">{JSON.stringify(entry.detail, null, 2)}</pre>
            ) : null}
          </div>
        ))}
        {visible.length === 0 ? <div className="muted">还没有事件</div> : null}
        <div ref={endRef} />
      </div>
    </div>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import type { IOEntry } from '../api/types'

interface Props {
  entries: IOEntry[]
  connected: boolean
  onClear: () => void
}

/** 通知到达时响一声：无音频设备就只靠高亮，不让播放失败影响渲染。 */
function beep(): void {
  try {
    const Ctor = (window as unknown as { AudioContext?: typeof AudioContext }).AudioContext
    if (!Ctor) return
    const ctx = new Ctor()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.frequency.value = 880
    gain.gain.value = 0.04
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start()
    osc.stop(ctx.currentTime + 0.12)
    osc.onended = () => void ctx.close()
  } catch {
    // 浏览器不给声音就只标红
  }
}

function sourceLabel(source: IOEntry['source']): string {
  return source === null || source === undefined ? '系统' : `cid ${String(source)}`
}

/**
 * 操作面板的主区左半：宿主侧输出流。
 * 与调试抽屉里的 EventLog 是两条独立的流——那条是面板对话与 Core 事件，
 * 这条是 `$MEM.outputs` 的输出，所以这里不复用 EventLog 的过滤与配对逻辑。
 */
export default function OutputConsole({ entries, connected, onClear }: Props) {
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const [bell, setBell] = useState(false)
  const [notifyOnly, setNotifyOnly] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const rungSeq = useRef(0)

  const sources = useMemo(() => {
    const set = new Set<string>()
    for (const entry of entries) set.add(String(entry.source ?? ''))
    return [...set].sort()
  }, [entries])

  const shown = useMemo(
    () =>
      entries.filter((entry) => {
        if (filter && String(entry.source ?? '') !== filter) return false
        if (notifyOnly && entry.kind !== 'notify') return false
        return true
      }),
    [entries, filter, notifyOnly],
  )

  useEffect(() => {
    if (follow && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [shown.length, follow])

  useEffect(() => {
    if (!bell) return
    const latest = entries.reduce(
      (max, entry) => (entry.kind === 'notify' ? Math.max(max, entry.seq) : max),
      0,
    )
    if (latest > rungSeq.current) {
      rungSeq.current = latest
      beep()
    }
  }, [entries, bell])

  return (
    <section className="pane output-console" data-testid="output-console">
      <div className="pane-title">
        输出流
        <span className="muted small" data-testid="output-connected">
          {connected ? 'SSE 已连接' : 'SSE 未连接'}
        </span>
        <span className="muted small">{entries.length} 条</span>
        <button type="button" className="link" data-testid="output-clear" onClick={onClear}>
          清屏
        </button>
      </div>
      {/* 不用 .filters：既有 e2e 用 `.filters input[type=checkbox]` 定位事件流的跟随开关 */}
      <div className="output-controls">
        <select
          aria-label="按来源过滤"
          data-testid="output-filter-source"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="">全部来源</option>
          {sources.map((src) => (
            <option key={src} value={src}>
              {src === '' ? '系统' : `cid ${src}`}
            </option>
          ))}
        </select>
        <label>
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
          />
          跟随
        </label>
        <label>
          <input
            type="checkbox"
            data-testid="output-bell"
            checked={bell}
            onChange={(e) => setBell(e.target.checked)}
          />
          通知响铃
        </label>
        <label>
          <input
            type="checkbox"
            checked={notifyOnly}
            onChange={(e) => setNotifyOnly(e.target.checked)}
          />
          只看通知
        </label>
      </div>
      <div className="scroll output-scroll" ref={scrollRef}>
        {shown.length === 0 ? (
          <div className="pane-body muted" data-testid="output-empty">
            还没有输出。对话写 `$MEM.outputs.out` 后会出现在这里。
          </div>
        ) : (
          shown.map((entry) => (
            <div
              key={entry.seq}
              className={`output-row${entry.kind === 'notify' ? ' output-row-notify' : ''}`}
              data-testid={entry.kind === 'notify' ? 'output-notify' : 'output-entry'}
            >
              <span className="output-seq mono muted small">#{entry.seq}</span>
              <span className="output-source mono muted small">{sourceLabel(entry.source)}</span>
              <span className="output-text">{entry.text}</span>
              {entry.kind === 'notify' ? <span className="tag tag-warn">通知</span> : null}
            </div>
          ))
        )}
      </div>
    </section>
  )
}

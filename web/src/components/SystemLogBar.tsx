import { useEffect, useMemo, useRef, useState } from 'react'
import type { IOEntry } from '../api/types'

interface Props {
  entries: IOEntry[]
  connected: boolean
  onClear: () => void
}

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
    // 浏览器不给声音就只靠高亮
  }
}

function sourceLabel(source: IOEntry['source']): string {
  return source === null || source === undefined ? '系统' : `cid ${String(source)}`
}

/**
 * 底部系统日志条：`$MEM.outputs` 的宿主侧输出流。
 * 默认只占一行（最新一条 + 未读/通知计数），点开才是列表；与命令台的历史是两条流。
 */
export default function SystemLogBar({ entries, connected, onClear }: Props) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const [bell, setBell] = useState(false)
  const [notifyOnly, setNotifyOnly] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const rungSeq = useRef(0)

  const sources = useMemo(() => {
    const set = new Set<string>()
    for (const entry of entries) set.add(String(entry.source ?? ''))
    return [...set].sort()
  }, [entries])

  const shown = useMemo(
    () =>
      entries.filter((entry) => {
        if (filter !== '' && String(entry.source ?? '') !== filter) return false
        if (notifyOnly && entry.kind !== 'notify') return false
        return true
      }),
    [entries, filter, notifyOnly],
  )

  const latest = entries.length > 0 ? entries[entries.length - 1] : null
  const notifyCount = useMemo(() => entries.filter((entry) => entry.kind === 'notify').length, [entries])

  useEffect(() => {
    if (open && follow && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [open, follow, shown.length])

  useEffect(() => {
    if (!bell) return
    const latestNotify = entries.reduce(
      (max, entry) => (entry.kind === 'notify' ? Math.max(max, entry.seq) : max),
      0,
    )
    if (latestNotify > rungSeq.current) {
      rungSeq.current = latestNotify
      beep()
    }
  }, [entries, bell])

  return (
    <section className="syslog" data-testid="system-log">
      <div className="syslog-bar" onClick={() => setOpen((value) => !value)}>
        <span className="syslog-label">系统日志</span>
        <span className="syslog-latest" data-testid="syslog-latest">
          {latest === null ? (
            '还没有输出'
          ) : (
            <>
              <span className="mono">#{latest.seq}</span> {sourceLabel(latest.source)} · {latest.text}
            </>
          )}
        </span>
        <span className="syslog-count" data-testid="syslog-count">{entries.length} 条</span>
        {notifyCount > 0 ? <span className="syslog-notify">● 通知 {notifyCount}</span> : null}
        <label className="syslog-bell" onClick={(event) => event.stopPropagation()}>
          <input type="checkbox" checked={bell} onChange={(event) => setBell(event.target.checked)} />
          响铃
        </label>
        <span className="syslog-toggle">{open ? '收起' : '展开'}</span>
      </div>
      {open ? (
        <div className="syslog-body" data-testid="syslog-body">
          <div className="syslog-controls">
            <span className="syslog-connected">{connected ? 'SSE 已连接' : 'SSE 未连接'}</span>
            <select
              aria-label="按来源过滤"
              data-testid="syslog-filter"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="">全部来源</option>
              {sources.map((source) => (
                <option key={source} value={source}>
                  {source === '' ? '系统' : `cid ${source}`}
                </option>
              ))}
            </select>
            <label className="syslog-check">
              <input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />
              跟随
            </label>
            <label className="syslog-check">
              <input
                type="checkbox"
                data-testid="syslog-notify-only"
                checked={notifyOnly}
                onChange={(event) => setNotifyOnly(event.target.checked)}
              />
              只看通知
            </label>
            <button type="button" className="link" data-testid="syslog-clear" onClick={onClear}>
              清屏
            </button>
          </div>
          <div className="syslog-list" ref={scrollRef}>
            {shown.length === 0 ? (
              <div className="syslog-empty">还没有输出。对话写 `$MEM.outputs.out` 后会出现在这里。</div>
            ) : (
              shown.map((entry) => (
                <div
                  key={entry.seq}
                  className={`syslog-row${entry.kind === 'notify' ? ' is-notify' : ''}`}
                  data-testid={entry.kind === 'notify' ? 'syslog-notify-row' : 'syslog-row'}
                >
                  <span className="syslog-seq mono">#{entry.seq}</span>
                  <span className="syslog-source mono">{sourceLabel(entry.source)}</span>
                  <span className="syslog-text">{entry.text}</span>
                  {entry.kind === 'notify' ? <span className="syslog-tag">通知</span> : null}
                </div>
              ))
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

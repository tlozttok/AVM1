import { useEffect, useMemo, useRef, useState, type WheelEvent } from 'react'
import {
  LOG_KINDS,
  LOG_KIND_LABEL,
  type FrameState,
  type LogEntry,
  type LogKind,
} from '../lib/frames'
import { copyTargets } from '../lib/correlation'

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

function jsonish(value: unknown): string {
  try {
    return JSON.stringify(value ?? {})
  } catch {
    return String(value)
  }
}

interface Props {
  frames: FrameState
  connected: boolean
  onClear: () => void
  /** 复制出口：写进面板暂存区，并尽量写进浏览器剪贴板。 */
  onCopy: (value: string) => void
  onSelectCid: (cid: number, source: string) => void
  /** 解析地址后切内存 / 设备视图；解析失败以 Promise 拒绝，行内给提示。 */
  onSelectRef: (ref: string, source: string) => Promise<void>
}

/** 右栏下：事件流日志。schedule 帧不入日志，只更新横幅。 */
export default function EventLog({
  frames,
  connected,
  onClear,
  onCopy,
  onSelectCid,
  onSelectRef,
}: Props) {
  const [hidden, setHidden] = useState<Set<LogKind>>(new Set())
  const [follow, setFollow] = useState(true)
  const [openId, setOpenId] = useState<string | null>(null)
  const [pairOpenId, setPairOpenId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [cidFilter, setCidFilter] = useState('')
  const [toolFilter, setToolFilter] = useState('')
  const [locateHint, setLocateHint] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (follow) endRef.current?.scrollIntoView({ block: 'end' })
  }, [frames.entries.length, follow])

  // 搜索命中 text 与 detail 的关键字段；整体缓存，避免每帧对 800 条重新序列化。
  const searchable = useMemo(() => {
    const map = new Map<string, string>()
    for (const entry of frames.entries) {
      map.set(entry.id, `${entry.text} ${jsonish(entry.detail)}`.toLowerCase())
    }
    return map
  }, [frames.entries])

  const cidOptions = useMemo(() => {
    const set = new Set<number>()
    for (const entry of frames.entries) {
      if (entry.link?.cid !== null && entry.link?.cid !== undefined) set.add(entry.link.cid)
    }
    return [...set].sort((a, b) => a - b)
  }, [frames.entries])

  const toolOptions = useMemo(() => {
    const set = new Set<string>()
    for (const entry of frames.entries) {
      if (entry.link?.tool) set.add(entry.link.tool)
    }
    return [...set].sort()
  }, [frames.entries])

  /** call_id → 调用与响应两行；只有成对出现才高亮、才给折叠摘要。 */
  const paired = useMemo(() => {
    const map = new Map<string, LogEntry[]>()
    for (const entry of frames.entries) {
      if (entry.kind !== 'tool_call' && entry.kind !== 'tool_result') continue
      const id = entry.link?.callId
      if (!id) continue
      map.set(id, [...(map.get(id) ?? []), entry])
    }
    return new Map([...map].filter(([, list]) => list.length > 1))
  }, [frames.entries])

  const counts = new Map<LogKind, number>()
  for (const entry of frames.entries) counts.set(entry.kind, (counts.get(entry.kind) ?? 0) + 1)
  const q = query.trim().toLowerCase()
  const visible = frames.entries.filter(
    (entry) =>
      !hidden.has(entry.kind) &&
      (q === '' || searchable.get(entry.id)?.includes(q)) &&
      (cidFilter === '' || String(entry.link?.cid ?? '') === cidFilter) &&
      (toolFilter === '' || entry.link?.tool === toolFilter),
  )

  const toggleKind = (kind: LogKind) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }

  // 只有「用户主动往上滚」才关闭跟随；回到底部自动恢复。
  // 不用 onScroll 判断关闭：程序化 scrollIntoView、中栏切换引起的布局变化都会触发 scroll，
  // 会被误判成上滚，把跟随自己关掉（单跑 e2e 时实测到过）。
  // TODO: 拖动滚动条、PageUp / 触控板上滑以外的手势目前不关闭跟随；
  //       若后续要求覆盖全部上滚方式，另立计划补 pointer / key 判定，本轮不扩。
  const onWheel = (e: WheelEvent<HTMLDivElement>) => {
    if (e.deltaY < 0) setFollow(false)
  }

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 24) setFollow(true)
  }

  const locate = (entry: LogEntry) => {
    const source = `事件 #${entry.seq ?? '—'} · ${entry.text}`
    // 同时带 cid 与地址时优先看「谁发的」；「看地址」在展开区里给出第二条路径。
    if (entry.link?.cid !== null && entry.link?.cid !== undefined) {
      onSelectCid(entry.link.cid, source)
      setLocateHint(null)
    } else if (entry.link?.ref) {
      void onSelectRef(entry.link.ref, source).then(
        () => setLocateHint(null),
        (err: unknown) => setLocateHint(`定位失败：${err instanceof Error ? err.message : String(err)}`),
      )
    } else {
      setLocateHint('这条事件没有可定位的 cid 或地址')
    }
  }

  const truncated = frames.entries.length >= frames.limit

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
      <div className="filters">
        <input
          data-testid="log-search"
          style={{ width: 160 }}
          value={query}
          placeholder="搜索事件 / 载荷"
          onChange={(e) => setQuery(e.target.value)}
        />
        <select
          data-testid="log-filter-cid"
          style={{ width: 92 }}
          value={cidFilter}
          onChange={(e) => setCidFilter(e.target.value)}
        >
          <option value="">全部 cid</option>
          {cidOptions.map((cid) => (
            <option key={cid} value={String(cid)}>
              cid {cid}
            </option>
          ))}
        </select>
        <select
          data-testid="log-filter-tool"
          style={{ width: 130 }}
          value={toolFilter}
          onChange={(e) => setToolFilter(e.target.value)}
        >
          <option value="">全部工具</option>
          {toolOptions.map((tool) => (
            <option key={tool} value={tool}>
              {tool}
            </option>
          ))}
        </select>
        <span className="muted small">{visible.length}/{frames.entries.length}</span>
      </div>
      {locateHint ? <div className="hint warn">{locateHint}</div> : null}
      {truncated ? (
        <div className="hint warn small" data-testid="log-truncated-hint">
          日志已保留最近 {frames.limit} 条，更早的已丢弃（不是实时全量）
        </div>
      ) : null}
      <div className="scroll log-scroll" ref={scrollRef} onScroll={onScroll} onWheel={onWheel}>
        {visible.map((entry) => {
          const targets = copyTargets(entry)
          const primary = targets[0]
          const ref = entry.link?.ref ?? null
          const callId = entry.link?.callId ?? null
          const pair = callId ? paired.get(callId) : undefined
          const partner = pair?.find((item) => item.id !== entry.id)
          return (
            <div
              key={entry.id}
              className={`log-entry ${KIND_CLASS[entry.kind]}`}
              style={pair ? { boxShadow: 'inset 2px 0 0 var(--accent)' } : undefined}
              onClick={() => setOpenId(openId === entry.id ? null : entry.id)}
              data-testid={`log-entry-${entry.kind}`}
            >
              <span className="muted small">{timeText(entry.time)}</span>
              {entry.seq !== null ? <span className="muted small">#{entry.seq}</span> : null}
              <span className="log-kind">{LOG_KIND_LABEL[entry.kind]}</span>
              <span className="log-text">{entry.text}</span>
              {primary ? (
                <button
                  type="button"
                  className="link small"
                  data-testid={primary.testId}
                  onClick={(e) => {
                    e.stopPropagation()
                    onCopy(primary.value)
                  }}
                >
                  {primary.label}
                </button>
              ) : null}
              {entry.link?.cid !== null && entry.link?.cid !== undefined ? (
                <button
                  type="button"
                  className="link small"
                  data-testid="log-locate"
                  onClick={(e) => {
                    e.stopPropagation()
                    locate(entry)
                  }}
                >
                  定位
                </button>
              ) : null}
              {partner ? (
                <button
                  type="button"
                  className="link small"
                  data-testid="log-pair-summary"
                  onClick={(e) => {
                    e.stopPropagation()
                    setPairOpenId(pairOpenId === entry.id ? null : entry.id)
                  }}
                >
                  配对 {callId}（{LOG_KIND_LABEL[partner.kind]}）
                </button>
              ) : null}
              {pairOpenId === entry.id && partner ? (
                <pre className="content muted" style={{ width: '100%' }}>
                  {partner.text}
                </pre>
              ) : null}
              {openId === entry.id && entry.detail ? (
                <div style={{ width: '100%' }}>
                  <div className="row" style={{ padding: '2px 0', gap: 6 }}>
                    {targets.map((target) => (
                      <button
                        type="button"
                        className="link small"
                        key={target.testId}
                        data-testid={`${target.testId}-detail`}
                        onClick={(e) => {
                          e.stopPropagation()
                          onCopy(target.value)
                        }}
                      >
                        {target.label}
                      </button>
                    ))}
                    {ref ? (
                      <button
                        type="button"
                        className="link small"
                        data-testid="log-locate-ref"
                        onClick={(e) => {
                          e.stopPropagation()
                          const source = `事件 #${entry.seq ?? '—'} · ${entry.text}`
                          void onSelectRef(ref, source).then(
                            () => setLocateHint(null),
                            (err: unknown) =>
                              setLocateHint(
                                `定位失败：${err instanceof Error ? err.message : String(err)}`,
                              ),
                          )
                        }}
                      >
                        看地址
                      </button>
                    ) : null}
                  </div>
                  <pre className="content muted">{JSON.stringify(entry.detail, null, 2)}</pre>
                </div>
              ) : null}
            </div>
          )
        })}
        {visible.length === 0 ? <div className="muted">还没有事件</div> : null}
        <div ref={endRef} />
      </div>
    </div>
  )
}

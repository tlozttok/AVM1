import { useEffect, useRef, useState } from 'react'
import type { IOEntry } from './types'

export interface IoStream {
  entries: IOEntry[]
  connected: boolean
  clear: () => void
}

const MAX_ENTRIES = 800

/** `/api/io/stream` 的输出流 SSE。断线重连带最后读到的 seq，去重靠 seq。 */
export function useIoStream(): IoStream {
  const [entries, setEntries] = useState<IOEntry[]>([])
  const [connected, setConnected] = useState(false)
  const lastSeq = useRef(0)

  useEffect(() => {
    let alive = true
    let source: EventSource | null = null
    let timer: number | null = null

    const connect = () => {
      if (!alive) return
      source = new EventSource(`/api/io/stream?after=${lastSeq.current}`)
      source.onopen = () => setConnected(true)
      source.onmessage = (ev: MessageEvent<string>) => {
        let frame: { type?: string; entry?: IOEntry }
        try {
          frame = JSON.parse(ev.data) as { type?: string; entry?: IOEntry }
        } catch {
          return  // 坏帧跳过，流不断
        }
        const entry = frame.entry
        if (frame.type !== 'io' || !entry) return
        lastSeq.current = Math.max(lastSeq.current, entry.seq)
        setEntries((prev) =>
          prev.some((item) => item.seq === entry.seq)
            ? prev
            : [...prev, entry].slice(-MAX_ENTRIES),
        )
      }
      source.onerror = () => {
        setConnected(false)
        source?.close()
        source = null
        if (alive && timer === null) {
          timer = window.setTimeout(() => {
            timer = null
            connect()
          }, 1000)
        }
      }
    }

    connect()
    return () => {
      alive = false
      source?.close()
      if (timer !== null) window.clearTimeout(timer)
    }
  }, [])

  // 清屏不动 lastSeq：重连重放的老帧不会再刷进来
  const clear = () => setEntries([])
  return { entries, connected, clear }
}

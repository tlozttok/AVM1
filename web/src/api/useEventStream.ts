import { useEffect, useState } from 'react'
import { initialFrameState, noteFrame, reduceFrame, type FrameState } from '../lib/frames'
import type { SseFrame } from './types'

export interface EventStream {
  frames: FrameState
  connected: boolean
  clear: () => void
}

/** `/api/events` 的 SSE 连接。断线重连交给浏览器，去重交给 reducer。 */
export function useEventStream(): EventStream {
  const [frames, setFrames] = useState<FrameState>(initialFrameState)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const source = new EventSource('/api/events')
    source.onopen = () => setConnected(true)
    source.onerror = () => setConnected(false)
    source.onmessage = (ev: MessageEvent<string>) => {
      let frame: SseFrame
      try {
        frame = JSON.parse(ev.data) as SseFrame
      } catch {
        setFrames((s) => noteFrame(s, `收到无法解析的帧：${ev.data.slice(0, 120)}`))
        return
      }
      setFrames((s) => reduceFrame(s, frame))
    }
    return () => source.close()
  }, [])

  // 清屏保留 lastSeq：重连重放的老帧不会再刷进来
  const clear = () => setFrames((s) => ({ ...initialFrameState, lastSeq: s.lastSeq, gaps: 0 }))

  return { frames, connected, clear }
}

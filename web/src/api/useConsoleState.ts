import { useCallback, useEffect, useState } from 'react'
import { apiGet, describeError } from './client'
import type { State } from './types'

export interface ConsoleState {
  state: State | null
  error: string | null
  refresh: () => void
}

/**
 * 1s 轮询 `/api/state`：调度、对话清单、面板状态、中断表、屏蔽账本都在这一份里，负载小。
 * 内存树与设备清单不进轮询（见 App：只在显式点击与动作提交后取）。
 */
export function useConsoleState(intervalMs = 1000): ConsoleState {
  const [state, setState] = useState<State | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const refresh = useCallback(() => setTick((n) => n + 1), [])

  useEffect(() => {
    let alive = true
    const ctrl = new AbortController()
    const poll = async () => {
      try {
        const next = await apiGet<State>('/api/state', ctrl.signal)
        if (!alive) return
        setState(next)
        setError(null)
      } catch (err) {
        if (!alive) return
        // 保留上一份好数据：控制台重启或 Core 结束时不把界面清空
        setError(describeError(err))
      }
    }
    void poll()
    const timer = window.setInterval(poll, intervalMs)
    return () => {
      alive = false
      window.clearInterval(timer)
      ctrl.abort()
    }
  }, [intervalMs, tick])

  return { state, error, refresh }
}

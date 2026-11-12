import { useCallback, useEffect, useRef, useState } from 'react'
import { apiSend } from '../api/client'
import type { ActionResponse } from '../api/types'
import { pairAction, startTrace, type ActionTrace } from '../lib/correlation'
import type { FrameState } from '../lib/frames'
import type { OperatorAction } from '../lib/operatorActions'

/**
 * 提交一次操作员指令，并在事件流里按 call_id 配对出工具响应。
 *
 * 提交路径就是既有的两个端点：`POST /api/action` 投一条工具调用；`release` 的动作
 * 立刻 `POST /api/resume` 让出执行权（命令台与对象操作的「做完就继续跑」）。
 * 未知工具不在这里拦：`/api/action` 会返回 `known_tool=false`，Core 不回工具响应，
 * 这条 trace 会停在 `no_response`，和既有调试语义一致。
 */
export interface OperatorTrace extends ActionTrace {
  id: number
  release: boolean
}

export interface SubmitReceipt {
  id: number
  warning: string | null
}

const TRACE_LIMIT = 20

export function useOperatorAction(frames: FrameState) {
  const [traces, setTraces] = useState<Record<number, OperatorTrace>>({})
  const seqRef = useRef(frames.lastSeq)
  seqRef.current = frames.lastSeq
  const nextId = useRef(1)

  const submit = useCallback(async (action: OperatorAction): Promise<SubmitReceipt> => {
    const id = nextId.current
    nextId.current += 1
    // 序列在请求前快照：HTTP 返回时 SSE 可能已经把 tool_call 推过来。
    const seqAfter = seqRef.current
    const response = await apiSend<ActionResponse>('/api/action', 'POST', {
      tool: action.tool,
      arguments: action.arguments,
    })
    if (action.release) {
      await apiSend('/api/resume', 'POST', {})
    }
    setTraces((prev) => {
      const next: Record<number, OperatorTrace> = {
        ...prev,
        [id]: {
          ...startTrace(action.tool, action.arguments, response.known_tool, seqAfter),
          id,
          release: action.release,
        },
      }
      const ids = Object.keys(next).map(Number).sort((a, b) => a - b)
      while (ids.length > TRACE_LIMIT) {
        const drop = ids.shift()
        if (drop !== undefined) delete next[drop]
      }
      return next
    })
    return { id, warning: response.warning }
  }, [])

  useEffect(() => {
    // pairAction 到终态返回原引用，所以没有变化时 setState 拿到同一份 state，不会形成循环。
    setTraces((prev) => {
      let changed = false
      const next: Record<number, OperatorTrace> = {}
      for (const trace of Object.values(prev)) {
        const paired = pairAction(trace, frames.entries)
        if (paired === trace) {
          next[trace.id] = trace
          continue
        }
        changed = true
        next[trace.id] = { ...paired, id: trace.id, release: trace.release }
      }
      return changed ? next : prev
    })
  }, [frames.entries])

  return { traces, submit }
}

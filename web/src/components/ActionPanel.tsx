import { useCallback, useEffect, useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ToolTable } from '../api/tools'
import type { DeviceInfo, PauseResponse, Schedule, State } from '../api/types'
import type { FrameState } from '../lib/frames'
import { pairAction, startTrace, type ActionTrace, type SubmittedAction } from '../lib/correlation'
import type { TrayValue } from '../lib/valueTray'
import ActionCard from './ActionCard'
import ActionForm from './ActionForm'
import DeviceApprovalPanel from './DeviceApprovalPanel'
import InputFeed from './InputFeed'
import ValueTray from './ValueTray'

interface Props {
  state: State | null
  /** SSE 里最近一帧 schedule：横幅不必等 1s 轮询。 */
  schedule: Schedule | null
  frames: FrameState
  tools: ToolTable
  devices: DeviceInfo[]
  tray: TrayValue[]
  traySelected: string | null
  onSelectTray: (value: string) => void
  copyNote: string | null
  onRefresh: () => void
  onDevicesChanged: () => void
  /** 动作卡配对到 tool_result：这一刻才补刷内存树与设备清单。 */
  onToolResult: () => void
}

function lastDispatchText(state: State | null): string {
  const records = state?.interrupts.last_dispatch ?? []
  if (records.length === 0) return '最近派发：无'
  return records
    .map((r) => {
      if (r.action === 'switch') return `switch ${r.type} → cid ${r.target}（打断 ${r.preempted ?? '—'}）`
      if (r.action === 'noop') return `noop ${r.type}（${r.reason ?? ''}）`
      return `dropped ${r.type}（${r.reason ?? ''}）`
    })
    .join('；')
}

/** 右栏上：暂停、动作、喂输入。三步交互的入口都在这。 */
export default function ActionPanel({
  state,
  schedule,
  frames,
  tools,
  devices,
  tray,
  traySelected,
  onSelectTray,
  copyNote,
  onRefresh,
  onDevicesChanged,
  onToolResult,
}: Props) {
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [trace, setTrace] = useState<ActionTrace | null>(null)

  const paused = schedule !== null && schedule.active_cid === schedule.panel_cid
  const running = state?.running ?? false
  const pendingInterrupts = state?.interrupts.pending ?? []
  const queuedActions = state?.panel.channel.pending_actions ?? 0
  const dropped = state?.panel.channel.dropped ?? 0

  let banner: string
  let bannerClass: string
  if (state === null) {
    banner = '连接中…'
    bannerClass = 'banner'
  } else if (!running) {
    banner = 'Core 已停止（core.run() 已返回）：端点仍在，可以查看'
    bannerClass = 'banner warn'
  } else if (paused) {
    banner = `已暂停：执行权在面板对话 cid ${schedule?.panel_cid ?? state.panel.cid}，系统停在这里等操作员`
    bannerClass = 'banner paused'
  } else if (pendingInterrupts.length > 0 || queuedActions > 0) {
    banner = `已排队，等下一个安全点（中断 ${pendingInterrupts.join('、') || '—'} · 面板动作 ${queuedActions}）`
    bannerClass = 'banner pending'
  } else {
    banner = `运行中 · 活跃对话 cid ${schedule?.active_cid ?? state.schedule.active_cid ?? '—'}`
    bannerClass = 'banner'
  }

  const call = async (path: string, ok: (resp: PauseResponse) => string) => {
    setBusy(true)
    setError(null)
    try {
      const resp = await apiSend<PauseResponse>(path, 'POST')
      setMessage(ok(resp))
      onRefresh()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  // 提交成功时起一条待配对动作；随后每来一批新帧就试着按序配对。
  // pairAction 是纯函数且已到终态返回原引用，所以这里不会形成 setState 循环。
  const onSubmitted = useCallback((info: SubmittedAction) => {
    setTrace(startTrace(info.tool, info.args, info.knownTool, info.seqAfter))
    onRefresh()
  }, [onRefresh])

  useEffect(() => {
    if (!trace) return
    const next = pairAction(trace, frames.entries)
    if (next.phase === 'responded' && trace.phase !== 'responded') onToolResult()
    if (next !== trace) setTrace(next)
  }, [frames.entries, trace, onToolResult])

  const guidance = paused
    ? trace?.phase === 'responded'
      ? '执行权在面板，响应已到：可继续提交，或点继续让出执行权'
      : '执行权在面板：提交动作后等工具响应，再点继续让出执行权'
    : '执行权不在面板：提交动作会自动请求中断，在下一个安全点生效'

  return (
    <div className="action-panel">
      <div className={bannerClass} data-testid="status-banner">
        {banner}
      </div>
      <div className="row">
        <button
          type="button"
          data-testid="pause-button"
          disabled={busy || paused || !running}
          onClick={() =>
            void call('/api/interrupt', (r) =>
              r.already_active ? '面板已经持有执行权' : '已请求暂停：中断在下一个安全点生效',
            )
          }
        >
          暂停
        </button>
        <button
          type="button"
          data-testid="resume-button"
          disabled={busy || !paused}
          onClick={() => void call('/api/resume', () => '已提交结束动作：执行权让回就绪队列')}
        >
          继续
        </button>
        <span className="muted small" data-testid="channel-status">
          面板输出 seq {state?.panel.channel.next_seq ?? '—'} · 缓冲 {state?.panel.channel.buffered ?? '—'}
          {dropped > 0 ? ` · 丢弃 ${dropped}` : ''}
        </span>
      </div>
      <div className="hint muted small">{lastDispatchText(state)}</div>
      <div className="hint muted small" data-testid="action-guidance">{guidance}</div>
      {message ? <div className="hint ok">{message}</div> : null}
      {error ? <div className="hint error">{error}</div> : null}

      <ActionCard trace={trace} />
      <ValueTray values={tray} selected={traySelected} onSelect={onSelectTray} note={copyNote} />
      <ActionForm
        tools={tools}
        currentSeq={frames.lastSeq}
        traySelected={traySelected}
        onSubmitted={onSubmitted}
      />
      <InputFeed devices={devices} onDevicesChanged={onDevicesChanged} />
      <DeviceApprovalPanel onRefresh={onRefresh} />
    </div>
  )
}

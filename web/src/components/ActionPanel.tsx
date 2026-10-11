import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ToolTable } from '../api/tools'
import type { DeviceInfo, PauseResponse, Schedule, State } from '../api/types'
import ActionForm from './ActionForm'
import InputFeed from './InputFeed'

interface Props {
  state: State | null
  /** SSE 里最近一帧 schedule：横幅不必等 1s 轮询。 */
  schedule: Schedule | null
  tools: ToolTable
  devices: DeviceInfo[]
  onRefresh: () => void
  onDevicesChanged: () => void
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
  tools,
  devices,
  onRefresh,
  onDevicesChanged,
}: Props) {
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

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
      {message ? <div className="hint ok">{message}</div> : null}
      {error ? <div className="hint error">{error}</div> : null}

      <ActionForm tools={tools} onSubmitted={onRefresh} />
      <InputFeed devices={devices} onDevicesChanged={onDevicesChanged} />
    </div>
  )
}

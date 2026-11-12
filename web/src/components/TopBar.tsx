import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { State } from '../api/types'
import type { SystemStatus } from '../lib/systemStatus'

interface Props {
  state: State | null
  status: SystemStatus
  onRefresh: () => void
}

function recoveryText(state: State | null): string {
  const recovery = state?.recovery
  if (!recovery) return '恢复信息不可用'
  const origin = recovery.source === 'snapshot'
    ? `快照 seq ${recovery.snapshot_seq ?? '—'}`
    : '冷启动'
  return `${origin} · 重放 ${recovery.replayed} 条 · ${recovery.warnings.length} 警告`
}

/** 顶栏：单一状态语义 + 生命周期按钮。状态判定在 lib/systemStatus.ts。 */
export default function TopBar({ state, status, onRefresh }: Props) {
  const [confirm, setConfirm] = useState<'shutdown' | 'force' | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const lifecycle = state?.lifecycle ?? (state?.running ? 'running' : 'stopped')
  const remaining = state?.force_stop_remaining ?? 0
  const forceAllowed = lifecycle === 'stopping' && remaining <= 0
  const canPause = status.phase === 'running'
  const canResume = status.phase === 'operator'

  const call = async (path: string, okText: string) => {
    setBusy(true)
    setError(null)
    try {
      await apiSend(path, 'POST', {})
      setMessage(okText)
      onRefresh()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
      setConfirm(null)
    }
  }

  return (
    <header className="topbar" data-testid="topbar" data-phase={status.phase}>
      <span className="topbar-title">AVM 操作面板</span>
      <span className="topbar-status" data-testid="topbar-status">{status.label}</span>
      {status.phase === 'api-paused' && state?.pause_reason ? (
        <span className="topbar-note" title={state.pause_reason}>{state.pause_reason}</span>
      ) : null}
      <span className="topbar-meta">
        活跃对话 <b data-testid="topbar-active">{status.activeLabel ?? '—'}</b>
      </span>
      <span className="topbar-meta">输出 seq <b>{state?.io?.output.next_seq ?? '—'}</b></span>
      <span className="topbar-meta" data-testid="topbar-recovery">{recoveryText(state)}</span>
      <span className="spacer" />
      {status.phase === 'api-paused' ? (
        <button
          type="button"
          data-testid="topbar-api-resume"
          disabled={busy}
          onClick={() => void call('/api/system/resume', '已恢复执行')}
        >
          恢复
        </button>
      ) : null}
      <button
        type="button"
        data-testid="topbar-pause"
        disabled={busy || !canPause}
        onClick={() => void call('/api/interrupt', '已请求暂停：下一个安全点生效')}
      >
        暂停
      </button>
      <button
        type="button"
        data-testid="topbar-resume"
        disabled={busy || !canResume}
        onClick={() => void call('/api/resume', '已让出执行权')}
      >
        继续
      </button>
      <button
        type="button"
        className="danger"
        data-testid="topbar-shutdown"
        disabled={busy || lifecycle === 'stopping' || lifecycle === 'stopped'}
        onClick={() => setConfirm('shutdown')}
      >
        关机
      </button>
      {confirm ? (
        <span className="topbar-confirm">
          <span className="topbar-confirm-text">
            {confirm === 'force' ? '强制退出不写快照，确认？' : '当前回合结束后写快照并退出，确认？'}
          </span>
          <button
            type="button"
            data-testid="topbar-shutdown-confirm"
            onClick={() =>
              void call(
                confirm === 'force' ? '/api/system/force-stop' : '/api/system/shutdown',
                confirm === 'force' ? '已请求强制退出' : '关机中',
              )
            }
          >
            确认
          </button>
          <button type="button" className="link" onClick={() => setConfirm(null)}>取消</button>
        </span>
      ) : null}
      {forceAllowed && !confirm ? (
        <button
          type="button"
          className="link"
          data-testid="topbar-force-stop"
          onClick={() => setConfirm('force')}
        >
          强制退出
        </button>
      ) : null}
      {lifecycle === 'stopping' && remaining > 0 ? (
        <span className="topbar-note" data-testid="topbar-force-wait">
          收尾中，{Math.ceil(remaining)} 秒后可强制退出
        </span>
      ) : null}
      {message ? <span className="topbar-ok">{message}</span> : null}
      {error ? <span className="topbar-error">{error}</span> : null}
    </header>
  )
}

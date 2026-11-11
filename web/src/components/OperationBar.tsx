import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { Schedule, State } from '../api/types'

interface Props {
  state: State | null
  schedule: Schedule | null
  onRefresh: () => void
}

function recoveryText(state: State | null): string {
  const r = state?.recovery
  if (!r) return '恢复信息不可用'
  const origin =
    r.source === 'snapshot'
      ? `从快照恢复（seq ${r.snapshot_seq ?? '—'}）`
      : '冷启动'
  return `${origin} · 重放 ${r.replayed} 条 · ${r.warnings.length} 个警告`
}

/** 顶栏：现在健康吗、跑到哪了、怎么收工。按钮走既有端点，不新增写内存路径。 */
export default function OperationBar({ state, schedule, onRefresh }: Props) {
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirm, setConfirm] = useState<'shutdown' | 'force' | null>(null)
  const [busy, setBusy] = useState(false)

  const lifecycle = state?.lifecycle ?? (state?.running ? 'running' : 'stopped')
  const remaining = state?.force_stop_remaining ?? 0
  // 强制退出不是常驻按钮：先请求关机，收尾等满后端给的秒数之后才出现
  const forceAllowed = lifecycle === 'stopping' && remaining <= 0
  const label =
    lifecycle === 'stopping'
      ? '关机中'
      : lifecycle === 'stopped'
        ? '已停止'
        : state?.running
          ? '运行中'
          : '未运行'

  const call = async (path: string, okText: string) => {
    setBusy(true)
    setError(null)
    try {
      await apiSend(path, 'POST', {})
      setResult(okText)
      onRefresh()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
      setConfirm(null)
    }
  }

  return (
    <div className="operation-bar" data-testid="operation-bar">
      <span className={`operation-status status-${lifecycle}`} data-testid="operation-status">
        {label}
      </span>
      <span className="muted small">
        活跃对话 <span className="mono">{schedule?.active_cid ?? '—'}</span>
      </span>
      <span className="muted small">
        输出 seq <span className="mono">{state?.io?.output.next_seq ?? '—'}</span>
      </span>
      <span className="muted small" data-testid="operation-recovery">
        {recoveryText(state)}
      </span>
      <span className="muted small" data-testid="operation-client">
        {state?.client?.single_client
          ? state.client.held ? '单客户端：已接入' : '单客户端：空闲'
          : '单客户端：未启用'}
      </span>
      <span className="operation-actions">
        <button
          type="button"
          data-testid="operation-pause"
          disabled={busy}
          onClick={() => void call('/api/interrupt', '已请求暂停')}
        >
          暂停
        </button>
        <button
          type="button"
          data-testid="operation-resume"
          disabled={busy}
          onClick={() => void call('/api/resume', '已继续')}
        >
          继续
        </button>
        <button
          type="button"
          data-testid="operation-shutdown"
          disabled={busy}
          onClick={() => setConfirm('shutdown')}
        >
          关机
        </button>
        {confirm ? (
          <span className="operation-confirm">
            <span className="hint warn inline">
              {confirm === 'force'
                ? '强制退出不写快照，确认？'
                : '当前回合结束后写快照并退出，确认？'}
            </span>
            <button
              type="button"
              data-testid="operation-shutdown-confirm"
              onClick={() =>
                void call(
                  confirm === 'force' ? '/api/system/force-stop' : '/api/system/shutdown',
                  confirm === 'force' ? '已请求强制退出' : '关机中',
                )
              }
            >
              确认
            </button>
            <button type="button" className="link" onClick={() => setConfirm(null)}>
              取消
            </button>
          </span>
        ) : null}
        {forceAllowed && !confirm ? (
          <button
            type="button"
            className="link small"
            data-testid="operation-force-stop"
            onClick={() => setConfirm('force')}
          >
            强制退出
          </button>
        ) : null}
      </span>
      {lifecycle === 'stopping' && remaining > 0 ? (
        <span className="hint warn inline" data-testid="operation-force-wait">
          收尾中，{Math.ceil(remaining)} 秒后才可强制退出
        </span>
      ) : null}
      {result ? <span className="hint ok inline">{result}</span> : null}
      {error ? <span className="hint error inline">{error}</span> : null}
    </div>
  )
}

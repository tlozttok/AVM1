import { useEffect, useState, type ReactNode } from 'react'
import { ApiError, apiSend, clearClientCookie, currentClientId, describeError, saveClientCookie } from '../api/client'
import LoginGate from './LoginGate'

interface Props {
  children: ReactNode
}

type GateState =
  | { kind: 'pending' }
  | { kind: 'ready' }
  | { kind: 'login' }
  | { kind: 'conflict'; message: string }
  | { kind: 'error'; message: string }

/**
 * 面板准入：一个宿主一次只允许一个客户端。
 *
 * - 身份是本标签页一份 sessionStorage 值；刷新复用，新标签页另算，被后端按租约挡住。
 * - 401 交给登录页；404（旧后端没有这个端点）按兼容放行；409 显示占用者信息。
 * - 关页时尽力释放席位；没发出来也不要紧，后端租约会在最后一次请求后过期。
 */
export default function ClientGate({ children }: Props) {
  const [gate, setGate] = useState<GateState>({ kind: 'pending' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let alive = true
    const clientId = currentClientId()
    setGate({ kind: 'pending' })
    void apiSend('/api/session/acquire', 'POST', { client_id: clientId })
      .then(() => {
        if (!alive) return
        saveClientCookie(clientId)
        setGate({ kind: 'ready' })
      })
      .catch((err: unknown) => {
        if (!alive) return
        if (err instanceof ApiError && err.status === 401) {
          setGate({ kind: 'login' })
        } else if (err instanceof ApiError && err.status === 404) {
          // 旧后端没有准入端点：不因为新前端把面板锁死
          setGate({ kind: 'ready' })
        } else if (err instanceof ApiError && err.status === 409) {
          setGate({ kind: 'conflict', message: err.message })
        } else {
          setGate({ kind: 'error', message: describeError(err) })
        }
      })
    return () => {
      alive = false
    }
  }, [attempt])

  useEffect(() => {
    const release = () => {
      const clientId = currentClientId()
      const body = JSON.stringify({ client_id: clientId })
      if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
        navigator.sendBeacon('/api/session/release', new Blob([body], { type: 'application/json' }))
      } else {
        void apiSend('/api/session/release', 'POST', { client_id: clientId }).catch(() => {})
      }
    }
    window.addEventListener('pagehide', release)
    return () => window.removeEventListener('pagehide', release)
  }, [])

  if (gate.kind === 'login') {
    return <LoginGate onLogin={() => window.location.reload()} />
  }

  if (gate.kind === 'pending') {
    return (
      <div className="client-gate" data-testid="client-gate-loading">
        <div className="login-card">
          <div className="pane-title">AVM 操作面板 · 接入中</div>
          <div className="pane-body muted small">正在确认本宿主是否已有客户端…</div>
        </div>
      </div>
    )
  }

  if (gate.kind === 'conflict' || gate.kind === 'error') {
    return (
      <div className="client-gate" data-testid="client-gate">
        <div className="login-card">
          <div className="pane-title">AVM 操作面板 · 无法接入</div>
          <div className="pane-body">
            <p className="small">
              {gate.kind === 'conflict'
                ? '这个宿主一次只允许一个面板客户端。'
                : '接入检查失败。'}
            </p>
            <div className="hint warn" data-testid="client-gate-message">
              {gate.message}
            </div>
            <div className="composer-actions">
              <button type="button" data-testid="client-gate-retry" onClick={() => setAttempt((n) => n + 1)}>
                重试
              </button>
              <button type="button" className="link" onClick={() => { clearClientCookie(); window.location.reload() }}>
                重新载入
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return <>{children}</>
}

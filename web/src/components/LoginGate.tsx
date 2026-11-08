import { useState } from 'react'
import { apiGet, describeError, saveToken } from '../api/client'

interface Props {
  onLogin: () => void
}

/**
 * 401 驱动的登录页：启动宿主时打印/写到 out/console.token 的一次性凭证。
 * 未启用认证的后端（单测、e2e mock）永远不会触发这一屏。
 */
export default function LoginGate({ onLogin }: Props) {
  const [token, setToken] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!token.trim() || busy) return
    setBusy(true)
    setError(null)
    saveToken(token.trim())
    try {
      await apiGet('/api/state')
      onLogin()
    } catch (err) {
      saveToken(null)
      setError(`凭证无效或服务端不可用：${describeError(err)}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-gate" data-testid="login-gate">
      <div className="login-card">
        <div className="pane-title">AVM 操作面板 · 登录</div>
        <div className="pane-body">
          <p className="muted small">
            本次运行的凭证由控制台启动时打印到终端，也写在镜像目录下的 out/console.token。
          </p>
          <label className="field">
            <span>凭证</span>
            <input
              type="password"
              data-testid="login-token"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void submit()
              }}
              placeholder="粘贴 token"
              autoFocus
            />
          </label>
          <div className="composer-actions">
            <button
              type="button"
              data-testid="login-submit"
              disabled={busy || !token.trim()}
              onClick={() => void submit()}
            >
              {busy ? '验证中…' : '进入面板'}
            </button>
          </div>
          {error ? (
            <div className="hint error" data-testid="login-error">
              {error}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}

import { useState } from 'react'
import type { DeviceInspection } from '../api/types'
import { previousHash, shortHash } from '../lib/deviceApproval'

interface Props {
  inspection: DeviceInspection
  busy: boolean
  error: string | null
  onApprove: () => void
  onReject: () => void
}

/**
 * 设备审批的全屏模态。
 *
 * 默认拒绝：焦点在"拒绝"；"允许"必须先用复选框主动确认，键盘没有直接允许的快捷键。
 * 这里只展示与确认，不自己提交；装载请求由外层经 `/api/action` 写 `$MEM.loader`。
 */
export default function DeviceApprovalModal({
  inspection,
  busy,
  error,
  onApprove,
  onReject,
}: Props) {
  const [confirmed, setConfirmed] = useState(false)
  const [showFullHash, setShowFullHash] = useState(false)
  const oldHash = previousHash(inspection)

  return (
    <div className="device-approval-overlay" data-testid="device-approval-modal">
      <div className="device-approval-modal" role="dialog" aria-modal="true"
        aria-labelledby="device-approval-title">
        <div className="device-approval-head">
          <h2 id="device-approval-title">允许装载这台设备？</h2>
          <span className="tag tag-warn">默认拒绝</span>
        </div>

        <div className="device-approval-body">
          <div className="kv">
            <div>
              <span className="key">设备文件</span>
              <span className="mono">{inspection.path}</span>
            </div>
            <div>
              <span className="key">大小</span>
              <span>{inspection.size} 字节</span>
            </div>
            <div>
              <span className="key">sha256</span>
              <span className="mono" data-testid="device-approval-hash">
                {showFullHash ? inspection.sha256 : shortHash(inspection.sha256)}
              </span>
              <button
                type="button"
                className="link small"
                data-testid="device-approval-hash-toggle"
                onClick={() => setShowFullHash((v) => !v)}
              >
                {showFullHash ? '收起' : '展开全文'}
              </button>
            </div>
            <div>
              <span className="key">设备类</span>
              <span className="mono">{inspection.class ?? '（未识别）'}</span>
            </div>
            <div>
              <span className="key">挂载点</span>
              <span className="mono">$MEM.{inspection.mount ?? '（待填写）'}</span>
            </div>
            <div>
              <span className="key">args</span>
              <pre className="content muted">
                {JSON.stringify(inspection.args ?? {}, null, 2)}
              </pre>
            </div>
            {oldHash ? (
              <div>
                <span className="key">覆盖旧版本</span>
                <span className="mono" data-testid="device-approval-old-hash">
                  旧哈希 {shortHash(oldHash)}
                  {inspection.existing?.approved_at
                    ? ` · 上次审批 ${inspection.existing.approved_at}`
                    : ''}
                </span>
              </div>
            ) : (
              <div>
                <span className="key">安装状态</span>
                <span>首次装载</span>
              </div>
            )}
          </div>

          <div className="device-approval-consequences">
            <div className="section-title">允许的后果</div>
            <ul>
              <li>这段代码会作为设备进入快照，并在<strong>每次启动</strong>时重新加载。</li>
              <li>
                设备以进程内权限运行：<code>attach_core</code> 拿到 Core 引用，能读写内存、关闭对话、
                注册监听器。
              </li>
              <li>批准绑定的是上面这个文件的 sha256；文件内容变化必须重新审批。</li>
              <li>
                审批是人的决定，不是内核强制的安全沙箱：能写 <code>$MEM.loader</code> 的对话知道哈希就能装载。
              </li>
            </ul>
          </div>
        </div>

        {error ? <div className="hint error">{error}</div> : null}

        <label className="device-approval-confirm">
          <input
            type="checkbox"
            data-testid="device-approval-confirm"
            checked={confirmed}
            onChange={(e) => setConfirmed(e.target.checked)}
          />
          我已核对路径与哈希，确认允许装载这段代码
        </label>

        <div className="device-approval-actions">
          <button
            type="button"
            data-testid="device-approval-reject"
            autoFocus
            disabled={busy}
            onClick={onReject}
          >
            拒绝
          </button>
          <button
            type="button"
            data-testid="device-approval-allow"
            disabled={!confirmed || busy}
            onClick={onApprove}
          >
            {busy ? '提交中…' : '允许装载'}
          </button>
        </div>
      </div>
    </div>
  )
}

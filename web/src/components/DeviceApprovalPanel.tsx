import { useState } from 'react'
import { apiSend, currentClientId, describeError } from '../api/client'
import type { ActionResponse, DeviceInspection } from '../api/types'
import { buildLoadDeviceCommand } from '../lib/deviceApproval'
import DeviceApprovalModal from './DeviceApprovalModal'

interface Props {
  onRefresh: () => void
}

/**
 * 装载设备的最小入口：填写文件/挂载/类，先只读检查，再弹全屏审批。
 *
 * 允许后把 load_device 命令（带 `approved_sha256`）经 `/api/action` 写进
 * `$MEM.loader`——与操作员手工提交一条 memory_write 完全相同，走面板对话、
 * 自动中断与可重放日志。拒绝则只在本地丢弃，不碰内核。
 */
export default function DeviceApprovalPanel({ onRefresh }: Props) {
  const [file, setFile] = useState('')
  const [mount, setMount] = useState('')
  const [className, setClassName] = useState('')
  const [argsText, setArgsText] = useState('{}')
  const [inspection, setInspection] = useState<DeviceInspection | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const parseArgs = (): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(argsText || '{}') as unknown
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setError('args 必须是 JSON 对象')
        return null
      }
      return parsed as Record<string, unknown>
    } catch {
      setError('args 不是合法 JSON')
      return null
    }
  }

  const inspect = async () => {
    if (!file.trim()) {
      setError('先填写设备文件路径')
      return
    }
    const args = parseArgs()
    if (args === null) return
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const result = await apiSend<DeviceInspection>('/api/device/inspect', 'POST', {
        path: file.trim(),
        mount: mount.trim() || undefined,
        class: className.trim() || undefined,
        args,
      })
      const resolvedMount = mount.trim() || result.mount || ''
      const resolvedClass = className.trim() || result.class || ''
      if (!resolvedMount || !resolvedClass) {
        setError('挂载点与设备类名都必须填写（弹窗只负责确认，不编辑字段）')
        return
      }
      // 覆盖旧版本且用户没写 args 时，用来源表里的旧 args 预填（弹窗里可见）。
      const existingArgs = result.existing?.args
      const shown =
        existingArgs && argsText === '{}' ? { ...result, args: existingArgs } : result
      setInspection(shown)
      if (!mount.trim() && result.mount) setMount(result.mount)
      if (!className.trim() && result.class) setClassName(result.class)
      if (existingArgs && argsText === '{}') {
        setArgsText(JSON.stringify(existingArgs, null, 2))
      }
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const approve = async () => {
    if (!inspection) return
    const resolvedMount = (inspection.mount ?? mount).trim()
    const resolvedClass = (className || inspection.class || '').trim()
    if (!resolvedMount || !resolvedClass) {
      setError('挂载点与设备类名都必须填写')
      return
    }
    const args = parseArgs()
    if (args === null) return
    setBusy(true)
    setError(null)
    try {
      const command = buildLoadDeviceCommand({
        inspection,
        mount: resolvedMount,
        className: resolvedClass,
        args,
        approvedAt: new Date().toISOString(),
        approvedVia: currentClientId(),
      })
      await apiSend<ActionResponse>('/api/action', 'POST', {
        tool: 'memory_write',
        arguments: { ref: '$MEM.loader', content: command },
      })
      setMessage('已提交装载请求（面板对话执行后出现在设备表）')
      setInspection(null)
      onRefresh()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const reject = () => {
    setInspection(null)
    setMessage('已拒绝装载：内核状态未改变')
  }

  return (
    <details className="device-approval-launcher" data-testid="device-approval-launcher">
      <summary>装载设备（审批）</summary>
      <div className="device-approval-form">
        <div className="row">
          <input
            type="text"
            data-testid="device-approval-file"
            placeholder="设备文件路径（镜像目录或包目录内）"
            value={file}
            onChange={(e) => setFile(e.target.value)}
          />
          <input
            type="text"
            data-testid="device-approval-mount"
            placeholder="挂载点，如 extra"
            value={mount}
            onChange={(e) => setMount(e.target.value)}
          />
          <input
            type="text"
            data-testid="device-approval-class"
            placeholder="设备类名，如 EchoDevice"
            value={className}
            onChange={(e) => setClassName(e.target.value)}
          />
        </div>
        <textarea
          data-testid="device-approval-args"
          rows={3}
          value={argsText}
          placeholder="构造参数（JSON 对象）"
          onChange={(e) => setArgsText(e.target.value)}
        />
        <div className="row">
          <button
            type="button"
            data-testid="device-approval-inspect"
            disabled={busy}
            onClick={() => void inspect()}
          >
            {busy ? '检查中…' : '检查并审批'}
          </button>
          <span className="hint muted small">先只读算哈希，再由操作员决定是否允许</span>
        </div>
        {message ? <div className="hint ok" data-testid="device-approval-message">{message}</div> : null}
        {error ? <div className="hint error">{error}</div> : null}

      </div>
      {inspection ? (
        <DeviceApprovalModal
          inspection={inspection}
          busy={busy}
          error={error}
          onApprove={() => void approve()}
          onReject={reject}
        />
      ) : null}
    </details>
  )
}

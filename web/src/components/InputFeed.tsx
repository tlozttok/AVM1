import { useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { DeviceInfo } from '../api/types'

interface Props {
  devices: DeviceInfo[]
  onDevicesChanged: () => void
}

/**
 * `$MEM.inputs` 的喂输入框。
 *
 * 输入框是多行的：设备把 `content` 整段排进队列（下一次 `$MEM.inputs.next` 整段取走），
 * 所以多行文本原样作为**一段**内容交给对话，不拆成多行排队。
 * 回车留给换行，投喂用 Ctrl+Enter 或按钮。
 *
 * 动作编号不硬编码：从 `GET /api/devices` 的清单里按设备类型与动作名找。被控制台托管时
 * （host_flags["console"]）设备的那一读正等在面板队列上，喂进来立刻生效。
 */
export default function InputFeed({ devices, onDevicesChanged }: Props) {
  const inputs =
    devices.find((d) => d.description.type === 'InputsListDevice') ??
    devices.find((d) => d.mount === 'inputs')
  const feed = inputs?.actions.find((a) => (a.name ?? '').includes('投喂'))
  const pending = inputs?.actions.find((a) => (a.name ?? '').includes('查询'))
  const [text, setText] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const call = async (
    index: number,
    args: Record<string, unknown>,
    describe: (result: unknown) => string,
  ) => {
    if (!inputs) return
    setBusy(true)
    setError(null)
    try {
      // 响应是调用记录：动作的返回值在 record.result 里，出错时 record.error 有原因
      const record = await apiSend<{ result?: unknown; error?: string | null }>(
        '/api/device-action',
        'POST',
        { mount: inputs.mount, index, arguments: args },
      )
      if (record.error) setError(record.error)
      setResult(describe(record.result))
      onDevicesChanged()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const send = () =>
    void call(feed?.index ?? 0, { content: text }, (result) => {
      const queued = (result as { queued?: number } | null)?.queued
      return queued === undefined ? '已投喂' : `已排队 ${queued} 行`
    })

  const query = () =>
    void call(pending?.index ?? 1, {}, (result) => {
      const q = result as { queued?: number; waiting?: boolean } | null
      return `队列 ${q?.queued ?? '?'} 行，设备${q?.waiting ? '正在等待输入' : '当前没有等着读'}`
    })

  return (
    <div className="input-feed">
      <div className="section-title">
        喂输入（$MEM.inputs）
        {pending ? (
          <button type="button" className="link" onClick={query} disabled={busy}>
            查询队列
          </button>
        ) : null}
      </div>
      {!inputs || !feed ? (
        <div className="hint warn">清单里没有找到输入设备，或它没有声明「投喂一行输入」动作</div>
      ) : (
        <>
          <div className="row">
            <textarea
              data-testid="input-feed-field"
              rows={4}
              value={text}
              placeholder="给对话的一段文本（回车换行，Ctrl+Enter 投喂）"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <button type="button" data-testid="input-feed-send" onClick={send} disabled={busy || text === ''}>
              投喂
            </button>
          </div>
          <div className="hint muted small">多行原样排成一段（不拆行），下一次 $MEM.inputs.next 整段取走</div>
        </>
      )}
      {error ? <div className="hint error">{error}</div> : null}
      {result ? (
        <div className="hint ok" data-testid="input-feed-result">
          {result}
        </div>
      ) : null}
    </div>
  )
}

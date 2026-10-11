import { useRef, useState } from 'react'
import { describeError } from '../api/client'
import type { DeviceInfo, DictNode, ListNode, MemoryNode } from '../api/types'

interface Props {
  root: MemoryNode | null
  /**
   * 已挂载设备。整树读取不含设备（它们在 Memory._devices 表里，不在数据树上），这里合并进来。
   *
   * TODO(issue)：`GET /api/memory` 的整树输出没有设备行，前端只能另取 `GET /api/devices` 补齐。
   * 三条路——保持前端合并（当前做法）、让整树带上设备行（改这个端点的响应形状）、
   * 另开一个只读的树端点——留给作者定夺；选定前不改后端。
   */
  devices: DeviceInfo[]
  busy: boolean
  error: string | null
  onRefresh: () => void
  onSelect: (ref: string, node: MemoryNode) => void
  onReadRef: (ref: string, signal: AbortSignal) => Promise<MemoryNode>
}

/** 设备伪列表（末段是下标，例如 $MEM.inputs.-1）：读它会阻塞到有人喂输入。 */
const BLOCKING_REF = /\.[0-9]+$/
const READ_TIMEOUT_MS = 8000

function kindLabel(node: MemoryNode): string {
  if (typeof node === 'string') return 'str'
  if (node.kind === 'device') return node.type
  return node.kind
}

function preview(node: MemoryNode): string {
  if (typeof node === 'string') return node.length > 40 ? `${node.slice(0, 40)}…` : node
  if (node.kind === 'device') return node.preview ?? ''
  const count = node.kind === 'dict' ? Object.keys(node.children).length : node.children.length
  return `${count} 项`
}

/** 挂在这一层下面的设备（挂载键 'sys.log' 挂在 $MEM.sys 下）。 */
function devicesUnder(ref: string, devices: DeviceInfo[]): DeviceInfo[] {
  const prefix = ref === '$MEM' ? '' : ref.slice('$MEM.'.length)
  return devices.filter((device) => {
    const mount = device.mount
    const parent = mount.includes('.') ? mount.slice(0, mount.lastIndexOf('.')) : ''
    return parent === prefix
  })
}

export default function MemoryIndex({
  root,
  devices,
  busy,
  error,
  onRefresh,
  onSelect,
  onReadRef,
}: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['$MEM']))
  const [refInput, setRefInput] = useState('')
  const [readState, setReadState] = useState<{
    ref: string
    pending: boolean
    message: string | null
  }>({ ref: '', pending: false, message: null })
  const ctrlRef = useRef<AbortController | null>(null)
  const blocking = BLOCKING_REF.test(refInput.trim())

  const toggle = (ref: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(ref)) next.delete(ref)
      else next.add(ref)
      return next
    })
  }

  const read = async () => {
    const ref = refInput.trim()
    if (!ref) return
    ctrlRef.current?.abort()
    const ctrl = new AbortController()
    ctrlRef.current = ctrl
    const blockingRead = BLOCKING_REF.test(ref)
    setReadState({
      ref,
      pending: true,
      message: blockingRead ? '设备伪列表读：会等到有人喂输入，8 秒后放弃等待' : null,
    })
    let timedOut = false
    const timer = window.setTimeout(() => {
      timedOut = true
      ctrl.abort()
    }, READ_TIMEOUT_MS)
    try {
      const node = await onReadRef(ref, ctrl.signal)
      setReadState({ ref, pending: false, message: null })
      onSelect(ref, node)
    } catch (err) {
      setReadState({
        ref,
        pending: false,
        message: timedOut
          ? '等待超时：请求已放弃，系统那边仍在等这一读（先喂输入再读）'
          : describeError(err),
      })
    } finally {
      window.clearTimeout(timer)
    }
  }

  const cancel = () => {
    ctrlRef.current?.abort()
    setReadState((prev) => ({ ...prev, pending: false, message: '已取消等待' }))
  }

  const renderDevice = (device: DeviceInfo, depth: number) => {
    const ref = `$MEM.${device.mount}`
    return (
      <div key={ref} className="tree-node" style={{ paddingLeft: depth * 12 }}>
        <div className="tree-row" data-ref={ref}>
          <button
            type="button"
            className="tree-toggle"
            title={ref}
            onClick={() => onSelect(ref, device.description)}
          >
            · {device.mount}
          </button>
          <span className="muted small">device {device.type}</span>
          <button type="button" className="link" onClick={() => onSelect(ref, device.description)}>
            看
          </button>
        </div>
      </div>
    )
  }

  const renderNode = (ref: string, name: string, node: MemoryNode, depth: number) => {
    const isContainer = typeof node !== 'string' && node.kind !== 'device'
    const open = expanded.has(ref)
    return (
      <div key={ref} className="tree-node" style={{ paddingLeft: depth * 12 }}>
        <div className="tree-row" data-ref={ref}>
          <button
            type="button"
            className="tree-toggle"
            onClick={() => (isContainer ? toggle(ref) : onSelect(ref, node))}
            title={ref}
          >
            {isContainer ? (open ? '▾' : '▸') : '·'} {name}
          </button>
          <span className="muted small">
            {kindLabel(node)} {preview(node)}
          </span>
          <button type="button" className="link" onClick={() => onSelect(ref, node)}>
            看
          </button>
        </div>
        {isContainer && open ? (
          <div>
            {node.kind === 'dict'
              ? Object.entries((node as DictNode).children).map(([key, child]) =>
                  renderNode(`${ref}.${key}`, key, child, depth + 1),
                )
              : (node as ListNode).children.map((child, index) =>
                  renderNode(`${ref}.${index}`, String(index), child, depth + 1),
                )}
            {devicesUnder(ref, devices).map((device) => renderDevice(device, depth + 1))}
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className="pane" data-testid="memory-tree">
      <div className="pane-title">
        内存
        <button type="button" className="link" onClick={onRefresh} disabled={busy}>
          {busy ? '读取中…' : '刷新'}
        </button>
      </div>
      <div className="ref-reader">
        <input
          data-testid="ref-input"
          value={refInput}
          placeholder="$MEM.路径"
          onChange={(e) => setRefInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void read()
          }}
        />
        <button type="button" data-testid="ref-read" onClick={() => void read()} disabled={readState.pending}>
          读路径
        </button>
        {readState.pending ? (
          <button type="button" className="link" onClick={cancel}>
            取消
          </button>
        ) : null}
      </div>
      {blocking ? (
        <div className="hint warn" data-testid="ref-blocking-hint">
          末段是下标：若它落在设备上，这一读会阻塞（$MEM.inputs.-1 就是「请求一次操作员输入」）
        </div>
      ) : null}
      {readState.message ? <div className="hint warn">{readState.message}</div> : null}
      {error ? <div className="hint error">内存读取失败：{error}</div> : null}
      <div className="scroll">
        {root ? renderNode('$MEM', '$MEM', root, 0) : <div className="muted">尚未读取内存</div>}
      </div>
    </div>
  )
}

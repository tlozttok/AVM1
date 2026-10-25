import { useRef, useState } from 'react'
import { describeError } from '../api/client'
import type { DeviceInfo, DictNode, ListNode, MemoryNode } from '../api/types'

interface Props {
  root: MemoryNode | null
  /**
   * 已挂载设备（按作者 2026-10-13 的定夺：设备显示在树里，行上标出类型）。
   *
   * 树里的挂载点就是链接节点，左栏照树画：挂载点那行的位置来自树，类型取这里的 `type`
   * （取不到就退回链接本身的信息），`看` 打开设备视图——描述与动作由 `GET /api/devices` 提供。
   * 不再按 `devices` 往树里注入行：那样位置来自挂载键的父段，与树里的位置是两套算法。
   */
  devices: DeviceInfo[]
  busy: boolean
  error: string | null
  onRefresh: () => void
  onSelect: (ref: string, node: MemoryNode) => void
  onReadRef: (ref: string, signal: AbortSignal) => Promise<MemoryNode>
}

/** 设备伪列表的阻塞读取：末段是已知会阻塞的命名子键（当前只有 InputsListDevice 的 `next`）。 */
const BLOCKING_SUBKEYS = new Set(['next'])

function isBlockingRead(ref: string): boolean {
  const trimmed = ref.trim()
  const dot = trimmed.lastIndexOf('.')
  return dot >= 0 && BLOCKING_SUBKEYS.has(trimmed.slice(dot + 1))
}
const READ_TIMEOUT_MS = 8000

function kindLabel(node: MemoryNode): string {
  if (typeof node === 'string') return 'str'
  if (node.kind === 'device') return node.type
  return node.kind
}

function preview(node: MemoryNode): string {
  if (typeof node === 'string') return node.length > 40 ? `${node.slice(0, 40)}…` : node
  if (node.kind === 'device') return node.preview ?? ''
  // 链接节点没有 children：它是树里的一个键，读它才进入 target
  if (node.kind === 'link') return `→ ${node.target}`
  const count = node.kind === 'dict' ? Object.keys(node.children).length : node.children.length
  return `${count} 项`
}

/** 挂在这个引用上的设备（挂载点也是链接节点，行由树给出）。 */
function deviceAt(ref: string, devices: DeviceInfo[]): DeviceInfo | undefined {
  const mount = ref === '$MEM' ? '' : ref.slice('$MEM.'.length)
  return devices.find((device) => device.mount === mount)
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
  const blocking = isBlockingRead(refInput)

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
    const blockingRead = isBlockingRead(ref)
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

  const renderNode = (ref: string, name: string, node: MemoryNode, depth: number) => {
    // 链接节点：树里的键，没有子节点（读它才进入 target）。挂载点上就是设备行——
    // 位置来自树，行上标出设备类型，`看` 打开设备视图。
    if (typeof node !== 'string' && node.kind === 'link') {
      const device = deviceAt(ref, devices)
      const label = device ? `device ${device.type}` : `link → ${node.target}`
      return (
        <div key={ref} className="tree-node" style={{ paddingLeft: depth * 12 }}>
          <div className="tree-row" data-ref={ref}>
            <button
              type="button"
              className="tree-toggle"
              title={ref}
              onClick={() => onSelect(ref, device ? device.description : node)}
            >
              · {name}
            </button>
            <span className="muted small">{label}</span>
            <button
              type="button"
              className="link"
              onClick={() => onSelect(ref, device ? device.description : node)}
            >
              看
            </button>
          </div>
        </div>
      )
    }
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
          末段是已知会阻塞的命名子键（`$MEM.inputs.next` 就是「请求一次操作员输入」）：这一读会等到有人喂输入
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

import { useState } from 'react'
import type { ReactElement } from 'react'
import type { DeviceInfo, MemoryNode } from '../api/types'
import { ctrlTypeOf, nodeShape, shapeLabel } from '../lib/nodeModel'

interface Props {
  root: MemoryNode | null
  devices: DeviceInfo[]
  busy: boolean
  error: string | null
  dirty: boolean
  selectedRef: string | null
  onRefresh: () => void
  onSelect: (ref: string, node: MemoryNode) => void
  onCopyRef: (ref: string) => void
}

function mountedDevice(ref: string, devices: DeviceInfo[]): DeviceInfo | undefined {
  const mount = ref === '$MEM' ? '' : ref.slice('$MEM.'.length)
  return devices.find((device) => device.mount === mount)
}

function childEntries(node: MemoryNode): [string, MemoryNode][] {
  if (typeof node === 'string') return []
  if (node.kind === 'dict') return Object.entries(node.children)
  if (node.kind === 'list') return node.children.map((child, index) => [String(index), child])
  return []
}

export default function MemoryIndex({
  root,
  devices,
  busy,
  error,
  dirty,
  selectedRef,
  onRefresh,
  onSelect,
  onCopyRef,
}: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set(['$MEM']))

  const toggle = (ref: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(ref)) next.delete(ref)
      else next.add(ref)
      return next
    })
  }

  const renderNode = (ref: string, name: string, node: MemoryNode, depth: number): ReactElement => {
    const children = childEntries(node)
    const isContainer = children.length > 0
    const isOpen = expanded.has(ref)
    const device = typeof node !== 'string' && node.kind === 'link' ? mountedDevice(ref, devices) : undefined
    const shape = nodeShape(node)
    let kindText: string
    if (device) kindText = device.type
    else if (typeof node !== 'string' && node.kind === 'link') kindText = `→ ${node.target}`
    else kindText = shapeLabel(shape)
    const ctrlType = ctrlTypeOf(node)

    return (
      <div key={ref} className="mem-node">
        <div
          className={`rail-row mem-row${selectedRef === ref ? ' selected' : ''}`}
          data-ref={ref}
          style={{ paddingLeft: 10 + depth * 12 }}
          onClick={() => onSelect(ref, node)}
          title={ref}
        >
          {isContainer ? (
            <button
              type="button"
              className="mem-twist"
              aria-label={isOpen ? '收起' : '展开'}
              onClick={(event) => {
                event.stopPropagation()
                toggle(ref)
              }}
            >
              {isOpen ? '▾' : '▸'}
            </button>
          ) : (
            <span className="mem-twist mem-twist-empty">·</span>
          )}
          <span className="rail-name mono">{depth === 0 ? '$MEM' : name}</span>
          <span className={`mem-kind${device ? ' is-device' : ''}`}>{kindText}</span>
          {ctrlType ? <span className="mem-ctrl-type">{ctrlType}</span> : null}
          <button
            type="button"
            className="rail-copy"
            title={`复制 ${ref}`}
            onClick={(event) => {
              event.stopPropagation()
              onCopyRef(ref)
            }}
          >
            ⧉
          </button>
        </div>
        {isContainer && isOpen
          ? children.map(([key, child]) => renderNode(`${ref}.${key}`, key, child, depth + 1))
          : null}
      </div>
    )
  }

  return (
    <div className="rail-block rail-mem">
      <div className="rail-head">
        内存
        {dirty ? (
          <span className="rail-dirty" data-testid="memory-dirty" title="事件流里出现过内存类事件，树可能已旧">
            有变更
          </span>
        ) : null}
        <button type="button" className="link rail-refresh" onClick={onRefresh} disabled={busy}>
          {busy ? '读取中…' : '刷新'}
        </button>
      </div>
      {error ? <div className="rail-error">内存读取失败：{error}</div> : null}
      <div className="rail-scroll" data-testid="memory-tree">
        {root ? renderNode('$MEM', '$MEM', root, 0) : <div className="rail-empty">尚未读取内存</div>}
      </div>
    </div>
  )
}

import type { DictNode, ListNode, MemoryNode } from '../api/types'

function NodeBody({ node, depth }: { node: MemoryNode; depth: number }) {
  if (typeof node === 'string') {
    return <div className="value">{node === '' ? <span className="muted">（空字符串）</span> : node}</div>
  }
  if (node.kind === 'device') {
    return (
      <div className="kv">
        <div>
          <span className="key">设备类型</span>
          <span>{node.type}</span>
        </div>
        <div>
          <span className="key">挂载键</span>
          <span>{node.mount}</span>
        </div>
        <div>
          <span className="key">pretend_as_type</span>
          <span>{node.pretend_as_type ?? '—'}</span>
        </div>
        <div>
          <span className="key">LLM 表示</span>
          <span>{node.preview ?? '（取不到）'}</span>
        </div>
        {node.error ? <div className="hint error">{node.error}</div> : null}
      </div>
    )
  }
  const entries: [string, MemoryNode][] =
    node.kind === 'dict'
      ? Object.entries((node as DictNode).children)
      : (node as ListNode).children.map((child, index) => [String(index), child])
  return (
    <div>
      <div className="meta">
        <span className="key">metadata</span>
        <span>{node.metadata === null || node.metadata === undefined ? '—' : String(node.metadata)}</span>
        <span className="key">ctrl</span>
        <span>{node.ctrl ? JSON.stringify(node.ctrl) : '—'}</span>
      </div>
      {entries.length === 0 ? <div className="muted">空 {node.kind}</div> : null}
      <table className="grid">
        <tbody>
          {entries.map(([key, child]) => (
            <tr key={key}>
              <td className="key-cell">{key}</td>
              <td>
                {typeof child === 'string' || child.kind === 'device' ? (
                  <NodeBody node={child} depth={depth + 1} />
                ) : depth >= 2 ? (
                  <span className="muted">（更深一层，去左栏展开）</span>
                ) : (
                  <NodeBody node={child} depth={depth + 1} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function MemoryView({ refName, node }: { refName: string; node: MemoryNode }) {
  return (
    <div className="pane-body" data-testid="memory-view">
      <div className="detail-head">
        <span className="mono">{refName}</span>
      </div>
      <NodeBody node={node} depth={0} />
    </div>
  )
}

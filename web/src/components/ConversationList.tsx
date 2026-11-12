import type { ConversationBrief } from '../api/types'

const STATUS_LABEL: Record<string, string> = {
  active: '活跃',
  ready: '就绪',
  dormant: '休眠',
  finished: '结束',
  unknown: '未知',
}

interface Props {
  /** 已排除面板对话（操作员通道，不属于 guest 世界）。 */
  conversations: ConversationBrief[]
  selectedCid: number | null
  activeCid: number | null
  onSelect: (cid: number) => void
  onCopyCid: (cid: number) => void
}

interface Row {
  conv: ConversationBrief
  depth: number
}

/** 资源栏对话树：程序对话 / LLM 对话分组，create_cmd 的子对话按 parent 缩进。 */
function orderedRows(conversations: ConversationBrief[]): Row[] {
  const ids = new Set(conversations.map((conv) => conv.cid))
  const children = new Map<number | null, ConversationBrief[]>()
  for (const conv of conversations) {
    const parent = conv.parent !== null && ids.has(conv.parent) ? conv.parent : null
    const list = children.get(parent)
    if (list) list.push(conv)
    else children.set(parent, [conv])
  }
  for (const list of children.values()) list.sort((a, b) => a.cid - b.cid)
  const rows: Row[] = []
  const walk = (parent: number | null, depth: number) => {
    for (const conv of children.get(parent) ?? []) {
      rows.push({ conv, depth })
      walk(conv.cid, depth + 1)
    }
  }
  walk(null, 0)
  return rows
}

export default function ConversationList({
  conversations,
  selectedCid,
  activeCid,
  onSelect,
  onCopyCid,
}: Props) {
  const programs = orderedRows(conversations.filter((conv) => conv.is_python))
  const llms = orderedRows(conversations.filter((conv) => !conv.is_python))
  const groups: [string, Row[]][] = [
    ['程序对话', programs],
    ['LLM 对话', llms],
  ]

  return (
    <div className="rail-block rail-convs">
      <div className="rail-head">
        对话 <span className="rail-count" data-testid="conversation-count">{conversations.length}</span>
      </div>
      <div className="rail-scroll" data-testid="conversation-list">
        {conversations.length === 0 ? <div className="rail-empty">没有对话</div> : null}
        {groups.map(([label, rows]) =>
          rows.length === 0 ? null : (
            <div key={label}>
              <div className="rail-group">{label}</div>
              {rows.map(({ conv, depth }) => {
                const status = conv.cid === activeCid ? 'active' : conv.status
                return (
                  <div
                    key={conv.cid}
                    className={`rail-row${conv.cid === selectedCid ? ' selected' : ''}`}
                    data-testid={`conversation-item-${conv.cid}`}
                    style={depth > 0 ? { paddingLeft: 10 + depth * 14 } : undefined}
                    onClick={() => onSelect(conv.cid)}
                    title={`cid ${conv.cid}${conv.identity ? ` · ${conv.identity}` : ''}`}
                  >
                    <span className={`conv-dot is-${status}`} />
                    <span className="rail-name">{conv.name ?? conv.identity}</span>
                    <span className="rail-cid mono">{conv.cid}</span>
                    <span className="rail-meta">{STATUS_LABEL[status] ?? status}</span>
                    <button
                      type="button"
                      className="rail-copy"
                      title={`复制 cid ${conv.cid}`}
                      onClick={(event) => {
                        event.stopPropagation()
                        onCopyCid(conv.cid)
                      }}
                    >
                      ⧉
                    </button>
                  </div>
                )
              })}
            </div>
          ),
        )}
      </div>
    </div>
  )
}

import type { ConvStatus, ConversationBrief } from '../api/types'

const STATUS_LABEL: Record<ConvStatus, string> = {
  active: '活跃',
  ready: '就绪',
  dormant: '休眠',
  finished: '结束',
  unknown: '未知',
}

interface Props {
  conversations: ConversationBrief[]
  panelCid: number | null
  selectedCid: number | null
  /** 当前活跃 cid 与最近一条事件里的 cid：谁在跑、谁刚发了指令。 */
  activeCid: number | null
  recentCid: number | null
  onSelect: (cid: number) => void
  onCopy: (value: string) => void
}

export default function ConversationList({
  conversations,
  panelCid,
  selectedCid,
  activeCid,
  recentCid,
  onSelect,
  onCopy,
}: Props) {
  return (
    <div className="pane" data-testid="conversation-list">
      <div className="pane-title">
        对话
        <span className="muted">{conversations.length} 条</span>
      </div>
      <div className="scroll">
        <table className="grid">
          <thead>
            <tr>
              <th>cid</th>
              <th>名字</th>
              <th>状态</th>
              <th>消息</th>
            </tr>
          </thead>
          <tbody>
            {conversations.map((conv) => (
              <tr
                key={conv.cid}
                data-testid={`conversation-item-${conv.cid}`}
                className={conv.cid === selectedCid ? 'selected' : ''}
                onClick={() => onSelect(conv.cid)}
              >
                <td>
                  {conv.cid}
                  {conv.cid === panelCid ? <span className="tag">面板</span> : null}
                  {conv.cid === activeCid ? (
                    <span className="tag" data-testid="conversation-active">活跃</span>
                  ) : null}
                  {conv.cid === recentCid && conv.cid !== activeCid ? (
                    <span className="tag" data-testid="conversation-recent">最近事件</span>
                  ) : null}
                  <button
                    type="button"
                    className="link small"
                    data-testid="copy-conversation-cid"
                    title={`复制 cid ${conv.cid}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      onCopy(String(conv.cid))
                    }}
                  >
                    复制
                  </button>
                </td>
                <td title={conv.identity}>
                  {conv.name ?? conv.identity}
                  <span className="muted small">
                    {conv.is_root ? ' root' : ''}
                    {conv.is_sub ? ' sub' : ''}
                    {conv.is_python ? ' py' : ''}
                    {conv.parent !== null ? ` ←${conv.parent}` : ''}
                  </span>
                </td>
                <td>
                  <span className={`status status-${conv.status}`}>{STATUS_LABEL[conv.status]}</span>
                </td>
                <td>{conv.messages}</td>
              </tr>
            ))}
            {conversations.length === 0 ? (
              <tr>
                <td colSpan={4} className="muted">
                  没有对话
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  )
}

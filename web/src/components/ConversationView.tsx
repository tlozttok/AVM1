import { useEffect, useState } from 'react'
import { apiGet, describeError } from '../api/client'
import type { ConversationDetail } from '../api/types'

interface Props {
  cid: number
  /** 手动刷新计数：动作提交、点击刷新时 +1。 */
  token: number
  /** 该对话当前的消息条数：变化即自动重取，正在跑的对话因此跟得上。 */
  messageCount: number
}

export default function ConversationView({ cid, token, messageCount }: Props) {
  const [detail, setDetail] = useState<ConversationDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    apiGet<ConversationDetail>(`/api/conversations/${cid}`)
      .then((next) => {
        if (!alive) return
        setDetail(next)
        setError(null)
      })
      .catch((err) => {
        if (alive) setError(describeError(err))
      })
    return () => {
      alive = false
    }
  }, [cid, token, messageCount])

  if (error) return <div className="pane-body hint error">对话 {cid} 读取失败：{error}</div>
  if (!detail) return <div className="pane-body muted">读取对话 {cid}…</div>

  return (
    <div className="pane-body" data-testid="conversation-detail">
      <div className="detail-head">
        <span className="mono">cid {detail.cid}</span>
        <span>{detail.name ?? detail.identity}</span>
        <span className={`status status-${detail.status}`}>{detail.status}</span>
        {detail.is_python ? <span className="tag">python</span> : null}
        {detail.is_sub ? <span className="tag">sub</span> : null}
        {detail.parent !== null ? <span className="tag">父 {detail.parent}</span> : null}
        <span className="muted small">para_ref {detail.para_ref}</span>
      </div>

      <div className="section-title">历史（{detail.history.length}）</div>
      {detail.history.map((msg, index) => (
        <div className="message" key={index}>
          <div className="message-head">
            <span className={`role role-${msg.role}`}>{msg.role}</span>
            {msg.tool_call_id ? <span className="muted small">call {msg.tool_call_id}</span> : null}
          </div>
          {msg.content ? <pre className="content">{msg.content}</pre> : null}
          {msg.tool_calls ? (
            <pre className="content muted">{JSON.stringify(msg.tool_calls, null, 2)}</pre>
          ) : null}
        </div>
      ))}

      <div className="section-title">
        user_batch（外部输入 {detail.user_batch.user_contents.length} · 工具响应{' '}
        {detail.user_batch.tool_responses.length}）
      </div>
      {detail.user_batch.user_contents.map((content, index) => (
        <div className="message" key={`u${index}`}>
          <div className="message-head">
            <span className="role role-user">user</span>
          </div>
          <pre className="content">{content}</pre>
        </div>
      ))}
      {detail.user_batch.tool_responses.map((resp, index) => (
        <div className="message" key={`t${index}`}>
          <div className="message-head">
            <span className="role role-tool">tool</span>
            <span className="muted small">call {resp.tool_call_id}</span>
          </div>
          <pre className="content">{resp.content}</pre>
        </div>
      ))}

      <div className="section-title">metadata</div>
      <pre className="content muted">{JSON.stringify(detail.metadata, null, 2)}</pre>
    </div>
  )
}

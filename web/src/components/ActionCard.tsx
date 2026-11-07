import type { ActionTrace } from '../lib/correlation'

interface Props {
  trace: ActionTrace | null
}

/**
 * 一次动作提交的三态卡片。响应内容直接取 `tool_result.content`，不再去事件流里翻。
 * 「按序关联」是刻意的措辞：后端 /api/action 响应不带请求标识，配对是启发式的，
 * 连续快速提交、重连重放、seq 跳号时可能错位（见 lib/correlation.ts）。
 */
export default function ActionCard({ trace }: Props) {
  if (!trace) {
    return (
      <div className="hint muted small" data-testid="action-card">
        还没有提交动作。提交后这里会显示这次提交与工具响应（按序关联）。
      </div>
    )
  }

  let status: string
  let className = 'hint small'
  if (trace.phase === 'submitted') status = '已提交：等待面板对话发出工具调用'
  else if (trace.phase === 'running') status = `执行中：call ${trace.callId ?? '—'}，等待工具响应`
  else if (trace.phase === 'no_response') status = '无响应：未知工具名不产生工具响应'
  else {
    status = `响应：call ${trace.callId ?? '—'}`
    className = 'hint ok small'
  }

  return (
    <div className={className} data-testid="action-card">
      <div>
        <span className="mono">{trace.tool}</span> · {status} · <span className="muted">按序关联</span>
      </div>
      {trace.content !== null ? (
        <pre className="content muted" data-testid="action-card-content">
          {trace.content}
        </pre>
      ) : null}
    </div>
  )
}

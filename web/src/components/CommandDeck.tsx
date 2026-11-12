import { useEffect, useMemo, useRef, useState } from 'react'
import { apiGet, describeError } from '../api/client'
import type { ConversationBrief, ConversationDetail } from '../api/types'
import { useOperatorAction } from '../api/useOperatorAction'
import type { FrameState } from '../lib/frames'
import { closeConversation, sendInstruction } from '../lib/operatorActions'
import type { SystemStatus } from '../lib/systemStatus'
import { transcriptLines, type TranscriptLine } from '../lib/transcript'

const STATUS_LABEL: Record<string, string> = {
  active: '活跃',
  ready: '就绪',
  dormant: '休眠',
  finished: '结束',
  unknown: '未知',
}

interface Props {
  cid: number
  conversations: ConversationBrief[]
  status: SystemStatus
  frames: FrameState
  onRefreshState: () => void
  onClosed: (cid: number) => void
}

function lineClass(line: TranscriptLine): string {
  switch (line.kind) {
    case 'cmd':
      return 'transcript-line is-cmd'
    case 'error':
      return 'transcript-line is-error'
    case 'tools':
      return 'transcript-line is-tools'
    case 'pending':
      return 'transcript-line is-pending'
    default:
      return 'transcript-line is-out'
  }
}

/**
 * 命令台：选中对话的历史 + 一行输入。
 *
 * 发送 = 以面板对话 cid -1 的身份 `send_instruction`，随后 `resume` 让出执行权，
 * 目标跑完把回复写进自己的历史；历史在消息计数变化时重取，正在执行的外部输入
 * （user_batch）单列成「已投递、待执行」。
 */
export default function CommandDeck({
  cid,
  conversations,
  status,
  frames,
  onRefreshState,
  onClosed,
}: Props) {
  const conv = conversations.find((item) => item.cid === cid) ?? null
  const messageCount = conv?.messages ?? 0
  const [detail, setDetail] = useState<ConversationDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [refreshToken, setRefreshToken] = useState(0)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [confirmClose, setConfirmClose] = useState(false)
  const [sentId, setSentId] = useState<number | null>(null)
  const [closeId, setCloseId] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const { traces, submit } = useOperatorAction(frames)

  // 历史在「消息计数变化」时重取：轮询已经在 1s 节拍上带 messages，不额外起轮询。
  useEffect(() => {
    const controller = new AbortController()
    apiGet<ConversationDetail>(`/api/conversations/${cid}`, controller.signal)
      .then((next) => {
        setDetail(next)
        setDetailError(null)
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        setDetailError(describeError(err))
      })
    return () => controller.abort()
  }, [cid, messageCount, refreshToken])

  const { lines, hidden } = useMemo(() => transcriptLines(detail), [detail])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [lines.length, cid])

  // 投递结果：responded 后重取历史（待执行输入立刻出现），no_response 是未知工具的既有路径。
  useEffect(() => {
    if (sentId === null) return
    const trace = traces[sentId]
    if (!trace) return
    if (trace.phase === 'responded') {
      setNote(trace.content ?? '已投递')
      setSentId(null)
      setRefreshToken((token) => token + 1)
      onRefreshState()
    } else if (trace.phase === 'no_response') {
      setNote('已提交：Core 没有回工具响应（未知工具或缺响应）')
      setSentId(null)
    }
  }, [sentId, traces, onRefreshState])

  useEffect(() => {
    if (closeId === null) return
    const trace = traces[closeId]
    if (!trace) return
    if (trace.phase === 'responded' || trace.phase === 'no_response') {
      setCloseId(null)
      setConfirmClose(false)
      onRefreshState()
      onClosed(cid)
    }
  }, [closeId, traces, onRefreshState, onClosed, cid])

  const send = async () => {
    const content = text.trim()
    if (content === '' || busy || !status.canOperate) return
    setBusy(true)
    setError(null)
    setNote(null)
    try {
      const receipt = await submit(sendInstruction(cid, content))
      setText('')
      setSentId(receipt.id)
      if (receipt.warning) setNote(receipt.warning)
      // 让出执行权之后再取一次历史：执行前的待执行输入也能看到。
      window.setTimeout(() => setRefreshToken((token) => token + 1), 350)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const close = async () => {
    setBusy(true)
    setError(null)
    try {
      const receipt = await submit(closeConversation(cid))
      setCloseId(receipt.id)
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const finished = conv?.status === 'finished'
  const canType = status.canOperate && !finished

  return (
    <section className="workbench-view command-deck" data-testid="command-deck">
      <header className="workbench-head">
        <span className="workbench-title mono">{conv?.name ?? conv?.identity ?? `cid ${cid}`}</span>
        <span className="chip mono">cid {cid}</span>
        <span className="chip">{conv?.is_python ? 'python' : 'LLM'}</span>
        <span className={`chip status-${conv?.status ?? 'unknown'}`}>
          {STATUS_LABEL[conv?.status ?? 'unknown']}
        </span>
        <span className="workbench-spacer" />
        <span className="workbench-note">操作员以 <code>cid -1</code> 身份发 <code>send_instruction</code></span>
        {confirmClose ? (
          <span className="workbench-confirm">
            <span>关闭 cid {cid}？关闭后不可再调度。</span>
            <button type="button" className="danger" data-testid="command-close-confirm" onClick={() => void close()} disabled={busy}>
              确认关闭
            </button>
            <button type="button" className="link" onClick={() => setConfirmClose(false)}>取消</button>
          </span>
        ) : (
          <button
            type="button"
            data-testid="command-close"
            disabled={busy || !status.canOperate || finished}
            onClick={() => setConfirmClose(true)}
          >
            关闭对话
          </button>
        )}
      </header>

      <div className="transcript" ref={scrollRef} data-testid="command-transcript">
        {hidden > 0 ? (
          <div className="transcript-more">
            更早的 {hidden} 行未显示（完整历史在调试抽屉的对话视图）
          </div>
        ) : null}
        {detailError ? <div className="transcript-error">历史读取失败：{detailError}</div> : null}
        {detail === null && !detailError ? <div className="transcript-empty">读取对话历史…</div> : null}
        {detail !== null && lines.length === 0 ? (
          <div className="transcript-empty">还没有交互。输入一行命令，Enter 发送。</div>
        ) : null}
        {lines.map((line, index) =>
          line.kind === 'tools' ? (
            <details key={index} className={lineClass(line)}>
              <summary>{line.text}</summary>
              <div className="transcript-tool-note">工具调用与响应原文在调试抽屉的「对话原始详情」里。</div>
            </details>
          ) : line.kind === 'cmd' ? (
            <div key={index} className={lineClass(line)}>
              <span className="transcript-prompt mono">❯</span>
              {line.text}
            </div>
          ) : line.kind === 'pending' ? (
            <div key={index} className={lineClass(line)}>
              <span className="transcript-pending-tag">已投递、待执行</span>
              {line.text}
            </div>
          ) : (
            <div key={index} className={lineClass(line)}>{line.text}</div>
          ),
        )}
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          void send()
        }}
      >
        <span className="composer-prompt mono">❯</span>
        <textarea
          data-testid="command-input"
          rows={1}
          value={text}
          placeholder={canType ? '输入一行命令，Enter 发送…' : '当前不可发送'}
          disabled={!canType}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              void send()
            }
          }}
        />
        <button type="submit" className="primary" data-testid="command-send" disabled={busy || !canType || text.trim() === ''}>
          {busy ? '发送中…' : '发送'}
        </button>
      </form>
      <div className={`composer-hint${status.canOperate ? '' : ' is-warn'}`} data-testid="command-hint">
        {status.canOperate
          ? 'Enter 发送 · Shift+Enter 换行 · 发送后自动让出执行权，目标跑完回到这里'
          : `${status.label}：面板不可发送`}
        {note ? <span className="composer-note" data-testid="command-note"> · {note}</span> : null}
        {error ? <span className="composer-error" data-testid="command-error"> · {error}</span> : null}
      </div>
    </section>
  )
}

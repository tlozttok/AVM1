import { useCallback, useEffect, useState } from 'react'
import { apiGet, apiSend, describeError } from '../api/client'
import type { Listener, ListenersResponse } from '../api/types'

/** 只有这些字段能从宿主侧改写；后端 GET /api/listeners 给权威清单。 */
const DEFAULTS: Record<string, string> = {
  event_type: '', target: '', phase: 'all', handler_ref: '',
  tier: 'normal', priority: '0',
}

interface Draft {
  enabled: boolean
  [key: string]: string | boolean
}

function draftFrom(listener: Listener, fields: string[]): Draft {
  const draft: Draft = { enabled: Boolean(listener.enabled) }
  for (const field of fields) {
    if (field === 'enabled') continue
    const value = listener[field]
    draft[field] = value === null || value === undefined
      ? (DEFAULTS[field] ?? '')
      : String(value)
  }
  return draft
}

function fieldsToPayload(draft: Draft, original: Listener): Record<string, unknown> {
  const payload: Record<string, unknown> = {}
  for (const [field, raw] of Object.entries(draft)) {
    if (field === 'enabled') {
      if (raw !== Boolean(original.enabled)) payload.enabled = raw
      continue
    }
    let value: unknown = raw
    if (field === 'priority') value = Number(raw)
    else if (field === 'tier' || field === 'event_type' || field === 'target'
      || field === 'phase' || field === 'handler_ref') value = String(raw)
    const before = original[field]
    const beforeNorm = before === null || before === undefined ? '' : String(before)
    if (String(value) !== beforeNorm) payload[field] = value
  }
  return payload
}

/**
 * 调试抽屉里的监听器编辑：`GET /api/listeners` + `PATCH /api/listeners/<id>`。
 * 只改后端声明为可编辑的字段，其它字段（owner_cid / listener_id 等）只读展示。
 */
export default function ListenerEditor() {
  const [data, setData] = useState<ListenersResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState<number | null>(null)
  const [draft, setDraft] = useState<Draft>({ enabled: true })
  const [result, setResult] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      setData(await apiGet<ListenersResponse>('/api/listeners'))
      setError(null)
    } catch (err) {
      setError(describeError(err))
    }
  }, [])

  useEffect(() => { void load() }, [load])

  const fields = data?.editable_fields ?? []
  const startEdit = (listener: Listener) => {
    setEditing(listener.listener_id)
    setDraft(draftFrom(listener, fields))
    setResult(null)
  }
  const setField = (field: string, value: string | boolean) => {
    setDraft((prev) => ({ ...prev, [field]: value }))
  }
  const save = async (listener: Listener) => {
    const payload = fieldsToPayload(draft, listener)
    setBusy(true)
    setError(null)
    try {
      if (Object.keys(payload).length === 0) {
        setResult('没有改动')
      } else {
        await apiSend(`/api/listeners/${listener.listener_id}`, 'PATCH', payload)
        setResult(`已更新 #${listener.listener_id}（${Object.keys(payload).join('、')}）`)
        setEditing(null)
        await load()
      }
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="pane listener-editor" data-testid="listener-editor">
      <div className="pane-title">
        监听器编辑
        <span className="muted small">
          {data ? `${data.listeners.length} 条 · 可改 ${fields.join(' / ') || '—'}` : '读取中'}
        </span>
        <button type="button" className="link" data-testid="listener-refresh" onClick={() => void load()}>
          刷新
        </button>
      </div>
      <div className="pane-body listener-body">
        {error ? <div className="hint error" data-testid="listener-error">{error}</div> : null}
        {result ? <div className="hint ok" data-testid="listener-result">{result}</div> : null}
        {data && data.listeners.length === 0 ? (
          <div className="muted small">当前没有注册的监听器。</div>
        ) : null}
        {data?.listeners.map((listener) => (
          <div className="listener-row" key={listener.listener_id} data-testid="listener-row">
            <div className="listener-summary">
              <span className="mono">#{listener.listener_id}</span>
              <span>{String(listener.event_type)}</span>
              <span className="mono">{String(listener.target)}</span>
              <span>{String(listener.phase)}</span>
              <span className={listener.enabled ? 'hint ok inline' : 'hint warn inline'}>
                {listener.enabled ? '启用' : '停用'}
              </span>
              <button
                type="button"
                className="link"
                data-testid="listener-edit"
                onClick={() => startEdit(listener)}
              >
                编辑
              </button>
            </div>
            {editing === listener.listener_id ? (
              <div className="listener-form" data-testid="listener-form">
                {fields.map((field) => (
                  <label className="field" key={field}>
                    <span>{field}</span>
                    {field === 'enabled' ? (
                      <input
                        type="checkbox"
                        data-testid={`listener-field-${field}`}
                        checked={Boolean(draft.enabled)}
                        onChange={(e) => setField(field, e.target.checked)}
                      />
                    ) : field === 'phase' || field === 'tier' ? (
                      <select
                        data-testid={`listener-field-${field}`}
                        value={String(draft[field] ?? '')}
                        onChange={(e) => setField(field, e.target.value)}
                      >
                        {(field === 'phase'
                          ? ['before', 'after', 'all']
                          : ['normal', 'kernel']).map((option) => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={field === 'priority' ? 'number' : 'text'}
                        data-testid={`listener-field-${field}`}
                        value={String(draft[field] ?? '')}
                        onChange={(e) => setField(field, e.target.value)}
                      />
                    )}
                  </label>
                ))}
                <div className="composer-actions">
                  <button
                    type="button"
                    data-testid="listener-save"
                    disabled={busy}
                    onClick={() => void save(listener)}
                  >
                    保存
                  </button>
                  <button type="button" className="link" onClick={() => setEditing(null)}>
                    取消
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ToolTable } from '../api/tools'
import type { ActionResponse } from '../api/types'
import type { SubmittedAction } from '../lib/correlation'
import {
  buildArguments,
  fieldsForTool,
  initialValues,
  type FieldValue,
} from '../lib/schemaForm'

interface Props {
  tools: ToolTable
  /** 渲染时事件流已见的最大 seq：提交时快照下来，配对只在这之后找 tool_call。 */
  currentSeq: number
  /** 值暂存区当前选中的值；由操作员点选，前端不自动匹配字段。 */
  traySelected: string | null
  onSubmitted: (info: SubmittedAction) => void
}

interface Payload {
  tool: string
  arguments: Record<string, unknown>
}

/**
 * 指令表的结构化表单：控件由 `tool_def.parameters` 生成，不硬编码参数名。
 * 工具名是一个下拉选择（点开就列全指令表里的每一条）加一个「其它（手输）」：未知工具名是后端有意允许的，
 * 用来观察 Core 的错误路径；指令表读不到时下拉里只剩这一项，操作员照样能提交。
 * 「原始 JSON」开关切过去直接编辑 `{tool, arguments}`，用来提交表单表达不了的参数。
 *
 * 参数值只从两处来：操作员手打，或点参数框旁的「填入」把值暂存区选中的文本写进来。
 * 填入不猜字段语义，写完仍走 `buildArguments` / JSON 解析这两条既有构造路径。
 * 从暂存值填入过的提交会先在 `action-summary` 展示将发送的 payload，确认后才发 /api/action
 * （裁决五）；手填提交保持原来的一步路径，既有 e2e 三步用例不受影响。
 */

const CUSTOM_TOOL = '__custom__'

export default function ActionForm({ tools, currentSeq, traySelected, onSubmitted }: Props) {
  const [toolName, setToolName] = useState(() => tools.tools[0]?.function.name ?? 'memory_read')
  const [manualTool, setManualTool] = useState(false)
  const [values, setValues] = useState<Record<string, FieldValue>>({})
  // 按工具名分桶缓存已填内容：切工具不再把表单清空。
  const valueCache = useRef<Record<string, Record<string, FieldValue>>>({})
  const [jsonMode, setJsonMode] = useState(false)
  const [jsonText, setJsonText] = useState('{}')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [summary, setSummary] = useState<Payload | null>(null)
  const [needsConfirm, setNeedsConfirm] = useState(false)

  // 指令表读到（或换了版本）时，把不在表里的当前名字校正回第一项。
  // 依赖里没有 toolName：否则每敲一个字都会被顶回去，手输未知名就变成不可能。
  useEffect(() => {
    if (manualTool || tools.tools.length === 0) return
    setToolName((current) =>
      tools.tools.some((t) => t.function.name === current)
        ? current
        : tools.tools[0].function.name,
    )
  }, [tools, manualTool])

  const known = tools.tools.some((t) => t.function.name === toolName)
  const manual = manualTool || !known

  const selected = useMemo(
    () => tools.tools.find((t) => t.function.name === toolName),
    [tools, toolName],
  )
  const fields = useMemo(() => fieldsForTool(selected), [selected])

  // 指令表异步到达、或切换工具时，取该工具的缓存值；没有缓存才用初始值。
  useEffect(() => {
    setValues(valueCache.current[toolName] ?? initialValues(fields))
  }, [fields, toolName])

  const updateField = (name: string, value: FieldValue) => {
    setValues((prev) => {
      const next = { ...prev, [name]: value }
      valueCache.current = { ...valueCache.current, [toolName]: next }
      return next
    })
  }

  const switchTool = (next: string) => {
    valueCache.current = { ...valueCache.current, [toolName]: values }
    setToolName(next)
    setSummary(null)
  }

  const fillField = (name: string) => {
    if (traySelected === null) {
      setError('先在值暂存区选一条值')
      return
    }
    setError(null)
    updateField(name, traySelected)
    setNeedsConfirm(true)
  }

  const enterJson = () => {
    const built = buildArguments(fields, values)
    setJsonText(JSON.stringify({ tool: toolName, arguments: built.args }, null, 2))
    setJsonMode(true)
  }

  /** 回表单时回填可识别字段；有表外字段或 tool 不一致就留在 JSON 侧并给提示，不静默丢参数。 */
  const exitJson = () => {
    try {
      const parsed = JSON.parse(jsonText) as { tool?: unknown; arguments?: unknown }
      if (typeof parsed.tool !== 'string' || parsed.tool !== toolName) {
        setError(`原始 JSON 的 tool 与当前工具 ${toolName} 不一致，留在原始 JSON`)
        return
      }
      const args = parsed.arguments ?? {}
      if (args === null || typeof args !== 'object' || Array.isArray(args)) {
        setError('arguments 不是对象，留在原始 JSON')
        return
      }
      const pairs = Object.entries(args as Record<string, unknown>)
      const unknown = pairs.filter(([name]) => !fields.some((field) => field.name === name))
      if (unknown.length > 0) {
        setError(`字段 ${unknown.map(([name]) => name).join('、')} 不在表单里，留在原始 JSON`)
        return
      }
      const next = initialValues(fields)
      for (const [name, value] of pairs) {
        next[name] =
          typeof value === 'boolean' ? value : typeof value === 'string' ? value : JSON.stringify(value)
      }
      setValues(next)
      valueCache.current = { ...valueCache.current, [toolName]: next }
      setJsonMode(false)
      setError(null)
    } catch (err) {
      setError(`JSON 不合法：${(err as Error).message}，留在原始 JSON`)
    }
  }

  const buildPayload = (): Payload | null => {
    if (jsonMode) {
      try {
        const parsed = JSON.parse(jsonText) as { tool?: unknown; arguments?: unknown }
        if (typeof parsed.tool !== 'string' || parsed.tool === '') {
          setError('JSON 里缺字符串字段 tool')
          return null
        }
        return {
          tool: parsed.tool,
          arguments: (parsed.arguments ?? {}) as Record<string, unknown>,
        }
      } catch (err) {
        setError(`JSON 不合法：${(err as Error).message}`)
        return null
      }
    }
    const built = buildArguments(fields, values)
    if (built.errors.length > 0) {
      setError(built.errors.join('；'))
      return null
    }
    return { tool: toolName, arguments: built.args }
  }

  const send = async (payload: Payload) => {
    // seqAfter 必须在发请求之前快照：HTTP 返回时 SSE 可能已经把 tool_call 推过来了。
    const seqAfter = currentSeq
    setBusy(true)
    try {
      const resp = await apiSend<ActionResponse>('/api/action', 'POST', payload)
      const notes = [resp.auto_paused ? '未暂停，已自动中断并排队' : '面板持有执行权，直接提交']
      if (!resp.known_tool) notes.push('未知工具：Core 会在 stderr 报错，且不回工具响应')
      if (resp.warning) notes.push(resp.warning)
      const note = `已提交 ${resp.tool} · ${notes.join(' · ')}`
      setResult(note)
      onSubmitted({
        tool: payload.tool,
        args: payload.arguments,
        knownTool: resp.known_tool,
        seqAfter,
        note,
      })
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
  }

  const submit = () => {
    setError(null)
    setResult(null)
    const payload = buildPayload()
    if (!payload) return
    // 裁决五：从暂存值填入过的提交先看摘要；手填提交保持一步直达。
    if (needsConfirm) {
      setSummary(payload)
      return
    }
    void send(payload)
  }

  const confirm = () => {
    if (!summary) return
    const payload = summary
    setSummary(null)
    setNeedsConfirm(false)
    void send(payload)
  }

  return (
    <div className="action-form">
      <div className="section-title">
        动作
        <span className="muted small" data-testid="action-tools-source">
          {tools.error
            ? '指令表读取失败'
            : `指令表来源：GET /api/tools（${tools.tools.length} 条）`}
        </span>
      </div>
      {tools.error ? (
        <div className="hint error" data-testid="action-tools-error">
          {tools.error}——工具名仍可手输，或用「原始 JSON」提交
        </div>
      ) : null}
      <label className="field">
        <span>工具</span>
        <select
          data-testid="action-tool"
          value={manual ? CUSTOM_TOOL : toolName}
          onChange={(e) => {
            const next = e.target.value
            if (next === CUSTOM_TOOL) {
              setManualTool(true)
            } else {
              setManualTool(false)
              switchTool(next)
            }
          }}
        >
          {tools.tools.map((tool) => (
            <option key={tool.function.name} value={tool.function.name}>
              {tool.function.name}
            </option>
          ))}
          <option value={CUSTOM_TOOL}>其它（手输）</option>
        </select>
      </label>
      {manual ? (
        <label className="field">
          <span>工具名</span>
          <input
            data-testid="action-tool-manual"
            value={toolName}
            placeholder="未知名也可提交，用来看 Core 的错误路径"
            onChange={(e) => setToolName(e.target.value)}
          />
        </label>
      ) : null}
      {selected?.function.description ? (
        <div className="hint">{selected.function.description}</div>
      ) : null}

      {jsonMode ? (
        <label className="field">
          <span>原始 JSON（{'{tool, arguments}'}）</span>
          <textarea
            data-testid="action-json"
            rows={10}
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
          />
        </label>
      ) : (
        <>
          {fields.map((field) => (
            <label className="field" key={field.name}>
              <span>
                {field.name}
                {field.required ? <b className="required">*</b> : null}
              </span>
              {field.kind === 'checkbox' ? (
                <input
                  data-testid={`action-field-${field.name}`}
                  type="checkbox"
                  checked={values[field.name] === true}
                  onChange={(e) => updateField(field.name, e.target.checked)}
                />
              ) : (
                <div style={{ display: 'flex', gap: 4, alignItems: 'center', minWidth: 0 }}>
                  {field.kind === 'select' ? (
                    <select
                      data-testid={`action-field-${field.name}`}
                      value={String(values[field.name] ?? '')}
                      onChange={(e) => updateField(field.name, e.target.value)}
                    >
                      {field.options?.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : field.kind === 'json' ? (
                    <textarea
                      data-testid={`action-field-${field.name}`}
                      rows={3}
                      placeholder={field.placeholder}
                      value={String(values[field.name] ?? '')}
                      onChange={(e) => updateField(field.name, e.target.value)}
                    />
                  ) : (
                    <input
                      data-testid={`action-field-${field.name}`}
                      type={field.kind === 'number' ? 'number' : 'text'}
                      placeholder={field.placeholder}
                      value={String(values[field.name] ?? '')}
                      onChange={(e) => updateField(field.name, e.target.value)}
                    />
                  )}
                  <button
                    type="button"
                    className="link small"
                    style={{ flexShrink: 0 }}
                    data-testid={`action-fill-${field.name}`}
                    title="把值暂存区选中的文本填进这个字段"
                    onClick={() => fillField(field.name)}
                  >
                    填入
                  </button>
                </div>
              )}
              {field.description ? <span className="muted small">{field.description}</span> : null}
            </label>
          ))}
        </>
      )}

      <div className="row">
        <button type="button" data-testid="action-submit" onClick={submit} disabled={busy}>
          {busy ? '提交中…' : '提交动作'}
        </button>
        <button
          type="button"
          className="link"
          onClick={() => (jsonMode ? exitJson() : enterJson())}
        >
          {jsonMode ? '回到表单' : '原始 JSON'}
        </button>
      </div>
      {summary ? (
        <div className="hint warn" data-testid="action-summary">
          <div>将发送（确认后才 POST /api/action）：</div>
          <pre className="content muted">{JSON.stringify(summary, null, 2)}</pre>
          <div className="row" style={{ padding: '2px 0', gap: 8 }}>
            <button type="button" data-testid="action-summary-confirm" onClick={confirm}>
              确认提交
            </button>
            <button
              type="button"
              className="link"
              data-testid="action-summary-back"
              onClick={() => setSummary(null)}
            >
              返回修改
            </button>
            <button
              type="button"
              className="link"
              data-testid="action-summary-cancel"
              onClick={() => {
                setSummary(null)
                setNeedsConfirm(false)
              }}
            >
              取消
            </button>
          </div>
        </div>
      ) : null}
      {error ? (
        <div className="hint error" data-testid="action-error">
          {error}
        </div>
      ) : null}
      {result ? (
        <div className="hint ok" data-testid="action-result">
          {result}
        </div>
      ) : null}
    </div>
  )
}

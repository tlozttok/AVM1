import { useEffect, useMemo, useState } from 'react'
import { apiSend, describeError } from '../api/client'
import type { ToolTable } from '../api/tools'
import type { ActionResponse } from '../api/types'
import {
  buildArguments,
  fieldsForTool,
  initialValues,
  type FieldValue,
} from '../lib/schemaForm'

interface Props {
  tools: ToolTable
  onSubmitted: () => void
}

/**
 * 九条指令的结构化表单：控件由 `tool_def.parameters` 生成，不硬编码参数名。
 * 工具名是一个下拉选择（点开就列全九条）加一个「其它（手输）」：未知工具名是后端有意允许的，
 * 用来观察 Core 的错误路径；指令表读不到时下拉里只剩这一项，操作员照样能提交。
 * 「原始 JSON」开关切过去直接编辑 `{tool, arguments}`，用来提交表单表达不了的参数。
 */

const CUSTOM_TOOL = '__custom__'

export default function ActionForm({ tools, onSubmitted }: Props) {
  const [toolName, setToolName] = useState(() => tools.tools[0]?.function.name ?? 'memory_read')
  const [manualTool, setManualTool] = useState(false)
  const [values, setValues] = useState<Record<string, FieldValue>>({})
  const [jsonMode, setJsonMode] = useState(false)
  const [jsonText, setJsonText] = useState('{}')
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

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

  useEffect(() => {
    setValues(initialValues(fields))
  }, [fields])

  const enterJson = () => {
    const built = buildArguments(fields, values)
    setJsonText(JSON.stringify({ tool: toolName, arguments: built.args }, null, 2))
    setJsonMode(true)
  }

  const submit = async () => {
    setError(null)
    setResult(null)
    let payload: { tool: string; arguments: Record<string, unknown> }
    if (jsonMode) {
      try {
        const parsed = JSON.parse(jsonText) as { tool?: unknown; arguments?: unknown }
        if (typeof parsed.tool !== 'string' || parsed.tool === '') {
          setError('JSON 里缺字符串字段 tool')
          return
        }
        payload = {
          tool: parsed.tool,
          arguments: (parsed.arguments ?? {}) as Record<string, unknown>,
        }
      } catch (err) {
        setError(`JSON 不合法：${(err as Error).message}`)
        return
      }
    } else {
      const built = buildArguments(fields, values)
      if (built.errors.length > 0) {
        setError(built.errors.join('；'))
        return
      }
      payload = { tool: toolName, arguments: built.args }
    }

    setBusy(true)
    try {
      const resp = await apiSend<ActionResponse>('/api/action', 'POST', payload)
      const notes = [resp.auto_paused ? '未暂停，已自动中断并排队' : '面板持有执行权，直接提交']
      if (!resp.known_tool) notes.push('未知工具：Core 会在 stderr 报错，且不回工具响应')
      if (resp.warning) notes.push(resp.warning)
      setResult(`已提交 ${resp.tool} · ${notes.join(' · ')}`)
      onSubmitted()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setBusy(false)
    }
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
              setToolName(next)
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
                  onChange={(e) =>
                    setValues((prev) => ({ ...prev, [field.name]: e.target.checked }))
                  }
                />
              ) : field.kind === 'select' ? (
                <select
                  data-testid={`action-field-${field.name}`}
                  value={String(values[field.name] ?? '')}
                  onChange={(e) =>
                    setValues((prev) => ({ ...prev, [field.name]: e.target.value }))
                  }
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
                  onChange={(e) =>
                    setValues((prev) => ({ ...prev, [field.name]: e.target.value }))
                  }
                />
              ) : (
                <input
                  data-testid={`action-field-${field.name}`}
                  type={field.kind === 'number' ? 'number' : 'text'}
                  placeholder={field.placeholder}
                  value={String(values[field.name] ?? '')}
                  onChange={(e) =>
                    setValues((prev) => ({ ...prev, [field.name]: e.target.value }))
                  }
                />
              )}
              {field.description ? <span className="muted small">{field.description}</span> : null}
            </label>
          ))}
        </>
      )}

      <div className="row">
        <button type="button" data-testid="action-submit" onClick={() => void submit()} disabled={busy}>
          {busy ? '提交中…' : '提交动作'}
        </button>
        <button
          type="button"
          className="link"
          onClick={() => (jsonMode ? setJsonMode(false) : enterJson())}
        >
          {jsonMode ? '回到表单' : '原始 JSON'}
        </button>
      </div>
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

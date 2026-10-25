import type { JsonSchemaProp, ToolDef } from '../api/types'

/**
 * schema → 表单的纯映射。
 *
 * 输入是指令表里每条指令的 `tool_def.parameters`（`_build_tools()` 的原样导出），
 * 因此这里不硬编码任何一条指令的参数名。`anyOf` / `oneOf`（例如 `send_instruction.cid`
 * 的「整数或整数列表」）交给 JSON 文本域，由用户写成 `2` 或 `[0,1]`。
 */

export type FieldKind = 'text' | 'number' | 'checkbox' | 'json' | 'select'

export interface FormField {
  name: string
  kind: FieldKind
  required: boolean
  description: string
  options?: string[]
  placeholder?: string
}

export type FieldValue = string | boolean

export function kindOf(prop: JsonSchemaProp | undefined): FieldKind {
  if (!prop) return 'json'
  if (Array.isArray(prop.enum) && prop.enum.length > 0) return 'select'
  switch (prop.type) {
    case 'string':
      return 'text'
    case 'integer':
    case 'number':
      return 'number'
    case 'boolean':
      return 'checkbox'
    case 'object':
    case 'array':
      return 'json'
    default:
      return 'json'
  }
}

export function fieldsForTool(tool: ToolDef | undefined): FormField[] {
  const parameters = tool?.function.parameters
  const properties = parameters?.properties ?? {}
  const required = new Set(parameters?.required ?? [])
  return Object.entries(properties).map(([name, prop]) => ({
    name,
    kind: kindOf(prop),
    required: required.has(name),
    description: prop?.description ?? '',
    options: prop?.enum,
    placeholder:
      prop?.anyOf || prop?.oneOf ? '整数或 JSON 数组，例如 2 或 [0,1]' : '',
  }))
}

/** 初始值：枚举选第一项，布尔不勾，其余空。 */
export function initialValues(fields: FormField[]): Record<string, FieldValue> {
  const out: Record<string, FieldValue> = {}
  for (const field of fields) {
    if (field.kind === 'select') out[field.name] = field.options?.[0] ?? ''
    else if (field.kind === 'checkbox') out[field.name] = false
    else out[field.name] = ''
  }
  return out
}

export interface BuildResult {
  args: Record<string, unknown>
  errors: string[]
}

/**
 * 表单值 → arguments。
 *
 * - 文本/数字/JSON/枚举：空值不发送，必填为空报错；
 * - 勾选框：勾了才发送（例外是必填布尔，按勾选状态原样发送，好让 `false` 也能表达）；
 * - 想要发送空字符串或 `false`，切到「原始 JSON」写。
 */
export function buildArguments(
  fields: FormField[],
  values: Record<string, FieldValue>,
): BuildResult {
  const args: Record<string, unknown> = {}
  const errors: string[] = []
  for (const field of fields) {
    const raw = values[field.name]
    if (field.kind === 'checkbox') {
      const checked = raw === true
      if (checked || field.required) args[field.name] = checked
      continue
    }
    const text = typeof raw === 'string' ? raw.trim() : ''
    if (text === '') {
      if (field.required) errors.push(`${field.name} 必填`)
      continue
    }
    if (field.kind === 'number') {
      const num = Number(text)
      if (!Number.isFinite(num)) {
        errors.push(`${field.name} 需要数字，得到 ${text}`)
        continue
      }
      args[field.name] = num
      continue
    }
    if (field.kind === 'json') {
      try {
        args[field.name] = JSON.parse(text)
      } catch (err) {
        errors.push(`${field.name} 不是合法 JSON：${(err as Error).message}`)
      }
      continue
    }
    args[field.name] = text
  }
  return { args, errors }
}

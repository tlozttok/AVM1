/**
 * 值暂存区的纯逻辑：只存文本值，不做字段语义判断。
 *
 * 指令表的参数规格只有后端一份（`GET /api/tools`），前端不生成、不预填参数；
 * 这里记录的是操作员从界面上复制出来的地址、cid 与参数 JSON，由人点选填进某个输入框。
 * 状态留在 React 里，页面刷新即失：不写 localStorage、不发端点、不新增持久化文件。
 */

export type TrayLabel = 'ref' | 'cid' | 'json' | 'text'

export interface TrayValue {
  id: string
  value: string
  label: TrayLabel
  /** 复制时刻，仅用于界面上的顺序说明；限量淘汰按列表顺序，不按时间重排。 */
  time: number
}

export const TRAY_LIMIT = 8

export const TRAY_LABEL_TEXT: Record<TrayLabel, string> = {
  ref: '地址',
  cid: 'cid',
  json: 'JSON',
  text: '文本',
}

/** 标签只读文本形状，不查指令表、不猜这个值该填到哪个参数。 */
export function labelForValue(value: string): TrayLabel {
  const trimmed = value.trim()
  if (trimmed.startsWith('$')) return 'ref'
  if (/^\d+$/.test(trimmed)) return 'cid'
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      JSON.parse(trimmed)
      return 'json'
    } catch {
      return 'text'
    }
  }
  return 'text'
}

/**
 * 写入一条值：按文本去重（重复复制只把它提到最前），限量 8 条淘汰最旧。
 * 纯函数，返回新数组；空值不写。
 */
export function pushTrayValue(
  list: TrayValue[],
  value: string,
  label: TrayLabel = labelForValue(value),
  time: number = Date.now(),
): TrayValue[] {
  const text = value.trim()
  if (text === '') return list
  const item: TrayValue = { id: `${label}:${text}`, value: text, label, time }
  return [item, ...list.filter((entry) => entry.value !== text)].slice(0, TRAY_LIMIT)
}

/** 暂存区里当前选中的值：选中的那条被淘汰后回落到 null，由调用方决定。 */
export function selectedTrayValue(list: TrayValue[], selected: string | null): string | null {
  if (selected === null) return null
  return list.some((entry) => entry.value === selected) ? selected : null
}

/** 界面上的短标签：长 JSON 截断显示，填入时仍用完整值。 */
export function trayPreview(value: string, max = 42): string {
  const flat = value.replace(/\s+/g, ' ')
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

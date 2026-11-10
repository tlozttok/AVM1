import type { InputAccepted } from '../api/types'

/**
 * 正式输入（`POST /api/input`）的请求体。
 *
 * 后端把它包装成 `{$MEM.sys.inbox}` 的一条 JSON 字符串条目，`target` 缺省为
 * null（无目标输入）。这里只做形状归一，不解析、不改写文本。
 */
export interface InputRequestBody {
  text: string
  target: string | null
}

/** 组装请求体：空白文本不发；text 原样保留，目标空白归一成 null（入口）。 */
export function buildInputBody(text: string, target: string): InputRequestBody | null {
  if (text.trim() === '') return null
  const trimmedTarget = target.trim()
  return { target: trimmedTarget === '' ? null : trimmedTarget, text }
}

/** 202 回执的一行文案：面板显示"已接收"，并带上宿主生成的条目 id。 */
export function describeAccepted(accepted: InputAccepted): string {
  return `已接收 · 条目 ${accepted.entry_id}`
}

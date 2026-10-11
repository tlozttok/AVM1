import { apiGet, describeError } from './client'
import type { ToolDef } from './types'

/**
 * 指令表：console 的只读端点 `GET /api/tools`（后端 `_build_tools()`），前端不留镜像。
 *
 * 指令的参数 schema 只有后端一份。端点读不到时如实报错，不拿一份可能过期的第二份契约顶上——
 * 作者 2026-10-10 定夺：删掉镜像，面板的指令表以后端为准。
 */
export interface ToolTable {
  tools: ToolDef[]
  /** null 表示读到了；否则是要给操作员看的原因，界面照原样显示 */
  error: string | null
}

export async function loadTools(): Promise<ToolTable> {
  try {
    const resp = await apiGet<{ tools?: ToolDef[] }>('/api/tools')
    const tools = Array.isArray(resp?.tools) ? resp.tools : []
    if (tools.length === 0) {
      return { tools: [], error: 'GET /api/tools 返回了空指令表' }
    }
    return { tools, error: null }
  } catch (err) {
    return { tools: [], error: `读不到 GET /api/tools：${describeError(err)}` }
  }
}

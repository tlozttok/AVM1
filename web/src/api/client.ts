/**
 * 薄 HTTP 客户端。
 *
 * 错误一律是 `{"error": "<消息>"}`，状态码语义见后端说明：
 * 400 参数不合法、404 资源不存在（含屏蔽期间读到被摘除的键）、409 还原冲突、500 意外错误。
 * 404 在这些端点上是可预期的结果，不当异常弹窗，由调用方决定怎么显示。
 */

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function readError(resp: Response): Promise<string> {
  const text = await resp.text().catch(() => '')
  if (!text) return `HTTP ${resp.status}`
  try {
    const parsed = JSON.parse(text) as { error?: unknown }
    if (typeof parsed.error === 'string') return parsed.error
  } catch {
    // 不是 JSON：原样给调用方看
  }
  return text
}

async function parseJson<T>(resp: Response): Promise<T> {
  const text = await resp.text()
  if (!text) return undefined as T
  return JSON.parse(text) as T
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const resp = await fetch(path, { method: 'GET', signal })
  if (!resp.ok) throw new ApiError(resp.status, await readError(resp))
  return parseJson<T>(resp)
}

export async function apiSend<T>(
  path: string,
  method: 'POST' | 'PATCH' | 'DELETE',
  body?: unknown,
): Promise<T> {
  const resp = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, await readError(resp))
  return parseJson<T>(resp)
}

/** 统一的错误文案：ApiError 带状态码，其余（取消、网络）照原样。 */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) return `${err.status}: ${err.message}`
  if (err instanceof DOMException && err.name === 'AbortError') return '已取消'
  if (err instanceof Error) return err.message
  return String(err)
}

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

const TOKEN_KEY = 'avm.console.token'
const CLIENT_KEY = 'avm.console.client'

/** 面板凭证：fetch 用 Bearer 头，EventSource 没法带自定义头，靠同源 cookie 兜底。 */
export function currentToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

/**
 * 本次标签页的客户端身份：sessionStorage 里一份，刷新不变、新标签页另算。
 * 单客户端准入靠它区分面板；未携带身份的请求被后端当作宿主本地调用放行。
 */
export function currentClientId(): string {
  try {
    const saved = sessionStorage.getItem(CLIENT_KEY)
    if (saved) return saved
    const fresh =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `client-${Date.now()}-${Math.random().toString(36).slice(2)}`
    sessionStorage.setItem(CLIENT_KEY, fresh)
    return fresh
  } catch {
    return 'local'
  }
}

/** SSE 不能带自定义头：把客户端身份写进同源 cookie，让 /api/events 与 /api/io/stream 过准入。 */
export function saveClientCookie(clientId: string): void {
  document.cookie = `avm_client=${encodeURIComponent(clientId)}; path=/; SameSite=Strict`
}

export function clearClientCookie(): void {
  document.cookie = 'avm_client=; path=/; Max-Age=0; SameSite=Strict'
}

export function saveToken(token: string | null): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token)
    else sessionStorage.removeItem(TOKEN_KEY)
  } catch {
    // 隐私模式下 sessionStorage 不可用，cookie 仍可让 SSE 通过
  }
  document.cookie = token
    ? `avm_token=${encodeURIComponent(token)}; path=/; SameSite=Strict`
    : 'avm_token=; path=/; Max-Age=0; SameSite=Strict'
}

function authHeaders(): Record<string, string> {
  const token = currentToken()
  const headers: Record<string, string> = { 'X-AVM-Client': currentClientId() }
  if (token) headers.Authorization = `Bearer ${token}`
  return headers
}

function flagUnauthorized(status: number): void {
  if (status === 401) window.dispatchEvent(new Event('avm:unauthorized'))
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const resp = await fetch(path, { method: 'GET', signal, headers: authHeaders() })
  if (!resp.ok) {
    flagUnauthorized(resp.status)
    throw new ApiError(resp.status, await readError(resp))
  }
  return parseJson<T>(resp)
}

export async function apiSend<T>(
  path: string,
  method: 'POST' | 'PATCH' | 'DELETE',
  body?: unknown,
): Promise<T> {
  const resp = await fetch(path, {
    method,
    headers: {
      ...authHeaders(),
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!resp.ok) {
    flagUnauthorized(resp.status)
    throw new ApiError(resp.status, await readError(resp))
  }
  return parseJson<T>(resp)
}

/** 统一的错误文案：ApiError 带状态码，其余（取消、网络）照原样。 */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) return `${err.status}: ${err.message}`
  if (err instanceof DOMException && err.name === 'AbortError') return '已取消'
  if (err instanceof Error) return err.message
  return String(err)
}

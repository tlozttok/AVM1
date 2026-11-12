/**
 * 十条指令在界面上的高层动作：每条动作对应一次工具调用，参数在构造时就位。
 *
 * `release` 表示提交后是否立刻让出执行权（`POST /api/resume`）：主界面的对象操作
 * 与命令台都是「做完就继续跑」，调试抽屉的原始动作保留手动暂停/继续的语义。
 * 这里不打包宏：一条动作就是一条指令。
 */

export type MemType = 'str' | 'dict' | 'list'

export interface OperatorAction {
  tool: string
  arguments: Record<string, unknown>
  release: boolean
}

function action(tool: string, args: Record<string, unknown>, release = true): OperatorAction {
  return { tool, arguments: args, release }
}

export function sendInstruction(cid: number, content: string): OperatorAction {
  return action('send_instruction', { cid, content })
}

export function closeConversation(cid: number): OperatorAction {
  return action('close_conversation', { cid })
}

export function memoryWrite(ref: string, content: string): OperatorAction {
  return action('memory_write', { ref, content })
}

export function memoryMake(ref: string, key: string, memType: MemType): OperatorAction {
  return action('memory_make', { ref, key, mem_type: memType })
}

export function memoryDelete(ref: string): OperatorAction {
  return action('memory_delete', { ref })
}

export function ctrlSet(ref: string, key: string, value: string): OperatorAction {
  return action('edit_metadata', { ref, type: 'set', key, value })
}

export function ctrlDel(ref: string, key: string): OperatorAction {
  return action('edit_metadata', { ref, type: 'del', key })
}

export function createCmd(systemRef: string, paraRef: string): OperatorAction {
  return action('create_cmd', { system_ref: systemRef, para_ref: paraRef })
}

export function listenerRegister(handlerRef: string, priority = 0): OperatorAction {
  return action('listen_event', { action: 'listen', handler_ref: handlerRef, priority })
}

export function listenerCancel(listenerId: number): OperatorAction {
  return action('listen_event', { action: 'cancel', listener_id: listenerId })
}

/** 节点键名规则与指令边界一致：非空、不含点号。 */
export function isValidMemoryKey(key: string): boolean {
  const trimmed = key.trim()
  return trimmed !== '' && !trimmed.includes('.')
}

/**
 * 新建子节点时真正写进 `key` 的值：list 追加位置固定 `-1`，dict 用操作员填的键。
 * 返回 null 表示键名不合法。
 */
export function makeKeyFor(shape: 'dict' | 'list', key: string): string | null {
  if (shape === 'list') return '-1'
  const trimmed = key.trim()
  return isValidMemoryKey(trimmed) ? trimmed : null
}

/** 子节点引用：容器 ref 加上键（list 用下标）。 */
export function childRef(ref: string, key: string): string {
  return `${ref}.${key}`
}

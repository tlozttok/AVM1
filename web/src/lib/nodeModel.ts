import type { MemoryNode } from '../api/types'

/**
 * 节点工作台的三个维度：
 *   ① 节点类型（内存结构）：str / dict / list / link / device —— 决定编辑操作；
 *   ② 数据类型：ctrl.type（python / settingup / handler / para …）—— 决定快捷修改与类型动作；
 *   ③ 设备：挂载点，行为由设备自己声明。
 *
 * 注意 str 节点在 `/api/memory` 里是纯字符串：它的 ctrl 不可见（后端只导出正文）。
 * 这是当前 API 的边界，界面上如实标注，不在前端猜。
 */
export type NodeShape = 'str' | 'dict' | 'list' | 'link' | 'device'

export function nodeShape(node: MemoryNode): NodeShape {
  if (typeof node === 'string') return 'str'
  return node.kind
}

export function shapeLabel(shape: NodeShape): string {
  switch (shape) {
    case 'str':
      return 'str 叶子'
    case 'dict':
      return 'dict 容器'
    case 'list':
      return 'list 容器'
    case 'link':
      return 'link 链接'
    case 'device':
      return '设备挂载点'
  }
}

export function nodeDescription(node: MemoryNode): string {
  if (typeof node === 'string' || node.kind === 'device') return ''
  const meta = node.metadata
  if (meta === null || meta === undefined) return ''
  return typeof meta === 'string' ? meta : JSON.stringify(meta)
}

export function nodeCtrl(node: MemoryNode): Record<string, unknown> | null {
  if (typeof node === 'string' || node.kind === 'device') return null
  return node.ctrl ?? null
}

export function ctrlValue(ctrl: Record<string, unknown> | null, key: string): string | null {
  if (ctrl === null) return null
  const value = ctrl[key]
  if (value === null || value === undefined) return null
  const text = String(value)
  return text === '' ? null : text
}

export function ctrlTypeOf(node: MemoryNode): string | null {
  return ctrlValue(nodeCtrl(node), 'type')
}

/** 界面上标数据类型用的短文案：没有类型时明确写「无」。 */
export function nodeCtrlTypeLabel(type: string | null): string {
  return type === null ? '无 ctrl.type' : `ctrl.type=${type}`
}

export function isProgramType(type: string | null): boolean {
  return type === 'settingup' || type === 'python'
}

export function isHandlerType(type: string | null): boolean {
  return type === 'handler'
}

export function isParaType(type: string | null): boolean {
  return type === 'para'
}

/** 类型的中文说明：界面上解释「ctrl.type 是什么」。 */
export function typeSummary(type: string | null): string | null {
  switch (type) {
    case 'python':
      return 'ctrl.type="python"：已注册的 Python 程序，与 Settingup 对等；para 的 model 字段承担 PLM 命名。'
    case 'settingup':
      return 'ctrl.type="settingup"：已注册的提示词程序（系统提示词）；启动时由 para 的 model 指定 LLM。'
    case 'handler':
      return 'ctrl.type="handler"：事件监听处理器。content 是处理器代码，condition 数据叶子声明监听条件；事件触发时按 priority 降序执行，首个 allow / replace 短路。'
    case 'para':
      return 'ctrl.type="para"：LLM / PLM 的参数文件；str 子字段直接作为 API 参数（model 决定程序命名空间）。'
    default:
      return null
  }
}

export function childNode(node: MemoryNode, key: string): MemoryNode | undefined {
  if (typeof node === 'string') return undefined
  if (node.kind === 'dict') return node.children[key]
  if (node.kind === 'list') return node.children[Number(key)]
  return undefined
}

export function childString(node: MemoryNode, key: string): string | null {
  const child = childNode(node, key)
  return typeof child === 'string' ? child : null
}

export function paraRefOf(node: MemoryNode): string | null {
  return childString(node, 'para_ref')
}

export function childEntries(node: MemoryNode): [string, MemoryNode][] {
  if (typeof node === 'string') return []
  if (node.kind === 'dict') return Object.entries(node.children)
  if (node.kind === 'list') return node.children.map((child, index) => [String(index), child])
  return []
}

export interface StructureCheck {
  label: string
  ok: boolean
}

/** 数据结构校验：按数据类型检查必需子节点是否就位。只做本地可见范围的检查。 */
export function structureChecks(node: MemoryNode, type: string | null): StructureCheck[] {
  if (isProgramType(type)) {
    return [
      { label: 'content 存在且为 str（程序正文）', ok: childString(node, 'content') !== null },
      { label: 'para_ref 存在且为 str（参数引用）', ok: childString(node, 'para_ref') !== null },
    ]
  }
  if (isHandlerType(type)) {
    return [
      { label: 'content 存在且为 str（处理器代码）', ok: childString(node, 'content') !== null },
      { label: 'condition 存在（监听条件）', ok: childNode(node, 'condition') !== undefined },
    ]
  }
  if (isParaType(type)) {
    return [{ label: 'model 存在且为 str（命名空间）', ok: childString(node, 'model') !== null }]
  }
  return []
}

export interface QuickField {
  key: string
  label: string
  kind: 'text' | 'textarea'
  value: string
}

/**
 * 类型快捷修改：改的是「该类型最常改的字段」，落点仍是 `memory_write`（子 str 节点）。
 * 没有类型时不提供快捷修改，走区域一的通用编辑与区域二的 ctrl 编辑。
 */
export function quickFields(node: MemoryNode, type: string | null): QuickField[] {
  if (isProgramType(type)) {
    return [
      { key: 'content', label: '程序正文（content）', kind: 'textarea', value: childString(node, 'content') ?? '' },
      { key: 'para_ref', label: '参数引用（para_ref）', kind: 'text', value: childString(node, 'para_ref') ?? '' },
    ]
  }
  if (isHandlerType(type)) {
    return [
      { key: 'content', label: '处理器代码（content）', kind: 'textarea', value: childString(node, 'content') ?? '' },
    ]
  }
  if (isParaType(type)) {
    return childEntries(node)
      .filter(([, child]) => typeof child === 'string')
      .slice(0, 8)
      .map(([key, child]) => ({ key, label: key, kind: 'text' as const, value: child as string }))
  }
  return []
}

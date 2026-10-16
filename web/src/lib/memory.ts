import type { MemoryNode } from '../api/types'

/** 按 `$MEM.a.b.0` 这样的引用在已取回的整棵树里定位节点；定位不到返回 null。 */
export function resolveRef(root: MemoryNode, ref: string): MemoryNode | null {
  if (!ref.startsWith('$MEM')) return null
  const parts = ref.split('.').slice(1)
  let node: MemoryNode = root
  for (const part of parts) {
    if (typeof node === 'string' || node.kind === 'device') return null
    // 链接节点在树里没有子节点：能定位到它自己，再往下不跟随（跟随是寻址的事）
    if (node.kind === 'link') return null
    if (node.kind === 'dict') {
      const next = node.children[part]
      if (next === undefined) return null
      node = next
    } else {
      const index = Number(part)
      if (!Number.isInteger(index) || index < 0 || index >= node.children.length) return null
      node = node.children[index]
    }
  }
  return node
}

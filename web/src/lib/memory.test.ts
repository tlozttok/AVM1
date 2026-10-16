import { describe, expect, it } from 'vitest'
import { resolveRef } from './memory'
import type { LinkNode, MemoryNode } from '../api/types'

const link: LinkNode = { kind: 'link', target: '$MEM.note', metadata: '链接', ctrl: null }

const root: MemoryNode = {
  kind: 'dict',
  metadata: null,
  ctrl: null,
  children: {
    note: {
      kind: 'dict',
      metadata: '一条笔记',
      ctrl: { owner: '0' },
      children: { v: 'hi' },
    },
    items: { kind: 'list', metadata: null, ctrl: null, children: ['a', 'b'] },
    alias: link,
  },
}

describe('resolveRef', () => {
  it('按段定位 dict 与 list 子节点', () => {
    expect(resolveRef(root, '$MEM.note.v')).toBe('hi')
    expect(resolveRef(root, '$MEM.items.1')).toBe('b')
    expect(resolveRef(root, '$MEM.items.2')).toBeNull()
    expect(resolveRef(root, '$MEM.note.ghost')).toBeNull()
  })

  it('链接节点本身可定位，再往下不跟随（树里没有子节点）', () => {
    expect(resolveRef(root, '$MEM.alias')).toBe(link)
    expect(resolveRef(root, '$MEM.alias.v')).toBeNull()
  })

  it('设备节点与越过叶子的段定位不到', () => {
    const withDevice = {
      kind: 'dict',
      metadata: null,
      ctrl: null,
      children: {
        dev: {
          kind: 'device',
          type: 'BoxDevice',
          mount: 'dev',
          pretend_as_type: null,
          preview: 'box',
        },
        leaf: 'text',
      },
    } as MemoryNode
    expect(resolveRef(withDevice, '$MEM.dev')).not.toBeNull()
    expect(resolveRef(withDevice, '$MEM.dev.inner')).toBeNull()
    expect(resolveRef(withDevice, '$MEM.leaf.deep')).toBeNull()
  })
})

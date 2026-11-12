import { describe, expect, it } from 'vitest'
import type { DictNode, MemoryNode } from '../api/types'
import {
  childString,
  ctrlTypeOf,
  isHandlerType,
  isParaType,
  isProgramType,
  nodeCtrl,
  nodeCtrlTypeLabel,
  nodeDescription,
  nodeShape,
  paraRefOf,
  quickFields,
  shapeLabel,
  structureChecks,
} from './nodeModel'

function dict(children: Record<string, MemoryNode>, ctrl: Record<string, unknown> | null = null): DictNode {
  return { kind: 'dict', metadata: '描述', ctrl, children }
}

describe('nodeModel', () => {
  it('节点类型与描述', () => {
    expect(nodeShape('x')).toBe('str')
    expect(nodeShape({ kind: 'list', metadata: null, ctrl: null, children: [] })).toBe('list')
    expect(nodeShape({ kind: 'link', target: '$MEM.a', metadata: null, ctrl: null })).toBe('link')
    expect(shapeLabel('dict')).toBe('dict 容器')
    expect(nodeDescription(dict({}))).toBe('描述')
    expect(nodeDescription('x')).toBe('')
    expect(nodeCtrl(dict({}))).toBeNull()
    expect(nodeCtrl('x')).toBeNull()
  })

  it('数据类型来自 ctrl.type', () => {
    const program = dict({ content: 'code', para_ref: '$MEM.para.python' }, { type: 'python' })
    expect(ctrlTypeOf(program)).toBe('python')
    expect(isProgramType(ctrlTypeOf(program))).toBe(true)
    expect(isHandlerType('handler')).toBe(true)
    expect(isParaType('para')).toBe(true)
    expect(nodeCtrlTypeLabel('python')).toBe('ctrl.type=python')
    expect(nodeCtrlTypeLabel(null)).toBe('无 ctrl.type')
  })

  it('子节点与 para_ref', () => {
    const program = dict({ content: 'code', para_ref: '$MEM.para.python', name: 'shell' }, { type: 'python' })
    expect(childString(program, 'content')).toBe('code')
    expect(childString(program, 'missing')).toBeNull()
    expect(paraRefOf(program)).toBe('$MEM.para.python')
  })

  it('结构校验按数据类型给', () => {
    const program = dict({ content: 'code', para_ref: '$MEM.p' }, { type: 'python' })
    expect(structureChecks(program, 'python').every((check) => check.ok)).toBe(true)
    const broken = dict({ content: 'code' }, { type: 'python' })
    expect(structureChecks(broken, 'python').filter((check) => !check.ok)).toHaveLength(1)
    const handler = dict({ content: 'code', condition: dict({}) }, { type: 'handler' })
    expect(structureChecks(handler, 'handler').every((check) => check.ok)).toBe(true)
    const para = dict({ model: 'plm.command' }, { type: 'para' })
    expect(structureChecks(para, 'para')).toEqual([{ label: 'model 存在且为 str（命名空间）', ok: true }])
  })

  it('快捷修改字段按数据类型给', () => {
    const program = dict({ content: 'code', para_ref: '$MEM.p' }, { type: 'python' })
    expect(quickFields(program, 'python').map((field) => field.key)).toEqual(['content', 'para_ref'])
    const handler = dict({ content: 'code' }, { type: 'handler' })
    expect(quickFields(handler, 'handler').map((field) => field.key)).toEqual(['content'])
    const para = dict({ model: 'plm.command', temperature: '0.2' }, { type: 'para' })
    expect(quickFields(para, 'para').map((field) => field.key)).toEqual(['model', 'temperature'])
    expect(quickFields(dict({}), null)).toEqual([])
  })
})

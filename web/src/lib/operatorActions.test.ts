import { describe, expect, it } from 'vitest'
import {
  childRef,
  closeConversation,
  createCmd,
  ctrlDel,
  ctrlSet,
  isValidMemoryKey,
  listenerCancel,
  listenerRegister,
  makeKeyFor,
  memoryDelete,
  memoryMake,
  memoryWrite,
  sendInstruction,
} from './operatorActions'

describe('operatorActions', () => {
  it('命令台发话：send_instruction 默认让出执行权', () => {
    expect(sendInstruction(5, 'help')).toEqual({
      tool: 'send_instruction',
      arguments: { cid: 5, content: 'help' },
      release: true,
    })
  })

  it('内存对象动作：写 / 建 / 删 / ctrl', () => {
    expect(memoryWrite('$MEM.a', 'v').arguments).toEqual({ ref: '$MEM.a', content: 'v' })
    expect(memoryMake('$MEM.a', 'b', 'str').arguments).toEqual({ ref: '$MEM.a', key: 'b', mem_type: 'str' })
    expect(memoryDelete('$MEM.a.b').arguments).toEqual({ ref: '$MEM.a.b' })
    expect(ctrlSet('$MEM.a', 'type', 'python').arguments).toEqual({
      ref: '$MEM.a', type: 'set', key: 'type', value: 'python',
    })
    expect(ctrlDel('$MEM.a', 'type').arguments).toEqual({ ref: '$MEM.a', type: 'del', key: 'type' })
  })

  it('对话与监听动作', () => {
    expect(closeConversation(7).arguments).toEqual({ cid: 7 })
    expect(createCmd('$MEM.p', '$MEM.para').arguments).toEqual({
      system_ref: '$MEM.p', para_ref: '$MEM.para',
    })
    expect(listenerRegister('$MEM.h').arguments).toEqual({
      action: 'listen', handler_ref: '$MEM.h', priority: 0,
    })
    expect(listenerCancel(3).arguments).toEqual({ action: 'cancel', listener_id: 3 })
  })

  it('键名规则与 list 追加', () => {
    expect(isValidMemoryKey('hello')).toBe(true)
    expect(isValidMemoryKey(' a.b ')).toBe(false)
    expect(isValidMemoryKey('  ')).toBe(false)
    expect(makeKeyFor('dict', ' hello ')).toBe('hello')
    expect(makeKeyFor('dict', 'a.b')).toBeNull()
    expect(makeKeyFor('list', 'anything')).toBe('-1')
    expect(childRef('$MEM.a', '-1')).toBe('$MEM.a.-1')
  })
})

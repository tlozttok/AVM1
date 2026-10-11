import { describe, expect, it } from 'vitest'
import type { ToolDef } from '../api/types'
import { buildArguments, fieldsForTool, initialValues, kindOf } from './schemaForm'

const writeTool: ToolDef = {
  type: 'function',
  function: {
    name: 'memory_write',
    parameters: {
      type: 'object',
      properties: {
        ref: { type: 'string', description: '内存引用' },
        content: { type: 'string' },
      },
      required: ['ref', 'content'],
    },
  },
}

const makeTool: ToolDef = {
  type: 'function',
  function: {
    name: 'memory_make',
    parameters: {
      type: 'object',
      properties: {
        ref: { type: 'string' },
        key: { type: 'string' },
        mem_type: { type: 'string', enum: ['str', 'dict', 'list'] },
      },
      required: ['ref', 'key', 'mem_type'],
    },
  },
}

const multicastTool: ToolDef = {
  type: 'function',
  function: {
    name: 'send_instruction',
    parameters: {
      type: 'object',
      properties: {
        cid: { anyOf: [{ type: 'integer' }, { type: 'array', items: { type: 'integer' } }] },
        content: { type: 'string' },
        once: { type: 'boolean' },
      },
      required: ['cid', 'content'],
    },
  },
}

describe('kindOf', () => {
  it('按 schema 选控件；anyOf 交给 JSON 文本域', () => {
    expect(kindOf({ type: 'string' })).toBe('text')
    expect(kindOf({ type: 'integer' })).toBe('number')
    expect(kindOf({ type: 'boolean' })).toBe('checkbox')
    expect(kindOf({ type: 'object' })).toBe('json')
    expect(kindOf({ type: 'string', enum: ['a'] })).toBe('select')
    expect(kindOf({ anyOf: [{ type: 'integer' }] })).toBe('json')
    expect(kindOf(undefined)).toBe('json')
  })
})

describe('fieldsForTool', () => {
  it('生成字段并标出必填', () => {
    const fields = fieldsForTool(writeTool)
    expect(fields.map((f) => f.name)).toEqual(['ref', 'content'])
    expect(fields.every((f) => f.required && f.kind === 'text')).toBe(true)
    expect(fields[0].description).toBe('内存引用')
  })

  it('枚举字段带上选项', () => {
    const fields = fieldsForTool(makeTool)
    const memType = fields.find((f) => f.name === 'mem_type')
    expect(memType?.kind).toBe('select')
    expect(memType?.options).toEqual(['str', 'dict', 'list'])
  })

  it('没有 schema 的工具给空表', () => {
    expect(fieldsForTool(undefined)).toEqual([])
    expect(fieldsForTool({ type: 'function', function: { name: 'x' } })).toEqual([])
  })
})

describe('initialValues', () => {
  it('枚举默认第一项，其余为空', () => {
    const values = initialValues(fieldsForTool(makeTool))
    expect(values.mem_type).toBe('str')
    expect(values.ref).toBe('')
  })
})

describe('buildArguments', () => {
  it('必填缺失报错，不发送空值', () => {
    const fields = fieldsForTool(writeTool)
    const built = buildArguments(fields, { ref: '$MEM.note', content: '' })
    expect(built.args).toEqual({ ref: '$MEM.note' })
    expect(built.errors).toEqual(['content 必填'])
  })

  it('数字字段转成数字，非法数字报错', () => {
    const fields = [{ name: 'cid', kind: 'number' as const, required: true, description: '' }]
    expect(buildArguments(fields, { cid: '3' }).args).toEqual({ cid: 3 })
    expect(buildArguments(fields, { cid: 'x' }).errors).toEqual(['cid 需要数字，得到 x'])
  })

  it('JSON 字段解析失败报错，成功则发送解析后的值', () => {
    const fields = [{ name: 'cid', kind: 'json' as const, required: true, description: '' }]
    expect(buildArguments(fields, { cid: '[0,1]' }).args).toEqual({ cid: [0, 1] })
    expect(buildArguments(fields, { cid: '2' }).args).toEqual({ cid: 2 })
    expect(buildArguments(fields, { cid: '{oops' }).errors[0]).toContain('不是合法 JSON')
  })

  it('勾选框：勾了才发送；必填布尔按原样发送，好让 false 也能表达', () => {
    const optional = [{ name: 'once', kind: 'checkbox' as const, required: false, description: '' }]
    const required = [{ name: 'once', kind: 'checkbox' as const, required: true, description: '' }]
    expect(buildArguments(optional, { once: false }).args).toEqual({})
    expect(buildArguments(optional, { once: true }).args).toEqual({ once: true })
    expect(buildArguments(required, { once: false }).args).toEqual({ once: false })
  })

  it('多播的 cid 字段可以写成整数或整数列表', () => {
    const fields = fieldsForTool(multicastTool)
    expect(buildArguments(fields, { cid: '[0,2]', content: '你好' }).args)
      .toEqual({ cid: [0, 2], content: '你好' })
  })
})

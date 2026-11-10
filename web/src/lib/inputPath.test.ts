import { describe, expect, it } from 'vitest'
import { buildInputBody, describeAccepted } from './inputPath'

describe('buildInputBody', () => {
  it('空白文本不产生请求体（按钮禁用与发送判定同一套）', () => {
    expect(buildInputBody('', '')).toBeNull()
    expect(buildInputBody('  \n\t ', '7')).toBeNull()
  })

  it('目标为空或空白时归一成 null（入口）', () => {
    expect(buildInputBody('你好', '')).toEqual({ text: '你好', target: null })
    expect(buildInputBody('你好', '  ')).toEqual({ text: '你好', target: null })
  })

  it('目标原样保留；文本不 trim、不改写', () => {
    expect(buildInputBody('  第一行\n第二行  ', ' 7 ')).toEqual({
      text: '  第一行\n第二行  ',
      target: '7',
    })
  })
})

describe('describeAccepted', () => {
  it('文案说明已接收并带上条目 id', () => {
    const text = describeAccepted({ accepted: true, entry_id: 'in-1-1' })
    expect(text).toContain('已接收')
    expect(text).toContain('in-1-1')
  })
})

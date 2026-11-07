import { describe, expect, it } from 'vitest'
import {
  TRAY_LIMIT,
  labelForValue,
  pushTrayValue,
  selectedTrayValue,
  trayPreview,
  type TrayValue,
} from './valueTray'

describe('labelForValue', () => {
  it('按文本形状给标签，不查指令表', () => {
    expect(labelForValue('$MEM.note.v')).toBe('ref')
    expect(labelForValue('  12  ')).toBe('cid')
    expect(labelForValue('{\n  "ref": "$MEM.note"\n}')).toBe('json')
    expect(labelForValue('[0,1]')).toBe('json')
    expect(labelForValue('{坏掉的 JSON')).toBe('text')
    expect(labelForValue('来自面板的一次写入')).toBe('text')
  })
})

describe('pushTrayValue', () => {
  it('新值排在最前，重复复制去重并提到最前', () => {
    let list: TrayValue[] = []
    list = pushTrayValue(list, '$MEM.note', undefined, 1)
    list = pushTrayValue(list, '3', undefined, 2)
    expect(list.map((v) => v.value)).toEqual(['3', '$MEM.note'])
    list = pushTrayValue(list, '$MEM.note', undefined, 3)
    expect(list.map((v) => v.value)).toEqual(['$MEM.note', '3'])
    expect(list[0].time).toBe(3)
  })

  it('超过 8 条淘汰最旧，空值不写入', () => {
    let list: TrayValue[] = []
    for (let i = 0; i < TRAY_LIMIT + 4; i += 1) {
      list = pushTrayValue(list, `$MEM.k${i}`, undefined, i)
    }
    expect(list).toHaveLength(TRAY_LIMIT)
    expect(list[0].value).toBe(`$MEM.k${TRAY_LIMIT + 3}`)
    expect(list[list.length - 1].value).toBe('$MEM.k4')
    const before = list
    expect(pushTrayValue(before, '   ', undefined, 99)).toBe(before)
  })

  it('整段 arguments JSON 作为一条值保存，不拆字段', () => {
    const json = JSON.stringify({ ref: '$MEM.note', content: 'x' }, null, 2)
    const list = pushTrayValue([], json, undefined, 1)
    expect(list).toHaveLength(1)
    expect(list[0].label).toBe('json')
    expect(list[0].value).toBe(json)
  })
})

describe('selectedTrayValue / trayPreview', () => {
  it('选中值被淘汰后回落 null', () => {
    const list = pushTrayValue([], '$MEM.note', undefined, 1)
    expect(selectedTrayValue(list, '$MEM.note')).toBe('$MEM.note')
    expect(selectedTrayValue(list, '$MEM.gone')).toBeNull()
    expect(selectedTrayValue([], null)).toBeNull()
  })

  it('长值截断显示，短值原样', () => {
    expect(trayPreview('$MEM.note')).toBe('$MEM.note')
    expect(trayPreview('x'.repeat(80)).endsWith('…')).toBe(true)
    expect(trayPreview('{\n "a": 1\n}')).toBe('{ "a": 1 }')
  })
})

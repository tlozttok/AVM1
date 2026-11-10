import { describe, expect, it } from 'vitest'
import type { DeviceInspection } from '../api/types'
import {
  buildLoadDeviceCommand,
  previousHash,
  shortHash,
} from './deviceApproval'

function inspection(overrides: Partial<DeviceInspection> = {}): DeviceInspection {
  return {
    path: '/image/echo.py',
    size: 120,
    sha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    class: 'EchoDevice',
    mount: null,
    args: null,
    existing: null,
    ...overrides,
  }
}

describe('shortHash', () => {
  it('取前 12 位，短哈希不越界', () => {
    expect(shortHash('abcdef0123456789')).toBe('abcdef012345')
    expect(shortHash('abc')).toBe('abc')
  })
})

describe('buildLoadDeviceCommand', () => {
  it('把审批哈希与挂载信息写进 load_device 命令', () => {
    const command = JSON.parse(
      buildLoadDeviceCommand({
        inspection: inspection(),
        mount: 'extra',
        className: 'EchoDevice',
        args: { value: '热装' },
        approvedAt: '2026-11-11T00:00:00+0800',
        approvedVia: 'client-1',
      }),
    )
    expect(command).toMatchObject({
      action: 'load_device',
      path: 'extra',
      file: '/image/echo.py',
      class: 'EchoDevice',
      args: { value: '热装' },
      approved_sha256: inspection().sha256,
      approved_at: '2026-11-11T00:00:00+0800',
      approved_via: 'client-1',
    })
  })

  it('不会把 password 写进命令（密码字段已从 loader 移除）', () => {
    const command = buildLoadDeviceCommand({
      inspection: inspection(),
      mount: 'extra',
      className: 'EchoDevice',
      args: {},
      approvedAt: 't',
      approvedVia: 'v',
    })
    expect(command).not.toContain('password')
  })
})

describe('previousHash', () => {
  it('首次安装没有旧哈希', () => {
    expect(previousHash(inspection())).toBeNull()
  })

  it('覆盖旧版本时返回旧记录哈希', () => {
    expect(
      previousHash(inspection({ existing: { mount: 'extra', sha256: 'old' } })),
    ).toBe('old')
  })
})

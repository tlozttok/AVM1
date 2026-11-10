/**
 * 设备审批的纯逻辑：哈希展示与 `memory_write $MEM.loader` 命令构造。
 *
 * 后端契约：审批值就是设备文件的内容哈希（`sha256`），装载时内核会重新计算并核对；
 * 面板只负责把"人看到的那段代码"的哈希随命令送进操作员通道。
 */

import type { DeviceInspection } from '../api/types'

/** 短哈希：首 12 位（与安装记录/首屏展示一致）。 */
export function shortHash(hash: string, length = 12): string {
  return hash.slice(0, length)
}

export interface LoadDeviceRequest {
  inspection: DeviceInspection
  mount: string
  className: string
  args: Record<string, unknown>
  approvedAt: string
  approvedVia: string
}

/** 构造 loader 的 load_device 命令（JSON 字符串，写入 `$MEM.loader`）。 */
export function buildLoadDeviceCommand(req: LoadDeviceRequest): string {
  const command: Record<string, unknown> = {
    action: 'load_device',
    path: req.mount,
    file: req.inspection.path,
    class: req.className,
    args: req.args,
    approved_sha256: req.inspection.sha256,
    approved_at: req.approvedAt,
    approved_via: req.approvedVia,
  }
  return JSON.stringify(command)
}

/** 旧哈希摘要：首次安装返回 null，覆盖旧版本时给弹窗显示。 */
export function previousHash(inspection: DeviceInspection): string | null {
  return inspection.existing?.sha256 ?? null
}

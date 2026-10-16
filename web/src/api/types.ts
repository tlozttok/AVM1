/**
 * 后端契约类型。形状对着 `avm/console.py` 的实现抄，不发明后端没有的字段。
 * 端点总表与响应形状见 docs/private/Agents/implemented/report-console-backend-2026-10-06.md。
 */

export type ConvStatus = 'active' | 'ready' | 'dormant' | 'finished' | 'unknown'

export interface Schedule {
  active_cid: number | null
  ready: number[]
  dormant: number[]
  finished: number[]
  instruction_budget: number
  activation_instructions: number
  panel_cid: number
}

export interface ConversationBrief {
  cid: number
  identity: string
  name: string | null
  status: ConvStatus
  is_python: boolean
  is_sub: boolean
  is_root: boolean
  parent: number | null
  messages: number
}

export interface Message {
  role: string
  content: string | null
  tool_calls: unknown
  tool_call_id: string | null
}

export interface ConversationDetail extends ConversationBrief {
  history: Message[]
  user_batch: {
    user_contents: string[]
    tool_responses: { content: string; tool_call_id: string }[]
  }
  metadata: Record<string, unknown>
  para_ref: string
}

export interface ChannelStatus {
  next_seq: number
  buffered: number
  dropped: number
  pending_actions: number
}

export interface PanelState {
  cid: number
  active: boolean
  messages: number
  channel: ChannelStatus
}

/** 中断派发记录：三种 action，见后端说明。 */
export interface DispatchRecord {
  type: string
  action: 'switch' | 'noop' | 'dropped'
  target?: number
  preempted?: number | null
  reason?: string
  payload?: unknown
}

export interface InterruptsState {
  vectors: Record<string, number>
  pending: string[]
  last_dispatch: DispatchRecord[]
}

export interface MaskState {
  memory: string[]
  devices: string[]
}

export interface State {
  running: boolean
  schedule: Schedule
  conversations: ConversationBrief[]
  panel: PanelState
  interrupts: InterruptsState
  mask: MaskState
  snapshots: number
}

/** `/api/memory` 的节点：dict / list / str / device / link 五种 kind。 */
export interface DictNode {
  kind: 'dict'
  metadata: unknown
  ctrl: Record<string, unknown> | null
  children: Record<string, MemoryNode>
}

export interface ListNode {
  kind: 'list'
  metadata: unknown
  ctrl: Record<string, unknown> | null
  children: MemoryNode[]
}

export interface DeviceNode {
  kind: 'device'
  type: string
  mount: string
  pretend_as_type: string | null
  preview: string | null
  error?: string
}

/**
 * 链接节点：树里的一个键，寻址遇到它就进入 target（`$MEM` 地址或设备标识符）。
 *
 * 面板看到的是树本身，所以这里不跟随：没有 children。设备挂载点也是链接节点，
 * 它们在左栏由设备行（`GET /api/devices`）呈现，不再重复画一行。
 */
export interface LinkNode {
  kind: 'link'
  target: string
  metadata: unknown
  ctrl: Record<string, unknown> | null
}

export type MemoryNode = DictNode | ListNode | DeviceNode | LinkNode | string

export interface MemoryResponse {
  ref: string
  node: MemoryNode
}

export interface DeviceAction {
  index: number
  name: string | null
  description: string | null
  parameters: unknown
  callable: boolean
}

export interface DeviceInfo {
  mount: string
  type: string
  description: DeviceNode
  actions: DeviceAction[]
  error: string | null
}

export interface ActionLogEntry {
  mount: string
  index: number
  name: string | null
  arguments: Record<string, unknown>
  result: unknown
  error: string | null
  time: number
}

export interface DevicesResponse {
  devices: DeviceInfo[]
  action_log: ActionLogEntry[]
}

export interface Listener {
  listener_id: number
  event_type: string
  target: string | null
  phase: string
  handler_ref: string | null
  enabled: boolean
  [key: string]: unknown
}

export interface ListenersResponse {
  listeners: Listener[]
  editable_fields: string[]
}

/** OpenAI function schema：九条指令各自带一份，`GET /api/tools` 原样导出。 */
export interface JsonSchemaProp {
  type?: string
  description?: string
  enum?: string[]
  items?: JsonSchemaProp
  anyOf?: JsonSchemaProp[]
  oneOf?: JsonSchemaProp[]
  properties?: Record<string, JsonSchemaProp>
}

export interface ToolDef {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: {
      type?: string
      properties?: Record<string, JsonSchemaProp>
      required?: string[]
    }
  }
}

export interface ToolsResponse {
  tools: ToolDef[]
}

/** 一次动作提交的响应。 */
export interface ActionResponse {
  submitted: boolean
  tool: string
  auto_paused: boolean
  known_tool: boolean
  warning: string | null
}

export interface PauseResponse {
  requested: boolean
  pending: string[]
  panel_cid: number
  already_active: boolean
}

/** SSE 帧：schedule 帧没有 entry，其余六种都带递增的 entry.seq。 */
export interface SseEntry {
  seq: number
  kind: string
  data: unknown
  time: number
}

export type SseFrame =
  | { type: 'schedule'; data: Schedule }
  | { type: 'tool_call' | 'tool_result' | 'event' | 'user_message' | 'action_error'; entry: SseEntry }

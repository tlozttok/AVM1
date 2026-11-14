import { useCallback, useEffect, useState } from 'react'
import type { ConversationBrief, DeviceInfo, State } from '../api/types'
import type { ToolTable } from '../api/tools'
import { pairAction, startTrace, type ActionTrace, type SubmittedAction } from '../lib/correlation'
import type { FrameState } from '../lib/frames'
import type { TrayValue } from '../lib/valueTray'
import ActionCard from './ActionCard'
import ActionForm from './ActionForm'
import ConversationView from './ConversationView'
import DebugInput from './DebugInput'
import DeviceApprovalPanel from './DeviceApprovalPanel'
import EventLog from './EventLog'
import InputFeed from './InputFeed'
import ListenerEditor from './ListenerEditor'
import OperatorDetail from './OperatorDetail'
import ValueTray from './ValueTray'

export type DrawerTab =
  | 'raw'
  | 'operator'
  | 'events'
  | 'listeners'
  | 'io'
  | 'tray'
  | 'approvals'
  | 'conversation'
  | 'system'

const TAB_LABEL: Record<DrawerTab, string> = {
  raw: '原始动作（cid -1）',
  operator: '面板对话（cid -1）',
  events: '事件流',
  listeners: '监听器编辑',
  io: 'IO 调试输入',
  tray: '值暂存区',
  approvals: '设备审批',
  conversation: '对话原始详情',
  system: '系统信息',
}

const TAB_ORDER: DrawerTab[] = [
  'raw',
  'operator',
  'events',
  'listeners',
  'io',
  'tray',
  'approvals',
  'conversation',
  'system',
]

interface Props {
  open: boolean
  tab: DrawerTab
  onToggle: () => void
  onTab: (tab: DrawerTab) => void
  state: State | null
  frames: FrameState
  eventsConnected: boolean
  tools: ToolTable
  devices: DeviceInfo[]
  conversations: ConversationBrief[]
  selectedCid: number | null
  selectedMessageCount: number
  tray: TrayValue[]
  traySelected: string | null
  copyNote: string | null
  onSelectTray: (value: string) => void
  onRefreshState: () => void
  onDevicesChanged: () => void
  onToolResult: () => void
  onCopy: (value: string) => void
  onSelectCid: (cid: number, source: string) => void
  onSelectRef: (ref: string, source: string) => Promise<void>
  onClearFrames: () => void
}

/**
 * 底部调试抽屉：原始动作（保留「直接作为 cid -1 对话执行指令」的通道）、事件流、
 * 监听器编辑、IO 调试输入、值暂存区、设备审批、对话原始详情、系统信息。
 * 默认收起，主界面不放调试能力。
 */
export default function DebugDrawer({
  open,
  tab,
  onToggle,
  onTab,
  state,
  frames,
  eventsConnected,
  tools,
  devices,
  conversations,
  selectedCid,
  selectedMessageCount,
  tray,
  traySelected,
  copyNote,
  onSelectTray,
  onRefreshState,
  onDevicesChanged,
  onToolResult,
  onCopy,
  onSelectCid,
  onSelectRef,
  onClearFrames,
}: Props) {
  const [trace, setTrace] = useState<ActionTrace | null>(null)
  const activeCid = state?.schedule.active_cid ?? null
  const channel = state?.panel.channel

  const onSubmitted = useCallback(
    (info: SubmittedAction) => {
      setTrace(startTrace(info.tool, info.args, info.knownTool, info.seqAfter))
      onRefreshState()
    },
    [onRefreshState],
  )

  useEffect(() => {
    if (!trace) return
    const next = pairAction(trace, frames.entries)
    if (next.phase === 'responded' && trace.phase !== 'responded') onToolResult()
    if (next !== trace) setTrace(next)
  }, [frames.entries, trace, onToolResult])

  return (
    <section className={`debug-drawer${open ? ' is-open' : ''}`} data-testid="debug-drawer">
      <div className="drawer-bar" data-testid="debug-toggle" onClick={onToggle}>
        <span className="drawer-label">调试抽屉</span>
        <span className="drawer-desc">
          原始动作（以 cid -1 直接执行） · 事件流 · 监听器编辑 · IO 调试输入 · 值暂存区 · 设备审批 ·
          对话原始详情 · 系统信息
        </span>
        <span className="drawer-toggle">{open ? '收起' : '展开'}</span>
      </div>
      {open ? (
        <div className="drawer-body" data-testid="debug-drawer-body">
          <nav className="drawer-tabs">
            {TAB_ORDER.map((item) => (
              <button
                key={item}
                type="button"
                className={item === tab ? 'is-active' : ''}
                data-testid={`drawer-tab-${item}`}
                onClick={() => onTab(item)}
              >
                {TAB_LABEL[item]}
              </button>
            ))}
          </nav>
          <div className="drawer-panes">
            {tab === 'raw' ? (
              <div className="drawer-pane" data-testid="drawer-pane-raw">
                <div className="drawer-note">
                  操作员以面板对话 <code>cid -1</code> 的身份直接执行任意指令——「直接作为 -1 对话」的保留通道。
                  未暂停时提交会自动中断并在下一个安全点执行；操作员接管时直接执行。主界面的对象操作在成功后自动让出执行权，
                  这里保持手动暂停/继续的调试语义。
                </div>
                <div className="drawer-meta">
                  面板通道：输出 seq {channel?.next_seq ?? '—'} · 缓冲 {channel?.buffered ?? '—'} · 待执行动作{' '}
                  {channel?.pending_actions ?? '—'}
                  {channel && channel.dropped > 0 ? ` · 丢弃 ${channel.dropped}` : ''}
                </div>
                <div className="drawer-meta">
                  每次动作的返回值在
                  <button type="button" className="link" data-testid="open-operator-detail" onClick={() => onTab('operator')}>
                    面板对话（cid -1）
                  </button>
                  页签按 call_id 列出。
                </div>
                <ActionCard trace={trace} />
                <ActionForm
                  tools={tools}
                  currentSeq={frames.lastSeq}
                  traySelected={traySelected}
                  onSubmitted={onSubmitted}
                />
              </div>
            ) : null}

            {tab === 'operator' ? (
              <OperatorDetail cid={state?.panel.cid ?? -1} frames={frames} />
            ) : null}

            {tab === 'events' ? (
              <div className="drawer-pane drawer-pane-fill" data-testid="drawer-pane-events">
                <EventLog
                  frames={frames}
                  connected={eventsConnected}
                  onClear={onClearFrames}
                  onCopy={onCopy}
                  onSelectCid={onSelectCid}
                  onSelectRef={onSelectRef}
                />
              </div>
            ) : null}

            {tab === 'listeners' ? (
              <div className="drawer-pane" data-testid="drawer-pane-listeners">
                <ListenerEditor />
              </div>
            ) : null}

            {tab === 'io' ? (
              <div className="drawer-pane" data-testid="drawer-pane-io">
                <InputFeed devices={devices} onDevicesChanged={onDevicesChanged} />
                <DebugInput conversations={conversations} activeCid={activeCid} />
              </div>
            ) : null}

            {tab === 'tray' ? (
              <div className="drawer-pane" data-testid="drawer-pane-tray">
                <ValueTray values={tray} selected={traySelected} onSelect={onSelectTray} note={copyNote} />
              </div>
            ) : null}

            {tab === 'approvals' ? (
              <div className="drawer-pane" data-testid="drawer-pane-approvals">
                <DeviceApprovalPanel onRefresh={onRefreshState} />
              </div>
            ) : null}

            {tab === 'conversation' ? (
              <div className="drawer-pane drawer-pane-fill" data-testid="drawer-pane-conversation">
                {selectedCid === null ? (
                  <div className="drawer-note">先在资源栏或事件流里选中一个对话。</div>
                ) : (
                  <ConversationView cid={selectedCid} token={frames.lastSeq} messageCount={selectedMessageCount} />
                )}
              </div>
            ) : null}

            {tab === 'system' ? (
              <div className="drawer-pane" data-testid="drawer-pane-system">
                <SystemInfo state={state} />
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function SystemInfo({ state }: { state: State | null }) {
  if (state === null) return <div className="drawer-note">还没有 /api/state 数据。</div>
  const recovery = state.recovery
  return (
    <div className="drawer-system">
      <dl className="drawer-kv">
        <dt>生命周期</dt>
        <dd>{state.lifecycle ?? (state.running ? 'running' : 'stopped')}</dd>
        <dt>API 暂停</dt>
        <dd>{state.paused ? `${state.pause_category ?? '未知分类'} · ${state.pause_reason ?? ''}` : '无'}</dd>
        <dt>面板对话</dt>
        <dd>cid {state.panel.cid} · {state.panel.active ? '持有执行权' : '休眠'} · {state.panel.messages} 条消息</dd>
        <dt>中断向量</dt>
        <dd className="mono">{JSON.stringify(state.interrupts.vectors)}</dd>
        <dt>待派发中断</dt>
        <dd>{state.interrupts.pending.join('、') || '无'}</dd>
        <dt>最近派发</dt>
        <dd className="mono">{JSON.stringify(state.interrupts.last_dispatch)}</dd>
        <dt>恢复</dt>
        <dd>
          {recovery
            ? `${recovery.source === 'snapshot' ? `快照 seq ${recovery.snapshot_seq ?? '—'}` : '冷启动'} · 重放 ${recovery.replayed} 条 · 警告 ${recovery.warnings.length} 条`
            : '不可用'}
          {recovery && recovery.warnings.length > 0 ? (
            <ul className="drawer-warnings">
              {recovery.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </dd>
        <dt>客户端席位</dt>
        <dd>
          {state.client
            ? `${state.client.single_client ? '单客户端' : '未启用'} · ${state.client.held ? '已接入' : '空闲'} · ${state.client.client_id ?? '—'}`
            : '不可用'}
        </dd>
        <dt>IO 总线</dt>
        <dd className="mono">{JSON.stringify(state.io ?? null)}</dd>
        <dt>屏蔽</dt>
        <dd className="mono">{JSON.stringify(state.mask)}</dd>
        <dt>内存快照点</dt>
        <dd>{state.snapshots}</dd>
      </dl>
    </div>
  )
}

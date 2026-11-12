import type { ConversationBrief, DispatchRecord, State } from '../api/types'

/**
 * 顶栏状态的唯一判定器。
 *
 * 术语按 `docs/runtime.md`：面板对话（cid -1）活跃即全局暂停；它由中断取得执行权，
 * 所以「空闲」与「操作员接管」按最近一次派发的中断类型区分，不靠猜测。
 */
export type SystemPhase =
  | 'connecting'
  | 'stopped'
  | 'stopping'
  | 'api-paused'
  | 'operator'
  | 'idle'
  | 'running'

export interface SystemStatus {
  phase: SystemPhase
  /** 顶栏状态文案：同一时刻只有一个说法。 */
  label: string
  /** 当前活跃的 guest 对话；面板占执行权时为 null。 */
  activeCid: number | null
  activeLabel: string | null
  /** 面板可以投递指令/动作（API 暂停与关机中不可）。 */
  canOperate: boolean
}

function convText(conversations: ConversationBrief[], cid: number | null): string | null {
  if (cid === null) return null
  const conv = conversations.find((item) => item.cid === cid)
  if (!conv) return `cid ${cid}`
  return `${conv.name ?? conv.identity}（cid ${cid}）`
}

/** 最近一次派发里命中面板对话的两种中断类型：operator = 暂停，idle = 空闲交接。 */
function panelDispatch(records: DispatchRecord[] | undefined, panelCid: number): 'operator' | 'idle' | null {
  if (!records) return null
  for (const record of [...records].reverse()) {
    if (record.target !== panelCid) continue
    if (record.type === 'operator') return 'operator'
    if (record.type === 'idle') return 'idle'
  }
  return null
}

export function deriveSystemStatus(
  state: State | null,
  conversations: ConversationBrief[],
): SystemStatus {
  if (state === null) {
    return {
      phase: 'connecting',
      label: '连接中…',
      activeCid: null,
      activeLabel: null,
      canOperate: false,
    }
  }

  const lifecycle = state.lifecycle ?? (state.running ? 'running' : 'stopped')
  const schedule = state.schedule
  const panelCid = schedule.panel_cid

  if (lifecycle === 'stopped') {
    return { phase: 'stopped', label: '已停止', activeCid: null, activeLabel: null, canOperate: false }
  }
  if (lifecycle === 'stopping') {
    return { phase: 'stopping', label: '关机中', activeCid: null, activeLabel: null, canOperate: false }
  }
  if (state.paused) {
    const category = state.pause_category ?? '未知分类'
    return {
      phase: 'api-paused',
      label: `API 暂停：${category}`,
      activeCid: null,
      activeLabel: null,
      canOperate: false,
    }
  }
  if (!state.running) {
    return { phase: 'stopped', label: '已停止', activeCid: null, activeLabel: null, canOperate: false }
  }

  if (schedule.active_cid === panelCid) {
    const dispatch = panelDispatch(state.interrupts.last_dispatch, panelCid)
    if (dispatch === 'operator') {
      return {
        phase: 'operator',
        label: '操作员接管（系统暂停）',
        activeCid: null,
        activeLabel: null,
        canOperate: true,
      }
    }
    return { phase: 'idle', label: '空闲（等待操作员）', activeCid: null, activeLabel: null, canOperate: true }
  }

  const activeCid = schedule.active_cid
  const activeLabel = convText(conversations, activeCid)
  return {
    phase: 'running',
    label: '运行中',
    activeCid,
    activeLabel,
    canOperate: true,
  }
}

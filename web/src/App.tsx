import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiGet, describeError } from './api/client'
import { loadTools, type ToolTable } from './api/tools'
import { useConsoleState } from './api/useConsoleState'
import { useEventStream } from './api/useEventStream'
import { useIoStream } from './api/useIoStream'
import type { DevicesResponse, DeviceInfo, MemoryNode, MemoryResponse } from './api/types'
import CommandDeck from './components/CommandDeck'
import DebugDrawer, { type DrawerTab } from './components/DebugDrawer'
import LoginGate from './components/LoginGate'
import NodeWorkbench from './components/NodeWorkbench'
import ResourceRail from './components/ResourceRail'
import SystemLogBar from './components/SystemLogBar'
import TopBar from './components/TopBar'
import { resolveRef } from './lib/memory'
import { labelForValue, pushTrayValue, selectedTrayValue, type TrayValue } from './lib/valueTray'
import { deriveSystemStatus } from './lib/systemStatus'

export type Selection =
  | { kind: 'none' }
  | { kind: 'conversation'; cid: number }
  | { kind: 'node'; ref: string; node: MemoryNode }

const LOCATE_TIMEOUT_MS = 8000

function isMemoryChange(tool: string | null | undefined): boolean {
  return tool !== null && tool !== undefined &&
    (tool.startsWith('memory_') || tool === 'edit_metadata')
}

/**
 * 面板装配：顶栏状态 + 左资源栏 + 中工作台 + 底部系统日志条 + 底部调试抽屉。
 *
 * 主入口是命令台：默认选中 shell，发送即投递并让出执行权；对象操作在节点工作台
 * 按「节点类型 / ctrl.type / 设备」三个区域呈现；调试能力全部收在默认收起的抽屉里。
 */
export default function App() {
  const { state, error: stateError, refresh } = useConsoleState()
  const { frames, connected: eventsConnected, clear: clearFrames } = useEventStream()
  const io = useIoStream()
  const [tools, setTools] = useState<ToolTable>({ tools: [], error: null })
  const [selection, setSelection] = useState<Selection>({ kind: 'none' })
  const [root, setRoot] = useState<MemoryNode | null>(null)
  const [memoryBusy, setMemoryBusy] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)
  const [memoryDirty, setMemoryDirty] = useState(false)
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [deviceError, setDeviceError] = useState<string | null>(null)
  const [tray, setTray] = useState<TrayValue[]>([])
  const [traySelected, setTraySelected] = useState<string | null>(null)
  const [copyNote, setCopyNote] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [drawerTab, setDrawerTab] = useState<DrawerTab>('raw')
  const [needLogin, setNeedLogin] = useState(false)
  const seenSeq = useRef(0)
  const autoSelected = useRef(false)
  const toastTimer = useRef<number | null>(null)

  const refreshMemory = useCallback(async () => {
    setMemoryBusy(true)
    try {
      const resp = await apiGet<MemoryResponse>('/api/memory')
      setRoot(resp.node)
      setMemoryError(null)
      setMemoryDirty(false)
      setSelection((prev) =>
        prev.kind === 'node'
          ? { ...prev, node: resolveRef(resp.node, prev.ref) ?? prev.node }
          : prev,
      )
    } catch (err) {
      setMemoryError(describeError(err))
    } finally {
      setMemoryBusy(false)
    }
  }, [])

  const refreshDevices = useCallback(async () => {
    try {
      const resp = await apiGet<DevicesResponse>('/api/devices')
      setDevices(resp.devices)
      setDeviceError(null)
    } catch (err) {
      setDeviceError(describeError(err))
    }
  }, [])

  useEffect(() => {
    void loadTools().then(setTools)
    void refreshMemory()
    void refreshDevices()
  }, [refreshMemory, refreshDevices])

  useEffect(() => {
    const onUnauthorized = () => setNeedLogin(true)
    window.addEventListener('avm:unauthorized', onUnauthorized)
    return () => window.removeEventListener('avm:unauthorized', onUnauthorized)
  }, [])

  const conversations = state?.conversations ?? []
  const panelCid = state?.panel.cid ?? null
  const guestConversations = useMemo(
    () => conversations.filter((conv) => conv.cid !== panelCid),
    [conversations, panelCid],
  )
  const status = deriveSystemStatus(state, conversations)
  const selectedCid = selection.kind === 'conversation' ? selection.cid : null
  const selectedRef = selection.kind === 'node' ? selection.ref : null
  const selectedMessages =
    selectedCid === null ? 0 : (conversations.find((conv) => conv.cid === selectedCid)?.messages ?? 0)

  // 首次拿到 state 后默认选中 shell（base 镜像的 boot 会创建它）；没有 shell 时留在空态。
  useEffect(() => {
    if (autoSelected.current || state === null) return
    if (selection.kind !== 'none') {
      autoSelected.current = true
      return
    }
    const shell = guestConversations.find(
      (conv) => conv.name === 'shell' || conv.identity === 'shell',
    )
    autoSelected.current = true
    if (shell) setSelection({ kind: 'conversation', cid: shell.cid })
  }, [state, selection, guestConversations])

  // 选中的对话被关闭/消失：回到空态。
  useEffect(() => {
    if (state === null || selection.kind !== 'conversation') return
    if (!conversations.some((conv) => conv.cid === selection.cid)) {
      setSelection({ kind: 'none' })
    }
  }, [state, selection, conversations])

  // 内存类事件出现时只打「有变更」，不自动重取整树。
  useEffect(() => {
    if (frames.lastSeq <= seenSeq.current) return
    const changed = frames.entries.some(
      (entry) =>
        entry.seq !== null &&
        entry.seq > seenSeq.current &&
        entry.kind === 'event' &&
        isMemoryChange(entry.link?.tool),
    )
    seenSeq.current = frames.lastSeq
    if (changed) setMemoryDirty(true)
  }, [frames.lastSeq, frames.entries])

  const showToast = useCallback((text: string | null) => {
    setCopyNote(text)
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current)
    if (text !== null) {
      toastTimer.current = window.setTimeout(() => setCopyNote(null), 1800)
    }
  }, [])

  /** 主界面复制：直接进剪贴板；不经过值暂存区。 */
  const copyText = useCallback(
    (value: string) => {
      if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) {
        showToast('浏览器剪贴板不可用')
        return
      }
      navigator.clipboard.writeText(value).then(
        () => showToast(`已复制 ${value.length > 24 ? `${value.slice(0, 24)}…` : value}`),
        () => showToast('浏览器剪贴板不可用'),
      )
    },
    [showToast],
  )

  /** 调试抽屉里的复制：留在值暂存区，供原始动作的「填入」使用。 */
  const copyToTray = useCallback((value: string) => {
    setTray((prev) => pushTrayValue(prev, value, labelForValue(value)))
    setTraySelected(value)
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(value).catch(() => showToast('浏览器剪贴板不可用，值已留在暂存区'))
    }
  }, [showToast])

  const selectConversation = useCallback((cid: number) => {
    setSelection({ kind: 'conversation', cid })
  }, [])

  const selectNode = useCallback((ref: string, node: MemoryNode) => {
    setSelection({ kind: 'node', ref, node })
  }, [])

  const readRef = useCallback(async (ref: string, signal: AbortSignal) => {
    const resp = await apiGet<MemoryResponse>(
      `/api/memory?ref=${encodeURIComponent(ref)}`,
      signal,
    )
    return resp.node
  }, [])

  /** 事件流按地址跳转：解析路径后切到节点工作台。 */
  const locateRef = useCallback(
    async (ref: string, _source: string) => {
      const controller = new AbortController()
      const timer = window.setTimeout(() => controller.abort(), LOCATE_TIMEOUT_MS)
      try {
        const node = await readRef(ref, controller.signal)
        selectNode(ref, node)
      } catch (err) {
        throw new Error(describeError(err))
      } finally {
        window.clearTimeout(timer)
      }
    },
    [readRef, selectNode],
  )

  const openDrawer = useCallback((tab: DrawerTab) => {
    setDrawerTab(tab)
    setDrawerOpen(true)
  }, [])

  const onNodeDeleted = useCallback(() => {
    setSelection({ kind: 'none' })
    void refreshMemory()
  }, [refreshMemory])

  const activeTrayValue = selectedTrayValue(tray, traySelected)

  if (needLogin) {
    return <LoginGate onLogin={() => window.location.reload()} />
  }

  return (
    <div className="console-app">
      <TopBar state={state} status={status} onRefresh={refresh} />
      {stateError ? <div className="console-banner is-warn">状态轮询失败：{stateError}</div> : null}
      {deviceError ? <div className="console-banner is-warn">设备清单失败：{deviceError}</div> : null}
      {copyNote ? <div className="console-toast" data-testid="copy-toast">{copyNote}</div> : null}
      <div className="console-body">
        <ResourceRail
          conversations={guestConversations}
          selectedCid={selectedCid}
          activeCid={status.activeCid}
          onSelectConversation={selectConversation}
          onCopyCid={(cid) => copyText(String(cid))}
          memoryRoot={root}
          devices={devices}
          memoryBusy={memoryBusy}
          memoryError={memoryError}
          memoryDirty={memoryDirty}
          selectedRef={selectedRef}
          onRefreshMemory={() => void refreshMemory()}
          onSelectNode={selectNode}
          onCopyRef={copyText}
        />
        <main className="workbench" data-testid="workbench">
          {selection.kind === 'conversation' ? (
            <CommandDeck
              cid={selection.cid}
              conversations={conversations}
              status={status}
              frames={frames}
              onRefreshState={refresh}
              onClosed={() => setSelection({ kind: 'none' })}
            />
          ) : selection.kind === 'node' ? (
            <NodeWorkbench
              refName={selection.ref}
              node={selection.node}
              devices={devices}
              frames={frames}
              status={status}
              onRefreshMemory={() => void refreshMemory()}
              onRefreshState={refresh}
              onRefreshDevices={() => void refreshDevices()}
              onSelectNode={selectNode}
              onDeleted={onNodeDeleted}
              onOpenListeners={() => openDrawer('listeners')}
              onCopyRef={copyText}
            />
          ) : (
            <div className="workbench-empty">
              <div className="workbench-empty-title">从左侧资源栏选一个对话或内存节点</div>
              <div className="workbench-empty-text">
                日常入口是命令台：默认目标 <span className="mono">shell</span>；选中其它对话即切换目标。
              </div>
            </div>
          )}
        </main>
      </div>
      <SystemLogBar entries={io.entries} connected={io.connected} onClear={io.clear} />
      <DebugDrawer
        open={drawerOpen}
        tab={drawerTab}
        onToggle={() => setDrawerOpen((value) => !value)}
        onTab={setDrawerTab}
        state={state}
        frames={frames}
        eventsConnected={eventsConnected}
        tools={tools}
        devices={devices}
        conversations={guestConversations}
        selectedCid={selectedCid}
        selectedMessageCount={selectedMessages}
        tray={tray}
        traySelected={activeTrayValue}
        copyNote={copyNote}
        onSelectTray={setTraySelected}
        onRefreshState={refresh}
        onDevicesChanged={() => void refreshDevices()}
        onToolResult={() => {
          void refreshMemory()
          void refreshDevices()
        }}
        onCopy={copyToTray}
        onSelectCid={(cid) => selectConversation(cid)}
        onSelectRef={locateRef}
        onClearFrames={clearFrames}
      />
    </div>
  )
}

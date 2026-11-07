import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiGet, describeError } from './api/client'
import { loadTools, type ToolTable } from './api/tools'
import { useConsoleState } from './api/useConsoleState'
import { useEventStream } from './api/useEventStream'
import type { DevicesResponse, DeviceInfo, MemoryNode, MemoryResponse } from './api/types'
import ActionPanel from './components/ActionPanel'
import ConversationList from './components/ConversationList'
import ConversationView from './components/ConversationView'
import DeviceView from './components/DeviceView'
import EventLog from './components/EventLog'
import MemoryIndex from './components/MemoryIndex'
import MemoryView from './components/MemoryView'
import { resolveRef } from './lib/memory'
import {
  labelForValue,
  pushTrayValue,
  selectedTrayValue,
  type TrayValue,
} from './lib/valueTray'

export type Selection =
  | { kind: 'none' }
  | { kind: 'conversation'; cid: number }
  | { kind: 'memory'; ref: string; node: MemoryNode }
  | { kind: 'device'; mount: string }

/** 事件流里这些指令会改内存树；只用于给左栏打「有变更」标记，不自动重取整树。 */
function isMemoryChange(tool: string | null | undefined): boolean {
  return tool !== null && tool !== undefined &&
    (tool.startsWith('memory_') || tool === 'edit_metadata')
}

const LOCATE_TIMEOUT_MS = 8000

export default function App() {
  const { state, error: stateError, refresh } = useConsoleState()
  const { frames, connected, clear } = useEventStream()
  const [tools, setTools] = useState<ToolTable>({ tools: [], error: null })
  const [selection, setSelection] = useState<Selection>({ kind: 'none' })
  const [selectionSource, setSelectionSource] = useState<string | null>(null)
  const [root, setRoot] = useState<MemoryNode | null>(null)
  const [memoryBusy, setMemoryBusy] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)
  const [memoryDirty, setMemoryDirty] = useState(false)
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [deviceError, setDeviceError] = useState<string | null>(null)
  const [detailToken, setDetailToken] = useState(0)
  // 值暂存区：React 状态，页面刷新即失；不写 localStorage、不发端点。
  const [tray, setTray] = useState<TrayValue[]>([])
  const [traySelected, setTraySelected] = useState<string | null>(null)
  const [copyNote, setCopyNote] = useState<string | null>(null)
  const seenSeq = useRef(0)

  const refreshMemory = useCallback(async () => {
    setMemoryBusy(true)
    try {
      const resp = await apiGet<MemoryResponse>('/api/memory')
      setRoot(resp.node)
      setMemoryError(null)
      setMemoryDirty(false)
      // 已选中的内存节点跟着刷新后的树重定位；重定位不到就保留旧节点（可能是屏蔽或删除）
      setSelection((prev) =>
        prev.kind === 'memory'
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

  const readRef = useCallback(async (ref: string, signal: AbortSignal) => {
    const resp = await apiGet<MemoryResponse>(
      `/api/memory?ref=${encodeURIComponent(ref)}`,
      signal,
    )
    return resp.node
  }, [])

  useEffect(() => {
    void loadTools().then(setTools)
    void refreshMemory()
    void refreshDevices()
  }, [refreshMemory, refreshDevices])

  const schedule = frames.schedule ?? state?.schedule ?? null
  const conversations = state?.conversations ?? []
  const selectedCid = selection.kind === 'conversation' ? selection.cid : null
  const selectedMessages =
    selectedCid !== null
      ? (conversations.find((conv) => conv.cid === selectedCid)?.messages ?? 0)
      : 0

  /** 复制出口的唯一写入口：剪贴板失败也要把值留在暂存区，复制链路不中断。 */
  const copyValue = useCallback((value: string) => {
    setTray((prev) => pushTrayValue(prev, value, labelForValue(value)))
    setTraySelected(value)
    const fallback = '浏览器剪贴板不可用，值已留在面板暂存区'
    if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) {
      setCopyNote(fallback)
      return
    }
    navigator.clipboard.writeText(value).then(
      () => setCopyNote(null),
      () => setCopyNote(fallback),
    )
  }, [])

  // 三栏选中态的统一写入口：左栏点选、事件流跳转都从这里过，顺手记下「来自哪条事件」。
  const selectConversation = useCallback((cid: number, source: string | null = null) => {
    setSelection({ kind: 'conversation', cid })
    setSelectionSource(source)
  }, [])

  const selectNode = useCallback((ref: string, node: MemoryNode, source: string | null = null) => {
    setSelection(
      typeof node !== 'string' && node.kind === 'device'
        ? { kind: 'device', mount: node.mount }
        : { kind: 'memory', ref, node },
    )
    setSelectionSource(source)
  }, [])

  /** 事件流按地址跳转：沿用现有 readRef 解析路径，解析失败把原因抛回行内提示。 */
  const locateRef = useCallback(async (ref: string, source: string) => {
    const ctrl = new AbortController()
    const timer = window.setTimeout(() => ctrl.abort(), LOCATE_TIMEOUT_MS)
    try {
      const node = await readRef(ref, ctrl.signal)
      selectNode(ref, node, source)
    } catch (err) {
      throw new Error(describeError(err))
    } finally {
      window.clearTimeout(timer)
    }
  }, [readRef, selectNode])

  // 提交动作只刷状态与对话详情；内存树 / 设备清单等 tool_result 到了再补刷（裁决三 A）。
  const afterAction = useCallback(() => {
    refresh()
    setDetailToken((n) => n + 1)
  }, [refresh])

  const onToolResult = useCallback(() => {
    void refreshMemory()
    void refreshDevices()
  }, [refreshMemory, refreshDevices])

  const onRefreshMemory = useCallback(() => {
    void refreshMemory()
  }, [refreshMemory])

  // 其余内存类事件只在左栏打「有变更」：0.2s 节拍下不逐条重取整树。
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

  const recentCid = useMemo(() => {
    for (let i = frames.entries.length - 1; i >= 0; i -= 1) {
      const cid = frames.entries[i].link?.cid
      if (cid !== null && cid !== undefined) return cid
    }
    return null
  }, [frames.entries])

  // 选中的值可能已被 8 条限量淘汰：回落到 null，不让「填入」写进过期文本。
  const activeTrayValue = selectedTrayValue(tray, traySelected)

  const maskMemory = state?.mask.memory ?? []
  const maskDevices = state?.mask.devices ?? []
  const panelCid = state?.panel.cid ?? null

  return (
    <div className="app">
      <header className="app-head">
        <span className="title">AVM 调试面板</span>
        <span className="muted small">
          观测与介入都走控制台后端；页面不直接读 Core
        </span>
        {stateError ? <span className="hint error inline">状态轮询失败：{stateError}</span> : null}
        {deviceError ? <span className="hint error inline">设备清单失败：{deviceError}</span> : null}
      </header>

      <div className="columns">
        <div className="col-left">
          <ConversationList
            conversations={conversations}
            panelCid={panelCid}
            selectedCid={selectedCid}
            activeCid={schedule?.active_cid ?? null}
            recentCid={recentCid}
            onSelect={(cid) => selectConversation(cid)}
            onCopy={copyValue}
          />
          <MemoryIndex
            root={root}
            devices={devices}
            busy={memoryBusy}
            error={memoryError}
            dirty={memoryDirty}
            onRefresh={onRefreshMemory}
            onSelect={(ref, node) => selectNode(ref, node)}
            onReadRef={readRef}
            onCopy={copyValue}
            traySelected={activeTrayValue}
          />
        </div>

        <div className="col-mid">
          {selectionSource ? (
            <div className="hint muted small" data-testid="selection-source">
              来自 {selectionSource}
              <button type="button" className="link" onClick={() => setSelectionSource(null)}>
                清除
              </button>
            </div>
          ) : null}
          {selection.kind === 'conversation' ? (
            <ConversationView
              cid={selection.cid}
              token={detailToken}
              messageCount={selectedMessages}
            />
          ) : null}
          {selection.kind === 'memory' ? (
            <MemoryView refName={selection.ref} node={selection.node} onCopy={copyValue} />
          ) : null}
          {selection.kind === 'device' ? (
            <DeviceView
              device={devices.find((d) => d.mount === selection.mount)}
              onCopy={copyValue}
            />
          ) : null}
          {selection.kind === 'none' ? (
            <div className="pane-body muted">
              点左栏的对话看历史与 user_batch，点内存节点看内容，点设备节点看设备信息。
            </div>
          ) : null}
          {maskMemory.length > 0 || maskDevices.length > 0 ? (
            <div className="hint warn">
              当前有屏蔽（只读展示，还原界面在下一轮）：内存 {maskMemory.join('、') || '—'} · 设备{' '}
              {maskDevices.join('、') || '—'}
            </div>
          ) : null}
        </div>

        <div className="col-right">
          <ActionPanel
            state={state}
            schedule={schedule}
            frames={frames}
            tools={tools}
            devices={devices}
            tray={tray}
            traySelected={activeTrayValue}
            onSelectTray={setTraySelected}
            copyNote={copyNote}
            onRefresh={afterAction}
            onDevicesChanged={() => void refreshDevices()}
            onToolResult={onToolResult}
          />
          <EventLog
            frames={frames}
            connected={connected}
            onClear={clear}
            onCopy={copyValue}
            onSelectCid={(cid, source) => selectConversation(cid, source)}
            onSelectRef={locateRef}
          />
        </div>
      </div>
    </div>
  )
}

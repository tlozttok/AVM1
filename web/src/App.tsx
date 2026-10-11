import { useCallback, useEffect, useState } from 'react'
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

export type Selection =
  | { kind: 'none' }
  | { kind: 'conversation'; cid: number }
  | { kind: 'memory'; ref: string; node: MemoryNode }
  | { kind: 'device'; mount: string }

export default function App() {
  const { state, error: stateError, refresh } = useConsoleState()
  const { frames, connected, clear } = useEventStream()
  const [tools, setTools] = useState<ToolTable>({ tools: [], error: null })
  const [selection, setSelection] = useState<Selection>({ kind: 'none' })
  const [root, setRoot] = useState<MemoryNode | null>(null)
  const [memoryBusy, setMemoryBusy] = useState(false)
  const [memoryError, setMemoryError] = useState<string | null>(null)
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [deviceError, setDeviceError] = useState<string | null>(null)
  const [detailToken, setDetailToken] = useState(0)

  const refreshMemory = useCallback(async () => {
    setMemoryBusy(true)
    try {
      const resp = await apiGet<MemoryResponse>('/api/memory')
      setRoot(resp.node)
      setMemoryError(null)
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

  const afterAction = useCallback(() => {
    refresh()
    setDetailToken((n) => n + 1)
    void refreshMemory()
    void refreshDevices()
  }, [refresh, refreshMemory, refreshDevices])

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
            onSelect={(cid) => setSelection({ kind: 'conversation', cid })}
          />
          <MemoryIndex
            root={root}
            devices={devices}
            busy={memoryBusy}
            error={memoryError}
            onRefresh={() => void refreshMemory()}
            onSelect={(ref, node) =>
              typeof node !== 'string' && node.kind === 'device'
                ? setSelection({ kind: 'device', mount: node.mount })
                : setSelection({ kind: 'memory', ref, node })
            }
            onReadRef={readRef}
          />
        </div>

        <div className="col-mid">
          {selection.kind === 'conversation' ? (
            <ConversationView
              cid={selection.cid}
              token={detailToken}
              messageCount={selectedMessages}
            />
          ) : null}
          {selection.kind === 'memory' ? (
            <MemoryView refName={selection.ref} node={selection.node} />
          ) : null}
          {selection.kind === 'device' ? (
            <DeviceView device={devices.find((d) => d.mount === selection.mount)} />
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
            tools={tools}
            devices={devices}
            onRefresh={afterAction}
            onDevicesChanged={() => void refreshDevices()}
          />
          <EventLog frames={frames} connected={connected} onClear={clear} />
        </div>
      </div>
    </div>
  )
}

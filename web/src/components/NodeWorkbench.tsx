import { useEffect, useRef, useState } from 'react'
import { apiGet, apiSend, describeError } from '../api/client'
import type { DeviceInfo, Listener, ListenersResponse, MemoryNode } from '../api/types'
import { useOperatorAction } from '../api/useOperatorAction'
import type { FrameState } from '../lib/frames'
import {
  childEntries,
  childNode,
  ctrlTypeOf,
  isHandlerType,
  isProgramType,
  nodeCtrl,
  nodeDescription,
  nodeShape,
  paraRefOf,
  quickFields,
  shapeLabel,
  structureChecks,
  typeSummary,
  type NodeShape,
  type QuickField,
} from '../lib/nodeModel'
import {
  childRef,
  createCmd,
  ctrlDel,
  ctrlSet,
  listenerCancel,
  listenerRegister,
  makeKeyFor,
  memoryDelete,
  memoryMake,
  memoryWrite,
  type MemType,
  type OperatorAction,
} from '../lib/operatorActions'
import type { SystemStatus } from '../lib/systemStatus'
import { isErrorMessage } from '../lib/transcript'

interface Props {
  refName: string
  node: MemoryNode
  devices: DeviceInfo[]
  frames: FrameState
  status: SystemStatus
  onRefreshMemory: () => void
  onRefreshState: () => void
  onRefreshDevices: () => void
  onSelectNode: (ref: string, node: MemoryNode) => void
  onDeleted: () => void
  onOpenListeners: () => void
  onCopyRef: (ref: string) => void
}

interface ResultNote {
  kind: 'note' | 'ok' | 'error'
  text: string
}

const SHAPE_HINT: Record<NodeShape, string> = {
  str: '叶子节点：只有正文与控制数据',
  dict: '字典容器：描述写在 metadata，子节点是键 → 值',
  list: '列表容器：描述写在 metadata，追加位置固定 -1',
  link: '链接节点：没有子节点；读写引用会进入目标',
  device: '挂载点：行为由设备自己声明',
}

/** 正文文本框的行数：按内容行数自适应，夹在 4..20 行之间，剩下的交给拖拽/滚动。 */
function autoRows(value: string): number {
  const lines = value === '' ? 1 : value.split('\n').length
  return Math.min(Math.max(lines + 1, 4), 20)
}

function formatResult(content: string | null): ResultNote {
  if (content === null || content === '') return { kind: 'note', text: '工具响应为空' }
  return { kind: isErrorMessage(content) ? 'error' : 'ok', text: content }
}

/**
 * 节点工作台：三个正交维度分区呈现。
 *
 * ① 节点类型（str/dict/list/link）：编辑操作（memory_write / memory_make / memory_delete）；
 * ② 数据类型 ctrl.type：快捷修改、结构校验、类型动作（程序启动、handler 监听）、ctrl 原样编辑；
 * ③ 设备：只有设备节点出现，行为来自 GET /api/devices 与 POST /api/device-action。
 *
 * 一条动作就是一条指令，不打包宏。
 */
export default function NodeWorkbench({
  refName,
  node,
  devices,
  frames,
  status,
  onRefreshMemory,
  onRefreshState,
  onRefreshDevices,
  onSelectNode,
  onDeleted,
  onOpenListeners,
  onCopyRef,
}: Props) {
  const shape = nodeShape(node)
  const mount = refName === '$MEM' ? '' : refName.slice('$MEM.'.length)
  const device = shape === 'link' ? devices.find((item) => item.mount === mount) : undefined
  const isDeviceNode = device !== undefined
  const ctrlType = ctrlTypeOf(node)
  const ctrl = nodeCtrl(node)
  const [desc, setDesc] = useState('')
  const [value, setValue] = useState('')
  const [newKey, setNewKey] = useState('')
  const [newType, setNewType] = useState<MemType>('str')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [quickValues, setQuickValues] = useState<Record<string, string>>({})
  const [ctrlDrafts, setCtrlDrafts] = useState<Record<string, string>>({})
  const [ctrlKey, setCtrlKey] = useState('')
  const [ctrlValue, setCtrlValue] = useState('')
  const [newCtrlType, setNewCtrlType] = useState('')
  const [listenCandidates, setListenCandidates] = useState<Listener[]>([])
  const [deviceArgs, setDeviceArgs] = useState<Record<number, string>>({})
  const [deviceResults, setDeviceResults] = useState<Record<number, string>>({})
  const [result, setResult] = useState<ResultNote | null>(null)
  const [busy, setBusy] = useState(false)
  const handled = useRef<Set<number>>(new Set())
  const { traces, submit } = useOperatorAction(frames)

  // 换节点：清空这次操作留下的界面状态。
  useEffect(() => {
    setNewKey('')
    setNewType('str')
    setConfirmDelete(false)
    setDeviceArgs({})
    setDeviceResults({})
    setResult(null)
    setBusy(false)
  }, [refName])

  // 节点数据刷新（保存成功后 App 会重取整树）：只同步字段草稿，不清操作结果。
  useEffect(() => {
    setDesc(nodeDescription(node))
    setValue(node.kind === 'str' ? node.value : '')
    setQuickValues(Object.fromEntries(quickFields(node, ctrlType).map((field) => [field.key, field.value])))
    setCtrlDrafts(
      Object.fromEntries(Object.entries(nodeCtrl(node) ?? {}).map(([key, raw]) => [key, String(raw)])),
    )
    setCtrlKey('')
    setCtrlValue('')
    setNewCtrlType('')
  }, [node, ctrlType])

  useEffect(() => {
    if (!isHandlerType(ctrlType)) {
      setListenCandidates([])
      return
    }
    let alive = true
    apiGet<ListenersResponse>('/api/listeners')
      .then((response) => {
        if (!alive) return
        setListenCandidates(response.listeners.filter((listener) => listener.handler_ref === refName))
      })
      .catch(() => {
        if (alive) setListenCandidates([])
      })
    return () => {
      alive = false
    }
  }, [refName, ctrlType])

  // 工具响应到达：刷新内存树与对话清单；删除成功后回到无选中。
  useEffect(() => {
    let deleted = false
    for (const trace of Object.values(traces)) {
      if (handled.current.has(trace.id)) continue
      if (trace.phase !== 'responded' && trace.phase !== 'no_response') continue
      handled.current.add(trace.id)
      if (trace.phase === 'no_response') {
        setResult({ kind: 'error', text: `${trace.tool}：没有工具响应（未知工具或缺响应）` })
        continue
      }
      setResult(formatResult(trace.content))
      onRefreshMemory()
      if (trace.tool === 'create_cmd' || trace.tool === 'listen_event') onRefreshState()
      if (trace.tool === 'memory_delete' && !isErrorMessage(trace.content ?? '')) deleted = true
    }
    if (deleted) onDeleted()
  }, [traces, onRefreshMemory, onRefreshState, onDeleted])

  const run = async (action: OperatorAction) => {
    if (!status.canOperate) {
      setResult({ kind: 'error', text: `${status.label}：当前不可操作` })
      return
    }
    setBusy(true)
    setResult({ kind: 'note', text: `已提交 ${action.tool}，等待工具响应…` })
    try {
      await submit(action)
    } catch (err) {
      setResult({ kind: 'error', text: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  /**
   * 按顺序提交多条指令，只有最后一条让出执行权。
   * 数据类型的快捷修改用它：目标子节点不存在时先 `memory_make` 建 str，再 `memory_write` 写值，
   * 两条指令在同一轮安全点前排队，不会在中间把执行权让出去。
   */
  const runSequence = async (actions: OperatorAction[]) => {
    if (!status.canOperate) {
      setResult({ kind: 'error', text: `${status.label}：当前不可操作` })
      return
    }
    if (actions.length === 0) return
    setBusy(true)
    setResult({ kind: 'note', text: `已提交 ${actions.map((action) => action.tool).join(' → ')}，等待工具响应…` })
    try {
      for (let index = 0; index < actions.length; index += 1) {
        const last = index === actions.length - 1
        await submit({ ...actions[index], release: last })
      }
    } catch (err) {
      setResult({ kind: 'error', text: describeError(err) })
    } finally {
      setBusy(false)
    }
  }

  /** 数据类型的快捷修改：目标子节点不存在时自动用 memory_make 建出来，再写值。 */
  const saveQuickField = (field: QuickField) => {
    const value = quickValues[field.key] ?? ''
    const target = childRef(refName, field.key)
    const existing = childNode(node, field.key)
    if (existing !== undefined && existing.kind !== 'str') {
      setResult({ kind: 'error', text: `${field.label} 对应的子节点不是 str，不能用快捷修改覆盖` })
      return
    }
    if (existing === undefined) {
      if (shape !== 'dict' && shape !== 'list') {
        setResult({ kind: 'error', text: `${shapeLabel(shape)} 不能有子节点：无法自动创建 ${field.key}` })
        return
      }
      void runSequence([memoryMake(refName, field.key, 'str'), memoryWrite(target, value)])
      return
    }
    void run(memoryWrite(target, value))
  }

  const runDeviceAction = async (index: number, actionName: string | null) => {
    if (!device) return
    let args: Record<string, unknown> = {}
    const raw = (deviceArgs[index] ?? '').trim()
    if (raw !== '') {
      try {
        const parsed: unknown = JSON.parse(raw)
        if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
          setDeviceResults((prev) => ({ ...prev, [index]: '参数必须是 JSON 对象' }))
          return
        }
        args = parsed as Record<string, unknown>
      } catch (err) {
        setDeviceResults((prev) => ({ ...prev, [index]: `JSON 不合法：${(err as Error).message}` }))
        return
      }
    }
    setBusy(true)
    try {
      const record = await apiSend<{ result?: unknown; error?: string | null }>(
        '/api/device-action',
        'POST',
        // 动作名优先（动态动作表在两次请求之间会变化）；没有名字时退回 index。
        { mount: device.mount, name: actionName ?? undefined, index, arguments: args },
      )
      const text = record.error
        ?? (record.result === undefined ? '已调用（没有返回值）' : JSON.stringify(record.result))
      setDeviceResults((prev) => ({ ...prev, [index]: text }))
      onRefreshDevices()
    } catch (err) {
      setDeviceResults((prev) => ({ ...prev, [index]: describeError(err) }))
    } finally {
      setBusy(false)
    }
  }

  const checks = structureChecks(node, ctrlType)
  const quick = quickFields(node, ctrlType)
  const missingQuick = quick.filter((field) => childNode(node, field.key) === undefined)
  const summary = typeSummary(ctrlType)
  const children = childEntries(node)
  const paraRef = quickValues.para_ref ?? paraRefOf(node) ?? ''

  return (
    <section className="workbench-view node-workbench" data-testid="node-workbench">
      <header className="workbench-head">
        <span className="workbench-title mono" data-testid="node-path">{refName}</span>
        <span className="chip" data-testid="node-shape">节点类型：{shapeLabel(shape)}</span>
        {isDeviceNode ? (
          <span className="chip" data-testid="node-device">设备节点</span>
        ) : (
          <span className="chip" data-testid="node-ctrl-type">
            数据类型：{ctrlType === null ? '无 ctrl.type' : `ctrl.type=${ctrlType}`}
          </span>
        )}
        <span className="workbench-spacer" />
        <button type="button" className="link" onClick={() => onCopyRef(refName)}>复制地址</button>
        <span className="workbench-note">操作员以 <code>cid -1</code> 身份执行内存指令</span>
      </header>

      <div className="node-view">
        <div className="node-sub">{SHAPE_HINT[shape]}</div>
        {result ? (
          <div className={`node-result is-${result.kind}`} data-testid="node-result">{result.text}</div>
        ) : null}

        {/* 区域 ①：节点类型 */}
        <section className="node-region" data-testid="node-region-shape">
          <header className="node-region-head">
            <span className="node-region-index">①</span>
            <span className="node-region-title">节点类型 · {shapeLabel(shape)}</span>
            <span className="node-region-hint">memory_write / memory_make / memory_delete 按节点类型作用</span>
          </header>

          {isDeviceNode ? (
            <div className="node-note">挂载点在内存树里表现为链接节点；设备没有内存编辑操作，设备行为见区域 ③。</div>
          ) : shape === 'str' ? (
            <>
              <div className="node-field">
                <label htmlFor="node-value">正文</label>
                <textarea
                  id="node-value"
                  className="node-value-textarea"
                  data-testid="node-value"
                  rows={autoRows(value)}
                  value={value}
                  spellCheck={false}
                  onChange={(event) => setValue(event.target.value)}
                />
              </div>
              <div className="node-field-hint" data-testid="node-value-meta">
                {value === '' ? 0 : value.split('\n').length} 行 · {value.length} 字符 · 等宽显示，可拖拽调高
              </div>
              <div className="node-actions">
                <button type="button" className="primary" data-testid="node-save-value" disabled={busy} onClick={() => void run(memoryWrite(refName, value))}>
                  保存正文（memory_write）
                </button>
              </div>
              <div className="node-note">str 节点只有控制数据、没有描述数据；ctrl 在下方控制数据区域查看与编辑。</div>
            </>
          ) : shape === 'link' ? (
            <div className="node-kv">
              <div><span className="node-key">链接目标</span><span className="mono">{typeof node !== 'string' && node.kind === 'link' ? node.target : '—'}</span></div>
              <div className="node-note">链接没有子节点：读写这个引用会进入目标；路径之外的兄弟节点不可见，这是 capability 边界。</div>
            </div>
          ) : (
            <>
              <div className="node-field">
                <label htmlFor="node-desc">描述 meta</label>
                <textarea id="node-desc" data-testid="node-desc" value={desc} onChange={(event) => setDesc(event.target.value)} />
              </div>
              <div className="node-actions">
                <button type="button" data-testid="node-save-desc" disabled={busy} onClick={() => void run(memoryWrite(refName, desc))}>
                  保存描述（memory_write）
                </button>
              </div>
              <div className="node-field">
                <label htmlFor="node-new-key">新建子节点</label>
                <span className="node-field-row">
                  <input
                    id="node-new-key"
                    data-testid="node-new-key"
                    value={shape === 'list' ? '-1' : newKey}
                    placeholder={shape === 'list' ? 'list 追加位置固定 -1' : '子节点键名（不含点号）'}
                    disabled={shape === 'list'}
                    onChange={(event) => setNewKey(event.target.value)}
                  />
                  <select data-testid="node-new-type" value={newType} onChange={(event) => setNewType(event.target.value as MemType)}>
                    <option value="str">str</option>
                    <option value="dict">dict</option>
                    <option value="list">list</option>
                  </select>
                  <button
                    type="button"
                    data-testid="node-make"
                    disabled={busy || (shape === 'dict' && makeKeyFor('dict', newKey) === null)}
                    onClick={() => {
                      const key = makeKeyFor(shape === 'list' ? 'list' : 'dict', newKey)
                      if (key === null) {
                        setResult({ kind: 'error', text: '键名不能为空、不能含点号' })
                        return
                      }
                      void run(memoryMake(refName, key, newType))
                    }}
                  >
                    memory_make
                  </button>
                </span>
              </div>
              {children.length > 0 ? (
                <div className="node-children" data-testid="node-children">
                  {children.map(([key, child]) => (
                    <button key={key} type="button" className="node-child" onClick={() => onSelectNode(childRef(refName, key), child)}>
                      <b>{key}</b>
                      <span>{child.kind}</span>
                      {child.kind !== 'device' && child.ctrl && typeof child.ctrl.type === 'string' ? (
                        <em>{child.ctrl.type}</em>
                      ) : null}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="node-note">空容器。</div>
              )}
              {confirmDelete ? (
                <div className="node-actions" data-testid="node-delete-confirm">
                  <span className="node-warn">删除 {refName}？空容器 / 叶子 / 树链接可删。</span>
                  <button type="button" className="danger" disabled={busy} data-testid="node-delete-confirm-button" onClick={() => { setConfirmDelete(false); void run(memoryDelete(refName)) }}>
                    确认删除
                  </button>
                  <button type="button" className="link" onClick={() => setConfirmDelete(false)}>取消</button>
                </div>
              ) : (
                <div className="node-actions">
                  <button type="button" className="danger" data-testid="node-delete" disabled={busy} onClick={() => setConfirmDelete(true)}>
                    删除节点（memory_delete）
                  </button>
                </div>
              )}
            </>
          )}
        </section>

        {/* 区域 ②：数据类型 ctrl.type */}
        {isDeviceNode ? null : (
          <section className="node-region" data-testid="node-region-type">
            <header className="node-region-head">
              <span className="node-region-index">②</span>
              <span className="node-region-title">
                数据类型 · {ctrlType === null ? '无 ctrl.type' : `ctrl.type = ${ctrlType}`}
              </span>
              <span className="node-region-hint">快捷修改 · 数据结构校验 · 类型动作</span>
            </header>

            {ctrlType === null ? (
              <div className="node-type-empty">
                <div className="node-note">
                  没有数据类型：区域一给通用编辑；这个节点也可以在这里直接设置 <code>ctrl.type</code>，设置后按该类型显示快捷修改与类型动作。
                </div>
                <div className="node-field-row" data-testid="node-set-ctrl-type-row">
                  <input
                    data-testid="node-set-ctrl-type-input"
                    placeholder="ctrl.type，如 python / handler / para"
                    value={newCtrlType}
                    onChange={(event) => setNewCtrlType(event.target.value)}
                  />
                  <button
                    type="button"
                    data-testid="node-set-ctrl-type"
                    disabled={busy || newCtrlType.trim() === ''}
                    onClick={() => void run(ctrlSet(refName, 'type', newCtrlType.trim()))}
                  >
                    设置 ctrl.type
                  </button>
                </div>
              </div>
            ) : (
              <>
                {summary ? <div className="node-type-summary">{summary}</div> : null}
                {checks.length > 0 ? (
                  <div className="node-checks" data-testid="node-checks">
                    {checks.map((check) => (
                      <div key={check.label} className={`node-check ${check.ok ? 'is-ok' : 'is-bad'}`}>
                        <span>{check.ok ? '✓' : '✗'}</span>
                        {check.label}
                      </div>
                    ))}
                  </div>
                ) : null}
                {quick.length > 0 ? (
                  <div className="node-quick" data-testid="node-quick">
                    {missingQuick.length > 0 ? (
                      <div className="node-note" data-testid="node-quick-missing">
                        字段 {missingQuick.map((field) => field.key).join('、')} 对应的子节点还不存在：保存时会先
                        `memory_make` 建 str 节点，再 `memory_write` 写入。
                      </div>
                    ) : null}
                    {quick.map((field) => (
                      <div className="node-field" key={field.key}>
                        <label htmlFor={`quick-${field.key}`}>{field.label}</label>
                        <span className="node-field-row">
                          {field.kind === 'textarea' ? (
                            <textarea
                              id={`quick-${field.key}`}
                              data-testid={`node-quick-${field.key}`}
                              rows={autoRows(quickValues[field.key] ?? '')}
                              value={quickValues[field.key] ?? ''}
                              spellCheck={false}
                              onChange={(event) => setQuickValues((prev) => ({ ...prev, [field.key]: event.target.value }))}
                            />
                          ) : (
                            <input
                              id={`quick-${field.key}`}
                              data-testid={`node-quick-${field.key}`}
                              value={quickValues[field.key] ?? ''}
                              onChange={(event) => setQuickValues((prev) => ({ ...prev, [field.key]: event.target.value }))}
                            />
                          )}
                          <button
                            type="button"
                            data-testid={`node-quick-save-${field.key}`}
                            disabled={busy}
                            onClick={() => saveQuickField(field)}
                          >
                            保存
                          </button>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : null}
                <div className="node-actions" data-testid="node-type-actions">
                  {isProgramType(ctrlType) ? (
                    <>
                      <span className="node-note">
                        para_ref = <code className="mono">{paraRef !== '' ? paraRef : '（缺失）'}</code>
                      </span>
                      <button
                        type="button"
                        className="primary"
                        data-testid="node-launch"
                        disabled={busy || paraRef === ''}
                        onClick={() => void run(createCmd(refName, paraRef))}
                      >
                        启动为常驻对话（create_cmd）
                      </button>
                    </>
                  ) : null}
                  {isHandlerType(ctrlType) ? (
                    <>
                      <button type="button" data-testid="node-view-listeners" disabled={busy} onClick={onOpenListeners}>
                        查看监听器
                      </button>
                      <button type="button" data-testid="node-register-listener" disabled={busy} onClick={() => void run(listenerRegister(refName))}>
                        注册监听（listen_event）
                      </button>
                      {listenCandidates.map((listener) => (
                        <span key={listener.listener_id} className="node-listener">
                          #{listener.listener_id} {String(listener.event_type)} · {String(listener.phase)}
                          <button
                            type="button"
                            className="link"
                            data-testid={`node-cancel-listener-${listener.listener_id}`}
                            disabled={busy}
                            onClick={() => void run(listenerCancel(listener.listener_id))}
                          >
                            取消
                          </button>
                        </span>
                      ))}
                    </>
                  ) : null}
                </div>
              </>
            )}

            <div className="node-ctrl" data-testid="node-ctrl">
              <div className="node-region-sub">控制数据 ctrl（元数据的控制部分，edit_metadata set / del）</div>
              {shape === 'link' ? (
                <>
                  <div className="node-note" data-testid="node-ctrl-link-note">
                    链接的引用会进入目标：这里显示的是链接节点自己的 ctrl，指令层没有直接改链接自身控制数据的路径，只读展示；
                    要改目标节点的 ctrl，请打开目标节点。
                  </div>
                  {Object.entries(ctrl ?? {}).map(([key, raw]) => (
                    <div className="node-ctrl-row" key={key}>
                      <span className="node-ctrl-key mono">{key}</span>
                      <input readOnly value={String(raw)} />
                    </div>
                  ))}
                  {Object.keys(ctrl ?? {}).length === 0 ? <div className="node-note">（没有控制数据）</div> : null}
                </>
              ) : (
                <>
                  {Object.entries(ctrl ?? {}).map(([key, raw]) => (
                    <div className="node-ctrl-row" key={key}>
                      <span className="node-ctrl-key mono">{key}</span>
                      <input
                        data-testid={`node-ctrl-input-${key}`}
                        value={ctrlDrafts[key] ?? String(raw)}
                        onChange={(event) => setCtrlDrafts((prev) => ({ ...prev, [key]: event.target.value }))}
                      />
                      <button type="button" disabled={busy} data-testid={`node-ctrl-set-${key}`} onClick={() => void run(ctrlSet(refName, key, ctrlDrafts[key] ?? String(raw)))}>
                        set
                      </button>
                      <button type="button" disabled={busy} data-testid={`node-ctrl-del-${key}`} onClick={() => void run(ctrlDel(refName, key))}>
                        del
                      </button>
                    </div>
                  ))}
                  <div className="node-note">
                    {ctrl === null || Object.keys(ctrl).length === 0
                      ? '还没有控制数据（ctrl 为空）；在下面填键值即可新增。'
                      : '新增或覆盖一条控制数据：'}
                  </div>
                  <div className="node-ctrl-row">
                    <input placeholder="新键" data-testid="node-ctrl-new-key" value={ctrlKey} onChange={(event) => setCtrlKey(event.target.value)} />
                    <input placeholder="值" data-testid="node-ctrl-new-value" value={ctrlValue} onChange={(event) => setCtrlValue(event.target.value)} />
                    <button
                      type="button"
                      data-testid="node-ctrl-add"
                      disabled={busy || ctrlKey.trim() === ''}
                      onClick={() => void run(ctrlSet(refName, ctrlKey.trim(), ctrlValue))}
                    >
                      新增键值
                    </button>
                  </div>
                </>
              )}
            </div>
          </section>
        )}

        {/* 区域 ③：设备 */}
        {isDeviceNode && device ? (
          <section className="node-region" data-testid="node-region-device">
            <header className="node-region-head">
              <span className="node-region-index">③</span>
              <span className="node-region-title">设备 · {device.type}</span>
              <span className="node-region-hint">GET /api/devices · 行为由设备自己声明</span>
            </header>
            {device.error ? <div className="node-warn">{device.error}</div> : null}
            <div className="node-kv">
              <div><span className="node-key">设备类型</span><span className="mono">{device.type}</span></div>
              <div><span className="node-key">挂载键</span><span className="mono">{device.mount}</span></div>
              <div><span className="node-key">pretend_as_type</span><span className="mono">{device.description.pretend_as_type ?? '—'}</span></div>
              <div><span className="node-key">LLM 表示</span><span className="mono">{device.description.preview ?? '（取不到）'}</span></div>
            </div>
            <div className="node-region-sub">设备声明的调试动作（{device.actions.length}）</div>
            {device.actions.length === 0 ? <div className="node-note">这台设备没有声明调试动作。</div> : null}
            {device.actions.map((action) => (
              <div className="node-device-action" key={action.index}>
                <div className="node-device-action-head">
                  <b>#{action.index} {action.name ?? '（无名）'}</b>
                  <span>{action.callable ? '可调用' : '不可调用'}</span>
                </div>
                {action.description ? <div className="node-note">{action.description}</div> : null}
                {action.parameters ? (
                  <pre className="node-device-schema">{JSON.stringify(action.parameters)}</pre>
                ) : null}
                <div className="node-field-row">
                  <input
                    data-testid={`node-device-args-${action.index}`}
                    placeholder="参数 JSON（默认 {}）"
                    value={deviceArgs[action.index] ?? ''}
                    onChange={(event) => setDeviceArgs((prev) => ({ ...prev, [action.index]: event.target.value }))}
                  />
                  <button
                    type="button"
                    data-testid={`node-device-run-${action.index}`}
                    disabled={busy || !action.callable}
                    onClick={() => void runDeviceAction(action.index, action.name)}
                  >
                    运行
                  </button>
                </div>
                {deviceResults[action.index] ? (
                  <div className="node-note" data-testid={`node-device-result-${action.index}`}>
                    {deviceResults[action.index]}
                  </div>
                ) : null}
              </div>
            ))}
            <div className="node-note">设备调试动作直接调用设备声明的 run：不入核心、不产生事件、不重放。</div>
          </section>
        ) : null}
      </div>
    </section>
  )
}

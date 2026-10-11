import type { DeviceInfo } from '../api/types'

/**
 * 设备详情：第一轮只读——类型、挂载键、LLM 表示，以及设备自己声明的调试动作清单。
 * 动作的调用（R10 全量）留到下一轮；`$MEM.inputs` 的喂输入框在右栏动作区。
 */
export default function DeviceView({ device }: { device: DeviceInfo | undefined }) {
  if (!device) return <div className="pane-body muted">这个设备已经不在挂载表里（可能被屏蔽了）</div>
  return (
    <div className="pane-body" data-testid="device-view">
      <div className="detail-head">
        <span className="mono">$MEM.{device.mount}</span>
        <span>{device.type}</span>
      </div>
      {device.error ? <div className="hint error">{device.error}</div> : null}
      <div className="kv">
        <div>
          <span className="key">pretend_as_type</span>
          <span>{device.description.pretend_as_type ?? '—'}</span>
        </div>
        <div>
          <span className="key">LLM 表示</span>
          <span>{device.description.preview ?? '（取不到）'}</span>
        </div>
      </div>
      <div className="section-title">设备声明的调试动作（{device.actions.length}）</div>
      {device.actions.length === 0 ? <div className="muted">这个设备没有声明调试动作</div> : null}
      {device.actions.map((action) => (
        <div className="message" key={action.index}>
          <div className="message-head">
            <span className="role role-tool">#{action.index}</span>
            <span>{action.name ?? '（无名）'}</span>
            <span className={`tag ${action.callable ? '' : 'tag-warn'}`}>
              {action.callable ? '可调用' : '不可调用'}
            </span>
          </div>
          <div className="muted small">{action.description ?? ''}</div>
          {action.parameters ? (
            <pre className="content muted">{JSON.stringify(action.parameters, null, 2)}</pre>
          ) : null}
        </div>
      ))}
    </div>
  )
}

import { TRAY_LABEL_TEXT, trayPreview, type TrayValue } from '../lib/valueTray'

interface Props {
  values: TrayValue[]
  selected: string | null
  onSelect: (value: string) => void
  /** 剪贴板写入失败等降级提示；不影响暂存区。 */
  note: string | null
}

/**
 * 值暂存区：只列最近复制出来的文本值，由操作员选中一条，再在参数框或 ref-input 旁点「填入」。
 * 不按参数名自动匹配、不预填、不生成参数——值的形状标签只是给人看的。
 */
export default function ValueTray({ values, selected, onSelect, note }: Props) {
  return (
    <div className="value-tray" data-testid="value-tray">
      <div className="section-title">
        值暂存区
        <span className="muted small">最近 {values.length} 条 · 选一条，在参数框旁点「填入」</span>
      </div>
      {values.length === 0 ? (
        <div className="hint muted small">
          还没有值。左栏内存树、对话列表、右栏事件流都有「复制」，复制完这里就能选。
        </div>
      ) : (
        <div className="row" style={{ padding: '2px 10px', gap: 4 }}>
          {values.map((item) => (
            <button
              type="button"
              key={item.id}
              data-testid="value-tray-item"
              className={item.value === selected ? 'chip' : 'chip chip-off'}
              title={item.value}
              style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis' }}
              onClick={() => onSelect(item.value)}
            >
              <span className="muted small">{TRAY_LABEL_TEXT[item.label]}</span>{' '}
              {trayPreview(item.value, 24)}
            </button>
          ))}
        </div>
      )}
      {selected ? (
        <div className="hint small" data-testid="value-tray-selected">
          已选：{trayPreview(selected, 70)}
        </div>
      ) : null}
      {note ? <div className="hint warn small">{note}</div> : null}
    </div>
  )
}

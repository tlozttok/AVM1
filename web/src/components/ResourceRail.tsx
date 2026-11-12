import type { ConversationBrief, DeviceInfo, MemoryNode } from '../api/types'
import ConversationList from './ConversationList'
import MemoryIndex from './MemoryIndex'

interface Props {
  conversations: ConversationBrief[]
  selectedCid: number | null
  activeCid: number | null
  onSelectConversation: (cid: number) => void
  onCopyCid: (cid: number) => void
  memoryRoot: MemoryNode | null
  devices: DeviceInfo[]
  memoryBusy: boolean
  memoryError: string | null
  memoryDirty: boolean
  selectedRef: string | null
  onRefreshMemory: () => void
  onSelectNode: (ref: string, node: MemoryNode) => void
  onCopyRef: (ref: string) => void
}

/** 左资源栏：对话树 + 内存树。点选中即在中间工作台打开。 */
export default function ResourceRail({
  conversations,
  selectedCid,
  activeCid,
  onSelectConversation,
  onCopyCid,
  memoryRoot,
  devices,
  memoryBusy,
  memoryError,
  memoryDirty,
  selectedRef,
  onRefreshMemory,
  onSelectNode,
  onCopyRef,
}: Props) {
  return (
    <aside className="rail" data-testid="resource-rail">
      <ConversationList
        conversations={conversations}
        selectedCid={selectedCid}
        activeCid={activeCid}
        onSelect={onSelectConversation}
        onCopyCid={onCopyCid}
      />
      <MemoryIndex
        root={memoryRoot}
        devices={devices}
        busy={memoryBusy}
        error={memoryError}
        dirty={memoryDirty}
        selectedRef={selectedRef}
        onRefresh={onRefreshMemory}
        onSelect={onSelectNode}
        onCopyRef={onCopyRef}
      />
    </aside>
  )
}

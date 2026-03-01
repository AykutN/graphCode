import { Handle, Position } from '@xyflow/react'
import { GraphNode } from '../../types/graph'

interface Props {
  data: GraphNode
  selected: boolean
  color: string
  badge: string
}

export function BaseNode({ data, selected, color, badge }: Props) {
  return (
    <div
      className={`rounded-lg border-2 px-4 py-2 min-w-[160px] shadow-sm
        ${selected ? 'border-blue-500' : 'border-gray-300'}
        ${color}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono bg-black/10 px-1 rounded">{badge}</span>
        <span className="font-semibold text-sm truncate">{data.label}</span>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

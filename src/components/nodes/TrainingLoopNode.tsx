import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function TrainingLoopNode({ data, selected }: NodeProps) {
  return (
    <BaseNode
      data={data as unknown as GraphNode}
      selected={!!selected}
      color="bg-amber-50 border-amber-200"
      badge="loop"
    />
  )
}

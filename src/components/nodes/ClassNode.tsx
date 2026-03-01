import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function ClassNode({ data, selected }: NodeProps) {
  return (
    <BaseNode
      data={data as unknown as GraphNode}
      selected={!!selected}
      color="bg-purple-50 border-purple-200"
      badge="cls"
    />
  )
}

import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function DataNode({ data, selected }: NodeProps) {
  return (
    <BaseNode
      data={data as unknown as GraphNode}
      selected={!!selected}
      color="bg-green-50 border-green-200"
      badge="data"
    />
  )
}

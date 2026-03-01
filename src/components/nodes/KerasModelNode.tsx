import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function KerasModelNode({ data, selected }: NodeProps) {
  return (
    <BaseNode
      data={data as unknown as GraphNode}
      selected={!!selected}
      color="bg-teal-50 border-teal-200"
      badge="tf"
    />
  )
}

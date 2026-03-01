import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function ModuleNode({ data, selected }: NodeProps) {
  return (
    <BaseNode
      data={data as unknown as GraphNode}
      selected={!!selected}
      color="bg-blue-50 border-blue-200"
      badge="nn"
    />
  )
}

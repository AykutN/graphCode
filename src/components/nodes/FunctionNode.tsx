import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function FunctionNode({ data, selected }: NodeProps) {
  return (
    <BaseNode data={data as unknown as GraphNode} selected={!!selected} color="bg-gray-100" badge="fn" />
  )
}

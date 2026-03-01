import { EdgeProps, getBezierPath } from '@xyflow/react'
import { EdgeKind } from '../types/graph'

const EDGE_STYLES: Record<EdgeKind, { stroke: string; strokeDasharray?: string; label: string }> = {
  sequence: { stroke: '#374151', label: '→' },
  dataflow: { stroke: '#3B82F6', strokeDasharray: '5 3', label: '⇢' },
  dependency: { stroke: '#9CA3AF', strokeDasharray: '2 4', label: '⋯' },
}

export function AnnotatedEdge({
  sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition,
  data,
}: EdgeProps) {
  const kind = (data?.kind as EdgeKind) ?? 'sequence'
  const style = EDGE_STYLES[kind]
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  })

  return (
    <>
      <path
        d={edgePath}
        fill="none"
        stroke={style.stroke}
        strokeWidth={1.5}
        strokeDasharray={style.strokeDasharray}
      />
      <text x={labelX} y={labelY} textAnchor="middle" fontSize={12} fill={style.stroke}>
        {style.label}
      </text>
    </>
  )
}

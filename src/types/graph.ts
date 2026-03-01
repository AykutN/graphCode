export type EdgeKind = 'sequence' | 'dataflow' | 'dependency'

export type NodeType =
  | 'nn.Module'
  | 'keras.Model'
  | 'function'
  | 'class'
  | 'training_loop'
  | 'data'

export interface LayerInfo {
  name: string
  in_features?: number
  out_features?: number
  activation?: string
}

export interface NodeMeta {
  input_size?: number | number[]
  output_size?: number | number[]
  hidden?: number[]
  layers?: LayerInfo[]
  optimizer?: string
  lr?: number
  loss?: string
  args?: string[]
  return_type?: string
  methods?: string[]
  source_range: { line_start: number; line_end: number }
}

export interface GraphNode {
  id: string
  type: NodeType
  label: string
  meta: NodeMeta
}

export interface GraphEdge {
  id: string
  from: string
  to: string
  kind: EdgeKind
}

export interface Graph {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

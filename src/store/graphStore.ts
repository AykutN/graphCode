import { create } from 'zustand'
import { Graph, GraphNode } from '../types/graph'
import { Node, Edge } from '@xyflow/react'

interface GraphState {
  graph: Graph | null
  selectedNodeId: string | null
  rfNodes: Node[]
  rfEdges: Edge[]
  setGraph: (graph: Graph) => void
  selectNode: (id: string | null) => void
  clearGraph: () => void
}

function toRFNodes(nodes: GraphNode[]): Node[] {
  return nodes.map((n, i) => ({
    id: n.id,
    type: n.type.replace('.', '_').replace('/', '_'),
    position: { x: (i % 4) * 260, y: Math.floor(i / 4) * 160 },
    data: n as unknown as Record<string, unknown>,
  }))
}

function toRFEdges(edges: Graph['edges']): Edge[] {
  return edges.map(e => ({
    id: e.id,
    source: e.from,
    target: e.to,
    type: 'annotated',
    data: { kind: e.kind },
  }))
}

export const useGraphStore = create<GraphState>((set) => ({
  graph: null,
  selectedNodeId: null,
  rfNodes: [],
  rfEdges: [],

  setGraph: (graph) =>
    set({
      graph,
      rfNodes: toRFNodes(graph.nodes),
      rfEdges: toRFEdges(graph.edges),
    }),

  selectNode: (id) => set({ selectedNodeId: id }),

  clearGraph: () =>
    set({ graph: null, selectedNodeId: null, rfNodes: [], rfEdges: [] }),
}))

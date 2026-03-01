import { useGraphStore } from '../store/graphStore'
import { GraphNode, LayerInfo } from '../types/graph'

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className="font-mono text-xs text-gray-900">{value}</span>
    </div>
  )
}

function MLModuleView({ node }: { node: GraphNode }) {
  const { meta } = node
  return (
    <div className="space-y-2 text-sm">
      <Row label="Type" value={node.type} />
      {meta.input_size !== undefined && (
        <Row label="Input" value={String(meta.input_size)} />
      )}
      {meta.output_size !== undefined && (
        <Row label="Output" value={String(meta.output_size)} />
      )}
      {meta.layers && meta.layers.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Layers</p>
          <div className="space-y-1">
            {meta.layers.map((l: LayerInfo, i: number) => (
              <div key={i} className="font-mono text-xs bg-gray-50 rounded px-2 py-1">
                {l.name}: {l.type}
                {l.in_features !== undefined && ` (${l.in_features} → ${l.out_features})`}
              </div>
            ))}
          </div>
        </div>
      )}
      {meta.optimizer && <Row label="Optimizer" value={meta.optimizer} />}
      {meta.lr !== undefined && <Row label="LR" value={String(meta.lr)} />}
      {meta.loss && <Row label="Loss" value={meta.loss} />}
    </div>
  )
}

export function InspectorPanel() {
  const { graph, selectedNodeId } = useGraphStore()

  if (!selectedNodeId || !graph) {
    return (
      <div className="w-64 border-l border-gray-200 p-4 flex items-center justify-center">
        <p className="text-gray-400 text-sm text-center">Select a block to inspect</p>
      </div>
    )
  }

  const node = graph.nodes.find(n => n.id === selectedNodeId)
  if (!node) return null

  return (
    <div className="w-64 border-l border-gray-200 p-4 overflow-y-auto">
      <h2 className="font-bold text-gray-900 mb-1">{node.label}</h2>
      <p className="text-xs text-gray-400 font-mono mb-3">{node.type}</p>
      {node.meta.source_range && (
        <p className="text-xs text-gray-400 mb-3">
          Lines {node.meta.source_range.line_start}–{node.meta.source_range.line_end}
        </p>
      )}
      {(node.type === 'nn.Module' || node.type === 'keras.Model') ? (
        <MLModuleView node={node} />
      ) : (
        <div className="text-sm text-gray-600 space-y-1">
          {node.meta.args && (
            <Row label="Args" value={node.meta.args.join(', ')} />
          )}
          {node.meta.methods && (
            <Row label="Methods" value={node.meta.methods.join(', ')} />
          )}
        </div>
      )}
    </div>
  )
}

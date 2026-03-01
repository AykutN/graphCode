import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore } from '../store/graphStore'
import { FunctionNode } from './nodes/FunctionNode'
import { ModuleNode } from './nodes/ModuleNode'
import { ClassNode } from './nodes/ClassNode'
import { TrainingLoopNode } from './nodes/TrainingLoopNode'
import { DataNode } from './nodes/DataNode'
import { KerasModelNode } from './nodes/KerasModelNode'
import { AnnotatedEdge } from '../edges/AnnotatedEdge'

const nodeTypes = {
  function: FunctionNode,
  class: ClassNode,
  nn_Module: ModuleNode,
  keras_Model: KerasModelNode,
  training_loop: TrainingLoopNode,
  data: DataNode,
}

const edgeTypes = {
  annotated: AnnotatedEdge,
}

export function GraphCanvas() {
  const { rfNodes, rfEdges, selectNode } = useGraphStore()

  return (
    <div className="flex-1 h-full">
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeClick={(_, node) => selectNode(node.id)}
        onPaneClick={() => selectNode(null)}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  )
}

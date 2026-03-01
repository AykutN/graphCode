import { GraphCanvas } from './components/GraphCanvas'
import { InspectorPanel } from './components/InspectorPanel'
import { Toolbar } from './components/Toolbar'

export default function App() {
  return (
    <div className="flex flex-col h-screen bg-white">
      <Toolbar />
      <div className="flex flex-1 overflow-hidden">
        <GraphCanvas />
        <InspectorPanel />
      </div>
    </div>
  )
}

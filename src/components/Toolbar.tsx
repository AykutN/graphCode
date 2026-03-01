import { useRef, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { useGraphStore } from '../store/graphStore'
import { Graph } from '../types/graph'

export function Toolbar() {
  const { setGraph, clearGraph } = useGraphStore()
  const [pasteOpen, setPasteOpen] = useState(false)
  const [pasteCode, setPasteCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function parseAndSet(source: string) {
    try {
      const result = await invoke<string>('parse_code', { source })
      const graph: Graph = JSON.parse(result)
      setGraph(graph)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }

  function handleOpenFile() {
    fileInputRef.current?.click()
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const source = ev.target?.result as string
      parseAndSet(source)
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  async function handlePaste() {
    await parseAndSet(pasteCode)
    setPasteOpen(false)
    setPasteCode('')
  }

  return (
    <div className="h-12 border-b border-gray-200 flex items-center px-4 gap-3 bg-white">
      <span className="font-bold text-gray-900 mr-2">graphTorch</span>
      <input
        ref={fileInputRef}
        type="file"
        accept=".py"
        className="hidden"
        onChange={handleFileChange}
      />
      <button
        onClick={handleOpenFile}
        className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200"
      >
        Open File
      </button>
      <button
        onClick={() => setPasteOpen(true)}
        className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200"
      >
        Paste Code
      </button>
      <button
        onClick={clearGraph}
        className="text-sm px-3 py-1 rounded bg-gray-100 hover:bg-gray-200"
      >
        Clear
      </button>
      {error && <span className="text-red-500 text-xs">{error}</span>}

      {pasteOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-[600px] shadow-xl">
            <h2 className="font-bold mb-3">Paste Python Code</h2>
            <textarea
              className="w-full h-64 font-mono text-sm border rounded p-2"
              value={pasteCode}
              onChange={e => setPasteCode(e.target.value)}
              placeholder="# paste your Python code here"
            />
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => setPasteOpen(false)}
                className="px-4 py-1.5 rounded border text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handlePaste}
                className="px-4 py-1.5 rounded bg-blue-600 text-white text-sm"
              >
                Visualize
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

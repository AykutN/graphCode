# graphTorch Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the MVP visualization layer — parse Python/PyTorch/TF code into an interactive block graph with a Tauri desktop app.

**Architecture:** A Python parser sidecar (AST + ML-aware passes) communicates with a React + ReactFlow frontend via Tauri IPC. Blocks represent code constructs; edges are annotated with type symbols. A right-side inspector panel shows block properties.

**Tech Stack:** Tauri (Rust), React + TypeScript, ReactFlow (`@xyflow/react`), Zustand, Tailwind CSS, Python `ast` module, pytest

---

## Project Structure

```
graphTorch/
├── src-tauri/
│   ├── src/main.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── components/
│   │   ├── GraphCanvas.tsx
│   │   ├── InspectorPanel.tsx
│   │   ├── Toolbar.tsx
│   │   └── nodes/
│   │       ├── ModuleNode.tsx
│   │       ├── FunctionNode.tsx
│   │       ├── ClassNode.tsx
│   │       ├── TrainingLoopNode.tsx
│   │       └── DataNode.tsx
│   ├── edges/
│   │   └── AnnotatedEdge.tsx
│   ├── store/
│   │   └── graphStore.ts
│   └── types/
│       └── graph.ts
├── parser/
│   ├── main.py
│   ├── ast_pass.py
│   ├── ml_pass.py
│   ├── edge_analysis.py
│   └── tests/
│       ├── test_ast_pass.py
│       ├── test_ml_pass.py
│       └── test_edge_analysis.py
├── docs/plans/
├── package.json
└── vite.config.ts
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `package.json`, `vite.config.ts`, `src/main.tsx`, `src-tauri/`
- Create: `src/types/graph.ts`

**Step 1: Scaffold Tauri + React project**

```bash
npm create tauri-app@latest . -- --template react-ts --manager npm
```

When prompted:
- App name: `graphTorch`
- Window title: `graphTorch`

**Step 2: Install frontend dependencies**

```bash
npm install @xyflow/react zustand tailwindcss @tailwindcss/vite
npx tailwindcss init
```

**Step 3: Configure Tailwind in `vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

**Step 4: Add Tailwind directives to `src/index.css`**

```css
@import "tailwindcss";
```

**Step 5: Create `src/types/graph.ts`**

```ts
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
```

**Step 6: Create parser directory**

```bash
mkdir -p parser/tests
touch parser/__init__.py parser/tests/__init__.py
```

**Step 7: Verify dev server starts**

```bash
npm run tauri dev
```

Expected: Tauri window opens with default React app.

**Step 8: Commit**

```bash
git add .
git commit -m "feat: scaffold Tauri + React + TypeScript project"
```

---

### Task 2: Python Parser — AST Pass

**Files:**
- Create: `parser/ast_pass.py`
- Create: `parser/tests/test_ast_pass.py`

**Step 1: Write failing tests**

```python
# parser/tests/test_ast_pass.py
import pytest
from parser.ast_pass import parse_ast

def test_detects_function():
    code = """
def train(model, data):
    pass
"""
    result = parse_ast(code)
    ids = [n['id'] for n in result['nodes']]
    assert 'train' in ids

def test_function_has_correct_type():
    code = "def forward(x): return x"
    result = parse_ast(code)
    node = next(n for n in result['nodes'] if n['id'] == 'forward')
    assert node['type'] == 'function'

def test_detects_class():
    code = """
class MyNet:
    def __init__(self):
        pass
"""
    result = parse_ast(code)
    ids = [n['id'] for n in result['nodes']]
    assert 'MyNet' in ids

def test_class_has_correct_type():
    code = "class MyNet: pass"
    result = parse_ast(code)
    node = next(n for n in result['nodes'] if n['id'] == 'MyNet')
    assert node['type'] == 'class'

def test_function_captures_source_range():
    code = "def foo():\n    pass"
    result = parse_ast(code)
    node = next(n for n in result['nodes'] if n['id'] == 'foo')
    assert node['meta']['source_range']['line_start'] == 1

def test_empty_code_returns_empty_graph():
    result = parse_ast("")
    assert result == {'nodes': [], 'edges': []}
```

**Step 2: Run tests to verify they fail**

```bash
cd parser && python -m pytest tests/test_ast_pass.py -v
```

Expected: `ImportError: cannot import name 'parse_ast'`

**Step 3: Implement `parser/ast_pass.py`**

```python
import ast
from typing import Any

def parse_ast(source: str) -> dict[str, list]:
    if not source.strip():
        return {'nodes': [], 'edges': []}

    tree = ast.parse(source)
    nodes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            nodes.append({
                'id': node.name,
                'type': 'function',
                'label': node.name,
                'meta': {
                    'args': [a.arg for a in node.args.args],
                    'source_range': {
                        'line_start': node.lineno,
                        'line_end': node.end_lineno,
                    }
                }
            })
        elif isinstance(node, ast.ClassDef):
            nodes.append({
                'id': node.name,
                'type': 'class',
                'label': node.name,
                'meta': {
                    'methods': [
                        n.name for n in ast.walk(node)
                        if isinstance(n, ast.FunctionDef)
                    ],
                    'bases': [
                        ast.unparse(b) for b in node.bases
                    ],
                    'source_range': {
                        'line_start': node.lineno,
                        'line_end': node.end_lineno,
                    }
                }
            })

    return {'nodes': nodes, 'edges': []}
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ast_pass.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add parser/ast_pass.py parser/tests/test_ast_pass.py
git commit -m "feat: add Python AST pass for functions and classes"
```

---

### Task 3: Python Parser — ML-Aware Pass

**Files:**
- Create: `parser/ml_pass.py`
- Create: `parser/tests/test_ml_pass.py`

**Step 1: Write failing tests**

```python
# parser/tests/test_ml_pass.py
import pytest
from parser.ml_pass import enrich_ml_nodes

PYTORCH_CODE = """
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

def test_nn_module_type_upgraded():
    nodes = [{'id': 'MyNet', 'type': 'class', 'label': 'MyNet',
              'meta': {'bases': ['nn.Module'], 'source_range': {'line_start': 4, 'line_end': 13}}}]
    result = enrich_ml_nodes(nodes, PYTORCH_CODE)
    node = next(n for n in result if n['id'] == 'MyNet')
    assert node['type'] == 'nn.Module'

def test_nn_module_layers_extracted():
    nodes = [{'id': 'MyNet', 'type': 'class', 'label': 'MyNet',
              'meta': {'bases': ['nn.Module'], 'source_range': {'line_start': 4, 'line_end': 13}}}]
    result = enrich_ml_nodes(nodes, PYTORCH_CODE)
    node = next(n for n in result if n['id'] == 'MyNet')
    layer_names = [l['name'] for l in node['meta']['layers']]
    assert 'fc1' in layer_names
    assert 'fc2' in layer_names

def test_linear_layer_features_extracted():
    nodes = [{'id': 'MyNet', 'type': 'class', 'label': 'MyNet',
              'meta': {'bases': ['nn.Module'], 'source_range': {'line_start': 4, 'line_end': 13}}}]
    result = enrich_ml_nodes(nodes, PYTORCH_CODE)
    node = next(n for n in result if n['id'] == 'MyNet')
    fc1 = next(l for l in node['meta']['layers'] if l['name'] == 'fc1')
    assert fc1['in_features'] == 784
    assert fc1['out_features'] == 256

def test_non_ml_class_unchanged():
    nodes = [{'id': 'Trainer', 'type': 'class', 'label': 'Trainer',
              'meta': {'bases': ['object'], 'source_range': {'line_start': 1, 'line_end': 5}}}]
    result = enrich_ml_nodes(nodes, "class Trainer: pass")
    node = next(n for n in result if n['id'] == 'Trainer')
    assert node['type'] == 'class'
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_ml_pass.py -v
```

Expected: `ImportError: cannot import name 'enrich_ml_nodes'`

**Step 3: Implement `parser/ml_pass.py`**

```python
import ast
import re

ML_BASE_PATTERNS = [
    r'nn\.Module',
    r'torch\.nn\.Module',
    r'tf\.keras\.Model',
    r'keras\.Model',
]

def _is_ml_class(bases: list[str]) -> tuple[bool, str]:
    for base in bases:
        for pattern in ML_BASE_PATTERNS:
            if re.search(pattern, base):
                if 'keras' in base.lower():
                    return True, 'keras.Model'
                return True, 'nn.Module'
    return False, ''

def _extract_layers(class_source: str) -> list[dict]:
    layers = []
    try:
        tree = ast.parse(class_source)
    except SyntaxError:
        return layers

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue

        call = node.value
        func_name = ast.unparse(call.func)

        layer = {'name': None, 'type': func_name}

        # target name (self.fc1 -> fc1)
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                layer['name'] = target.attr

        if layer['name'] is None:
            continue

        # extract positional args for known layer types
        if func_name in ('nn.Linear', 'torch.nn.Linear'):
            args = call.args
            if len(args) >= 2:
                layer['in_features'] = ast.literal_eval(args[0])
                layer['out_features'] = ast.literal_eval(args[1])

        layers.append(layer)

    return layers

def enrich_ml_nodes(nodes: list[dict], source: str) -> list[dict]:
    lines = source.splitlines()
    enriched = []

    for node in nodes:
        if node['type'] != 'class':
            enriched.append(node)
            continue

        bases = node['meta'].get('bases', [])
        is_ml, ml_type = _is_ml_class(bases)

        if not is_ml:
            enriched.append(node)
            continue

        # extract class source for layer analysis
        sr = node['meta']['source_range']
        class_lines = lines[sr['line_start'] - 1: sr['line_end']]
        class_source = '\n'.join(class_lines)

        layers = _extract_layers(class_source)

        updated = dict(node)
        updated['type'] = ml_type
        updated['meta'] = dict(node['meta'])
        updated['meta']['layers'] = layers
        enriched.append(updated)

    return enriched
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_ml_pass.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add parser/ml_pass.py parser/tests/test_ml_pass.py
git commit -m "feat: add ML-aware pass for nn.Module detection and layer extraction"
```

---

### Task 4: Python Parser — Edge Analysis

**Files:**
- Create: `parser/edge_analysis.py`
- Create: `parser/tests/test_edge_analysis.py`

**Step 1: Write failing tests**

```python
# parser/tests/test_edge_analysis.py
import pytest
from parser.edge_analysis import build_edges

def test_call_produces_sequence_edge():
    code = """
def train():
    result = evaluate()
"""
    nodes = [
        {'id': 'train', 'type': 'function', 'label': 'train',
         'meta': {'source_range': {'line_start': 2, 'line_end': 3}}},
        {'id': 'evaluate', 'type': 'function', 'label': 'evaluate',
         'meta': {'source_range': {'line_start': 1, 'line_end': 1}}},
    ]
    edges = build_edges(nodes, code)
    seq_edges = [e for e in edges if e['kind'] == 'sequence']
    froms = [e['from'] for e in seq_edges]
    assert 'train' in froms

def test_import_produces_dependency_edge():
    code = """
import torch
class MyNet: pass
"""
    nodes = [
        {'id': 'MyNet', 'type': 'class', 'label': 'MyNet',
         'meta': {'bases': [], 'source_range': {'line_start': 3, 'line_end': 3}}}
    ]
    edges = build_edges(nodes, code)
    dep_edges = [e for e in edges if e['kind'] == 'dependency']
    # torch becomes a pseudo-node, MyNet depends on it
    assert any(e['to'] == 'MyNet' or e['from'] == 'MyNet' for e in dep_edges)

def test_edges_have_unique_ids():
    code = "def a(): b()\ndef b(): pass"
    nodes = [
        {'id': 'a', 'type': 'function', 'label': 'a',
         'meta': {'source_range': {'line_start': 1, 'line_end': 1}}},
        {'id': 'b', 'type': 'function', 'label': 'b',
         'meta': {'source_range': {'line_start': 2, 'line_end': 2}}},
    ]
    edges = build_edges(nodes, code)
    ids = [e['id'] for e in edges]
    assert len(ids) == len(set(ids))
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_edge_analysis.py -v
```

Expected: `ImportError`

**Step 3: Implement `parser/edge_analysis.py`**

```python
import ast

def build_edges(nodes: list[dict], source: str) -> list[dict]:
    edges = []
    node_ids = {n['id'] for n in nodes}
    edge_counter = 0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return edges

    def make_edge(from_id, to_id, kind):
        nonlocal edge_counter
        edge_counter += 1
        return {
            'id': f'e{edge_counter}',
            'from': from_id,
            'to': to_id,
            'kind': kind,
        }

    # sequence edges: function A calls function B
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        caller = node.name
        if caller not in node_ids:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                callee = child.func.id
            elif isinstance(child.func, ast.Attribute):
                callee = child.func.attr
            else:
                continue
            if callee in node_ids and callee != caller:
                edges.append(make_edge(caller, callee, 'sequence'))

    # dependency edges: imports
    import_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                import_names.append(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                import_names.append(name)

    for imp in import_names:
        # find nodes that reference this import
        for n in nodes:
            bases = n['meta'].get('bases', [])
            if any(imp in b for b in bases):
                edges.append(make_edge(imp, n['id'], 'dependency'))

    return edges
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_edge_analysis.py -v
```

Expected: All 3 tests PASS.

**Step 5: Run full parser test suite**

```bash
python -m pytest tests/ -v
```

Expected: All 13 tests PASS.

**Step 6: Commit**

```bash
git add parser/edge_analysis.py parser/tests/test_edge_analysis.py
git commit -m "feat: add edge analysis pass (sequence, dataflow, dependency)"
```

---

### Task 5: Parser Entry Point (stdin/stdout IPC)

**Files:**
- Create: `parser/main.py`
- Create: `parser/tests/test_main.py`

**Step 1: Write failing test**

```python
# parser/tests/test_main.py
import json
import subprocess
import sys

def test_parser_returns_graph_json():
    code = "def train(): pass"
    payload = json.dumps({'source': code})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    graph = json.loads(result.stdout)
    assert 'nodes' in graph
    assert 'edges' in graph

def test_parser_detects_function():
    code = "def forward(x): return x"
    payload = json.dumps({'source': code})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    graph = json.loads(result.stdout)
    ids = [n['id'] for n in graph['nodes']]
    assert 'forward' in ids
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'parser.main'`

**Step 3: Implement `parser/main.py`**

```python
import json
import sys
from parser.ast_pass import parse_ast
from parser.ml_pass import enrich_ml_nodes
from parser.edge_analysis import build_edges


def parse(source: str) -> dict:
    graph = parse_ast(source)
    nodes = enrich_ml_nodes(graph['nodes'], source)
    edges = build_edges(nodes, source)
    return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
    payload = json.loads(sys.stdin.read())
    source = payload.get('source', '')
    result = parse(source)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_main.py -v
```

Expected: Both tests PASS.

**Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All 15 tests PASS.

**Step 6: Commit**

```bash
git add parser/main.py parser/tests/test_main.py
git commit -m "feat: add parser entry point with stdin/stdout IPC"
```

---

### Task 6: Tauri IPC — Sidecar Command

**Files:**
- Modify: `src-tauri/src/main.rs`
- Modify: `src-tauri/tauri.conf.json`

**Step 1: Add sidecar config to `tauri.conf.json`**

Under `"tauri"` → `"allowlist"`:

```json
{
  "tauri": {
    "allowlist": {
      "shell": {
        "all": false,
        "sidecar": true,
        "open": false
      },
      "dialog": {
        "open": true
      }
    }
  }
}
```

**Step 2: Add Tauri command to `src-tauri/src/main.rs`**

```rust
use tauri::Manager;
use std::process::{Command, Stdio};
use std::io::Write;

#[tauri::command]
fn parse_code(source: String) -> Result<String, String> {
    let payload = format!("{{\"source\": {}}}", serde_json::to_string(&source).unwrap());

    let mut child = Command::new("python3")
        .arg("-m")
        .arg("parser.main")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(payload.as_bytes()).map_err(|e| e.to_string())?;
    }

    let output = child.wait_with_output().map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![parse_code])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Step 3: Add serde_json to Cargo.toml**

```toml
[dependencies]
serde_json = "1"
tauri = { version = "1", features = ["dialog-open", "shell-sidecar"] }
```

**Step 4: Build and verify**

```bash
npm run tauri build -- --debug 2>&1 | tail -20
```

Expected: Build succeeds, no compile errors.

**Step 5: Commit**

```bash
git add src-tauri/
git commit -m "feat: add Tauri IPC command to invoke Python parser"
```

---

### Task 7: Zustand Graph Store

**Files:**
- Create: `src/store/graphStore.ts`

**Step 1: Implement store**

```ts
// src/store/graphStore.ts
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
    data: n,
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
```

**Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: No errors.

**Step 3: Commit**

```bash
git add src/store/graphStore.ts
git commit -m "feat: add Zustand graph store with ReactFlow conversion"
```

---

### Task 8: Custom Node Components

**Files:**
- Create: `src/components/nodes/ModuleNode.tsx`
- Create: `src/components/nodes/FunctionNode.tsx`
- Create: `src/components/nodes/BaseNode.tsx`

**Step 1: Create `src/components/nodes/BaseNode.tsx`**

```tsx
import { Handle, Position } from '@xyflow/react'
import { GraphNode } from '../../types/graph'

interface Props {
  data: GraphNode
  selected: boolean
  color: string
  badge: string
}

export function BaseNode({ data, selected, color, badge }: Props) {
  return (
    <div
      className={`rounded-lg border-2 px-4 py-2 min-w-[160px] shadow-sm
        ${selected ? 'border-blue-500' : 'border-gray-300'}
        ${color}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono bg-black/10 px-1 rounded">{badge}</span>
        <span className="font-semibold text-sm truncate">{data.label}</span>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
```

**Step 2: Create `src/components/nodes/FunctionNode.tsx`**

```tsx
import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function FunctionNode({ data, selected }: NodeProps<GraphNode>) {
  return (
    <BaseNode data={data} selected={selected} color="bg-gray-100" badge="fn" />
  )
}
```

**Step 3: Create `src/components/nodes/ModuleNode.tsx`**

```tsx
import { NodeProps } from '@xyflow/react'
import { GraphNode } from '../../types/graph'
import { BaseNode } from './BaseNode'

export function ModuleNode({ data, selected }: NodeProps<GraphNode>) {
  return (
    <BaseNode
      data={data}
      selected={selected}
      color="bg-blue-50 border-blue-200"
      badge="nn"
    />
  )
}
```

**Step 4: Create remaining node types in `src/components/nodes/`**

`ClassNode.tsx` — purple, badge `cls`
`TrainingLoopNode.tsx` — orange/amber, badge `loop`
`DataNode.tsx` — green, badge `data`
`KerasModelNode.tsx` — blue/teal, badge `tf`

All follow the same pattern as FunctionNode with different `color` and `badge` props.

**Step 5: Commit**

```bash
git add src/components/nodes/
git commit -m "feat: add custom ReactFlow node components per block type"
```

---

### Task 9: Annotated Edge Component

**Files:**
- Create: `src/edges/AnnotatedEdge.tsx`

**Step 1: Implement `src/edges/AnnotatedEdge.tsx`**

```tsx
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
  const kind: EdgeKind = data?.kind ?? 'sequence'
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
```

**Step 2: Commit**

```bash
git add src/edges/AnnotatedEdge.tsx
git commit -m "feat: add annotated edge component with sequence/dataflow/dependency styles"
```

---

### Task 10: Graph Canvas Component

**Files:**
- Create: `src/components/GraphCanvas.tsx`

**Step 1: Implement `src/components/GraphCanvas.tsx`**

```tsx
import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useGraphStore } from '../store/graphStore'
import { FunctionNode } from './nodes/FunctionNode'
import { ModuleNode } from './nodes/ModuleNode'
import { ClassNode } from './nodes/ClassNode'
import { TrainingLoopNode } from './nodes/TrainingLoopNode'
import { DataNode } from './nodes/DataNode'
import { AnnotatedEdge } from '../edges/AnnotatedEdge'

const nodeTypes = {
  function: FunctionNode,
  class: ClassNode,
  nn_Module: ModuleNode,
  keras_Model: ModuleNode,
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
```

**Step 2: Commit**

```bash
git add src/components/GraphCanvas.tsx
git commit -m "feat: add ReactFlow graph canvas with custom nodes and edges"
```

---

### Task 11: Block Inspector Panel

**Files:**
- Create: `src/components/InspectorPanel.tsx`

**Step 1: Implement `src/components/InspectorPanel.tsx`**

```tsx
import { useGraphStore } from '../store/graphStore'
import { GraphNode, LayerInfo } from '../types/graph'

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

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500 text-xs">{label}</span>
      <span className="font-mono text-xs text-gray-900">{value}</span>
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
        <div className="text-sm text-gray-600">
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
```

**Step 2: Commit**

```bash
git add src/components/InspectorPanel.tsx
git commit -m "feat: add block inspector panel with ML-aware property display"
```

---

### Task 12: Toolbar — File Import and Paste

**Files:**
- Create: `src/components/Toolbar.tsx`

**Step 1: Implement `src/components/Toolbar.tsx`**

```tsx
import { useState } from 'react'
import { invoke } from '@tauri-apps/api/tauri'
import { open } from '@tauri-apps/api/dialog'
import { readTextFile } from '@tauri-apps/api/fs'
import { useGraphStore } from '../store/graphStore'
import { Graph } from '../types/graph'

export function Toolbar() {
  const { setGraph, clearGraph } = useGraphStore()
  const [pasteOpen, setPasteOpen] = useState(false)
  const [pasteCode, setPasteCode] = useState('')
  const [error, setError] = useState<string | null>(null)

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

  async function handleOpenFile() {
    const path = await open({ filters: [{ name: 'Python', extensions: ['py'] }] })
    if (typeof path === 'string') {
      const source = await readTextFile(path)
      await parseAndSet(source)
    }
  }

  async function handlePaste() {
    await parseAndSet(pasteCode)
    setPasteOpen(false)
    setPasteCode('')
  }

  return (
    <div className="h-12 border-b border-gray-200 flex items-center px-4 gap-3 bg-white">
      <span className="font-bold text-gray-900 mr-2">graphTorch</span>
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
```

**Step 2: Commit**

```bash
git add src/components/Toolbar.tsx
git commit -m "feat: add toolbar with file open and paste code entry points"
```

---

### Task 13: Wire Up App Layout

**Files:**
- Modify: `src/App.tsx`

**Step 1: Read existing `src/App.tsx`**

**Step 2: Replace content**

```tsx
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
```

**Step 3: Launch and manually test**

```bash
npm run tauri dev
```

Manual test checklist:
- [ ] App window opens
- [ ] Click "Open File" → file dialog opens → select a `.py` file → graph renders
- [ ] Click "Paste Code" → paste `class MyNet(nn.Module): pass` → click Visualize → node appears
- [ ] Click a node → Inspector Panel shows node details
- [ ] Click canvas background → Inspector Panel shows placeholder text
- [ ] Click "Clear" → graph disappears

**Step 4: Commit**

```bash
git add src/App.tsx
git commit -m "feat: wire up app layout with graph canvas, inspector, and toolbar"
```

---

### Task 14: End-to-End Smoke Test with Real PyTorch Code

**Files:**
- Create: `parser/tests/fixtures/simple_net.py`
- Create: `parser/tests/test_e2e.py`

**Step 1: Create fixture**

```python
# parser/tests/fixtures/simple_net.py
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, loader, optimizer, criterion):
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**Step 2: Write end-to-end test**

```python
# parser/tests/test_e2e.py
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / 'fixtures' / 'simple_net.py'

def test_e2e_parses_pytorch_fixture():
    source = FIXTURE.read_text()
    payload = json.dumps({'source': source})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    graph = json.loads(result.stdout)

    node_ids = [n['id'] for n in graph['nodes']]
    assert 'SimpleNet' in node_ids
    assert 'train' in node_ids

def test_e2e_simplenet_is_nn_module():
    source = FIXTURE.read_text()
    payload = json.dumps({'source': source})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    graph = json.loads(result.stdout)
    node = next(n for n in graph['nodes'] if n['id'] == 'SimpleNet')
    assert node['type'] == 'nn.Module'

def test_e2e_layers_extracted():
    source = FIXTURE.read_text()
    payload = json.dumps({'source': source})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    graph = json.loads(result.stdout)
    node = next(n for n in graph['nodes'] if n['id'] == 'SimpleNet')
    layer_names = [l['name'] for l in node['meta']['layers']]
    assert 'fc1' in layer_names
    assert 'fc2' in layer_names

def test_e2e_has_edges():
    source = FIXTURE.read_text()
    payload = json.dumps({'source': source})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    graph = json.loads(result.stdout)
    assert len(graph['edges']) > 0
```

**Step 3: Run end-to-end tests**

```bash
python -m pytest tests/test_e2e.py -v
```

Expected: All 4 tests PASS.

**Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All 19 tests PASS.

**Step 5: Commit**

```bash
git add parser/tests/
git commit -m "test: add end-to-end tests with real PyTorch fixture"
```

---

## Summary

| Task | Deliverable |
|---|---|
| 1 | Tauri + React + TS project scaffold |
| 2 | Python AST pass (functions, classes) |
| 3 | ML-aware pass (nn.Module layer extraction) |
| 4 | Edge analysis (sequence, dataflow, dependency) |
| 5 | Parser stdin/stdout IPC entry point |
| 6 | Tauri IPC command invoking Python parser |
| 7 | Zustand graph store |
| 8 | Custom ReactFlow node components |
| 9 | Annotated edge component |
| 10 | Graph canvas wired to store |
| 11 | Block inspector panel |
| 12 | Toolbar with file open + paste |
| 13 | App layout assembly |
| 14 | End-to-end smoke tests |

**Phase 1 complete = 19 passing tests + working Tauri app that parses and visualizes PyTorch code.**

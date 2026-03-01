# graphTorch — Design Document

**Date:** 2026-03-01
**Status:** Approved

---

## Overview

graphTorch is a visual code graph tool that parses Python code (with first-class support for PyTorch and TensorFlow) into an interactive block-based graph. Users can explore, understand, and eventually edit code through a visual canvas augmented by an AI copilot.

**Target users:** Both ML beginners (visual learners) and practitioners (fast prototyping).

---

## Architecture

Three layers:

1. **Tauri desktop shell** — lightweight Rust-based desktop window, file system access, manages the Python parser sidecar process
2. **React frontend (TypeScript)** — graph canvas, block inspector, AI chat panel; shared between desktop app and VS Code extension
3. **Python parser service** — child process, parses `.py` files, returns Graph JSON over IPC (stdin/stdout)

```
┌─────────────────────────────────────────────────────┐
│                   Tauri Desktop Shell                │
│  ┌───────────────────────────────────────────────┐  │
│  │              React Frontend (TS)              │  │
│  │  ┌──────────────┐  ┌────────────────────────┐ │  │
│  │  │  Graph Canvas │  │  Block Inspector Panel │ │  │
│  │  │  (ReactFlow)  │  │  (properties, shapes)  │ │  │
│  │  └──────────────┘  └────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │         AI Copilot Chat Panel           │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │         Python Parser Service (sidecar)       │  │
│  │   AST parser │ ML analyzer │ File watcher     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

VS Code Extension: same React frontend as a webview panel
         │
         └── communicates with same Python parser via IPC
```

---

## Core Components

### 1. Python Parser Service

Takes `.py` source code as input, returns a Graph JSON object.

**Stages:**
- **AST pass** — Python's built-in `ast` module extracts functions, classes, imports, call relationships
- **ML-aware pass** — detects `nn.Module` subclasses (PyTorch) and `tf.keras.Model` subclasses (TF); extracts layer configs from `__init__` and `forward`/`call` methods
- **Edge analysis** — builds three edge types: `sequence`, `dataflow`, `dependency`

**Graph JSON schema:**
```json
{
  "nodes": [
    {
      "id": "MyNet",
      "type": "nn.Module",
      "meta": { "input_size": 784, "hidden": [256, 128], "output_size": 10 },
      "source_range": { "line_start": 4, "line_end": 22 }
    }
  ],
  "edges": [
    { "from": "DataLoader", "to": "MyNet", "kind": "dataflow" },
    { "from": "MyNet", "to": "criterion", "kind": "sequence" }
  ]
}
```

### 2. Graph Canvas (ReactFlow)

**Block types:**
| Type | Appearance | Drilldown |
|---|---|---|
| `nn.Module` / `keras.Model` | Blue, neural net icon | Expands to layer list with shapes |
| `function` | Gray, `fn` badge | Shows signature and callees |
| `class` | Purple, class icon | Shows methods as sub-blocks |
| `training_loop` | Orange, loop icon | Shows optimizer, loss, epochs |
| `data` | Green, dataset icon | Shows shape, transforms |

**Edge symbols:**
- `→` solid arrow — sequence / execution order
- `⇢` dashed arrow — data flow (tensor/variable passing)
- `⋯` dotted line — dependency (import / inheritance)

### 3. Block Inspector Panel

Right panel showing properties of the selected block. For `nn.Module`:
```
Name:         MyNet
Type:         nn.Module
Input shape:  (batch, 784)
Output shape: (batch, 10)
Layers:       Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)
Optimizer:    Adam (lr=0.001)
Loss:         CrossEntropyLoss
Source:       model.py:4–22
```

### 4. AI Copilot Panel

Context-aware (selected block + full graph state + source code).

**Three modes:**
- **Explain** — plain language description of a selected block
- **Build** — natural language → graph mutations (Phase 3)
- **Suggest** — proactive soft overlay annotations on nodes (missing activation, no dropout, etc.)

**AI model:** `claude-sonnet-4-6`

**Context payload sent to AI:**
```json
{
  "message": "...",
  "selected_block": { ... },
  "full_graph": { ... },
  "source_code": "..."
}
```

---

## Data Flow

### Code → Graph
1. User imports file / pastes code / file changes on disk
2. Tauri sends source to Python parser via IPC
3. Parser runs AST → ML-aware → edge analysis passes
4. Returns Graph JSON to frontend
5. ReactFlow renders nodes and annotated edges
6. User clicks block → Inspector Panel populates

### AI Copilot
1. User types message in chat
2. Frontend builds context payload
3. Sent to Claude API
4. Response is text explanation, graph mutation JSON diff, or suggestion overlay

### Live Sync
1. Tauri file watcher detects `.py` file change
2. Debounce 300ms
3. Re-parse → diff against current graph state
4. ReactFlow animates only changed nodes/edges

---

## Phased Delivery

### Phase 1 — Visualize (MVP)
- Python parser (AST + PyTorch/TF ML-aware)
- ReactFlow graph canvas with all block/edge types
- Block Inspector Panel
- File import + paste entry points
- Tauri desktop app

### Phase 2 — Interact
- AI Copilot (explain + suggest modes)
- Live file sync
- VS Code extension (webview reusing Phase 1 frontend)

### Phase 3 — Bidirectional
- Graph edits → code generation (export clean Python)
- AI Build mode (natural language → graph mutations)
- Multi-file project graph

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Tauri (Rust) |
| Frontend | React + TypeScript + ReactFlow |
| Parser | Python (`ast`, `tree-sitter`) |
| AI | Claude API (`claude-sonnet-4-6`) |
| VS Code extension | VS Code Webview API |
| IPC | stdin/stdout JSON (Tauri sidecar) |

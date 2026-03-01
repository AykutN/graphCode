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

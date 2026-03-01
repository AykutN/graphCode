import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / 'fixtures' / 'simple_net.py'


def _run_parser(source: str) -> dict:
    payload = json.dumps({'source': source})
    result = subprocess.run(
        [sys.executable, '-m', 'parser.main'],
        input=payload,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_e2e_parses_pytorch_fixture():
    graph = _run_parser(FIXTURE.read_text())
    node_ids = [n['id'] for n in graph['nodes']]
    assert 'SimpleNet' in node_ids
    assert 'train' in node_ids


def test_e2e_simplenet_is_nn_module():
    graph = _run_parser(FIXTURE.read_text())
    node = next(n for n in graph['nodes'] if n['id'] == 'SimpleNet')
    assert node['type'] == 'nn.Module'


def test_e2e_layers_extracted():
    graph = _run_parser(FIXTURE.read_text())
    node = next(n for n in graph['nodes'] if n['id'] == 'SimpleNet')
    layer_names = [l['name'] for l in node['meta']['layers']]
    assert 'fc1' in layer_names
    assert 'fc2' in layer_names


def test_e2e_has_edges():
    graph = _run_parser(FIXTURE.read_text())
    assert len(graph['edges']) > 0

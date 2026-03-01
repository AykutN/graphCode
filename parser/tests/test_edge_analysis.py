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

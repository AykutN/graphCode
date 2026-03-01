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

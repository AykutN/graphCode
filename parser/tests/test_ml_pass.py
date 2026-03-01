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

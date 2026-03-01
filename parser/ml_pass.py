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

        for target in node.targets:
            if isinstance(target, ast.Attribute):
                layer['name'] = target.attr

        if layer['name'] is None:
            continue

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

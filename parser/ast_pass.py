import ast


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

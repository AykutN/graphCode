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
        for n in nodes:
            edges.append(make_edge(imp, n['id'], 'dependency'))

    return edges

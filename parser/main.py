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

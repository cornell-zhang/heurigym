import json
import networkx as nx
from structure import LogicNetwork

# ---------- read BLIF format ----------
def read_blif(blif_path: str) -> LogicNetwork:
    G = nx.DiGraph()
    PI, PO = set(), set()
    with open(blif_path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == '.model':
            continue
        elif cmd == '.inputs':
            for sig in parts[1:]:
                PI.add(sig)
                G.add_node(sig, type='PI')
        elif cmd == '.outputs':
            for sig in parts[1:]:
                PO.add(sig)
                G.add_node(sig, type='PO')
        elif cmd == '.names':
            inputs = parts[1:-1]
            out = parts[-1]
            G.add_node(out, type='names')
            truth = []
            while i < len(lines) and lines[i] and not lines[i].startswith('.'):
                tok = lines[i].strip().split()
                if len(tok) == 2:
                    truth.append((tok[0], tok[1]))
                i += 1
            for inp in inputs:
                if inp not in G:
                    G.add_node(inp, type='wire')
                G.add_edge(inp, out)
            G.nodes[out]['truth_table'] = truth
            continue
        elif cmd == '.end':
            break
    return LogicNetwork(graph=G, PI=PI, PO=PO)

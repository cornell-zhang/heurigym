#!/usr/bin/env python3
"""
Verifier for E-Graph extraction solutions.

Checks the four constraints (root-selection, child-selection, acyclicity,
integrity).  Does **not** compute the objective cost.
"""

from __future__ import annotations
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys

# -----------------------------------------------------------------------------#
# I/O helpers (left public so evaluator can re-use them)                        #
# -----------------------------------------------------------------------------#


def load_graph(path: str) -> Tuple[Dict[str, dict], Set[str]]:
    """Return (nodes, root_eclasses) loaded from *graph.json*."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        return data["nodes"], set(data["root_eclasses"])
    except (KeyError, TypeError):
        raise ValueError("JSON must contain 'nodes' and 'root_eclasses'")


def load_solution(path: str) -> List[str]:
    """Read solution from string in file"""
    with open(path, "rb") as fh:
        obj = fh.read()
        import ast

        obj = ast.literal_eval(obj.decode("utf-8"))

    if not isinstance(obj, (list, set, tuple)):
        raise TypeError("Solution pickle must contain a list/tuple/set of IDs")
    return obj


# -----------------------------------------------------------------------------#
# Core verification logic                                                      #
# -----------------------------------------------------------------------------#


def _build_aux(nodes: Dict[str, dict]):
    """Pre-compute helper maps for fast constraint checks."""
    eclass_to_nodes: Dict[str, List[str]] = defaultdict(list)
    child_ecs_of_node: Dict[str, Set[str]] = {}

    for nid, rec in nodes.items():
        ec = rec["eclass"]
        eclass_to_nodes[ec].append(nid)

        child_ecs: Set[str] = set()
        for child in rec["children"]:
            if child not in nodes:
                raise ValueError(f"Node '{nid}' references unknown child '{child}'")
            child_ecs.add(nodes[child]["eclass"])
        child_ecs_of_node[nid] = child_ecs

    return eclass_to_nodes, child_ecs_of_node


def verify(input_file: str, output_file: str) -> Tuple[bool, str]:
    nodes, root_eclasses = load_graph(input_file)
    selected_nodes = load_solution(output_file)

    violations: str = ""
    selected_set: Set[str] = set(selected_nodes)

    # Integrity — unknown IDs --------------------------------------------------
    unknown = [n for n in selected_set if n not in nodes]
    for n in unknown:
        violations += f"Unknown node selected: '{n}'\n"

    # Build helpers ------------------------------------------------------------
    eclass_to_nodes, child_ecs = _build_aux(nodes)

    # Convenient mapping: e-class → selected node list -------------------------
    selected_by_ec: Dict[str, List[str]] = defaultdict(list)
    for nid in selected_set & nodes.keys():
        selected_by_ec[nodes[nid]["eclass"]].append(nid)

    # 1. Root selection --------------------------------------------------------
    for ec in root_eclasses:
        cnt = len(selected_by_ec.get(ec, []))
        if cnt != 1:
            violations += f"Root e-class '{ec}' has {cnt} selected nodes (expected 1)."

    # 2. Child selection -------------------------------------------------------
    for nid in selected_set & nodes.keys():
        for child_ec in child_ecs[nid]:
            cnt = len(selected_by_ec.get(child_ec, []))
            if cnt != 1:
                violations += f"Node '{nid}': child e-class '{child_ec}' has {cnt} selected nodes (expected 1)."

    # 3. Acyclicity ------------------------------------------------------------
    adj: Dict[str, List[str]] = defaultdict(list)
    for nid in selected_set & nodes.keys():
        for child_ec in child_ecs[nid]:
            child_list = selected_by_ec.get(child_ec, [])
            if child_list:  # exactly one if child-sel passed
                adj[nid].append(child_list[0])

    def _has_cycle() -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        colour = {n: WHITE for n in selected_set}

        def dfs(u: str) -> bool:
            colour[u] = GRAY
            for v in adj.get(u, []):
                if colour[v] == GRAY:
                    return True
                if colour[v] == WHITE and dfs(v):
                    return True
            colour[u] = BLACK
            return False

        return any(dfs(n) for n in selected_set if colour[n] == WHITE)

    if _has_cycle():
        violations += "Selected sub-graph contains cycles (acyclicity violated)."

    return len(violations) == 0, violations


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        valid, violations = verify(input_file, output_file)
        print(f"Verification result: {valid}")
        print(f"Violations:\n{violations}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

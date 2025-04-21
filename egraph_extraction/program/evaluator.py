#!/usr/bin/env python3
"""
Evaluator for E‑Graph extraction solutions.

Given an e‑graph description (JSON) and a candidate extraction (pickle), this
script checks whether the candidate satisfies the legality constraints and
computes its total cost.

Constraints checked
-------------------
1. **Root selection** – Exactly one e‑node must be selected from every root
   e‑class.
2. **Child selection** – For every *selected* e‑node, exactly one e‑node must
   be selected from each of its child e‑classes.
3. **Acyclicity** – The sub‑graph induced by the selected nodes must be a DAG
   (no cyclic dependencies).
4. **Integrity** – All selected node IDs must exist in the e‑graph.

If *any* of the above constraints are violated the solution is reported as
*INVALID* (exit code 1) and the offending conditions are listed.  Costs are still
reported so that partially valid solutions can be compared.

Usage
-----
```bash
python evaluator.py graph.json solution.pkl
```
where
* **graph.json** is the input file described in the task spec, and
* **solution.pkl** is a pickle file containing a Python *list* (or *set*) of
  selected node‑ID strings.

The script prints a short validation summary and exits with code 0 for valid
solutions or 1 for invalid ones.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def _load_graph(path: str) -> Tuple[Dict[str, dict], Set[str]]:
    """Return *(nodes, root_eclasses)* from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        nodes: Dict[str, dict] = data["nodes"]
        root_eclasses: Set[str] = set(data["root_eclasses"])
    except (KeyError, TypeError):
        raise ValueError(
            "Input JSON must contain 'nodes' and 'root_eclasses'.")
    return nodes, root_eclasses


def _load_solution(path: str) -> List[str]:
    """Unpickle and normalise the solution container to a list of node IDs."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, (list, set, tuple)):
        raise TypeError(
            "Solution pickle must contain a list/tuple/set of node IDs.")
    return list(obj)


# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------


def _build_aux_structures(nodes: Dict[str, dict]):
    """Pre‑compute helper mappings for fast constraint checking."""
    eclass_to_nodes: Dict[str, List[str]] = defaultdict(list)
    child_eclasses_of_node: Dict[str, Set[str]] = {}

    for nid, rec in nodes.items():
        ec = rec["eclass"]
        eclass_to_nodes[ec].append(nid)

        # Resolve child e‑classes (deduplicated) referenced by this node
        child_ecs: Set[str] = set()
        for child_nid in rec["children"]:
            if child_nid not in nodes:
                raise ValueError(
                    f"Node '{nid}' references unknown child node '{child_nid}'."
                )
            child_ecs.add(nodes[child_nid]["eclass"])
        child_eclasses_of_node[nid] = child_ecs

    return eclass_to_nodes, child_eclasses_of_node


def evaluate(
    nodes: Dict[str, dict],
    root_eclasses: Set[str],
    selected_nodes: List[str],
):
    """Return *(violations, total_cost)* for *selected_nodes* on the given graph."""

    violations: List[str] = []
    selected_set: Set[str] = set(selected_nodes)

    # Integrity – unknown IDs --------------------------------------------------
    unknown = [nid for nid in selected_set if nid not in nodes]
    if unknown:
        violations.extend(
            [f"Unknown node selected: '{nid}'" for nid in unknown])

    # Build helper structures --------------------------------------------------
    eclass_to_nodes, child_ecs = _build_aux_structures(nodes)

    # Convenience: mapping e‑class → selected node list ------------------------
    selected_by_eclass: Dict[str, List[str]] = defaultdict(list)
    for nid in selected_set & nodes.keys():
        selected_by_eclass[nodes[nid]["eclass"]].append(nid)

    # 1. Root selection --------------------------------------------------------
    for ec in root_eclasses:
        cnt = len(selected_by_eclass.get(ec, []))
        if cnt != 1:
            violations.append(
                f"Root e‑class '{ec}' has {cnt} selected nodes (expected exactly 1)."
            )

    # 2. Child selection -------------------------------------------------------
    for nid in selected_set & nodes.keys():
        for child_ec in child_ecs[nid]:
            cnt = len(selected_by_eclass.get(child_ec, []))
            if cnt != 1:
                violations.append(
                    f"Node '{nid}': child e‑class '{child_ec}' has {cnt} selected nodes (expected 1)."
                )

    # 3. Acyclicity ------------------------------------------------------------
    # Build adjacency (edge: parent node  →  selected node in each child e‑class)
    adj: Dict[str, List[str]] = defaultdict(list)
    for nid in selected_set & nodes.keys():
        for child_ec in child_ecs[nid]:
            child_list = selected_by_eclass.get(child_ec, [])
            if child_list:
                adj[nid].append(child_list[0])  # exactly one asserted above

    def _has_cycle() -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2  # DFS colours
        state = {nid: WHITE for nid in selected_set}

        def dfs(u: str) -> bool:
            state[u] = GRAY
            for v in adj.get(u, []):
                if state[v] == GRAY:
                    return True
                if state[v] == WHITE and dfs(v):
                    return True
            state[u] = BLACK
            return False

        return any(dfs(n) for n in selected_set if state[n] == WHITE)

    if _has_cycle():
        violations.append(
            "Selected sub‑graph contains cycles (violates acyclicity).")

    # Total cost --------------------------------------------------------------
    total_cost = sum(nodes[nid]["cost"] for nid in selected_set
                     if nid in nodes)

    return violations, total_cost


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Evaluate an E‑Graph extraction solution for correctness and cost.", )
    parser.add_argument("graph_json",
                        help="Path to the input *graph.json* file")
    parser.add_argument("solution_pkl",
                        help="Path to the output *solution.pkl* file")
    args = parser.parse_args()

    try:
        nodes, roots = _load_graph(args.graph_json)
        solution = _load_solution(args.solution_pkl)
        violations, cost = evaluate(nodes, roots, solution)
    except Exception as exc:
        print(f"❌  Evaluation failed: {exc}")
        sys.exit(1)

    if violations:
        print("❌  INVALID solution – constraints violated:\n")
        for v in violations:
            print("  •", v)
        print(f"\n⚠️  Reported cost (ignoring invalidity): {cost}")
        sys.exit(1)
    else:
        print("✅  VALID solution – all constraints satisfied.")
        print(f"Total cost: {cost}")
        sys.exit(0)


if __name__ == "__main__":
    main()

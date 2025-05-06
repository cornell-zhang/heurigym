import os
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple, List


@dataclass
class Node:
    id: str  # e.g., "n1"
    resource: str  # e.g., "mul" or "sub"
    preds: List[str] = field(default_factory=list)  # Predecessor nodes
    succs: List[str] = field(default_factory=list)  # Successor nodes
    start_time: int = 0  # Scheduled start cycle (to be computed)
    in_degree: int = 0  # Number of incoming edges (for topological ordering)


def get_filename(path: str) -> str:
    """Extract filename from path."""
    return os.path.basename(path).split(".")[0]


def parse_json(filename: str) -> Tuple[Dict[str, Node], Dict[str, int], Dict[str, int]]:
    """Parse the JSON file and build the graph.

    Returns:
        A tuple of:
        - Dictionary mapping node IDs to Node objects
        - Dictionary mapping resource types to their delays
        - Dictionary mapping resource types to the number of available functional units
    """
    nodes = {}

    try:
        with open(filename, "r") as file:
            data = json.load(file)

            # Extract delay and resource constraints
            delay = data.get("delay", {})
            resource_constraints = data.get("resource", {})

            # Create nodes
            for node_info in data.get("nodes", []):
                if len(node_info) >= 2:
                    node_id = node_info[0]
                    resource = node_info[1]
                    node = Node(id=node_id, resource=resource)
                    nodes[node_id] = node

            # Add edges
            for edge_info in data.get("edges", []):
                if len(edge_info) >= 2:
                    src = edge_info[0]
                    dst = edge_info[1]
                    # Add the edge src -> dst
                    if src in nodes and dst in nodes:
                        nodes[src].succs.append(dst)
                        nodes[dst].preds.append(src)

    except FileNotFoundError:
        print(f"Error opening JSON file: {filename}", file=sys.stderr)
        sys.exit(1)

    # Initialize in-degrees for each node based on its predecessors
    for node in nodes.values():
        node.in_degree = len(node.preds)

    return nodes, delay, resource_constraints


def parse_schedule(filename: str) -> Dict[str, int]:
    """Parse the schedule file.

    Returns:
        Dictionary mapping node IDs to their scheduled start times
    """
    schedule = {}

    try:
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Format: node_id:start_time
                parts = line.split(":")
                if len(parts) != 2:
                    print(f"Invalid schedule format: {line}", file=sys.stderr)
                    continue

                node_id = parts[0]
                try:
                    start_time = int(parts[1])
                    schedule[node_id] = start_time
                except ValueError:
                    print(f"Invalid start time: {parts[1]}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error opening schedule file: {filename}", file=sys.stderr)
        sys.exit(1)

    return schedule

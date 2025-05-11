import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Strategy:
    cost: int
    usage: Optional[int] = None  # Usage is optional for edges
    num_id: Optional[int] = None

    def __lt__(self, other):
        return self.usage < other.usage


@dataclass
class Edge:
    nodes: List[int]
    strategies: List[Strategy] = field(default_factory=list)


@dataclass
class Node:
    interval: tuple
    strategies: List[Strategy] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)


@dataclass
class Problem:
    name: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    usage_limit: Optional[int] = None
    strategies_to_node: Dict[int, Tuple[int,
                                        int]] = field(default_factory=dict)

    def set_strategy_ids(self):
        count = 0
        for node_idx, node in enumerate(self.nodes):
            for s_idx, s in enumerate(node.strategies):
                s.num_id = count
                self.strategies_to_node[count] = (node_idx, s_idx)
                count += 1


@dataclass
class Solution:
    strategies: List[int]  # Index of strategy chosen for each node


def transpose(col_major, rows, cols):
    assert len(col_major) == rows * cols, "Invalid dimensions."

    row_major = [0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            col_index = j * rows + i
            row_index = i * cols + j
            row_major[row_index] = col_major[col_index]

    return row_major


def load_problem(filename: str) -> Problem:
    with open(filename, 'r') as f:
        data = json.load(f)

    prob_data = data["problem"]
    problem = Problem(name=prob_data["name"])

    # Parse nodes
    for interval in prob_data["nodes"]["intervals"]:
        problem.nodes.append(Node(interval=tuple(interval)))

    for node_idx, (costs, usages) in enumerate(
            zip(prob_data["nodes"]["costs"], prob_data["nodes"]["usages"])):
        node = problem.nodes[node_idx]
        for cost, usage in zip(costs, usages):
            node.strategies.append(Strategy(cost=cost, usage=usage))

    # Parse edges
    seen_edges = {}
    edge_idx = 0
    for edge_nodes, costs in zip(prob_data["edges"]["nodes"],
                                 prob_data["edges"]["costs"]):
        edge = Edge(nodes=edge_nodes)

        # get edges in interval order (topological)
        n0, n1 = edge.nodes
        if problem.nodes[n0].interval[0] > problem.nodes[n1].interval[0]:
            edge.nodes.reverse()
            n0, n1 = edge.nodes
            costs = transpose(costs, len(problem.nodes[n0].strategies),
                              len(problem.nodes[n1].strategies))

        # check if seen edge
        edge_id = (n0, n1)
        if edge_id in seen_edges:
            old_edge_idx = seen_edges[edge_id]
            for i, s in enumerate(problem.edges[old_edge_idx].strategies):
                s.cost += costs[i]
        else:
            # mark edge as seen
            seen_edges[edge_id] = edge_idx

            # create new edge
            problem.edges.append(edge)
            problem.nodes[n0].edges.append(edge)
            problem.nodes[n1].edges.append(edge)
            for cost in costs:
                edge.strategies.append(Strategy(cost=cost))
            edge_idx += 1

    # Parse usage limit
    problem.usage_limit = prob_data.get("usage_limit")
    problem.set_strategy_ids()

    return problem


def load_solution(filename: str) -> List[int]:
    with open(filename, 'r') as file:
        # Read the content and remove whitespace
        content = file.read().strip()

        # Remove the square brackets
        if content.startswith('[') and content.endswith(']'):
            content = content[1:-1]

        # Split by commas and convert to integers
        numbers = [int(num) for num in content.split(',')]

    return Solution(numbers)


def check_usage_legal(problem: Problem, solution: Solution):
    # function to check that a provided cost-legal solution is usage-legal

    max_time = max(node.interval[1] for node in problem.nodes)
    total_usages = [0] * max_time

    # get usages
    illegal_times = []
    for node_idx, strategy_idx in enumerate(solution.strategies):
        node = problem.nodes[node_idx]
        strategy = node.strategies[strategy_idx]

        for t in range(node.interval[0], node.interval[1]):
            if strategy.usage is not None:
                total_usages[t] += strategy.usage
                if total_usages[t] > problem.usage_limit:
                    illegal_times.append(t)

    return illegal_times


def verify(input_file: str, output_file: str) -> Tuple[bool, str]:
    problem = load_problem(input_file)
    solution = load_solution(output_file)

    if len(solution.strategies) != len(problem.nodes):
        return False, "Solution length does not match number of nodes"

    # Check if the solution is cost-legal
    illegal_times = check_usage_legal(problem, solution)
    if len(illegal_times) != 0:
        return False, f"Solution is not usage-legal on {illegal_times}."

    for node_idx, strategy_idx in enumerate(solution.strategies):
        node = problem.nodes[node_idx]
        if not (0 <= strategy_idx < len(node.strategies)):
            return False, f"Strategy index {strategy_idx} out of range for node {node_idx}."

    for edge in problem.edges:
        strategy_idx = 0
        for node_idx in edge.nodes:
            strategy_idx *= len(problem.nodes[node_idx].strategies)
            strategy_idx += solution.strategies[node_idx]

        if not (0 <= strategy_idx < len(edge.strategies)):
            return False, f"Strategy index {strategy_idx} out of range for edge with nodes {edge.nodes}."

    return True, ""


if __name__ == '__main__':
    print(verify("dataset/demo/example.json", "dataset/demo/example.sol"))

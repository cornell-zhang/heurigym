from math import sqrt
from structure import Instance

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def evaluate_solution(instance: Instance, file_path: str) -> float:
    """
    Parse the solution file and evaluate the total cost of the solution.

    Args:
        instance: The PDPTW instance containing node information.
        file_path: Path to the solution file.

    Returns:
        The total cost (distance) of the solution.
    """
    nodes = instance.nodes  # Use the dictionary directly
    total_cost = 0.0

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_num, line in enumerate(lines[1:], start=1):  # Skip the first line (cost line)
            route = list(map(int, line.strip().split()))
            if not route or route[0] != instance.depot_node[0] or route[-1] != instance.depot_node[0]:
                raise ValueError(f"Route on line {line_num} must start and end at the depot.")

            route_cost = 0.0
            for i in range(len(route) - 1):
                current_node = nodes[route[i]]
                next_node = nodes[route[i + 1]]
                route_cost += calculate_distance(current_node.x, current_node.y, next_node.x, next_node.y)

            print(f"Cost for route {line_num}: {route_cost}")
            total_cost += route_cost

    return total_cost

import sys
from utils import read_instance, calculate_distance


def evaluate(input_file: str, solution_file: str) -> float:
    """Cost calculation function: calculates the solution cost.

    Args:
        input_file: Path to the input file containing the instance
        solution_file: Path to the solution file

    Returns:
        float: The cost of the solution, or infinity if invalid
    """
    instance = read_instance(input_file)
    nodes = instance.nodes  # Use the dictionary directly
    total_cost = 0.0

    with open(solution_file, "r") as file:
        lines = file.readlines()
        for line_num, line in enumerate(
            lines[1:], start=1
        ):  # Skip the first line (cost line)
            route = list(map(int, line.strip().split()))
            if (
                not route
                or route[0] != instance.depot_node[0]
                or route[-1] != instance.depot_node[0]
            ):
                raise ValueError(
                    f"Route on line {line_num} must start and end at the depot."
                )

            route_cost = 0.0
            for i in range(len(route) - 1):
                current_node = nodes[route[i]]
                next_node = nodes[route[i + 1]]
                route_cost += calculate_distance(
                    current_node.x, current_node.y, next_node.x, next_node.y
                )

            print(f"Cost for route {line_num}: {route_cost}")
            total_cost += route_cost

    return total_cost


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <solution_file>", file=sys.stderr)
        sys.exit(1)

    cost = evaluate(sys.argv[1], sys.argv[2])
    print(cost)

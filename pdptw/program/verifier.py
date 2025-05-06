import sys
from typing import Tuple
from utils import read_instance, calculate_distance


def verify(input_file: str, solution_file: str) -> Tuple[bool, str]:
    """Verification function: checks if the solution is valid.

    Args:
        input_file: Path to the input file containing the instance
        solution_file: Path to the solution file

    Returns:
        Tuple[bool, str]: A tuple of (is_valid, error_message)
        is_valid: True if the solution is valid, False otherwise
        error_message: If the solution is invalid, provides a detailed error message;
        If the solution is valid, returns an empty string.
    """
    instance = read_instance(input_file)
    nodes = instance.nodes  # Use the dictionary directly
    visited_nodes = set()
    error_message = ""

    with open(solution_file, "r") as file:
        lines = file.readlines()

    # Skip the first line
    for line_num, line in enumerate(lines[1:], start=1):
        route = list(map(int, line.strip().split()))
        if (
            not route
            or route[0] != instance.depot_node[0]
            or route[-1] != instance.depot_node[0]
        ):
            error_message = f"Route {line_num} (Vehicle {line_num}): Does not start and end at the depot."
            return False, error_message

        current_time = 0
        current_capacity = 0

        for i in range(
            len(route) - 1
        ):  # Skip the first and last elements (depot nodes)
            current_node = nodes[route[i]]
            next_node = nodes[route[i + 1]]
            if next_node.index == instance.depot_node[0]:
                break
            visited_nodes.add(next_node.index)  # Mark the node as visited

            # Check time window constraints
            arrival_time = current_time + calculate_distance(
                current_node.x, current_node.y, next_node.x, next_node.y
            )
            if arrival_time < next_node.earliest_time:
                current_time = next_node.earliest_time
            elif arrival_time > next_node.latest_time:
                error_message = f"Route {line_num} (Vehicle {line_num}): Node {next_node.index} is visited outside its time window."
                return False, error_message
            else:
                current_time = arrival_time

            # Add service time
            current_time += next_node.service_time

            # Check capacity constraints
            current_capacity += next_node.demand
            if current_capacity > instance.capacity:
                error_message = (
                    f"Route {line_num}: Capacity exceeded at Node {next_node.index}."
                )
                return False, error_message

            # Check pick-up and delivery constraints
            if next_node.is_pickup:
                if next_node.delivery_sibling not in route[i + 1 :]:
                    error_message = f"Route {line_num}: Delivery for pickup Node {next_node.index} is missing."
                    return False, error_message
            else:
                if next_node.pickup_sibling not in route[: i + 1]:
                    error_message = f"Route {line_num}: Pickup for delivery Node {next_node.index} has not been visited."
                    return False, error_message

    # Check if all nodes are visited
    all_nodes = set(nodes.keys()) - set(instance.depot_node)  # Exclude depot nodes
    if visited_nodes != all_nodes:
        if len(visited_nodes) > len(all_nodes):
            error_message = f"Solution is invalid: nodes {visited_nodes - all_nodes} are visited more than once."
            return False, error_message
        if len(visited_nodes) < len(all_nodes):
            error_message = f"Solution is invalid: nodes {all_nodes - visited_nodes} are not visited."
            return False, error_message

    return True, error_message


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <solution_file>", file=sys.stderr)
        sys.exit(1)

    is_valid, message = verify(sys.argv[1], sys.argv[2])
    print(f"Valid: {is_valid}")
    if message:
        print(f"Message: {message}")

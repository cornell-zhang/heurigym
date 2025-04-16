from structure import Instance

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return round(((x2 - x1) ** 2 + (y1 - y2) ** 2) ** 0.5, 0)


def verify_solution(instance: Instance, solution_file: str) -> bool:
    """
    Verify the feasibility of a solution for the given PDPTW instance.

    Args:
        instance: The PDPTW instance containing node information.
        solution_file: Path to the solution file.

    Returns:
        True if the solution is feasible, False otherwise.
    """
    nodes = instance.nodes  # Use the dictionary directly
    visited_nodes = set()
    
    with open(solution_file, 'r') as file:
        lines = file.readlines()

    # Skip the first line
    for line_num, line in enumerate(lines[1:], start=1):
        route = list(map(int, line.strip().split()))
        if not route or route[0] != instance.depot_node[0] or route[-1] != instance.depot_node[0]:
            print(f"Route {line_num} (Vehicle {line_num}): Does not start and end at the depot.")
            return False

        current_time = 0
        current_capacity = 0

        for i in range(len(route) - 1):  # Skip the first and last elements (depot nodes)            
            current_node = nodes[route[i]]
            next_node = nodes[route[i + 1]]
            if next_node.index == instance.depot_node[0]:
                break
            visited_nodes.add(next_node.index)  # Mark the node as visited

            # Check time window constraints
            arrival_time = current_time + calculate_distance(current_node.x, current_node.y, next_node.x, next_node.y)
            if arrival_time < next_node.earliest_time:
                current_time = next_node.earliest_time
            elif arrival_time > next_node.latest_time:
                print(f"Route {line_num} (Vehicle {line_num}): Node {next_node.index} is visited outside its time window.")
                return False
            else:
                current_time = arrival_time

            # Add service time
            current_time += next_node.service_time

            # Check capacity constraints
            current_capacity += next_node.demand
            if current_capacity > instance.capacity:
                print(f"Route {line_num}: Capacity exceeded at Node {next_node.index}.")
                return False

            # Check pick-up and delivery constraints
            if next_node.is_pickup:
                if next_node.delivery_sibling not in route[i + 1:]:
                    print(f"Route {line_num}: Delivery for pickup Node {next_node.index} is missing.")
                    return False
            else:
                if next_node.pickup_sibling not in route[:i + 1]:
                    print(f"Route {line_num}: Pickup for delivery Node {next_node.index} has not been visited.")
                    return False

    # Check if all nodes are visited
    all_nodes = set(nodes.keys()) - set(instance.depot_node)  # Exclude depot nodes
    if visited_nodes != all_nodes:
        if len(visited_nodes) > len(all_nodes):
            print(f"Solution is invalid: nodes {visited_nodes - all_nodes} are visited more than once.")
            return False
        if len(visited_nodes) < len(all_nodes):
            print(f"Solution is invalid: nodes {all_nodes - visited_nodes} are not visited.")
        return False

    return True

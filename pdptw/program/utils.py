import sys
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Node:
    index: int
    x: float
    y: float
    demand: float
    earliest_time: float
    latest_time: float
    service_time: float
    pickup_sibling: int
    delivery_sibling: int
    is_pickup: bool  # Added indicator for pickup or delivery

@dataclass
class Instance:
    name: str
    type: str
    dimension: int
    vehicles: int
    capacity: float
    edge_weight_type: str
    nodes: Dict[int, Node]  # Changed to a dictionary
    depot_node: List[int]

def read_instance(file_path: str) -> Instance:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    name = ""
    type_ = ""
    dimension = 0
    vehicles = 0
    capacity = 0.0
    edge_weight_type = ""
    nodes = {}
    depot_node = []

    section = None
    for line in lines:
        line = line.strip()
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("TYPE"):
            type_ = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("VEHICLES"):
            vehicles = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            capacity = float(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            section = "NODE_COORD_SECTION"
        elif line.startswith("PICKUP_AND_DELIVERY_SECTION"):
            section = "PICKUP_AND_DELIVERY_SECTION"
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT_SECTION"
        elif line.startswith("EOF"):
            break
        elif section == "NODE_COORD_SECTION":
            parts = line.split()
            index = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            nodes[index] = Node(
                index=index,
                x=x,
                y=y,
                demand=0.0,
                earliest_time=0.0,
                latest_time=0.0,
                service_time=0.0,
                pickup_sibling=0,
                delivery_sibling=0,
                is_pickup=False  # Default value
            )
        elif section == "PICKUP_AND_DELIVERY_SECTION":
            parts = line.split()
            index = int(parts[0])
            if index in nodes:
                nodes[index].demand = float(parts[1])
                nodes[index].earliest_time = float(parts[2])
                nodes[index].latest_time = float(parts[3])
                nodes[index].service_time = float(parts[4])
                nodes[index].pickup_sibling = int(parts[5])
                nodes[index].delivery_sibling = int(parts[6])
                nodes[index].is_pickup = nodes[index].pickup_sibling == 0  # Set indicator
        elif section == "DEPOT_SECTION":
            if line != "-1":
                depot_node.append(int(line))

    return Instance(
        name=name,
        type=type_,
        dimension=dimension,
        vehicles=vehicles,
        capacity=capacity,
        edge_weight_type=edge_weight_type,
        nodes=nodes,
        depot_node=depot_node
    )

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
            print(f"Route {line_num} (Vehicle {line_num}): Does not start and end at the depot.", file=sys.stderr)
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
                print(f"Route {line_num} (Vehicle {line_num}): Node {next_node.index} is visited outside its time window.", file=sys.stderr)
                return False
            else:
                current_time = arrival_time

            # Add service time
            current_time += next_node.service_time

            # Check capacity constraints
            current_capacity += next_node.demand
            if current_capacity > instance.capacity:
                print(f"Route {line_num}: Capacity exceeded at Node {next_node.index}.", file=sys.stderr)
                return False

            # Check pick-up and delivery constraints
            if next_node.is_pickup:
                if next_node.delivery_sibling not in route[i + 1:]:
                    print(f"Route {line_num}: Delivery for pickup Node {next_node.index} is missing.", file=sys.stderr)
                    return False
            else:
                if next_node.pickup_sibling not in route[:i + 1]:
                    print(f"Route {line_num}: Pickup for delivery Node {next_node.index} has not been visited.", file=sys.stderr)
                    return False

    # Check if all nodes are visited
    all_nodes = set(nodes.keys()) - set(instance.depot_node)  # Exclude depot nodes
    if visited_nodes != all_nodes:
        if len(visited_nodes) > len(all_nodes):
            print(f"Solution is invalid: nodes {visited_nodes - all_nodes} are visited more than once.", file=sys.stderr)
            return False
        if len(visited_nodes) < len(all_nodes):
            print(f"Solution is invalid: nodes {all_nodes - visited_nodes} are not visited.", file=sys.stderr)
        return False

    return True

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
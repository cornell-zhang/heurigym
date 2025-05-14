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
    with open(file_path, "r") as file:
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
                is_pickup=False,  # Default value
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
                nodes[index].is_pickup = (
                    nodes[index].pickup_sibling == 0
                )  # Set indicator
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
        depot_node=depot_node,
    )


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return round(((x2 - x1) ** 2 + (y1 - y2) ** 2) ** 0.5, 2)

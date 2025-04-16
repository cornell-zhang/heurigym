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

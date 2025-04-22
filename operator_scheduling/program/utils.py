#!/usr/bin/env python3
import os
import json
import sys
from typing import Dict, List, Tuple
from solver import Node

def get_filename(path: str) -> str:
    """Extract filename from path."""
    return os.path.basename(path).split('.')[0]

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
        with open(filename, 'r') as file:
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

def verify(nodes: Dict[str, Node],
           schedule: Dict[str, int],
           delay: Dict[str, int],
           resource_constraints: Dict[str, int]) -> bool:
    """Verification function: checks dependency and resource constraints.
    
    Dependency: For each edge, finish time of predecessor (start + delay)
    must be less than or equal to the start time of the successor.
    
    Resource: At each cycle, the active operations for a resource type must
    not exceed the available functional units.
    """
    valid = True
    
    # Check data dependency constraints
    for node_id, node in nodes.items():
        node_delay = delay[node.resource]
        for succ_id in node.succs:
            if schedule[node_id] + node_delay > schedule[succ_id]:
                print(f"Dependency constraint violated: {node_id} "
                      f"finishes at {schedule[node_id] + node_delay} "
                      f"but {succ_id} starts at {schedule[succ_id]}", 
                      file=sys.stderr)
                valid = False
    
    # Determine overall latency (final cycle when operations end)
    final_cycle = 0
    for node_id, node in nodes.items():
        finish_time = schedule[node_id] + delay[node.resource]
        final_cycle = max(final_cycle, finish_time)
    
    # Check resource constraints at each cycle from 0 to finalCycle
    for t in range(final_cycle + 1):
        # Count active operations per resource type
        resource_usage = {resource: 0 for resource in resource_constraints.keys()}
        
        # For each node, if its active time covers cycle t, increment usage
        for node_id, node in nodes.items():
            start = schedule[node_id]
            finish = schedule[node_id] + delay[node.resource]
            if start <= t < finish:
                resource_usage[node.resource] += 1
        
        # Verify that usage does not exceed available units
        for resource, available in resource_constraints.items():
            if resource_usage[resource] > available:
                print(f"Resource constraint violated for resource {resource} "
                      f"at time {t}: used {resource_usage[resource]}, "
                      f"available {available}", file=sys.stderr)
                valid = False
    
    return valid

def calculate_cost(nodes: Dict[str, Node],
                  schedule: Dict[str, int],
                  delay: Dict[str, int]) -> int:
    """Cost calculation function: calculates the final latency.
    
    Final latency is defined as the maximum over operations of (start cycle + delay).
    """
    latency = 0
    for node_id, node in nodes.items():
        finish_time = schedule[node_id] + delay[node.resource]
        latency = max(latency, finish_time)
    return latency

def parse_schedule(filename: str) -> Dict[str, int]:
    """Parse the schedule file.
    
    Returns:
        Dictionary mapping node IDs to their scheduled start times
    """
    schedule = {}
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                # Format: node_id:start_time
                parts = line.split(':')
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
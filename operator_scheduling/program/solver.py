import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Node:
    id: str  # e.g., "n1"
    resource: str  # e.g., "mul" or "sub"
    preds: List[str] = field(default_factory=list)  # Predecessor nodes
    succs: List[str] = field(default_factory=list)  # Successor nodes
    start_time: int = 0  # Scheduled start cycle (to be computed)
    in_degree: int = 0  # Number of incoming edges (for topological ordering)

def solve(nodes: Dict[str, Node], 
          delay: Dict[str, int], 
          resource_constraints: Dict[str, int]) -> Dict[str, int]:
    """
    Solve the operator scheduling problem.
    
    Args:
        nodes: Dictionary mapping node IDs to Node objects
        delay: Dictionary mapping resource types to their delays
        resource_constraints: Dictionary mapping resource types to available functional units
        
    Returns:
        Dictionary mapping node IDs to their scheduled start times
    """
    schedule = {}
    
    # Generate topological order
    topological_order = []
    in_degree_copy = {node_id: node.in_degree for node_id, node in nodes.items()}
    q = [node_id for node_id, in_deg in in_degree_copy.items() if in_deg == 0]
    
    while q:
        u = q.pop(0)
        topological_order.append(u)
        for succ in nodes[u].succs:
            in_degree_copy[succ] -= 1
            if in_degree_copy[succ] == 0:
                q.append(succ)
    
    # Calculate ASAP (As Soon As Possible) schedule
    asap = {}
    for node_id in topological_order:
        asap[node_id] = 0
        for pred in nodes[node_id].preds:
            asap[node_id] = max(asap[node_id], asap[pred] + delay[nodes[pred].resource])
    
    # Calculate depth for each node
    depth = {}
    reverse_topo = list(reversed(topological_order))
    for node_id in reverse_topo:
        max_succ_depth = 0
        for succ in nodes[node_id].succs:
            max_succ_depth = max(max_succ_depth, depth.get(succ, 0))
        depth[node_id] = delay[nodes[node_id].resource] + max_succ_depth
    
    # Initialize current in-degree
    current_in_degree = {node_id: len(node.preds) for node_id, node in nodes.items()}
    
    # Priority queue for scheduling
    pq = []
    for node_id, in_deg in current_in_degree.items():
        if in_deg == 0:
            heapq.heappush(pq, (-depth[node_id], 0, node_id))
    
    # Track resource usage
    resource_usage = {resource: {} for resource in resource_constraints.keys()}
    
    while pq:
        neg_depth_val, earliest_start, node_id = heapq.heappop(pq)
        
        u = nodes[node_id]
        r = u.resource
        d = delay[r]
        G = resource_constraints[r]
        
        # Find earliest feasible start time
        t = earliest_start
        while True:
            feasible = True
            for step in range(t, t + d):
                if resource_usage[r].get(step, 0) >= G:
                    feasible = False
                    break
            if feasible:
                break
            t += 1
        
        # Schedule the node
        schedule[node_id] = t
        for step in range(t, t + d):
            resource_usage[r][step] = resource_usage[r].get(step, 0) + 1
        
        # Update successors
        for succ_id in u.succs:
            current_in_degree[succ_id] -= 1
            if current_in_degree[succ_id] == 0:
                earliest_start_succ = 0
                for pred in nodes[succ_id].preds:
                    earliest_start_succ = max(earliest_start_succ, 
                                            schedule[pred] + delay[nodes[pred].resource])
                heapq.heappush(pq, (-depth[succ_id], earliest_start_succ, succ_id))
    
    return schedule 
from typing import List, Tuple, Dict, Set, FrozenSet
from structure import LogicNetwork, KLut
import networkx as nx
import itertools

# DEBUG = True
DEBUG = False

class Cut:
    """Represents a cut in the logic network"""
    def __init__(self, node: str, inputs: FrozenSet[str], level: int = 0):
        self.node = node
        self.inputs = inputs
        self.level = level  # Depth/level of this cut in the network

    def __repr__(self):
        return f"Cut({self.node}, inputs={list(self.inputs)}, level={self.level})"


def enumerate_cuts(G: nx.DiGraph, node: str, k: int, cut_set: Dict[str, List[Cut]], visited: Set[str] = None) -> List[Cut]:
    """
    Enumerate all possible k-feasible cuts for a node using recursive approach
    """
    if visited is None:
        visited = set()
    
    # Return empty list if node has been visited (to avoid cycles)
    if node in visited:
        return []
    
    # If node is a PI, the only cut is the node itself with level 0
    if G.in_degree(node) == 0:
        trivial_cut = Cut(node, frozenset([node]), level=0)
        return [trivial_cut]
    
    # If we've already computed cuts for this node, return them
    if node in cut_set:
        return cut_set[node]
    
    # Initialize with the trivial cut (node itself) with level 0
    trivial_cut = Cut(node, frozenset([node]), level=0)
    all_cuts = [trivial_cut]
    
    # Use a set to track unique input sets to avoid redundant cuts
    unique_inputs = {frozenset([node])}
    
    # Get predecessors
    predecessors = list(G.predecessors(node))
    
    # If node has at most k inputs, add the cut with all inputs
    if len(predecessors) <= k:
        # For direct inputs, level is 1
        input_cut = Cut(node, frozenset(predecessors), level=1)
        all_cuts.append(input_cut)
        unique_inputs.add(frozenset(predecessors))
    
    # For more complex cases, recursively compute cuts for each input
    # and merge them to form new cuts
    if len(predecessors) > 1:
        # Compute cuts for each predecessor
        pred_cuts: Dict[str, List[Cut]] = {}
        for pred in predecessors:
            visited_new = visited.copy()
            visited_new.add(node)  # Mark current node as visited to avoid cycles
            pred_cuts[pred] = enumerate_cuts(G, pred, k, cut_set, visited_new)
        
        # Generate all combinations of predecessor cuts
        for combo in itertools.product(*[pred_cuts[pred] for pred in predecessors]):
            # The inputs of the new cut are the union of inputs of all predecessor cuts
            inputs = frozenset().union(*[cut.inputs for cut in combo])
            
            # Only consider k-feasible cuts that we haven't seen before
            if len(inputs) <= k and inputs not in unique_inputs:
                # Level is 1 + maximum level of input cuts
                max_input_level = max(cut.level for cut in combo)
                level = 1 + max_input_level
                
                new_cut = Cut(node, inputs, level=level)
                all_cuts.append(new_cut)
                unique_inputs.add(inputs)
    
    # Store the computed cuts for this node
    cut_set[node] = all_cuts
    return all_cuts



def create_lut_for_cut(G: nx.DiGraph, node: str, cut: Cut) -> KLut:
    """
    Create a KLut for a given cut
    
    Args:
        G: The logic network graph
        node: The node for which to create the LUT
        cut: The cut to use for the LUT creation
        
    Returns:
        A KLut object with inputs and truth table
    """
    inputs = list(cut.inputs)
    # placeholder for correct truth table logic
    truth_table = {}
    return KLut(k=len(inputs), inputs=inputs, truth_table=truth_table)


def solve(netlist: LogicNetwork, k: int, delay_budget: float = None, optimize_for: str = "size") -> Tuple[float, List[Tuple[str, KLut]]]:
    """
    k-LUT based technology mapping using cut enumeration and dynamic programming.
    
    Args:
        netlist: The input logic network.
        k: The maximum number of inputs for each LUT.
        delay_budget: Optional timing constraint.
        optimize_for: Whether to optimize for "size" (number of LUTs) or "depth" (number of levels)
        
    Returns:
        - total_area (float): Total area of the mapped network (number of LUTs).
        - mapping (List[(node_id, lut)]): Mapping results for each node.
    """
    G = netlist.graph
    mapping: List[Tuple[str, KLut]] = []
    
    # Phase 1: Cut enumeration for all nodes
    all_cuts: Dict[str, List[Cut]] = {}
    
    if DEBUG:
        print("===== Phase 1: All cuts ======")

    # Enumerate cuts in topological order
    for node in netlist.topological_order():
        if node in netlist.PI:
            # PIs don't need cuts, but we'll add a trivial cut for consistency
            all_cuts[node] = [Cut(node, frozenset([node]), level=0)]
            continue
        
        # For normal nodes, enumerate all possible k-feasible cuts
        enumerate_cuts(G, node, k, all_cuts)
    
    if DEBUG:
        # Debug: print all cuts
        for node, cuts in all_cuts.items():
            print(f"Node {node} has {len(cuts)} cuts")
            for cut in cuts:
                print(cut)

    # Phase 2: Dynamic programming to select the best cut for each node
    if DEBUG:
        print("===== Phase 2: Dynamic programming ======")


    best_cut: Dict[str, Cut] = {}
    best_lut_count: Dict[str, int] = {}  # Number of LUTs in the best implementation
    best_level: Dict[str, int] = {}      # Level (depth) in the best implementation
    
    # Initialize metrics for PIs
    for pi in netlist.PI:
        best_lut_count[pi] = 0  # Primary inputs don't need LUTs
        best_level[pi] = 0      # Primary inputs are at level 0
    
    # Dynamic programming approach based on optimization target
    for node in netlist.topological_order():
        if node in netlist.PI:
            continue
        
        if optimize_for == "size":
            # Optimize for minimum number of LUTs
            min_lut_count = float('inf')
            min_lut_count_cut = None
            min_level = float('inf')  # Track level for tie-breaking

            if DEBUG:
                print(f"Node {node} has {len(all_cuts.get(node, []))} cuts")

            for cut in all_cuts.get(node, []):
                if len(cut.inputs) == 1 and list(cut.inputs)[0] == node:
                    # Skip trivial cut of node itself
                    continue

                # Calculate LUT count for this implementation. Sum up the LUT count of all inputs and this LUT. 
                lut_count = 1  # This LUT
                for input_node in cut.inputs:
                    if input_node not in netlist.PI:
                        lut_count += best_lut_count.get(input_node, 0)
                
                # Calculate level for tie-breaking. Max of the level of all inputs + 1. 
                level = 1  # This LUT
                for input_node in cut.inputs:
                    if input_node not in netlist.PI:
                        input_level = best_level.get(input_node, 0)
                        level = max(level, 1 + input_level)
                
                # Check delay budget if specified
                if delay_budget is not None and level > delay_budget:
                    continue
                
                # Choose the cut with minimum LUT count
                if lut_count < min_lut_count:
                    min_lut_count = lut_count
                    min_lut_count_cut = cut
                    min_level = level
                # If LUT counts are equal, choose the one with lower level
                elif lut_count == min_lut_count and level < min_level:
                    min_lut_count_cut = cut
                    min_level = level

                # Debug
                if DEBUG:
                    print(f"cut: {cut} with lut_count {lut_count} and level {level}")

            # Store the best cut and its metrics
            if min_lut_count_cut:
                best_cut[node] = min_lut_count_cut
                best_lut_count[node] = min_lut_count
                best_level[node] = min_level
            
            # Debug
            if DEBUG:
                print(f"best_cut {best_cut[node]} with lut_count {best_lut_count[node]} and level {best_level[node]}")
                
        else:  # optimize_for == "depth"
            # Optimize for minimum level (depth)
            min_level = float('inf')
            min_level_cut = None
            min_lut_count = float('inf')  # Track LUT count for tie-breaking
            
            for cut in all_cuts.get(node, []):
                if len(cut.inputs) == 1 and list(cut.inputs)[0] == node:
                    # Skip trivial cut of node itself
                    continue
                
                # Calculate level for this implementation
                level = 1  # This LUT
                for input_node in cut.inputs:
                    if input_node not in netlist.PI:
                        input_level = best_level.get(input_node, 0)
                        level = max(level, 1 + input_level)
                
                # Calculate LUT count for tie-breaking
                lut_count = 1  # This LUT
                for input_node in cut.inputs:
                    if input_node not in netlist.PI:
                        lut_count += best_lut_count.get(input_node, 0)
                
                # Choose the cut with minimum level
                if level < min_level:
                    min_level = level
                    min_level_cut = cut
                    min_lut_count = lut_count
                # If levels are equal, choose the one with lower LUT count
                elif level == min_level and lut_count < min_lut_count:
                    min_level_cut = cut
                    min_lut_count = lut_count
            
            # Store the best cut and its metrics
            if min_level_cut:
                best_cut[node] = min_level_cut
                best_level[node] = min_level
                best_lut_count[node] = min_lut_count
    
    # Debug: print best cut for each node
    if DEBUG:
        print("===== Phase 2 Completed: Best cut for each node ======")
        for node, cut in best_cut.items():
            print(f"Node {node} has best cut {cut}")



    # Phase 3: Generate mapping from the selected cuts
    nodes_ready_to_map = set(netlist.PO) # start from POs. This set includes all nodes that are mapped to LUTs. 
    while nodes_ready_to_map:
        node = nodes_ready_to_map.pop()
        if node in best_cut: # only PI node is not included in best_cut. 
            cut = best_cut[node]
            lut = create_lut_for_cut(G, node, cut)
            mapping.append((node, lut))
            # add the inputs of this LUT to the nodes_ready_to_map
            for input_node in cut.inputs:
                if input_node not in netlist.PI and input_node not in nodes_ready_to_map:
                    nodes_ready_to_map.add(input_node)
    

    # Calculate total area - in this simplified model, it's just the count of LUTs
    total_area = len(mapping)
    
    # Calculate maximum level for statistics. It is the max level of the POs. 
    max_level = 0
    for po in netlist.PO:
        max_level = max(max_level, best_level[po])
    
    if DEBUG:
        print("===== Phase 3: Mapping ======")
        for node, lut in mapping:
            # print the node and LUT info with its inputs and truth table
            print(f"Node {node} has LUT with inputs {lut.inputs}")
            # print(f"truth table {lut.truth_table}")


    # Print statistics
    print(f"======== Mapping statistics ========")
    print(f"  Number of LUTs: {total_area}")
    print(f"  Maximum depth (levels): {max_level}")
    print(f"  Optimization target: {optimize_for}")
    
    return total_area, mapping


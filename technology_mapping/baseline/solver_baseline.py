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
    
    # Handle special case: if the cut contains only the node itself, it's a wire
    if len(inputs) == 1 and inputs[0] == node:
        # For a wire, output is always equal to input
        truth_table = [('1', '1')]
        return KLut(k=len(inputs), inputs=inputs, truth_table=truth_table)
    
    # Generate all possible input combinations for the cut
    input_combinations = list(itertools.product(['0', '1'], repeat=len(inputs)))
    
    # Create a simulation context to evaluate the output for each input combination
    truth_table = []
    for input_combo in input_combinations:
        # Map each input to its value
        input_values = {inputs[i]: input_combo[i] for i in range(len(inputs))}
        
        # Simulate the logic network to determine the output for this input combination
        output = simulate_cut(G, node, cut.inputs, input_values)
        
        # Add to the truth table if output is '1' (BLIF format typically only lists '1' outputs)
        input_pattern = ''.join(input_combo)
        if output == '1':
            truth_table.append((input_pattern, output))
    
    # If the truth table is empty (all outputs are '0'), use empty table
    # BLIF format handles this case as constant 0
    # If the truth table has all 2^k entries (all outputs are '1'), we can optimize
    # by just using the single entry "1" for constant 1
    if len(truth_table) == 2**len(inputs):
        truth_table = [('', '1')]  # Constant 1 in BLIF format
    
    return KLut(k=len(inputs), inputs=inputs, truth_table=truth_table)

def simulate_cut(G: nx.DiGraph, target_node: str, cut_inputs: FrozenSet[str], input_values: Dict[str, str]) -> str:
    """
    Simulates a cut subgraph to determine the output value for given input values
    
    Args:
        G: The logic network graph
        target_node: The node to evaluate (output of the cut)
        cut_inputs: The inputs to the cut (boundary)
        input_values: Values for the cut inputs
        
    Returns:
        The output value ('0' or '1') for the node
    """
    # Cache to avoid re-computing node values
    value_cache = {}
    
    # Add initial input values to cache
    for node, value in input_values.items():
        value_cache[node] = value
    
    # Define a recursive evaluation function for this specific cut
    def evaluate_node_in_cut(node: str) -> str:
        # If we've already calculated this node's value, return it
        if node in value_cache:
            return value_cache[node]
        
        # If this node is a cut input, use the provided value
        if node in cut_inputs:
            if node not in input_values:
                raise ValueError(f"Missing value for cut input {node}")
            value_cache[node] = input_values[node]
            return input_values[node]
        
        # For internal nodes, compute based on inputs and truth table
        predecessors = list(G.predecessors(node))
        
        # First evaluate all inputs
        input_pattern = []
        for pred in predecessors:
            input_pattern.append(evaluate_node_in_cut(pred))
        
        # Check if this node has a truth table
        if 'truth_table' in G.nodes[node]:
            truth_table = G.nodes[node]['truth_table']
            input_str = ''.join(input_pattern)
            
            # Special case for constant 1
            if len(truth_table) == 1 and truth_table[0][0] == '' and truth_table[0][1] == '1':
                value_cache[node] = '1'
                return '1'
                
            # Find the matching entry in the truth table
            for pattern, result in truth_table:
                if pattern == input_str:
                    value_cache[node] = result
                    return result
            
            # If node has 'explicit_truth_table', check input against it with don't-care handling
            if 'explicit_truth_table' in G.nodes[node]:
                explicit_truth_table = G.nodes[node]['explicit_truth_table']
                
                # Check if we have any output '0' entries
                has_output_0 = any(output == '0' for _, output in explicit_truth_table)
                default_output = '1' if has_output_0 and not any(output == '1' for _, output in explicit_truth_table) else '0'
                
                for pattern, result in explicit_truth_table:
                    # Handle don't-care in the pattern
                    if '-' in pattern:
                        match = True
                        for i, pat_bit in enumerate(pattern):
                            if pat_bit != '-' and pat_bit != input_pattern[i]:
                                match = False
                                break
                                
                        if match:
                            value_cache[node] = result
                            return result
                
                # Return default if no match in explicit table
                value_cache[node] = default_output
                return default_output
            
            # If no match found (shouldn't happen with complete table), default to '0'
            value_cache[node] = '0'
            return '0'
            
        # If node has no truth table but only one input, it's a wire/buffer
        if len(predecessors) == 1:
            value = evaluate_node_in_cut(predecessors[0])
            value_cache[node] = value
            return value
            
        # Default case (should rarely happen)
        value_cache[node] = '0'
        return '0'
    
    # Evaluate the target node, which triggers recursive evaluation of the subgraph
    return evaluate_node_in_cut(target_node)


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

                # Calculate LUT count for this implementation
                lut_count = 1  # This LUT
                for input_node in cut.inputs:
                    if input_node not in netlist.PI:
                        lut_count += best_lut_count.get(input_node, 0)
                
                # Calculate level for tie-breaking
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
    nodes_ready_to_map = set(netlist.PO)  # start from POs
    visited_nodes = set()  # Track visited nodes to avoid duplicates
    
    while nodes_ready_to_map:
        node = nodes_ready_to_map.pop()
        
        # Skip if already visited
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        
        # If node is not a PI, create a LUT for it
        if node not in netlist.PI and node in best_cut:
            cut = best_cut[node]
            lut = create_lut_for_cut(G, node, cut)
            mapping.append((node, lut))
            
            # Add the inputs of this LUT to the nodes_ready_to_map if they're not PIs
            for input_node in cut.inputs:
                if input_node not in netlist.PI and input_node not in visited_nodes:
                    nodes_ready_to_map.add(input_node)
    
    # Calculate total area - in this simplified model, it's just the count of LUTs
    total_area = len(mapping)
    
    # Calculate maximum level for statistics
    max_level = 0
    for po in netlist.PO:
        if po in best_level:
            max_level = max(max_level, best_level[po])
    
    if DEBUG:
        print("===== Phase 3: Mapping ======")
        for node, lut in mapping:
            print(f"Node {node} has LUT with inputs {lut.inputs}")
            if hasattr(lut, 'truth_table'):
                print(f"  Truth table entries: {len(lut.truth_table)}")

    # Print statistics
    print(f"======== Mapping statistics ========")
    print(f"  Number of LUTs: {total_area}")
    print(f"  Maximum depth (levels): {max_level}")
    print(f"  Optimization target: {optimize_for}")
    
    return total_area, mapping


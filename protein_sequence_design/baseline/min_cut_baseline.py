import sys
from pathlib import Path
import networkx as nx
from Bio.SeqUtils import seq1
import warnings

# d_ij, s_i, g and Phi are shared with the evaluator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "program"))
from utils import ALPHA, BETA, gc_fitness, gc_terms, get_natural_hp_sequence

def calculate_percentage_agreement(seq1, seq2):
    """Calculates the percentage agreement between two sequences."""
    if len(seq1) != len(seq2):
        warnings.warn(f"Sequences have different lengths ({len(seq1)} vs {len(seq2)}), cannot calculate agreement.", UserWarning)
        return 0.0
    if not seq1: return 100.0
    matches = sum(1 for i in range(len(seq1)) if seq1[i] == seq2[i])
    return (matches / len(seq1)) * 100.0

def build_graph(contacts, sasa_values, alpha, beta):
    """Builds the graph G for the min-cut problem. contacts[i, j] is g(d_ij) for j >= i + 3."""
    n = len(sasa_values)
    G = nx.DiGraph()
    source, sink = 's', 't'
    G.add_nodes_from([source, sink])
    v_nodes = [f'v_{i}' for i in range(n)]
    G.add_nodes_from(v_nodes)
    B = 0.0
    u_nodes_data = []
    for i in range(n):
        for j in range(i + 3, n):
            g_dij = contacts[i, j]
            if g_dij > 0:
                if alpha >= 0: warnings.warn(f"Alpha ({alpha}) not negative.", UserWarning)
                B += abs(alpha) * g_dij
                u_nodes_data.append({'name': f'u_{i}_{j}', 'i': i, 'j': j, 'g_dij': g_dij})
    large_capacity = B + 1.0 if B > 0 else 1.0
    for u_data in u_nodes_data:
        u_node, i, j, g_dij = u_data['name'], u_data['i'], u_data['j'], u_data['g_dij']
        G.add_node(u_node)
        cap_s_u = abs(alpha) * g_dij
        if cap_s_u > 0: G.add_edge(source, u_node, capacity=cap_s_u)
        G.add_edge(u_node, v_nodes[i], capacity=large_capacity)
        G.add_edge(u_node, v_nodes[j], capacity=large_capacity)
    if beta <= 0: warnings.warn(f"Beta ({beta}) not positive.", UserWarning)
    for i in range(n):
        s_i = sasa_values[i]
        if s_i > 0:
            cap_v_t = beta * s_i
            if cap_v_t > 0: G.add_edge(v_nodes[i], sink, capacity=cap_v_t)
    return G, source, sink

def get_optimal_sequence(graph, source, sink, n_residues):
    """Computes min-cut and returns the H/P sequence."""
    try:
        cut_value, partition = nx.minimum_cut(graph, source, sink)
        reachable, non_reachable = partition
    except Exception as e:
         print(f"Error computing minimum cut: {e}")
         has_path = False
         try: has_path = nx.has_path(graph, source, sink)
         except nx.NodeNotFound: pass
         if not has_path: print("Source/Sink path check failed. Defaulting to all 'P'."); return 'P' * n_residues
         else: print("Min-cut failed despite path existing."); raise e
    sequence = ['P'] * n_residues
    v_nodes = [f'v_{i}' for i in range(n_residues)]
    for i in range(n_residues):
        if v_nodes[i] in reachable: sequence[i] = 'H'
    return "".join(sequence)

def main():
    pdb_file = None
    if len(sys.argv) == 2:
        pdb_file = sys.argv[1]
    else:
        print("Usage: python .py <pdb_file>")
        sys.exit(1)

    alpha = ALPHA; beta = BETA

    print("1. Parsing PDB, calculating distances and SASA on model 0 standard residues...")
    residues, contacts, sasa_values = gc_terms(pdb_file)
    n_residues = len(residues)
    if n_residues == 0: print("Error: No standard residues found."); sys.exit(1)
    original_sequence_1L = "".join(seq1(residue.get_resname()) for residue in residues)
    print(f"Found {n_residues} standard residues for analysis.")

    print("2. Building the graph...")
    graph, source, sink = build_graph(contacts, sasa_values, alpha, beta)
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    print("3. Computing minimum s-t cut (Optimal H/P Sequence)...")
    optimal_sequence = get_optimal_sequence(graph, source, sink, n_residues)

    print("4. Calculating Natural H/P Sequence and Agreement...")
    natural_hp_sequence = get_natural_hp_sequence(original_sequence_1L)
    percent_agreement = calculate_percentage_agreement(optimal_sequence, natural_hp_sequence)

    print("\n--- Results ---")
    print(f"PDB File Analyzed       : {pdb_file}")
    print(f"Residues Analyzed       : {n_residues}")
    seq_len_limit = 100
    print(f"Original Sequence (1L)  : {original_sequence_1L if n_residues <= seq_len_limit else original_sequence_1L[:50] + '...' + original_sequence_1L[-50:]}")
    print(f"Natural H/P Sequence    : {natural_hp_sequence if n_residues <= seq_len_limit else natural_hp_sequence[:50] + '...' + natural_hp_sequence[-50:]}")
    print(f"Optimal H/P Sequence    : {optimal_sequence if n_residues <= seq_len_limit else optimal_sequence[:50] + '...' + optimal_sequence[-50:]}")
    print(f"-------------------------------------------")
    print(f"Percentage Agreement    : {percent_agreement:.2f}%")
    print(f"Optimal Fitness (Phi)   : {gc_fitness(optimal_sequence, contacts, sasa_values):.4f}")
    print(f"-------------------------------------------")
    h_opt = optimal_sequence.count('H'); p_opt = n_residues - h_opt
    h_nat = natural_hp_sequence.count('H'); p_nat = n_residues - h_nat
    print(f"Optimal H/P Counts    : H={h_opt}, P={p_opt}")
    print(f"Natural H/P Counts    : H={h_nat}, P={p_nat}")

if __name__ == "__main__":

    main()

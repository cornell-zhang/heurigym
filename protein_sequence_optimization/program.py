import sys
import numpy as np
import networkx as nx
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from Bio.PDB.SASA import ShrakeRupley
import warnings
import warnings

HP_MAP = {
    'A': 'H', 'C': 'H', 'F': 'H', 'I': 'H', 'L': 'H',
    'M': 'H', 'V': 'H', 'W': 'H', 'Y': 'H',
    'R': 'P', 'N': 'P', 'D': 'P', 'Q': 'P', 'E': 'P',
    'G': 'P', 'H': 'P', 'K': 'P', 'P': 'P', 'S': 'P',
    'T': 'P'
}

def get_residue_coordinates_and_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0]
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return None, None, None

    residue_data = []
    sequence = []

    for chain in model:
        for residue in chain:
            if is_aa(residue, standard=True) and residue.id[0] == ' ':
                res_name = residue.get_resname().upper()
                atom_name = 'CA' if res_name == 'GLY' else ('CB' if 'CB' in residue else 'CA')
                if atom_name in residue:
                    coord = residue[atom_name].get_coord()
                    residue_data.append({'residue': residue, 'coord': coord})

                    # seq1 will map ALA→A, etc., and unknown→X
                    one_letter = seq1(res_name)
                    if one_letter == 'X':
                        warnings.warn(
                            f"seq1 could not map {res_name}; using 'X'.",
                            UserWarning
                        )
                    sequence.append(one_letter)

    if not residue_data:
        print("Error: No suitable standard residues with coordinates found.")
        return None, None, None

    return structure, residue_data, "".join(sequence)

def get_natural_hp_sequence(original_sequence_1_letter):
    """Converts a 1-letter AA sequence to its natural H/P sequence."""
    hp_sequence = [HP_MAP.get(aa, 'P') for aa in original_sequence_1_letter]
    return "".join(hp_sequence)

def calculate_percentage_agreement(seq1, seq2):
    """Calculates the percentage agreement between two sequences."""
    if len(seq1) != len(seq2):
        warnings.warn(f"Sequences have different lengths ({len(seq1)} vs {len(seq2)}), cannot calculate agreement.", UserWarning)
        return 0.0
    if not seq1: return 100.0
    matches = sum(1 for i in range(len(seq1)) if seq1[i] == seq2[i])
    return (matches / len(seq1)) * 100.0

def calculate_distances(residue_data):
    """Calculates pairwise distances between residues."""
    n_residues = len(residue_data)
    dist_matrix = np.zeros((n_residues, n_residues))
    coords = [rd['coord'] for rd in residue_data]
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist; dist_matrix[j, i] = dist
    return dist_matrix

def g_function(distance):
    """Sigmoidal function g(d_ij) from the paper."""
    if distance is None or distance > 6.5: return 0.0
    return 1.0 / (1.0 + np.exp(distance - 6.5))

def calculate_sasa(structure, residue_data):
    """
    Calculates SASA using Bio.PDB.SASA.ShrakeRupley.
    Args:
        structure (Bio.PDB.Structure): The parsed PDB structure object.
        residue_data (list): List of dicts [{'residue': res_obj, 'coord': coord_array}, ...].
    Returns:
        list: A list of SASA values corresponding to residue_data order. None if fails.
    """
    print("Attempting SASA calculation using Bio.PDB.SASA.ShrakeRupley...")
    try:
        sr = ShrakeRupley(probe_radius=1.4, n_points=100)
        sr.compute(structure, level="A") # Calculate at Atom level
        sasa_values = []
        for rd in residue_data:
            res = rd['residue']
            residue_sasa = 0.0
            try:
                for atom in res:
                    if hasattr(atom, 'sasa'): residue_sasa += atom.sasa
                sasa_values.append(residue_sasa)
            except Exception as atom_err:
                 warnings.warn(f"Error processing atoms for residue {res.id}: {atom_err}. Assigning SASA 0.0.", UserWarning)
                 sasa_values.append(0.0)

        if len(sasa_values) != len(residue_data):
             print(f"Error: SASA value list length mismatch")
             return None
        print(f"   Successfully obtained {len(sasa_values)} per-residue SASA values.")
        return sasa_values
    except Exception as e:
        print(f"Error during Bio.PDB.SASA calculation: {e}")
        return None

def build_graph(residue_data, dist_matrix, sasa_values, alpha, beta):
    """Builds the graph G for the min-cut problem."""
    n = len(residue_data)
    G = nx.DiGraph()
    source, sink = 's', 't'
    G.add_nodes_from([source, sink])
    v_nodes = [f'v_{i}' for i in range(n)]
    G.add_nodes_from(v_nodes)
    B = 0.0
    u_nodes_data = []
    for i in range(n):
        for j in range(i + 1, n):
            if j >= i + 3: # i < j-2 condition
                dist = dist_matrix[i, j]; g_dij = g_function(dist)
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
        # Simplified usage message as --dssp is removed
        print("Usage: python .py <pdb_file>")
        sys.exit(1)

    alpha = -2.0; beta = 1.0 / 3.0

    print("1. Parsing PDB, extracting coordinates and sequence...")
    structure, residue_data, original_sequence_1L = get_residue_coordinates_and_sequence(pdb_file)
    if residue_data is None: sys.exit(1)
    n_residues = len(residue_data)
    print(f"   Found {n_residues} standard residues for analysis.")

    print("2. Calculating distances...")
    dist_matrix = calculate_distances(residue_data)

    print("3. Calculating Solvent Accessible Surface Areas (SASA)...")
    # *** Corrected call to calculate_sasa: removed dssp_executable ***
    sasa_values = calculate_sasa(structure, residue_data)
    # ******************************************************************
    if sasa_values is None: print("   SASA calculation failed."); sys.exit(1)
    if len(sasa_values) != n_residues: print("Error: SASA values length mismatch."); sys.exit(1)

    print("4. Building the graph...")
    graph, source, sink = build_graph(residue_data, dist_matrix, sasa_values, alpha, beta)
    print(f"   Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    print("5. Computing minimum s-t cut (Optimal H/P Sequence)...")
    optimal_sequence = get_optimal_sequence(graph, source, sink, n_residues)

    print("6. Calculating Natural H/P Sequence and Agreement...")
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
    print(f"-------------------------------------------")
    h_opt = optimal_sequence.count('H'); p_opt = n_residues - h_opt
    h_nat = natural_hp_sequence.count('H'); p_nat = n_residues - h_nat
    print(f"Optimal H/P Counts    : H={h_opt}, P={p_opt}")
    print(f"Natural H/P Counts    : H={h_nat}, P={p_nat}")

if __name__ == "__main__":
    main()
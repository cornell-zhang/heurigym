# main.py
import sys
import warnings

# Import functions from our modules
from utils import (get_residue_coordinates_and_sequence,
                   get_natural_hp_sequence,
                   calculate_distances,
                   calculate_sasa)
from solver import solve_hp_sequence
from verifier import verify_hp_sequence
from evaluator import evaluate_agreement

def print_sequence_comparison(label, seq, limit=100):
    """Helper function to print sequences, truncating if long."""
    if len(seq) <= limit:
        print(f"{label:<25}: {seq}")
    else:
        print(f"{label:<25}: {seq[:50]}...{seq[-50:]}")

def main():
    pdb_file = None
    if len(sys.argv) == 2:
        pdb_file = sys.argv[1]
    else:
        print("Usage: python main.py <pdb_file>")
        sys.exit(1)

    # --- Parameters for the model ---
    alpha = -2.0 # Weight for hydrophobic interactions (negative)
    beta = 1.0 / 3.0 # Weight for solvent exposure (positive)
    # ---------------------------------

    print(f"Processing PDB file: {pdb_file}")
    print(f"Using parameters: alpha = {alpha}, beta = {beta}")

    print("\n1. Parsing PDB, extracting coordinates and sequence...")
    structure, residue_data, original_sequence_1L = get_residue_coordinates_and_sequence(pdb_file)
    if residue_data is None or structure is None or original_sequence_1L is None:
        print("Exiting due to errors in PDB parsing or data extraction.")
        sys.exit(1)
    n_residues = len(residue_data)
    print(f"  Found {n_residues} standard residues for analysis.")

    print("\n2. Calculating Natural H/P Sequence...")
    natural_hp_sequence = get_natural_hp_sequence(original_sequence_1L)
    if len(natural_hp_sequence) != n_residues:
         warnings.warn("Length mismatch between original AA sequence and natural HP sequence. Check HP_MAP or residue filtering.", UserWarning)
         # Decide how critical this is - maybe exit? For now, continue.

    print("\n3. Calculating pairwise distances...")
    dist_matrix = calculate_distances(residue_data)
    print(f"  Distance matrix calculated ({dist_matrix.shape}).")

    print("\n4. Calculating Solvent Accessible Surface Areas (SASA)...")
    # Pass the structure object obtained from parsing
    sasa_values = calculate_sasa(structure, residue_data)
    if sasa_values is None:
        print("  SASA calculation failed. Exiting.")
        sys.exit(1)
    if len(sasa_values) != n_residues:
        print(f"Error: SASA values list length ({len(sasa_values)}) does not match residue count ({n_residues}). Exiting.")
        sys.exit(1)

    # --- Call the Solver ---
    print("\n5. Solving for Optimal H/P Sequence...")
    # Pass necessary data to the solver function
    optimal_sequence = solve_hp_sequence(residue_data, dist_matrix, sasa_values, alpha, beta)

    if optimal_sequence is None:
         print("  Optimal sequence could not be determined. Exiting.")
         sys.exit(1)

    # --- Verification ---
    print("\n6. Verifying Optimal H/P Sequence...")
    is_valid = verify_hp_sequence(optimal_sequence, n_residues)
    if not is_valid:
        print("  The generated optimal sequence is invalid. Results might be incorrect.")
        # Decide whether to exit or continue with potentially flawed sequence
        # sys.exit(1) # Example: exit if verification fails

    # --- Evaluation ---
    print("\n7. Evaluating Agreement with Natural Sequence...")
    percent_agreement = evaluate_agreement(optimal_sequence, natural_hp_sequence)

    # --- Final Results ---
    print("\n--- Results ---")
    print(f"PDB File Analyzed           : {pdb_file}")
    print(f"Residues Analyzed           : {n_residues}")
    print("-" * 60)
    seq_len_limit = 100 # Truncate long sequences for display
    print_sequence_comparison("Original Sequence (1L)", original_sequence_1L, seq_len_limit)
    print_sequence_comparison("Natural H/P Sequence", natural_hp_sequence, seq_len_limit)
    print_sequence_comparison("Optimal H/P Sequence", optimal_sequence, seq_len_limit)
    print("-" * 60)
    print(f"Percentage Agreement        : {percent_agreement:.2f}%")
    print("-" * 60)
    h_opt = optimal_sequence.count('H'); p_opt = n_residues - h_opt
    h_nat = natural_hp_sequence.count('H'); p_nat = n_residues - h_nat
    print(f"Optimal H/P Counts          : H={h_opt}, P={p_opt} (Total: {h_opt+p_opt})")
    print(f"Natural H/P Counts          : H={h_nat}, P={p_nat} (Total: {h_nat+p_nat})")
    print("-" * 60)

if __name__ == "__main__":
    main()
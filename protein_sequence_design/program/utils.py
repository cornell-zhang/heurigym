# utils.py
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from Bio.PDB.SASA import ShrakeRupley
import warnings

# Hydrophobic/Polar mapping for standard amino acids
HP_MAP = {
    'A': 'H', 'C': 'H', 'F': 'H', 'I': 'H', 'L': 'H',
    'M': 'H', 'V': 'H', 'W': 'H', 'Y': 'H',
    'R': 'P', 'N': 'P', 'D': 'P', 'Q': 'P', 'E': 'P',
    'G': 'P', 'H': 'P', 'K': 'P', 'P': 'P', 'S': 'P',
    'T': 'P'
}

def get_residue_coordinates_and_sequence(pdb_file):
    """
    Parses a PDB file to extract residue objects, their representative coordinates
    (CB or CA), and the 1-letter amino acid sequence.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        tuple: (Bio.PDB.Structure object, list of residue data dicts, str)
               Returns (None, None, None) on failure.
               Residue data dict format: {'residue': Bio.PDB.Residue, 'coord': np.array}
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        model = structure[0] # Assuming single model PDB
    except Exception as e:
        print(f"Error parsing PDB file {pdb_file}: {e}")
        return None, None, None

    residue_data = []
    sequence = []

    for chain in model:
        for residue in chain:
            # Check if it's a standard amino acid and not a HETATM residue
            # residue.id[0] == ' ' filters out HETATMs and water
            if is_aa(residue, standard=True) and residue.id[0] == ' ':
                res_name = residue.get_resname().upper()
                # Use CB atom if available (except for GLY), otherwise use CA
                atom_name = 'CA' if res_name == 'GLY' else ('CB' if 'CB' in residue else 'CA')

                if atom_name in residue:
                    coord = residue[atom_name].get_coord()
                    residue_data.append({'residue': residue, 'coord': coord})

                    # Convert 3-letter code to 1-letter code
                    one_letter = seq1(res_name)
                    if one_letter == 'X': # Handle unknown residues if seq1 fails
                        warnings.warn(
                            f"seq1 could not map {res_name}; using 'X'.",
                            UserWarning
                        )
                    sequence.append(one_letter)
                # else:
                #     warnings.warn(f"Residue {residue.id} ({res_name}) missing required atom {atom_name}. Skipping.", UserWarning)


    if not residue_data:
        print("Error: No suitable standard residues with coordinates found.")
        return None, None, None

    return structure, residue_data, "".join(sequence)


def calculate_distances(residue_data):
    """
    Calculates the pairwise Euclidean distances between residue coordinates.

    Args:
        residue_data (list): List of residue data dicts from
                             get_residue_coordinates_and_sequence.

    Returns:
        np.ndarray: A square matrix where element (i, j) is the distance
                    between residue i and residue j.
    """
    n_residues = len(residue_data)
    dist_matrix = np.zeros((n_residues, n_residues))
    coords = [rd['coord'] for rd in residue_data]

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            # Calculate Euclidean distance
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist # Matrix is symmetric
    return dist_matrix


def get_natural_hp_sequence(original_sequence_1_letter):
    """
    Converts a 1-letter amino acid sequence to its corresponding
    Hydrophobic (H) / Polar (P) sequence based on HP_MAP.

    Args:
        original_sequence_1_letter (str): The 1-letter amino acid sequence.

    Returns:
        str: The H/P sequence. Unknown amino acids ('X') default to 'P'.
    """
    hp_sequence = [HP_MAP.get(aa, 'P') for aa in original_sequence_1_letter]
    return "".join(hp_sequence)


def calculate_sasa(structure, residue_data):
    """
    Calculates Solvent Accessible Surface Area (SASA) per residue using
    Bio.PDB.SASA.ShrakeRupley.

    Args:
        structure (Bio.PDB.Structure): The parsed PDB structure object.
        residue_data (list): List of residue data dicts, used to ensure
                             SASA values correspond to the correct residues.

    Returns:
        list or None: A list of SASA values (float) in the same order as
                      residue_data. Returns None if the calculation fails.
    """
    print("Attempting SASA calculation using Bio.PDB.SASA.ShrakeRupley...")
    try:
        # Initialize ShrakeRupley algorithm
        sr = ShrakeRupley(probe_radius=1.4, n_points=100) # Standard parameters
        # Compute SASA for the whole structure at the Atom level
        sr.compute(structure, level="A")

        sasa_values = []
        residue_map = {rd['residue'].get_full_id(): i for i, rd in enumerate(residue_data)}
        temp_sasa = [0.0] * len(residue_data)

        # Sum atom SASAs for each residue *that we are tracking*
        for chain in structure.get_chains():
            for res in chain.get_residues():
                 # Check if this residue is one we extracted earlier
                 res_full_id = res.get_full_id()
                 if res_full_id in residue_map:
                    residue_sasa = 0.0
                    for atom in res:
                         if hasattr(atom, 'sasa') and atom.sasa is not None:
                             residue_sasa += atom.sasa
                         # else:
                             # print(f"DEBUG: Atom {atom.id} in {res.id} missing sasa attribute or value is None")

                    # Store the calculated SASA in the correct position
                    idx = residue_map[res_full_id]
                    temp_sasa[idx] = residue_sasa


        # Verify and finalize SASA values based on the original residue_data order
        sasa_values = [temp_sasa[residue_map[rd['residue'].get_full_id()]] for rd in residue_data]


        if len(sasa_values) != len(residue_data):
            print(f"Error: SASA value list length mismatch ({len(sasa_values)} vs {len(residue_data)}). This indicates an issue mapping SASA back to residues.")
            return None

        print(f"  Successfully obtained {len(sasa_values)} per-residue SASA values.")
        return sasa_values

    except Exception as e:
        print(f"Error during Bio.PDB.SASA calculation: {e}")
        import traceback
        traceback.print_exc()
        return None
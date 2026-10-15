import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

# Hydrophobic/Polar mapping for standard amino acids
HP_MAP = {
    "A": "H",
    "C": "H",
    "F": "H",
    "I": "H",
    "L": "H",
    "M": "H",
    "V": "H",
    "W": "H",
    "Y": "H",
    "R": "P",
    "N": "P",
    "D": "P",
    "Q": "P",
    "E": "P",
    "G": "P",
    "H": "P",
    "K": "P",
    "P": "P",
    "S": "P",
    "T": "P",
}


def get_natural_hp_sequence(original_sequence_1_letter: str) -> str:
    """
    Converts a 1-letter amino acid sequence to its corresponding
    Hydrophobic (H) / Polar (P) sequence based on HP_MAP.

    Args:
        original_sequence_1_letter (str): The 1-letter amino acid sequence.

    Returns:
        str: The H/P sequence. Unknown amino acids ('X') default to 'P'.
    """
    hp_sequence = [HP_MAP.get(aa, "P") for aa in original_sequence_1_letter]
    return "".join(hp_sequence)


# GC model parameters
ALPHA = -2.0  # weight of H-H contacts
BETA = 1.0 / 3.0  # weight of H solvent exposure
CONTACT_CUTOFF = 6.5  # Angstrom, g(d) = 0 beyond it
MIN_SEPARATION = 3  # a pair (i, j) is a contact only if j >= i + 3


def load_residues(pdb_file: str):
    """
    Parses the PDB file and keeps only the standard amino acid residues of model 0
    (no waters, ligands or other NMR models).

    Returns:
        tuple: (model, residues), where residues are in file order.
    """
    model = PDBParser(QUIET=True).get_structure("protein", pdb_file)[0]
    for chain in list(model):
        for residue in list(chain):
            if not (is_aa(residue, standard=True) and residue.id[0] == " "):
                chain.detach_child(residue.id)
    residues = [residue for chain in model for residue in chain]
    return model, residues


def side_chain_center(residue) -> np.ndarray:
    """CB atom, or CA for glycine and residues without CB. Falls back to the first atom."""
    name = "CA" if residue.get_resname().upper() == "GLY" or "CB" not in residue else "CB"
    if name in residue:
        return residue[name].get_coord()
    atoms = list(residue.get_atoms())
    return atoms[0].get_coord() if atoms else np.zeros(3)


def distance_matrix(residues) -> np.ndarray:
    """Pairwise distances d_ij between side chain centers."""
    coords = np.array([side_chain_center(residue) for residue in residues], dtype=float)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def contact_matrix(dist: np.ndarray) -> np.ndarray:
    """g(d_ij) = 1 / (1 + exp(d_ij - 6.5)) for d_ij <= 6.5, else 0. Only j >= i + 3 is kept."""
    g = np.where(dist <= CONTACT_CUTOFF, 1.0 / (1.0 + np.exp(dist - CONTACT_CUTOFF)), 0.0)
    return np.triu(g, k=MIN_SEPARATION)


def solvent_accessibility(model, residues) -> np.ndarray:
    """Per-residue SASA s_i: Shrake-Rupley with a 1.4 A probe and 100 points per atom."""
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(model, level="A")
    return np.array([sum(getattr(atom, "sasa", 0.0) for atom in residue) for residue in residues])


def gc_terms(pdb_file: str):
    """Returns (residues, contacts, sasa) used by the fitness function."""
    model, residues = load_residues(pdb_file)
    contacts = contact_matrix(distance_matrix(residues))
    sasa = solvent_accessibility(model, residues)
    return residues, contacts, sasa


def gc_fitness(hp_sequence: str, contacts: np.ndarray, sasa: np.ndarray) -> float:
    """Phi(S) = ALPHA * sum of g(d_ij) over H-H contacts + BETA * sum of s_i over H residues."""
    h = np.array([c == "H" for c in hp_sequence])
    return float(ALPHA * contacts[np.ix_(h, h)].sum() + BETA * sasa[h].sum())

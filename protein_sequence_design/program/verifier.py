from typing import Tuple
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


def verify(input_file: str, solution_file: str) -> Tuple[bool, str]:
    """
    Verifies if the generated sequence in solution_file consists only of 'H' and 'P'
    characters and matches the expected length from the input PDB file.

    Args:
        input_file (str): Path to the input PDB file
        solution_file (str): Path to the solution file containing the H/P sequence

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the sequence is valid, False otherwise
            - str: A message describing the verification result
    """
    # Parse the input PDB file to get the expected sequence length
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_file)
    model = structure[0]  # Assuming single model PDB

    # Count standard residues
    n_residues = 0
    for chain in model:
        for residue in chain:
            if is_aa(residue, standard=True) and residue.id[0] == " ":
                n_residues += 1

    # Read the solution file
    with open(solution_file, "r") as f:
        generated_sequence = f.read().strip()

    if not generated_sequence:
        return False, "Generated sequence is empty"

    # Check length
    if len(generated_sequence) != n_residues:
        return (
            False,
            f"Generated sequence length ({len(generated_sequence)}) does not match expected length ({n_residues})",
        )

    # Check characters
    allowed_chars = set("HP")
    invalid_chars = set(generated_sequence) - allowed_chars
    if invalid_chars:
        return (
            False,
            f"Generated sequence contains invalid characters: {invalid_chars}. Only 'H' and 'P' are allowed",
        )

    return True, "Generated sequence format and length verified successfully"

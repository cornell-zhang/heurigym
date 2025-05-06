from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from utils import get_natural_hp_sequence


def evaluate(input_file: str, solution_file: str) -> float:
    """
    Calculates the percentage agreement between the natural H/P sequence from the PDB file
    and the generated H/P sequence from the solution file.

    Args:
        input_file (str): Path to the input PDB file
        solution_file (str): Path to the solution file containing the generated H/P sequence

    Returns:
        float: The percentage agreement (0.0 to 100.0). Returns 0.0 if
               lengths differ or sequences are empty.
    """
    # Parse the input PDB file to get the natural sequence
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_file)
    model = structure[0]  # Assuming single model PDB

    # Extract the natural sequence
    natural_sequence = []
    for chain in model:
        for residue in chain:
            if is_aa(residue, standard=True) and residue.id[0] == " ":
                res_name = residue.get_resname().upper()
                one_letter = seq1(res_name)
                natural_sequence.append(one_letter)

    natural_sequence = "".join(natural_sequence)
    natural_hp_sequence = get_natural_hp_sequence(natural_sequence)

    # Read the solution file
    with open(solution_file, "r") as f:
        generated_sequence = f.read().strip()

    # Calculate agreement
    len1 = len(generated_sequence)
    len2 = len(natural_hp_sequence)

    if len1 != len2:
        return 0.0

    if len1 == 0:
        return 100.0

    matches = sum(
        1 for i in range(len1) if generated_sequence[i] == natural_hp_sequence[i]
    )
    agreement = (matches / len1) * 100.0
    return agreement

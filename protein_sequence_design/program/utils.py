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
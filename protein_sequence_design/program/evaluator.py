# evaluator.py
import warnings

def evaluate_agreement(generated_sequence, natural_sequence):
    """
    Calculates the percentage agreement between two sequences (e.g.,
    the generated H/P sequence and the natural H/P sequence).

    Args:
        generated_sequence (str): The first sequence (e.g., optimal H/P).
        natural_sequence (str): The second sequence (e.g., natural H/P).

    Returns:
        float: The percentage agreement (0.0 to 100.0). Returns 0.0 if
               lengths differ or sequences are empty.
    """
    len1 = len(generated_sequence)
    len2 = len(natural_sequence)

    if len1 != len2:
        warnings.warn(f"Sequences have different lengths ({len1} vs {len2}), cannot calculate agreement accurately. Returning 0.0.", UserWarning)
        return 0.0

    if len1 == 0: # Handle empty sequences case
        return 100.0 # Or 0.0 depending on desired behavior for empty inputs

    matches = sum(1 for i in range(len1) if generated_sequence[i] == natural_sequence[i])
    agreement = (matches / len1) * 100.0
    return agreement
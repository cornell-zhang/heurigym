# verifier.py
import warnings

def verify_hp_sequence(generated_sequence, expected_length):
    """
    Verifies if the generated sequence consists only of 'H' and 'P'
    characters and matches the expected length.

    Args:
        generated_sequence (str): The H/P sequence produced by the solver.
        expected_length (int): The expected length (usually the length of the
                               original protein sequence).

    Returns:
        bool: True if the sequence is valid, False otherwise.
    """
    valid = True
    if not generated_sequence:
        warnings.warn("Generated sequence is empty.", UserWarning)
        return False # Consider empty sequence invalid

    # Check length
    if len(generated_sequence) != expected_length:
        warnings.warn(f"Generated sequence length ({len(generated_sequence)}) does not match expected length ({expected_length}).", UserWarning)
        valid = False

    # Check characters
    allowed_chars = set('HP')
    invalid_chars = set(generated_sequence) - allowed_chars
    if invalid_chars:
        warnings.warn(f"Generated sequence contains invalid characters: {invalid_chars}. Only 'H' and 'P' are allowed.", UserWarning)
        valid = False

    if valid:
        print("  Generated sequence format and length verified successfully.")
    else:
        print("  Generated sequence verification failed.")

    return valid
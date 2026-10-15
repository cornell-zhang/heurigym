from utils import gc_fitness, gc_terms


def evaluate(input_file: str, solution_file: str) -> float:
    """
    Calculates the GC model fitness Phi of the generated H/P sequence on the input structure.

    Args:
        input_file (str): Path to the input PDB file
        solution_file (str): Path to the solution file containing the generated H/P sequence

    Returns:
        float: Phi(S) as defined in the README (lower is better). The all-P sequence
               has Phi = 0.
    """
    _, contacts, sasa = gc_terms(input_file)

    with open(solution_file, "r") as f:
        generated_sequence = f.read().strip()

    return gc_fitness(generated_sequence, contacts, sasa)

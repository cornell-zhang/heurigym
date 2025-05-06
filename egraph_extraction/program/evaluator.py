"""
Evaluator for E-Graph extraction solutions.

Compute the total cost of the chosen nodes.
"""

from __future__ import annotations
import sys
import verifier  # local import – both files live side-by-side

# -----------------------------------------------------------------------------#
# Cost helper                                                                   #
# -----------------------------------------------------------------------------#


def evaluate(input_file: str, output_file: str):
    """Sum the 'cost' field of every selected node that actually exists."""
    nodes, roots = verifier.load_graph(input_file)
    selected = verifier.load_solution(output_file)
    return sum(nodes[n]["cost"] for n in selected if n in nodes)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    cost = evaluate(input_file, output_file)
    print(f'Cost on "{input_file}" with solution "{output_file}": {cost}')

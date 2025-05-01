#!/usr/bin/env python3
import os
import sys

# solver needs to be implemented by LLM
from solver import solve


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file>", file=sys.stderr)
        return 1

    input_file = sys.argv[1]
    # Extract dataset name from input file path
    dataset = os.path.basename(input_file).split(".")[0]
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{dataset}.output"

    # Generate the solution
    solve(input_file, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

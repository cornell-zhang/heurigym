#!/usr/bin/env python3
import sys

# solver needs to be implemented by LLM
from solver import solve


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>", file=sys.stderr)
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # Generate the solution
    solve(input_file, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

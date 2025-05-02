#!/usr/bin/env python3
import sys

# solver needs to be implemented by LLM
from solver import solve


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <input_file1> [input_file2 ...] <output_file>",
            file=sys.stderr,
        )
        return 1

    # Get all input files and output file
    input_files = sys.argv[1:-1]  # All arguments except the last one are input files
    output_file = sys.argv[-1]  # Last argument is the output file

    # Generate the solution
    solve(*input_files, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

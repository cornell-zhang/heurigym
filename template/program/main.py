#!/usr/bin/env python3
import os
import sys
from solver import solve


def get_filename(path: str) -> str:
    """Extract filename from path."""
    return os.path.basename(path).split(".")[0]


def main():
    # TODO: Free to modify
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset>", file=sys.stderr)
        return 1

    input_file = sys.argv[1]
    dataset = get_filename(input_file)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{dataset}.output"

    # TODO: Implement the parser yourself
    formatted_input = parser(input_file)

    # Generate the output
    output = solve(*formatted_input)

    # Write the output to the output file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as out_file:
            raise NotImplementedError(
                "You need to implement the output writer yourself."
            )
    except IOError:
        print(f"Error opening output file: {output_file}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

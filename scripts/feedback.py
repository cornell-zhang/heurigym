#!/usr/bin/env python3
import sys
import json
import os
import io
from contextlib import redirect_stderr

# evaluator and verifier need to be provided by the user
from evaluator import evaluate
from verifier import verify


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>", file=sys.stderr)
        return 1

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Capture stderr output during verification
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        valid = verify(input_file, output_file)

    # Calculate the cost
    if valid:
        cost = evaluate(input_file, output_file)
    else:
        cost = float("inf")

    # Prepare the output data
    output_data = {
        "validity": valid,
        "cost": cost,
        "message": stderr_capture.getvalue().strip(),
    }

    # Write the output to a JSON file
    output_file = f"{os.path.splitext(output_file)[0]}.cost"
    try:
        with open(output_file, "w") as out_file:
            json.dump(output_data, out_file, indent=2)
    except IOError:
        print(f"Error opening output file: {output_file}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

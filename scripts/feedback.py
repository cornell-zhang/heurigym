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
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <input_file1> [input_file2 ...] <output_file>",
            file=sys.stderr,
        )
        return 1

    # Get all input files and output file
    input_files = sys.argv[1:-1]  # All arguments except the last one are input files
    output_file = sys.argv[-1]  # Last argument is the output file

    # Capture stderr output during verification
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        # Pass all input files to verify function
        valid, error_message = verify(*input_files, output_file)
    # Check if there were any stderr messages and append them to error_message
    stderr_output = stderr_capture.getvalue()
    if stderr_output:
        error_message = (error_message + "\n" + stderr_output).strip()

    # Calculate the cost
    if valid:
        # Pass all input files to evaluate function
        cost = evaluate(*input_files, output_file)
    else:
        cost = float("inf")

    # Prepare the output data
    output_data = {
        "validity": valid,
        "cost": cost,
        "message": "Verification failed: " + error_message,
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

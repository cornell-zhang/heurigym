import sys
import io
import os
import json
from contextlib import redirect_stderr
from utils import read_instance, verify_solution, evaluate_solution

def main():
    
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <solution_file>", file=sys.stderr)
        return 1
    
    input_file = sys.argv[1]
    solution_file = sys.argv[2]
    
    # Read the instance from the input file
    instance = read_instance(input_file)

    # Verify the solution validity        
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        is_valid = verify_solution(instance, solution_file)

    # Calculate the cost
    if is_valid:
        cost = evaluate_solution(instance, solution_file)
    else:
        cost = float('inf')
    
    # Prepare the output data
    output_data = {
        "validity": is_valid,
        "cost": cost,
        "message": stderr_capture.getvalue().strip()
    }
    
    # Write the output to a JSON file
    output_file = f"{os.path.splitext(solution_file)[0]}.cost"
    try:
        with open(output_file, 'w') as out_file:
            json.dump(output_data, out_file, indent=2)
    except IOError:
        print(f"Error opening output file: {output_file}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    main()
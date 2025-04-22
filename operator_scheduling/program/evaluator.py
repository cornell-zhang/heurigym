#!/usr/bin/env python3
import sys
import json
import os
import io
from contextlib import redirect_stderr
from utils import parse_json, verify, calculate_cost, parse_schedule

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset> <schedule_file>", file=sys.stderr)
        return 1
    
    json_file = sys.argv[1]
    schedule_file = sys.argv[2]
    
    # Parse the input graph and the JSON configuration
    nodes, delay, resource_constraints = parse_json(json_file)
    
    # Parse the schedule
    schedule = parse_schedule(schedule_file)
    
    # Capture stderr output during verification
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        valid = verify(nodes, schedule, delay, resource_constraints)
    
    # Calculate the cost
    if valid:
        cost = calculate_cost(nodes, schedule, delay)
    else:
        cost = float('inf')
    
    # Prepare the output data
    output_data = {
        "validity": valid,
        "cost": cost,
        "message": stderr_capture.getvalue().strip()
    }
    
    # Write the output to a JSON file
    output_file = f"{os.path.splitext(schedule_file)[0]}.cost"
    try:
        with open(output_file, 'w') as out_file:
            json.dump(output_data, out_file, indent=2)
    except IOError:
        print(f"Error opening output file: {output_file}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
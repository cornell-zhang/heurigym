#!/usr/bin/env python3
import os
import sys
from solver import solve
from utils import get_filename, parse_json

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset>", file=sys.stderr)
        return 1
    
    json_file = sys.argv[1]
    dataset = get_filename(json_file)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{dataset}.output"
    
    # Parse the input graph and the JSON configuration
    nodes, delay, resource_constraints = parse_json(json_file)
    
    # Generate the schedule
    schedule = solve(nodes, delay, resource_constraints)
    
    # Write the schedule to the output file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as out_file:
            for node_id, start_time in schedule.items():
                out_file.write(f"{node_id}:{start_time}\n")
    except IOError:
        print(f"Error opening output file: {output_file}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
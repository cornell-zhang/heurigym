import sys
import os
from solver import solve
from utils import read_instance

def main():
    
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_file>")
        sys.exit(1)

    # Define input and output file paths
    input_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{base_name}.output"
    
    # Read the instance from the input file
    instance = read_instance(input_file)

    # Solve the instance using the solver
    cost, routes = solve(instance)
    
    # Write the solution to the output file
    with open(output_file, 'w') as output_file:
        output_file.write(f"{instance.name}, cost: {cost}\n")
        for route in routes:
            output_file.write(" ".join(map(str, route)) + "\n")

if __name__ == "__main__":
    main()

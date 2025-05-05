import os

def get_objective_value(file_path: str) -> float:
    """Extract the objective value from the solution file."""
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('# Objective value ='):
                return float(line.split('=')[1].strip())
    return None

def main():
    # Get all .sol files in the results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    sol_files = [f for f in os.listdir(results_dir) if f.endswith('.sol')]
    
    print("=" * 50)
    print(f"{'Dataset':<30} | {'Cost':<10}")
    print("-" * 50)
    
    for sol_file in sorted(sol_files):
        file_path = os.path.join(results_dir, sol_file)
        objective_value = int(get_objective_value(file_path))
        # Remove .sol extension and format the name
        dataset_name = sol_file.replace('.sol', '')
        print(f"{dataset_name:<30} | {objective_value:<10}")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 
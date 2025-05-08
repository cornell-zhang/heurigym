import os
import sys
import json
import subprocess
from collections import defaultdict

def find_run_files(base_dir):
    """Find all run.py files in iteration directories."""
    run_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        # Look for iteration directories
        if "iteration" in root and "run.py" in files:
            run_files.append(os.path.join(root, "run.py"))
    return run_files

def run_optimization(run_file, dataset_path, timeout=10):
    """Run the optimization script and return the results."""
    try:
        # Get the directory containing run.py
        run_dir = os.path.dirname(run_file)
        
        # Run the script with dataset path and timeout
        result = subprocess.run(
            ["python3", "run.py", dataset_path, "--timeout", str(timeout)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Look for results.json in the output directory
        results_file = os.path.join(run_dir, "results.json")
        
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error running {run_file}: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 collect_best_results.py <llm_solutions_dir> <dataset_path> [--timeout TIMEOUT]")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    dataset_path = os.path.abspath(sys.argv[2])
    
    # Parse timeout argument if provided
    timeout = 10  # default timeout
    if len(sys.argv) > 3:
        for i in range(3, len(sys.argv)):
            if sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
                try:
                    timeout = int(sys.argv[i + 1])
                except ValueError:
                    print("Error: Timeout must be an integer")
                    sys.exit(1)
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        sys.exit(1)
        
    print(f"Using dataset path: {dataset_path}")
    print(f"Using timeout: {timeout} seconds")
        
    best_results = defaultdict(lambda: {"cost": float("inf"), "source": None})
    
    # Find all run.py files
    run_files = find_run_files(base_dir)
    print(f"Found {len(run_files)} run.py files to process")
    
    # Process each run.py file
    for run_file in run_files:
        print(f"Processing {run_file}...")
        results = run_optimization(run_file, dataset_path, timeout)
        
        if results:
            # Update best results for each dataset
            for dataset, data in results.items():
                cost = int(data) if data != "X" else float("inf")
                if cost < best_results[dataset]["cost"]:
                    best_results[dataset] = {
                        "cost": cost,
                        "source": run_file
                    }
    
    # Print summary of best results
    print("\nBest Results Summary:")
    print("=" * 80)
    print(f"{'Dataset':<40} | {'Best Cost':<10} | {'Source':<30}")
    print("-" * 80)
    
    for dataset, result in sorted(best_results.items()):
        if result["source"]:
            source = os.path.relpath(result["source"], base_dir)
            print(f"{dataset:<40} | {result['cost']:<10} | {source:<30}")
        else:
            print(f"{dataset:<40} | {result['cost']:<10} | \"None\"")
    
    print("=" * 80)
    
    # Save best results to file
    output_file = os.path.join(base_dir, "best_results.json")
    with open(output_file, "w") as f:
        json.dump({
            dataset: {
                "cost": result["cost"],
                "source": os.path.relpath(result["source"], base_dir) if result["source"] else None
            }
            for dataset, result in best_results.items()
        }, f, indent=2)
    
    print(f"\nBest results saved to {output_file}")

if __name__ == "__main__":
    main() 
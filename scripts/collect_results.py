import os
import sys
import json
import subprocess
import re
from collections import defaultdict

def find_iteration_dirs(base_dir):
    """Find all iteration directories."""
    iteration_dirs = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        # Look for iteration directories
        if "iteration" in root:
            iteration_dirs.append(root)
    return iteration_dirs

def classify_error(message):
    """Classify error message into operator scheduling specific categories."""
    message = message.lower()
    
    # Define error categories specific to operator scheduling
    error_patterns = {
        "Stage I: Execution Error": r"execution error|runtime error",
        "Stage II: Output Error": r"evaluator error|timed out",
        "Stage III: Verification Error": r"verification failed|dependency constraint violated",
        "Other Error": r".*"  # Catch-all for other errors
    }
    
    for category, pattern in error_patterns.items():
        if re.search(pattern, message):
            return category
    
    return "Unknown Error"

def extract_errors(iteration_dir):
    """Extract error information from .cost files in the iteration directory."""
    errors = {}
    
    # Look for .cost files in the output directory
    output_dir = os.path.join(iteration_dir, "output")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(".cost"):
                test_case = file[:-5]  # Remove .cost extension
                cost_file = os.path.join(output_dir, file)
                try:
                    with open(cost_file, "r") as f:
                        cost_data = json.load(f)
                        errors[test_case] = {
                            "validity": cost_data.get("validity", False),
                            "cost": cost_data.get("cost", None),
                            "message": cost_data.get("message", ""),
                            "error_type": classify_error(cost_data.get("message", "")) if not cost_data.get("validity", False) else "Stage IV: No Error!!"
                        }
                except Exception as e:
                    errors[test_case] = {
                        "error": f"Error reading cost file: {str(e)}",
                        "error_type": "File Reading Error"
                    }
    
    return errors

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
        print("Usage: python3 collect_results.py <llm_solutions_dir> <dataset_path> [--timeout TIMEOUT]")
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
    
    # Initialize data structures for both results and errors
    best_results = defaultdict(lambda: {"cost": float("inf"), "source": None})
    all_errors = defaultdict(dict)
    error_stats = defaultdict(lambda: defaultdict(int))
    test_case_stats = defaultdict(lambda: defaultdict(int))
    
    # Find all iteration directories and run files
    iteration_dirs = find_iteration_dirs(base_dir)
    run_files = find_run_files(base_dir)
    print(f"Found {len(iteration_dirs)} iteration directories and {len(run_files)} run.py files to process")
    
    # First, process each run.py file for optimization results
    print("\nRunning optimizations...")
    for run_file in run_files:
        print(f"Processing optimization in {run_file}...")
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
    
    for dataset, result in best_results.items():
        if result["source"]:
            source = os.path.relpath(result["source"], base_dir)
            print(f"{dataset:<40} | {result['cost']:<10} | {source:<30}")
        else:
            print(f"{dataset:<40} | {result['cost']:<10} | \"None\"")
    
    print("=" * 80)
    
    # Then, process each iteration directory for errors
    print("\nCollecting error information...")
    for iteration_dir in iteration_dirs:
        iteration_name = os.path.basename(iteration_dir)
        errors = extract_errors(iteration_dir)
        
        if errors:
            all_errors[iteration_name] = errors
            
            # Update error statistics
            for test_case, error_info in errors.items():
                error_type = error_info.get("error_type", "Unknown Error")
                error_stats[iteration_name][error_type] += 1
                test_case_stats[test_case][error_type] += 1
    
    # Print error statistics
    print("\nError Statistics by Iteration:")
    print("=" * 100)
    
    # Get all unique error types across all iterations
    all_error_types = set()
    for stats in error_stats.values():
        all_error_types.update(stats.keys())
    
    # Print header
    header = "Iteration".ljust(15)
    for error_type in sorted(all_error_types):
        header += f" | {error_type.ljust(20)}"
    print(header)
    print("-" * len(header))
    
    # Print statistics for each iteration
    for iteration in sorted(error_stats.keys()):
        line = iteration.ljust(15)
        for error_type in sorted(all_error_types):
            count = error_stats[iteration].get(error_type, 0)
            line += f" | {str(count).ljust(20)}"
        print(line)
    
    # Print overall error statistics
    print("\nOverall Error Statistics:")
    print("=" * 80)
    total_stats = defaultdict(int)
    for stats in error_stats.values():
        for error_type, count in stats.items():
            total_stats[error_type] += count
    
    for error_type, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{error_type}: {count}")
    
    # Save all results to files
    results_output = os.path.join(base_dir, "best_results.json")
    with open(results_output, "w") as f:
        json.dump({
            dataset: {
                "cost": result["cost"],
                "source": os.path.relpath(result["source"], base_dir) if result["source"] else None
            }
            for dataset, result in best_results.items()
        }, f, indent=2)
    
    errors_output = os.path.join(base_dir, "error_summary.json")
    with open(errors_output, "w") as f:
        json.dump({
            "detailed_errors": all_errors,
            "iteration_statistics": error_stats,
            "test_case_statistics": test_case_stats,
            "total_statistics": total_stats
        }, f, indent=2)
    
    print(f"\nBest results saved to {results_output}")
    print(f"Error information saved to {errors_output}")

if __name__ == "__main__":
    main() 
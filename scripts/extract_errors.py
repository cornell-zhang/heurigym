import os
import sys
import json
from collections import defaultdict
import re

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
    if not message:
        return "No Error"
    
    message = message.lower()
    
    # Define error categories specific to operator scheduling
    error_patterns = {
        "Stage I: Execution Error": r"execution error|runtime error",
        "Stage II: Evaluation Error": r"evaluator error",
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
                            "error_type": classify_error(cost_data.get("message", ""))
                        }
                except Exception as e:
                    errors[test_case] = {
                        "error": f"Error reading cost file: {str(e)}",
                        "error_type": "File Reading Error"
                    }
    
    return errors

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract_errors.py <llm_solutions_dir>")
        sys.exit(1)
        
    base_dir = sys.argv[1]
    
    # Find all iteration directories
    iteration_dirs = find_iteration_dirs(base_dir)
    print(f"Found {len(iteration_dirs)} iteration directories to process")
    
    # Collect errors from each iteration
    all_errors = defaultdict(dict)
    error_stats = defaultdict(lambda: defaultdict(int))
    test_case_stats = defaultdict(lambda: defaultdict(int))
    
    for iteration_dir in iteration_dirs:
        print(f"Processing {iteration_dir}...")
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
    print("=" * 80)
    
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
    
    # Print test case statistics
    print("\nError Statistics by Test Case:")
    print("=" * 80)
    header = "Test Case".ljust(15)
    for error_type in sorted(all_error_types):
        header += f" | {error_type.ljust(20)}"
    print(header)
    print("-" * len(header))
    
    for test_case in sorted(test_case_stats.keys()):
        line = test_case.ljust(15)
        for error_type in sorted(all_error_types):
            count = test_case_stats[test_case].get(error_type, 0)
            line += f" | {str(count).ljust(20)}"
        print(line)
    
    # Print overall statistics
    print("\nOverall Error Statistics:")
    print("=" * 80)
    total_stats = defaultdict(int)
    for stats in error_stats.values():
        for error_type, count in stats.items():
            total_stats[error_type] += count
    
    for error_type, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{error_type}: {count}")
    
    # Save detailed errors to file
    output_file = os.path.join(base_dir, "error_summary.json")
    with open(output_file, "w") as f:
        json.dump({
            "detailed_errors": all_errors,
            "iteration_statistics": error_stats,
            "test_case_statistics": test_case_stats,
            "total_statistics": total_stats
        }, f, indent=2)
    
    print(f"\nDetailed error information saved to {output_file}")

if __name__ == "__main__":
    main() 
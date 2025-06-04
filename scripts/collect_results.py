import os
import sys
import json
import subprocess
import re
import math
import shutil
import argparse
import importlib.util
from collections import defaultdict

def load_metric_module(metric_path):
    """Dynamically load metric module from the given path or fall back to scripts directory."""
    if os.path.exists(metric_path):
        # Load from problem-specific path
        spec = importlib.util.spec_from_file_location("metric", metric_path)
        metric_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric_module)
        return metric_module
    else:
        # Fall back to scripts directory
        scripts_metric_path = os.path.join(os.path.dirname(__file__), "metric.py")
        spec = importlib.util.spec_from_file_location("metric", scripts_metric_path)
        metric_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric_module)
        return metric_module

def find_iteration_dirs(base_dir):
    """Find all iteration directories and their sample subdirectories."""
    iteration_dirs = set()  # Use set to prevent duplications
    
    # First, find all iteration directories
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and unnecessary subdirectories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "output"]]
        
        # Look for iteration directories
        if "iteration" in root:
            # If this is a main iteration directory, check for sample subdirectories
            if not any("sample" in d for d in dirs):
                iteration_dirs.add(root)
            # If it has sample subdirectories, add those instead
            else:
                for dir_name in dirs:
                    if dir_name.startswith("sample"):
                        sample_dir = os.path.join(root, dir_name)
                        iteration_dirs.add(sample_dir)

    return sorted(list(iteration_dirs))  # Convert set to sorted list for consistent order

def get_stage_number(error_type):
    """Get the stage number from error type."""
    # order is important
    if "Stage IV" in error_type:
        return 4
    if "Stage III" in error_type:
        return 3
    if "Stage II" in error_type:
        return 2
    if "Stage I" in error_type:
        return 1
    return 0


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
                        # Limit message to 10000 characters and add truncation indicator
                        message = cost_data.get("message", "")
                        if len(message) > 10000:
                            message = message[:10000] + "(truncated)"
                        errors[test_case] = {
                            "validity": cost_data.get("validity", False),
                            "cost": cost_data.get("cost", None),
                            "message": message,
                            "error_type": classify_error(message) if not cost_data.get("validity", False) else "Stage IV: No Error!!"
                        }
                except Exception as e:
                    errors[test_case] = {
                        "error": f"Error reading cost file: {str(e)}",
                        "error_type": "File Reading Error"
                    }

    return errors

def find_run_files(base_dir):
    """Find all run.py files in iteration directories and their sample subdirectories."""
    run_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Look for iteration directories
        if "iteration" in root and "run.py" in files:
            # Check if this is a sample subdirectory
            if "sample" in root:
                run_files.append(os.path.join(root, "run.py"))
            # If this is a main iteration directory without samples, add it
            elif not any("sample" in d for d in dirs):
                run_files.append(os.path.join(root, "run.py"))
    return run_files

def run_optimization(run_file, dataset_path, timeout=10, num_cores=8):
    """Run the optimization script and return the results."""
    try:
        # Get the directory containing run.py
        run_dir = os.path.dirname(run_file)

        # Run the script with dataset path and timeout, streaming output to terminal
        result = subprocess.run(
            ["python3", "run.py", dataset_path, "--timeout", str(timeout), "--num_cores", str(num_cores)],
            cwd=run_dir,
            text=True,
            check=True,
            stdout=subprocess.PIPE,  # Stream stdout to terminal
            stderr=subprocess.PIPE   # Stream stderr to terminal
        )
        print(result.stdout)
        print(result.stderr)
        # Look for results.json in the output directory
        results_file = os.path.join(run_dir, "results.json")

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                return json.load(f)
        else:
            raise Exception(f"Results file not found at {results_file}")
    except Exception as e:
        print(f"Error running {run_file}: {e}")
        return None

def read_baseline_values(baseline_path):
    """Read baseline values from baseline.json in the problem folder."""
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline file not found at {baseline_path}")
        return {}
    
    try:
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
            return baseline_data
    except Exception as e:
        print(f"Error reading baseline file: {e}")
        return {}

def calculate_geomean(results, baseline_values, normalize_score):
    """Calculate geometric mean, coverage, and QCI for an iteration's results using baseline values."""
    valid_values = []
    total_datasets = len(results)
    valid_datasets = 0

    for dataset, value in results.items():
        # Skip if no baseline value available
        if dataset not in baseline_values:
            raise ValueError(f"Baseline value not found for dataset: {dataset}")
        # Skip "X" cases
        if value == "X":
            continue
        # Normalize the score using the baseline value
        normalized_score = normalize_score(float(value), baseline_values[dataset])
        valid_values.append(normalized_score)
        valid_datasets += 1

    if not valid_values:
        return float('inf'), 0.0, 0.0

    # Calculate geometric mean of normalized scores (quality)
    quality = math.exp(sum(math.log(x) for x in valid_values) / len(valid_values))
    
    # Calculate coverage (pass rate)
    coverage = valid_datasets / total_datasets
    
    # Calculate QCI (Quality-Coverage Index) using F1-like formula
    if quality + coverage == 0:
        qci = 0.0
    else:
        qci = 2 * quality * coverage / (quality + coverage)

    return quality, coverage, qci

def calculate_solve_at_i(all_errors, i):
    """Calculate solve@i metrics for the first i iterations."""
    # Get the first i iterations (grouped by iteration number, ignoring samples)
    iteration_groups = defaultdict(list)
    for key in sorted(all_errors.keys()):
        # Extract iteration number from key (e.g., "iteration0/sample0" -> "iteration0")
        iteration_num = key.split('/')[0]
        iteration_groups[iteration_num].append(key)
    
    # Get the first i iteration groups
    first_i_iterations = sorted(iteration_groups.keys())[:i]

    # Track the best stage each test case passes in the first i iterations
    test_case_best_stages = defaultdict(int)

    # For each iteration group, look at all its samples
    for iteration in first_i_iterations:
        # Get all samples for this iteration
        samples = iteration_groups[iteration]
        
        # For each test case, check if any sample passes each stage
        test_case_stages = defaultdict(int)
        
        # First, find the best stage achieved by any sample for each test case
        for sample in samples:
            iteration_errors = all_errors[sample]
            for test_case, error_info in iteration_errors.items():
                error_type = error_info.get("error_type", "Unknown Error")
                stage_num = get_stage_number(error_type)
                if stage_num > 0:  # If it's a valid stage
                    test_case_stages[test_case] = max(test_case_stages[test_case], stage_num - 1)
        
        # Update the overall best stages with this iteration's results
        for test_case, stage in test_case_stages.items():
            test_case_best_stages[test_case] = max(test_case_best_stages[test_case], stage)

    # Calculate stage pass statistics
    stage_pass_stats = defaultdict(int)
    for stage in test_case_best_stages.values():
        # If a test case passes stage N, it also passes all stages 0 to N-1
        for s in range(stage + 1):
            stage_pass_stats[s] += 1

    return stage_pass_stats

def remove_output_folders(base_dir):
    """Remove all output folders under iteration directories."""
    iteration_dirs = find_iteration_dirs(base_dir)
    removed_count = 0

    for iteration_dir in iteration_dirs:
        output_dir = os.path.join(iteration_dir, "output")
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                removed_count += 1
                print(f"Removed output folder: {output_dir}")
            except Exception as e:
                print(f"Error removing {output_dir}: {e}")

    return removed_count

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect and analyze optimization results')
    parser.add_argument('llm_solutions_dir', help='Path to LLM solutions directory')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout in seconds for program execution (default: 10)')
    parser.add_argument('--num_cores', type=int, default=8,
                        help='Number of CPU cores to use for program execution (default: 8)')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output folders before processing')
    
    args = parser.parse_args()

    base_dir = args.llm_solutions_dir
    dataset_path = os.path.abspath(args.dataset_path)

    # Read baseline values
    problem_name = os.path.basename(dataset_path)
    metric_path = os.path.join(problem_name, "program", "metric.py")
    baseline_path = os.path.join(problem_name, "baseline", "baseline.json")
    
    # Load metric module
    metric_module = load_metric_module(metric_path)
    normalize_score = metric_module.normalize_score
    
    baseline_values = read_baseline_values(baseline_path)
    if not baseline_values:
        print("Error: No baseline values found. Exiting.")
        sys.exit(1)

    if args.clean:
        removed_count = remove_output_folders(base_dir)
        print(f"\nRemoved {removed_count} output folders")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        sys.exit(1)

    preprocess_path = os.path.join(dataset_path, "data_preprocess.py")
    if os.path.exists(preprocess_path):
        subprocess.run(["python3", preprocess_path], check=True)
        os.remove(preprocess_path)

    print(f"Using dataset path: {dataset_path}")
    print(f"Using timeout: {args.timeout} seconds")
    print(f"Using CPU cores: {args.num_cores}")

    # Initialize data structures
    iteration_results = {}  # Store results for each iteration and sample
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
        results = run_optimization(run_file, os.path.abspath(dataset_path), args.timeout, args.num_cores)
        
        # Get iteration name and sample number from path
        path_parts = run_file.split(os.sep)
        # Find the iteration directory index
        iteration_idx = next(i for i, p in enumerate(path_parts) if p.startswith("iteration"))
        iteration_name = path_parts[iteration_idx]
        
        # Check if this is a sample run by looking at the next directory
        if iteration_idx + 1 < len(path_parts) and path_parts[iteration_idx + 1].startswith("sample"):
            sample_name = path_parts[iteration_idx + 1]
            iteration_key = f"{iteration_name}/{sample_name}"
        else:
            iteration_key = iteration_name

        if results:
            # Store results for this iteration
            iteration_results[iteration_key] = results
        else:
            iteration_results[iteration_key] = {}

    # Calculate geomean for each iteration and find the best one
    best_qci = 0.0
    best_iteration = None
    iteration_metrics = {}  # Changed from iteration_geomeans to iteration_metrics

    print("\nResults Summary:")
    print("=" * 100)
    print(f"{'Iteration':<30} | {'Quality':<10} | {'Coverage':<10} | {'QCI':<10}")
    print("-" * 100)

    for iteration, results in iteration_results.items():
        quality, coverage, qci = calculate_geomean(results, baseline_values, normalize_score)
        iteration_metrics[iteration] = {
            'quality': quality,
            'coverage': coverage,
            'qci': qci
        }
        print(f"{iteration:<30} | {quality:<10.4f} | {coverage:<10.4f} | {qci:<10.4f}")

        if qci >= best_qci: # take the last best iteration
            best_qci = qci
            best_iteration = iteration

    print("=" * 100)
    
    # If all iterations have zero QCI, use the last iteration
    if best_qci == 0.0:  # Changed condition to check for zero QCI
        best_iteration = list(iteration_results.keys())[-1]
        best_qci = iteration_metrics[best_iteration]['qci']
        print(f"All iterations have zero QCI. Using last iteration: {best_iteration}")
    
    print(f"Best iteration: {best_iteration} with QCI: {best_qci:.4f}")

    # Use results from the best iteration
    best_results = defaultdict(lambda: {"cost": float("inf"), "source": None})
    if best_iteration:
        for dataset, value in iteration_results[best_iteration].items():
            cost = float(value) if value != "X" else float("inf")
            # Get the full path to the run.py file
            if "/" in best_iteration:  # This is a sample directory
                iteration_name, sample_name = best_iteration.split("/")
                source_path = os.path.join(iteration_name, sample_name, "run.py")
            else:
                source_path = os.path.join(best_iteration, "run.py")
            best_results[dataset] = {
                "cost": cost,
                "source": source_path
            }

    # Print summary of best results
    print("\nBest Results Summary (from best iteration):")
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

    # Then, process each iteration directory for errors
    print("\nCollecting error information...")
    for iteration_dir in iteration_dirs:
        # Get iteration name and sample number from path
        path_parts = iteration_dir.split(os.sep)
        # Find the iteration directory index
        iteration_idx = next(i for i, p in enumerate(path_parts) if p.startswith("iteration"))
        iteration_name = path_parts[iteration_idx]
        
        # Check if this is a sample directory by looking at the next directory
        if iteration_idx + 1 < len(path_parts) and path_parts[iteration_idx + 1].startswith("sample"):
            sample_name = path_parts[iteration_idx + 1]
            iteration_key = f"{iteration_name}/{sample_name}"
        else:
            iteration_key = iteration_name

        errors = extract_errors(iteration_dir)

        if errors:
            all_errors[iteration_key] = errors

            # Update error statistics and track best passed stages
            for test_case, error_info in errors.items():
                error_type = error_info.get("error_type", "Unknown Error")
                error_stats[iteration_key][error_type] += 1
                test_case_stats[test_case][error_type] += 1

    # Print error statistics
    print("\nError Statistics by Iteration:")
    print("=" * 100)

    # Get all unique error types across all iterations
    all_error_types = set()
    for stats in error_stats.values():
        all_error_types.update(stats.keys())

    # Print header
    header = "Iteration".ljust(30)
    for error_type in sorted(all_error_types):
        header += f" | {error_type.ljust(20)}"
    print(header)
    print("-" * len(header))

    # Print statistics for each iteration
    for iteration in sorted(error_stats.keys()):
        line = iteration.ljust(30)
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

    # Calculate stage pass statistics
    total_cases = len(test_case_stats)  # Use test_case_stats to get total number of test cases

    # Calculate and print solve@i statistics for different values of i
    print("\nsolve@i Statistics:")
    print("=" * 80)
    print(f"Total test cases: {total_cases}")
    print("-" * 80)

    # Store solve@i metrics
    solve_at_i_metrics = {}
    for i in [10, 5, 3, 1]:
        if i <= len(all_errors):
            stage_pass_stats_i = calculate_solve_at_i(all_errors, i)
            print(f"\nsolve@{i} Statistics:")
            solve_at_i_metrics[i] = {}
            for stage in range(1, 4):
                passed = stage_pass_stats_i[stage]
                print(f"solve_s{stage}@{i}: {passed}/{total_cases} passed ({passed/total_cases*100:.1f}%)")
                solve_at_i_metrics[i][f"stage_{stage}"] = {
                    "passed": passed,
                    "total": total_cases,
                    "percentage": passed/total_cases*100
                }

    # Save all results to files
    results_output = os.path.join(base_dir, "best_results.json")
    with open(results_output, "w") as f:
        json.dump({
            dataset: {
                "cost": result["cost"],
                "source": os.path.relpath(result["source"], base_dir) if result["source"] else None,
                "iteration": best_iteration,
                "metrics": iteration_metrics[best_iteration]  # Add metrics to the output
            }
            for dataset, result in sorted(best_results.items())
        }, f, indent=2)

    # Dump metrics to log file
    log_output = os.path.join(base_dir, "metrics.log")
    with open(log_output, "w") as f:
        # Write costs for each dataset
        for dataset, result in sorted(best_results.items()):
            f.write(f"{result['cost']:.4f}\n")

        # Write best QCI
        f.write(f"{best_qci:.4f}\n")

        # Write current solution directory (timestamp)
        f.write(f"{sys.argv[1].split('/')[1]}\n")

        # Write solve@i metrics
        for i in [10, 5, 3, 1]:
            if i <= len(all_errors):
                stage_pass_stats_i = calculate_solve_at_i(all_errors, i)
                for stage in range(1, 4):
                    passed = stage_pass_stats_i[stage]
                    f.write(f"{passed}\n")

    errors_output = os.path.join(base_dir, "error_summary.json")
    with open(errors_output, "w") as f:
        json.dump({
            "detailed_errors": all_errors,
            "iteration_statistics": error_stats,
            "test_case_statistics": test_case_stats,
            "total_statistics": total_stats,
            "iteration_metrics": iteration_metrics,  # Changed from iteration_geomeans
            "best_iteration": best_iteration,
            "solve_at_i_metrics": solve_at_i_metrics,
            "best_results": {
                dataset: {
                    "cost": result["cost"],
                    "source": os.path.relpath(result["source"], base_dir) if result["source"] else None,
                    "iteration": best_iteration,
                    "metrics": iteration_metrics[best_iteration]  # Add metrics to the output
                }
                for dataset, result in sorted(best_results.items())
            }
        }, f, indent=2)

    print(f"\nBest results saved to {results_output}")
    print(f"Error information saved to {errors_output}")
    print(f"Metrics saved to {log_output}")

if __name__ == "__main__":
    main()

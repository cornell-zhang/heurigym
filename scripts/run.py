import os
import subprocess
import sys
import json


def find_all_datasets(base_dir):
    """
    Find all potential datasets under the given base directory and group them by base name.

    Args:
        base_dir (str): Base directory to search

    Returns:
        dict: Dictionary mapping base names to lists of file paths
    """
    file_groups = {}

    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Group files by base name
        for file in files:
            if not file.startswith(".") and not file.endswith(".py"):  # Skip hidden files and .py files
                base_name = os.path.splitext(file)[0]  # Get base name without extension
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(os.path.join(root, file))

    return file_groups


def run_optimization(input_files, output_dir="output", timeout=10):
    """
    Run the optimization program on the given dataset and return the result.

    Args:
        input_files (list): List of paths to input files
        output_dir (str): Directory to store output files
        timeout (int): Timeout in seconds for program execution (default: 10)

    Returns:
        tuple: (success, cost) where success is a boolean and cost is an integer or None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use the base name of the first input file for output naming
        base_name = os.path.splitext(os.path.basename(input_files[0]))[0]

        # Define output file paths
        output_file = os.path.join(output_dir, f"{base_name}.output")
        cost_file = os.path.join(output_dir, f"{base_name}.cost")

        # Check if cost files exist
        if os.path.exists(cost_file):
            # Read the cost file to get the result
            with open(cost_file, "r") as f:
                cost_data = json.load(f)
                if cost_data.get("validity", False):
                    print(f"Skipping {base_name}: Using existing valid result with cost {cost_data.get('cost')}")
                    return True, cost_data.get("cost")
                else:
                    print(f"Skipping {base_name}: Using existing invalid result with error: {cost_data.get('message', 'Unknown error')}")
                    return False, None

        # Run the main program to generate output
        try:
            cmd = ["python3", "main.py"]
            cmd.extend(sorted(input_files))  # Add all input files
            cmd.append(output_file)  # Add output file
            
            main_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            error_data = {
                "message": f"Program execution timed out after {timeout} seconds",
                "validity": False,
                "cost": None
            }
            with open(cost_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Error: Program execution timed out after {timeout} seconds")
            return False, None

        # Check if main.py executed successfully
        if main_result.returncode != 0:
            error_data = {
                "message": f"Python execution error: {main_result.stderr}",
                "validity": False,
                "cost": None
            }
            with open(cost_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Error running main.py: {main_result.stderr}")
            return False, None

        # Check if output file is empty
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            error_data = {
                "message": "Evaluator error: No output file generated or output file is empty",
                "validity": False,
                "cost": None
            }
            with open(cost_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Evaluator error: No output file generated or output file is empty")
            return False, None

        # Run the evaluator to evaluate the output
        eval_cmd = ["python3", "feedback.py"]
        eval_cmd.extend(sorted(input_files))  # Add all input files
        eval_cmd.append(output_file)  # Add output file
        
        eval_result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        # Check if feedback.py executed successfully
        if eval_result.returncode != 0:
            error_data = {
                "message": f"Evaluator error: {eval_result.stderr}",
                "validity": False,
                "cost": None
            }
            with open(cost_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Error running feedback.py: {eval_result.stderr}")
            return False, None

        # Read the cost from the cost file
        if os.path.exists(cost_file):
            with open(cost_file, "r") as f:
                cost_data = json.load(f)
                if cost_data.get("validity", False):
                    return True, cost_data.get("cost")
                else:
                    # If solution is invalid, return False with the error message
                    print(f"Invalid solution: {cost_data.get('message', 'Unknown error')}")
                    return False, None

        # If we get here, something went wrong
        error_data = {
            "message": "No cost file generated",
            "validity": False,
            "cost": None
        }
        with open(cost_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        print(f"Warning: Could not extract cost")
        return False, None

    except Exception as e:
        # Create error cost file for any unexpected exceptions
        error_data = {
            "message": f"Unexpected error: {str(e)}",
            "validity": False,
            "cost": None
        }
        with open(cost_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        print(f"Error processing files: {e}")
        return False, None


def main():
    # Path to dataset directory
    dataset_dir = "../dataset/full"
    timeout = 10  # Default timeout in seconds

    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
            try:
                timeout = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print("Error: Timeout must be an integer")
                sys.exit(1)
        else:
            dataset_dir = sys.argv[i]
            i += 1

    # Make sure datasets directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Get all potential datasets grouped by base name
    file_groups = find_all_datasets(dataset_dir)

    # If no datasets found
    if not file_groups:
        print(f"Warning: No datasets found under '{dataset_dir}'.")
        return

    # Results will be stored as (dataset_name, cost) tuples
    results = []

    # Run optimization on each group of files
    for base_name, input_files in file_groups.items():
        success, cost = run_optimization(input_files, timeout=timeout)

        if success:
            results.append((base_name, str(cost)))
        else:
            results.append((base_name, "X"))

    # Print results as a table
    print("\nResults Summary:")
    print("=" * 50)
    print(f"{'Dataset':<30} | {'Cost':<10}")
    print("-" * 50)

    for dataset_name, cost in results:
        print(f"{dataset_name:<30} | {cost:<10}")

    print("=" * 50)

    # Save results to JSON file
    results_dict = {dataset: cost for dataset, cost in results}
    with open("results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("\nResults have been saved to results.json")


if __name__ == "__main__":
    main()

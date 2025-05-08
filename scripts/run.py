import os
import subprocess
import sys
import json


def run_optimization(input_path, output_dir="output", timeout=10):
    """
    Run the optimization program on the given dataset and return the result.

    Args:
        input_path (str): Path to the input file
        output_dir (str): Directory to store output files
        timeout (int): Timeout in seconds for program execution (default: 10)

    Returns:
        tuple: (success, cost) where success is a boolean and cost is an integer or None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract dataset name from path
        dataset_name = os.path.basename(input_path)
        if dataset_name.endswith(".json"):
            dataset_name = dataset_name[:-5]  # Remove .json extension

        # Define output file paths
        output_file = os.path.join(output_dir, f"{dataset_name}.output")
        cost_file = os.path.join(output_dir, f"{dataset_name}.cost")

        # Run the main program to generate output
        try:
            main_result = subprocess.run(
                ["python3", "main.py", input_path, output_file],
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
            print(f"Error running main.py on {input_path}: {main_result.stderr}")
            return False, None

        # Check if output file is empty
        if os.path.getsize(output_file) == 0:
            error_data = {
                "message": "Evaluator error: Output file is empty",
                "validity": False,
                "cost": None
            }
            with open(cost_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Error: Output file is empty for {input_path}")
            return False, None

        # Run the evaluator to evaluate the output
        eval_result = subprocess.run(
            ["python3", "feedback.py", input_path, output_file],
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
            print(f"Error running feedback.py on {input_path}: {eval_result.stderr}")
            return False, None

        # Read the cost from the cost file
        if os.path.exists(cost_file):
            with open(cost_file, "r") as f:
                cost_data = json.load(f)
                if cost_data.get("validity", False):
                    return True, cost_data.get("cost")
                else:
                    # If solution is invalid, return False with the error message
                    print(f"Invalid solution for {input_path}: {cost_data.get('message', 'Unknown error')}")
                    return False, None

        # If we get here, something went wrong
        error_data = {
            "message": "No cost file generated",
            "validity": False,
            "cost": None
        }
        with open(cost_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        print(f"Warning: Could not extract cost for {input_path}")
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
        print(f"Error processing {input_path}: {e}")
        return False, None


def find_all_datasets(base_dir):
    """
    Find all potential datasets under the given base directory.

    Args:
        base_dir (str): Base directory to search

    Returns:
        list: List of paths to potential datasets
    """
    datasets = set()

    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Add all JSON files in this directory
        for file in files:
            if file.endswith(".json"):
                datasets.add(os.path.join(root, file))

    return list(datasets)


def main():
    # Path to dataset directory
    dataset_dir = "../dataset/full"

    # Allow dataset directory to be specified via command line
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]

    # Make sure datasets directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Get all potential datasets
    datasets = find_all_datasets(dataset_dir)

    # If no datasets found
    if not datasets:
        print(f"Warning: No datasets found under '{dataset_dir}'.")
        return

    # Results will be stored as (dataset_name, cost) tuples
    results = []

    # Run optimization on each dataset
    for dataset_path in sorted(datasets):
        dataset_name = os.path.relpath(dataset_path, start=dataset_dir)
        success, cost = run_optimization(dataset_path)

        if success:
            results.append((dataset_name, str(cost)))
        else:
            results.append((dataset_name, "X"))

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

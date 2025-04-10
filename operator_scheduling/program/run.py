import os
import subprocess
import re
import sys

def run_optimization(executable, dataset_path):
    """
    Run the optimization program on the given dataset and return the result.
    
    Args:
        executable (str): Path to the optimization program
        dataset_path (str): Path to the dataset
        
    Returns:
        tuple: (success, latency) where success is a boolean and latency is an integer or None
    """
    try:
        # Run the executable with the dataset path as argument
        result = subprocess.run([executable, dataset_path], 
                                capture_output=True, 
                                text=True, 
                                check=False)
        
        # Check if the output contains "Final latency"
        latency_match = re.search(r"Final latency: (\d+)", result.stdout)
        if latency_match:
            return True, int(latency_match.group(1))
        
        # If no latency found but contains "Failed verification", it's a failed verification
        if "Failed verification" in result.stdout:
            return False, None
        
        # If neither pattern is found, print a warning and assume failure
        print(f"Warning: Unexpected output for {dataset_path}:\n{result.stdout}")
        return False, None
    
    except Exception as e:
        print(f"Error running optimization on {dataset_path}: {e}")
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
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Add all files in this directory
        for file in files:
            datasets.add(os.path.join(root, file).rsplit('.', 1)[0])
        
        # Add directories that might be datasets themselves
        # for dir_name in dirs:
        #     dir_path = os.path.join(root, dir_name)
        #     datasets.append(dir_path)
    
    return list(datasets)

def main():
    # Path to executable and dataset directory
    executable = "./main.out"
    dataset_dir = "../dataset"
    
    # Allow dataset directory to be specified via command line
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    
    # Make sure datasets directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return
    
    # Make sure executable exists
    if not os.path.isfile(executable):
        print(f"Error: Executable '{executable}' not found.")
        return
    
    # Get all potential datasets
    datasets = find_all_datasets(dataset_dir)
    
    # If no datasets found
    if not datasets:
        print(f"Warning: No datasets found under '{dataset_dir}'.")
        return
    
    # Results will be stored as (dataset_name, latency) tuples
    results = []
    
    # Run optimization on each dataset
    for dataset_path in sorted(datasets):
        dataset_name = os.path.relpath(dataset_path, start=dataset_dir)
        success, latency = run_optimization(executable, dataset_path)
        
        if success:
            results.append((dataset_name, str(latency)))
        else:
            results.append((dataset_name, "X"))
    
    # Print results as a table
    print("\nResults Summary:")
    print("=" * 50)
    print(f"{'Dataset':<30} | {'Latency':<10}")
    print("-" * 50)
    
    for dataset_name, latency in results:
        print(f"{dataset_name:<30} | {latency:<10}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
import shutil
import argparse
from pathlib import Path

def copy_run_to_iterations(llm_solutions_dir):
    # Get the current directory
    current_dir = Path(__file__).resolve().parent.parent
    
    # Path to the run.py file
    run_file = current_dir / 'scripts' / 'run.py'
    
    # Convert llm_solutions_dir to Path object
    llm_solutions_dir = Path(llm_solutions_dir)
    
    # Check if run.py exists
    if not run_file.exists():
        print(f"Error: {run_file} not found!")
        return
    
    # Check if llm_solutions directory exists
    if not llm_solutions_dir.exists():
        print(f"Error: {llm_solutions_dir} not found!")
        return
    
    # Iterate through all iteration folders
    for iteration_dir in llm_solutions_dir.iterdir():
        if iteration_dir.is_dir() and iteration_dir.name.startswith('iteration'):
            shutil.copy2(run_file, iteration_dir)
            print(f"Copied run.py to {iteration_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy run.py to operator_scheduling directories')
    parser.add_argument('llm_solutions_dir', help='Path to the llm_solutions directory')
    args = parser.parse_args()
    
    copy_run_to_iterations(args.llm_solutions_dir) 
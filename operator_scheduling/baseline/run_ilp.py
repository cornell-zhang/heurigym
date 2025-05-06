import subprocess
from pathlib import Path
import time
import json

def run_gurobi_cl(ilp_file, output_file):
    """Run gurobi_cl on an ILP file and save results to output file"""
    try:
        start_time = time.time()
        result = subprocess.run(
            ['gurobi_cl', 'ResultFile=' + str(output_file), str(ilp_file)],
            capture_output=True,
            text=True
        )
        end_time = time.time()
        
        # Check if solution was found
        if "Optimal solution found" in result.stdout:
            status = "optimal"
        elif "Infeasible" in result.stdout:
            status = "infeasible"
        else:
            status = "unknown"
            
        return {
            "status": status,
            "runtime": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get all ILP files
    ilp_dir = Path("ilp")
    if not ilp_dir.exists():
        print("ILP directory not found!")
        return
        
    ilp_files = list(ilp_dir.glob("*.lp"))
    if not ilp_files:
        print("No ILP files found!")
        return
        
    print(f"Found {len(ilp_files)} ILP files to process")
    
    # Process each ILP file
    results = {}
    for ilp_file in ilp_files:
        print(f"\nProcessing {ilp_file.name}...")
        
        # Create output file path
        output_file = results_dir / f"{ilp_file.stem}.sol"
        
        # Run gurobi_cl
        result = run_gurobi_cl(ilp_file, output_file)
        results[ilp_file.name] = result
        
        # Print status
        print(f"Status: {result['status']}")
        if 'runtime' in result:
            print(f"Runtime: {result['runtime']:.2f} seconds")
        if result['status'] == 'error':
            print(f"Error: {result['error']}")
            
    # Save summary to JSON file
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

if __name__ == "__main__":
    main() 
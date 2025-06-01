'''
python benchmark_cpp.py --base-path <path_to_heurigen_workspace>
'''

import os
import subprocess
import glob
import json
import time
import argparse
from pathlib import Path
from ..program.verifier import verify
from ..program.evaluator import evaluate
import logging


class Result:
    def __init__(self, verify_result, cost, elapsed_time):
        self.verify_result = verify_result # True or False
        self.cost = cost # None if verify_result is False, int if verify_result is True
        self.elapsed_time = elapsed_time # float: seconds


def setup_logging(logs_dir):
    # Create logs directory
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure logging
    log_file = f"{logs_dir}/benchmark_cpp.log"
    
    # Remove existing log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark C++ technology mapping program')
    parser.add_argument('--base-path', default='.', 
                       help='Base path for the project (default: current directory)')
    args = parser.parse_args()
    
    # Set up paths based on the base path argument
    base_path = args.base_path
    datasets_dir = f"{base_path}/_datasets/technology_mapping"
    output_dir = f"{base_path}/llm_solutions/tech_mapping_cpp/gemini-2.5-pro-preview-05-06"
    logs_dir = f"{output_dir}/logs"
    main_executable = f"{base_path}/technology_mapping/cpp_program/main"
    verifier_script = f"{base_path}/technology_mapping/program/verifier.py"
    evaluator_script = f"{base_path}/technology_mapping/program/evaluator.py"

    # Setup logging
    logger = setup_logging(logs_dir)
    
    # result is a dictionary of benchmark_name -> Result
    results = {}


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .blif files in the datasets directory and its subdirectories
    blif_files = glob.glob(f"{datasets_dir}/**/*.blif", recursive=True)
    results = {}

    logger.info(f"Found {len(blif_files)} BLIF files to process")
    
    # For testing, uncomment to limit files processed
    blif_files = blif_files[:1]

    for input_file in blif_files:
        # Extract benchmark name from the file path
        benchmark_name = os.path.basename(input_file).replace(".blif", "")
        logger.info(f"===== Processing benchmark: {benchmark_name} =====")
        logger.info(f"Input file: {input_file}")
        
        # Define output file path
        output_file = f"{output_dir}/{benchmark_name}.blif"
        logger.info(f"Output file: {output_file}")
        
        # Run main executable
        try:
            logger.info("Running main executable...")
            
            # Record start time
            start_time = time.time()
            
            main_result = subprocess.run([main_executable, input_file, output_file], 
                        check=True, capture_output=True, text=True)
            
            # Record end time and calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"Tech Mapping C++ elapsed time: {elapsed_time:.2f} seconds")
                
            
            # Run verifier
            logger.info("Running verifier...")
            verifier_result = subprocess.run(
                ["python", verifier_script, input_file, output_file],
                check=True, capture_output=True, text=True
            )
            logger.info(f"Verifier stdout:\n{verifier_result.stdout}")
            if verifier_result.stderr:
                logger.info(f"Verifier stderr:\n{verifier_result.stderr}")
            
            # Run evaluator
            logger.info("Running evaluator...")
            evaluator_result = subprocess.run(
                ["python", evaluator_script, output_file],
                check=True, capture_output=True, text=True
            )
            logger.info(f"Evaluator stdout:\n{evaluator_result.stdout}")
            if evaluator_result.stderr:
                logger.info(f"Evaluator stderr:\n{evaluator_result.stderr}")
            
            # Store the raw output in the results
            try:
                cost = float(evaluator_result.stdout.strip())
                results[benchmark_name] = {"cost": cost, "elapsed_time": elapsed_time}
                logger.info(f"Parsed cost: {cost}")
            except ValueError:
                results[benchmark_name] = {"cost": float('inf'), "elapsed_time": elapsed_time}
                logger.warning(f"Failed to parse cost for {benchmark_name}, setting to infinity")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {benchmark_name}: {e}")
            logger.error(f"Command output (stdout):\n{e.stdout if hasattr(e, 'stdout') else 'N/A'}")
            logger.error(f"Command output (stderr):\n{e.stderr if hasattr(e, 'stderr') else 'N/A'}")
            results[benchmark_name] = {"cost": float('inf'), "elapsed_time": -1}
        
        logger.info(f"Processing of {benchmark_name} complete")
        logger.info("-" * 50)

    # Save results to JSON file
    with open(f"{output_dir}/best_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark complete. Results saved to {output_dir}/best_results.json")


if __name__ == "__main__":
    main()
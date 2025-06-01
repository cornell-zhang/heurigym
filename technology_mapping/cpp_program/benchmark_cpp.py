import os
import subprocess
import glob
import json
import time
from pathlib import Path
from ..program.verifier import verify
from ..program.evaluator import evaluate
import logging

# Paths
BASE_PATH = "/work/shared/users/phd/jl4257/Project/heurigen"
DATASETS_DIR = f"{BASE_PATH}/_datasets/technology_mapping"
OUTPUT_DIR = f"{BASE_PATH}/llm_solutions/tech_mapping_cpp/gemini-2.5-pro-preview-05-06"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
MAIN_EXECUTABLE = f"{BASE_PATH}/technology_mapping/cpp_program/main"
VERIFIER_SCRIPT = f"{BASE_PATH}/technology_mapping/program/verifier.py"
EVALUATOR_SCRIPT = f"{BASE_PATH}/technology_mapping/program/evaluator.py"


class Result:
    def __init__(self, verify_result, cost, elapsed_time):
        self.verify_result = verify_result # True or False
        self.cost = cost # None if verify_result is False, int if verify_result is True
        self.elapsed_time = elapsed_time # float: seconds


def setup_logging():
    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Configure logging
    log_file = f"{LOGS_DIR}/benchmark_cpp.log"
    
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

    # Setup logging
    logger = setup_logging()
    
    # result is a dictionary of benchmark_name -> Result
    results = {}


    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .blif files in the datasets directory and its subdirectories
    blif_files = glob.glob(f"{DATASETS_DIR}/**/*.blif", recursive=True)
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
        output_file = f"{OUTPUT_DIR}/{benchmark_name}.blif"
        logger.info(f"Output file: {output_file}")
        
        # Run main executable
        try:
            logger.info("Running main executable...")
            
            # Record start time
            start_time = time.time()
            
            main_result = subprocess.run([MAIN_EXECUTABLE, input_file, output_file], 
                        check=True, capture_output=True, text=True)
            
            # Record end time and calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"Tech Mapping C++ elapsed time: {elapsed_time:.2f} seconds")
                
            
            # Run verifier
            logger.info("Running verifier...")
            verifier_result = subprocess.run(
                ["python", VERIFIER_SCRIPT, input_file, output_file],
                check=True, capture_output=True, text=True
            )
            logger.info(f"Verifier stdout:\n{verifier_result.stdout}")
            if verifier_result.stderr:
                logger.info(f"Verifier stderr:\n{verifier_result.stderr}")
            
            # Run evaluator
            logger.info("Running evaluator...")
            evaluator_result = subprocess.run(
                ["python", EVALUATOR_SCRIPT, output_file],
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
    with open(f"{OUTPUT_DIR}/best_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark complete. Results saved to {OUTPUT_DIR}/best_results.json")


if __name__ == "__main__":
    main()
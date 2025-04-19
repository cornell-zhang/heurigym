import os
import re
import json
import logging
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProblemReader:
    """Reads and parses problem descriptions from README files."""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        
    def get_problem_folders(self) -> List[Path]:
        """Returns a list of problem folders in the workspace."""
        return [p for p in self.workspace_root.iterdir() 
                if p.is_dir() and not p.name.startswith('.')]
    
    def read_problem_description(self, problem_folder: Path) -> Dict[str, str]:
        """Reads and parses the README.md file in the given problem folder."""
        readme_path = problem_folder / "README.md"
        if not readme_path.exists():
            raise FileNotFoundError(f"No README.md found in {problem_folder}")
            
        with open(readme_path, 'r') as f:
            md_content = f.read()
            
        # Parse markdown content directly
        sections = {}
        current_section = "overview"
        current_content = []
        
        for line in md_content.split('\n'):
            # Check for headers (## or #)
            if line.startswith('##') or line.startswith('#'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Extract section name from header
                current_section = line.lstrip('#').strip().lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
            
        return {
            'name': problem_folder.name,
            'sections': sections
        }

class ProgramExecutor:
    """Handles program compilation, execution, and result extraction."""
    
    def __init__(self, problem_folder: Path):
        self.problem_folder = problem_folder
        
    def save_program(self, program: str, language: str) -> Path:
        """Saves the LLM's program to the appropriate file."""
        # Remove markdown code block formatting if present
        program = program.strip()
        if program.startswith("```"):
            # Find the first newline after the opening ```
            first_newline = program.find("\n")
            if first_newline != -1:
                # Remove the opening ``` and language identifier
                program = program[first_newline + 1:]
                # Remove the closing ``` if present
                if program.endswith("```"):
                    program = program[:-3]
                program = program.strip()
        
        if language == "cpp":
            target_file = self.problem_folder / "solver.cpp"
            # Check if solver.h is included
            if "#include \"solver.h\"" not in program and "#include 'solver.h'" not in program:
                # Insert solver.h include at the beginning of the file
                program = "#include \"solver.h\"\n\n" + program
        else:  # python
            target_file = self.problem_folder / "solver.py"
            
        with open(target_file, 'w') as f:
            f.write(program)
        logger.info(f"Saved program to {target_file}")
        return target_file
        
    def compile_and_run(self, language: str) -> Tuple[bool, str]:
        """Compiles and runs the program, returns success status and output."""
        try:
            if language == "cpp":
                # Compile C++ program
                compile_result = subprocess.run(
                    ['make', '-C', str(self.problem_folder)],
                    capture_output=True,
                    text=True
                )
                if compile_result.returncode != 0:
                    return False, f"Compilation failed: {compile_result.stderr}"
                
                # Get all test cases from the demo folder
                demo_folder = Path(str(self.problem_folder.parent) + "/dataset/demo")
                if not demo_folder.exists():
                    return False, f"Demo folder not found: {demo_folder}"
                
                # Find all .dot and .json files in the demo folder
                dot_files = [f for f in demo_folder.iterdir() if f.is_file() and f.suffix == '.dot']
                json_files = [f for f in demo_folder.iterdir() if f.is_file() and f.suffix == '.json']
                
                if not dot_files and not json_files:
                    return False, f"No test cases found in {demo_folder}"
                
                # Create a mapping of base names to file pairs
                file_pairs = {}
                
                # Process .dot files
                for dot_file in dot_files:
                    base_name = dot_file.stem
                    file_pairs[base_name] = {'input': dot_file, 'constraint': None}
                
                # Process .json files and match with .dot files
                for json_file in json_files:
                    base_name = json_file.stem
                    if base_name in file_pairs:
                        file_pairs[base_name]['constraint'] = json_file
                    else:
                        # If no matching .dot file, add it as a standalone constraint file
                        file_pairs[base_name] = {'input': None, 'constraint': json_file}
                
                # Run the program for each file pair
                all_outputs = []
                for base_name, files in file_pairs.items():
                    # Skip if no input file
                    if not files['input']:
                        all_outputs.append(f"Test case {base_name}: No input file found")
                        continue
                    
                    # Run the compiled program
                    run_result = subprocess.run(
                        ['./main.out', f'../dataset/demo/{base_name}'],
                        cwd=str(self.problem_folder),
                        capture_output=True,
                        text=True
                    )
                    
                    if run_result.returncode != 0:
                        all_outputs.append(f"Test case {base_name}:\n{run_result.stderr}")
                        continue
                    
                    # Run the verifier and evaluator
                    eval_result = subprocess.run(
                        ['./evaluator.out', f'../dataset/demo/{base_name}', f'output/{base_name}.schedule'],
                        cwd=str(self.problem_folder),
                        capture_output=True,
                        text=True
                    )
                    
                    if eval_result.returncode != 0:
                        all_outputs.append(f"Test case {base_name}:\n{eval_result.stderr}")
                        continue
                    
                    all_outputs.append(f"Test case {base_name}:\n{eval_result.stdout}")
                
                # Combine all outputs
                combined_output = "\n\n".join(all_outputs)
                return True, combined_output
                
            else:  # python
                # Run Python program directly
                run_result = subprocess.run(
                    ['python3', 'main.py'],
                    cwd=str(self.problem_folder),
                    capture_output=True,
                    text=True
                )
            
            if run_result.returncode != 0:
                return False, f"Execution failed: {run_result.stderr}"
                
            return True, run_result.stdout
            
        except Exception as e:
            return False, f"Error: {str(e)}"
            
    def extract_results(self, output: str) -> Dict:
        """Extracts and parses the results from program output."""
        try:
            results = {}
            
            # Split the output by test cases
            test_case_outputs = output.split("Test case ")
            
            for test_output in test_case_outputs:
                if not test_output.strip():
                    continue
                
                # Extract test case name
                name_match = re.match(r'([^:]+):', test_output)
                if not name_match:
                    continue
                
                test_case_name = name_match.group(1).strip()
                # Extract the cost for this test case
                cost_match = re.search(r'Cost: (\d+)', test_output)
                if cost_match:
                    cost = int(cost_match.group(1))
                    results[test_case_name] = cost
                else:
                    results[test_case_name] = float('inf')
                
            return results
        except Exception as e:
            logger.error(f"Error extracting results: {str(e)}")
            return {'Cost': float('inf')}
            

class LLMInterface:
    """Interface for interacting with different LLM providers."""
    
    def __init__(self, models_to_use: List[str]):
        load_dotenv()
        
        # Initialize clients only for models that will be used
        self.openai_client = None
        self.deepseek_client = None
        self.anthropic_client = None
        
        # Check which models are being used and initialize appropriate clients
        openai_models = [m for m in models_to_use if m.startswith("gpt")]
        deepseek_models = [m for m in models_to_use if m.startswith("deepseek")]
        anthropic_models = [m for m in models_to_use if m.startswith("claude")]
        
        if openai_models:
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY is required for OpenAI models but not provided")
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
        if deepseek_models:
            if not os.getenv('DEEPSEEK_API_KEY'):
                raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek models but not provided")
            self.deepseek_client = OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
            
        if anthropic_models:
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models but not provided")
            self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Loads the prompt template from prompt.md."""
        template_path = Path(os.getcwd()) / "prompt.md"
        if not template_path.exists():
            logger.warning(f"Prompt template not found at {template_path}. Using default template.")
            return self._get_default_template()
            
        with open(template_path, 'r') as f:
            return f.read()
            
    def _get_default_template(self) -> str:
        """Returns a default prompt template if prompt.md is not found."""
        raise NotImplementedError("Prompt template not found. Please create a prompt.md file in the root directory.")
        
    def _format_problem_info(self, problem_desc: Dict[str, str]) -> str:
        """Formats the problem information section for the prompt."""
        return f"""Problem: {problem_desc['name']}

Description:
{problem_desc['sections'].get('background', '')}

Formalization:
{problem_desc['sections'].get('formalization', '')}

Objective:
{problem_desc['sections'].get('objective', '')}

Input Format:
{problem_desc['sections'].get('input_format', '')}

Output Format:
{problem_desc['sections'].get('output_format', '')}"""
        
    def _get_example_program(self, problem_desc: Dict[str, str]) -> str:
        """Gets an example program template for the problem."""
        problem_folder = Path(problem_desc['name']) / "program"
        
        # Check if it's a C++ problem
        if (problem_folder / "main.cpp").exists() and (problem_folder / "solver.h").exists():
            # Read C++ template files
            with open(problem_folder / "main.cpp", 'r') as f:
                main_cpp = f.read()
            with open(problem_folder / "solver.h", 'r') as f:
                solver_h = f.read()
            return f"""// Main program:
{main_cpp}

// Solver header:
{solver_h}
"""
        else:
            # Read Python template files
            with open(problem_folder / "main.py", 'r') as f:
                main_py = f.read()
            with open(problem_folder / "solver.py", 'r') as f:
                solver_py = f.read()
            return f"""# Main program:
{main_py}

# Solver module:
{solver_py}
"""
        
    def _get_demo_dataset_info(self, problem_desc: Dict[str, str]) -> str:
        """Loads and formats information about the demo dataset files."""
        # Get the path to the demo folder
        demo_folder = Path(str(Path(problem_desc['name'])) + "/dataset/demo")
        if not demo_folder.exists():
            return "Demo dataset folder not found."
        
        # Find all .dot and .json files in the demo folder
        dot_files = [f for f in demo_folder.iterdir() if f.is_file() and f.suffix == '.dot']
        json_files = [f for f in demo_folder.iterdir() if f.is_file() and f.suffix == '.json']
        
        if not dot_files and not json_files:
            return "No input or constraint files found in the demo dataset folder."
        
        # Format information about the files
        dataset_info = "## Demo Dataset Information\n\n"
        
        # Create a mapping of base names to file pairs
        file_pairs = {}
        
        # Process .dot files
        for dot_file in dot_files:
            base_name = dot_file.stem
            file_pairs[base_name] = {'input': dot_file, 'constraint': None}
        
        # Process .json files and match with .dot files
        for json_file in json_files:
            base_name = json_file.stem
            if base_name in file_pairs:
                file_pairs[base_name]['constraint'] = json_file
            else:
                # If no matching .dot file, add it as a standalone constraint file
                file_pairs[base_name] = {'input': None, 'constraint': json_file}
        
        # Add information about each file pair
        for base_name, files in file_pairs.items():
            dataset_info += f"### Input-Constraint Pair: {base_name}\n\n"
            
            # Add input file content
            if files['input']:
                try:
                    with open(files['input'], 'r') as f:
                        content = f.read()
                        
                    # Truncate very large files
                    if len(content) > 1000:
                        content = content[:1000] + "...\n[Content truncated]"
                        
                    dataset_info += f"**Input File ({files['input'].name}):**\n```\n{content}\n```\n\n"
                except Exception as e:
                    dataset_info += f"**Input File ({files['input'].name}):** Error reading file: {str(e)}\n\n"
            else:
                dataset_info += "**Input File:** Not found\n\n"
            
            # Add constraint file content
            if files['constraint']:
                try:
                    with open(files['constraint'], 'r') as f:
                        content = f.read()
                        
                    # Truncate very large files
                    if len(content) > 1000:
                        content = content[:1000] + "...\n[Content truncated]"
                        
                    dataset_info += f"**Constraint File ({files['constraint'].name}):**\n```\n{content}\n```\n\n"
                except Exception as e:
                    dataset_info += f"**Constraint File ({files['constraint'].name}):** Error reading file: {str(e)}\n\n"
            else:
                dataset_info += "**Constraint File:** Not found\n\n"
        
        return dataset_info
        
    def format_prompt(self, 
                     problem_desc: Dict[str, str], 
                     iteration: int = 0, 
                     previous_program: str = None,
                     previous_output: str = None,
                     previous_costs: Dict = None,
                     language: str = None) -> str:
        """Formats the problem description into a prompt for the LLM."""
        # Format the problem information
        problem_info = self._format_problem_info(problem_desc)
        
        # Get the example program
        example_program = self._get_example_program(problem_desc)
        
        # Replace placeholders in the template
        prompt = self.prompt_template.replace("{PROBLEM}", problem_info)
        prompt = prompt.replace("{EXAMPLE_PROGRAM}", example_program)
        
        # If this is an iteration beyond the first, add the previous program, its costs, and dataset information
        if iteration > 0 and previous_program:
            # Get the demo dataset information
            dataset_info = self._get_demo_dataset_info(problem_desc)
            
            prompt += f"""
## Feedback from Previous Iteration (Iteration {iteration-1})
This is the program you generated in the previous iteration:
```{language}
{previous_program}
```

This is the demo dataset we just executed:
{dataset_info}

This is the execution output on the demo dataset from the previous iteration:
{previous_output}

The program achieved the following costs for each test case:"""

            # Add costs for each test case
            if previous_costs:
                for test_case, cost in previous_costs.items():
                    # Skip non-test case keys
                    if test_case in ['Cost', 'Average_Cost', 'Total_Cost', 'Test_Case_Count']:
                        continue
                    
                    if cost == float('inf'):
                        prompt += f"\n- {test_case}: Failed to produce a valid solution"
                    else:
                        prompt += f"\n- {test_case}: {cost}"
            else:
                prompt += "\nNo cost information available."

            # Add error message if any costs are infinity
            if previous_costs:
                any_failed = any(cost == float('inf') for cost in previous_costs.values() 
                                if cost not in ['Cost', 'Average_Cost', 'Total_Cost', 'Test_Case_Count'])
                
                if any_failed:
                    prompt += """

The program failed to produce valid solutions for some test cases. Please fix the following issues:
1. Check for compilation errors or runtime exceptions
2. Ensure the program handles all edge cases and meets the problem constraints correctly
3. Verify that the output format matches the expected format
4. Make sure all required functions are implemented correctly"""
                else:
                    prompt += """

Please carefully observe the problem structure and improve upon this program by:
1. Addressing any weaknesses in the previous approach
2. Introducing more advanced or efficient algorithms
3. Focusing on improving performance for test cases with higher costs

Your goal is to improve the solution for as many test cases as possible, with special attention to those where the previous solution performed poorly."""
            else:
                prompt += """

Please carefully observe the problem structure and improve upon this program by:
1. Addressing any weaknesses in the previous approach
2. Introducing more advanced or efficient algorithms

Your goal is to improve the solution for all test cases."""
        
        return prompt
    
    def get_program(self, 
                    problem_desc: Dict[str, str], 
                    model: str = "gpt-4-turbo-preview",
                    iteration: int = 0,
                    previous_program: str = None,
                    previous_output: str = None,
                    previous_costs: Dict = None,
                    language: str = None) -> str:
        """Gets a program from the specified LLM."""
        prompt = self.format_prompt(problem_desc, iteration, previous_program, previous_output, previous_costs, language)
        
        # Create solutions directory if it doesn't exist
        solutions_dir = Path("llm_solutions")
        solutions_dir.mkdir(exist_ok=True)
        
        # Save the prompt to the log file if it's provided
        if hasattr(self, 'current_log_file'):
            with open(self.current_log_file, 'a') as f:
                f.write(f"PROMPT FOR ITERATION {iteration}:\n")
                f.write(prompt)
                f.write("\n\n")
        
        if model.startswith("gpt"):
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized. OPENAI_API_KEY is required for OpenAI models.")
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert optimization algorithm designer."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
            
        elif model.startswith("claude"):
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized. ANTHROPIC_API_KEY is required for Claude models.")
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            raw_response = response.content[0].text
        
        elif model.startswith("deepseek"):
            if not self.deepseek_client:
                raise ValueError("DeepSeek client not initialized. DEEPSEEK_API_KEY is required for DeepSeek models.")
            # Use OpenAI SDK with DeepSeek's base URL
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert optimization algorithm designer."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        return raw_response
    
    def get_iterative_program(self,
                              problem_desc: Dict[str, str],
                              model: str,
                              executor: ProgramExecutor,
                              language: str,
                              max_iterations: int = 3) -> Tuple[str, int]:
        """Gets an iteratively improved program from the specified LLM."""
        current_program = None
        previous_program = None
        previous_output = None
        best_program = None
        best_costs = {}  # Dictionary to store costs for each test case
        best_iteration = 0
        
        # Create solutions directory if it doesn't exist
        solutions_dir = Path("llm_solutions")
        solutions_dir.mkdir(exist_ok=True)
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a single log file for this problem and model with timestamp
        log_file = solutions_dir / f"{problem_desc['name']}_{model}_{timestamp}_log.txt"
        
        # Store the log file path in the LLMInterface instance
        self.current_log_file = log_file
        
        # Initialize the log file with a header
        with open(log_file, 'w') as f:
            f.write(f"Problem: {problem_desc['name']}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Language: {language}\n")
            f.write(f"Max Iterations: {max_iterations}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"Getting program from {model} (iteration {iteration+1}/{max_iterations})")
                
                # Get program from LLM
                current_program = self.get_program(
                    problem_desc, 
                    model, 
                    iteration, 
                    previous_program,
                    previous_output,
                    best_costs if iteration > 0 else None,
                    language
                )
                
                # Append the program to the log file
                with open(log_file, 'a') as f:
                    f.write(f"\n{'=' * 40} ITERATION {iteration} {'=' * 40}\n\n")
                    f.write("PROGRAM:\n")
                    f.write(current_program if current_program else "No program generated yet")
                    f.write("\n\n")
                
                # Save and execute the program
                program_file = executor.save_program(current_program, language)
                success, output = executor.compile_and_run(language)
                
                # Store the output and program for the next iteration
                previous_output = output
                previous_program = current_program
                
                # Append the execution output to the log file
                with open(log_file, 'a') as f:
                    f.write("EXECUTION OUTPUT:\n")
                    f.write(output)
                    f.write("\n\n")
                
                if not success:
                    logger.error(f"Failed to run program: {output}")
                    # Otherwise, keep the previous best program
                    continue
                
                # Extract and parse results
                results = executor.extract_results(output)
                
                # Check if this program is better than the best so far
                is_better = False
                
                # If this is the first iteration, initialize best_costs
                if iteration == 0:
                    best_costs = results
                    is_better = True
                else:
                    # Compare costs for each test case
                    for test_case, cost in results.items():
                        # Skip non-test case keys
                        if test_case in ['Cost', 'Average_Cost', 'Total_Cost', 'Test_Case_Count']:
                            continue
                            
                        # If this test case is better than the best so far
                        if test_case not in best_costs or cost < best_costs[test_case]:
                            is_better = True
                            best_costs[test_case] = cost
                
                # Log the current costs
                logger.info(f"Current costs: {results}")
                
                # Append the results to the log file
                with open(log_file, 'a') as f:
                    f.write("RESULTS:\n")
                    f.write(json.dumps(results, indent=2))
                    f.write("\n\n")
                
                # Update best program if this one is better
                if is_better:
                    best_program = current_program
                    best_iteration = iteration
                    logger.info(f"New best program found! Costs: {best_costs}")
                    
                    # Append the best program info to the log file
                    with open(log_file, 'a') as f:
                        f.write("NEW BEST PROGRAM FOUND!\n")
                        f.write(f"Best Iteration: {best_iteration}\n")
                        f.write(f"Best Costs: {json.dumps(best_costs, indent=2)}\n")
                        f.write("=" * 80 + "\n\n")
                
                # Add a delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration+1}: {str(e)}")
                # Log the error to the log file
                with open(log_file, 'a') as f:
                    f.write(f"ERROR IN ITERATION {iteration+1}: {str(e)}\n")
                    f.write("=" * 80 + "\n\n")
                # If we have a program from a previous iteration, return that
                if best_program:
                    return best_program, best_iteration
                else:
                    raise
        
        # Append a summary to the log file
        with open(log_file, 'a') as f:
            f.write(f"\n{'=' * 40} FINAL SUMMARY {'=' * 40}\n\n")
            f.write(f"Best Iteration: {best_iteration}\n")
            f.write(f"Best Costs: {json.dumps(best_costs, indent=2)}\n")
            f.write("=" * 80 + "\n")
        
        return best_program, best_iteration

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Solver Agent for optimization problems')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=["gpt-4-turbo-preview", "claude-3-opus-20240229", "deepseek-chat", "deepseek-coder"],
                        help='List of models to use (default: all supported models)')
    
    parser.add_argument('--iterations', type=int, default=3,
                        help='Maximum number of iterations for each model (default: 3)')
    
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip problems that already have solutions')
    
    parser.add_argument('--problem', type=str,
                        default='operator_scheduling',
                        help='Specific problem to solve (folder name)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    workspace_root = os.getcwd()
    
    # Initialize components
    problem_reader = ProblemReader(workspace_root)
    llm_interface = LLMInterface(args.models)  # Pass the models to use
    
    # Get problem folders
    problem_folders = problem_reader.get_problem_folders()
    
    # Filter by specific problem if provided
    if args.problem:
        problem_folders = [p for p in problem_folders if p.name == args.problem]
        if not problem_folders:
            logger.error(f"Problem '{args.problem}' not found")
            return
    
    # Process each problem
    for problem_folder in problem_folders:
        try:
            # Read problem description
            problem_desc = problem_reader.read_problem_description(problem_folder)
            logger.info(f"Processing problem: {problem_desc['name']}")
            
            # Initialize program executor
            executor = ProgramExecutor(Path(problem_desc['name']) / "program")
            
            # Determine language from problem folder
            language = "cpp" if (Path(problem_desc['name']) / "program" / "Makefile").exists() else "python"
            
            # Get solutions from each model
            for model in args.models:
                # Check if we already have a solution
                if args.skip_existing:
                    solutions_dir = Path("llm_solutions")
                    # We can't check for existing logs with timestamps, so we'll skip this check
                    # and let the user decide if they want to run again
                    logger.info(f"Skip-existing flag is set, but we can't check for existing logs with timestamps")
                    continue
                
                # Get iterative program
                logger.info(f"Getting iterative program from {model}")
                program, final_iteration = llm_interface.get_iterative_program(
                    problem_desc, 
                    model,
                    executor,
                    language,
                    args.iterations
                )
                
        except Exception as e:
            logger.error(f"Error processing {problem_folder}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
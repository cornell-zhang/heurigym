import os
import re
import json
import logging
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
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
            
        # Parse markdown content
        sections = {}
        current_section = "overview"
        current_content = []
        
        for line in md_content.split('\n'):
            # Check for main headers (level 1 or 2)
            if line.startswith('#') and not line.startswith('###'):
                # Save previous section content if exists
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                
                # Extract section name from header
                current_section = line.lstrip('#').strip().lower().replace(' ', '_')
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
    """Handles program execution and result extraction."""
    
    def __init__(self, problem_folder: Path, solution_folder: Path, timeout: int = 10):
        self.problem_folder = problem_folder
        self.program_folder = problem_folder / "program"
        self.solution_folder = solution_folder
        self.timeout = timeout
        
    def save_program(self, program: str, iteration: int = 0) -> Tuple[Path, str]:
        """Saves the LLM's program to solver.py in the solution folder and copies all necessary Python files."""
        # Find the first code block enclosed with ``` in the generated text
        program = program.strip()
        
        # Look for the first code block
        start_marker = "```"
        end_marker = "```"
        
        start_idx = program.find(start_marker)
        if start_idx != -1:
            # Find the end of the language identifier (if any)
            first_newline = program.find("\n", start_idx)
            if first_newline != -1:
                # Find the closing ```
                end_idx = program.find(end_marker, first_newline)
                if end_idx != -1:
                    # Extract just the code between the markers
                    program = program[first_newline + 1:end_idx].strip()
        
        # Create output directory in the solution folder for this iteration
        output_dir = self.solution_folder / f"output{iteration}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy all Python files from the original program folder to the solution folder
        for py_file in self.program_folder.glob("*.py"):
            if py_file.name != "solve.py":  # Skip solve.py as we'll write it separately
                target_file = self.solution_folder / py_file.name
                with open(py_file, 'r') as src, open(target_file, 'w') as dst:
                    dst.write(src.read())
        
        # Save the LLM's program to solver.py
        target_file = self.solution_folder / "solver.py"
        with open(target_file, 'w') as f:
            f.write(program)
        logger.info(f"Saved program to {target_file}")
        return target_file, program
        
    def execute_program(self, iteration: int = 0) -> Tuple[bool, str]:
        """Runs the Python program and returns success status and output."""
        try:
            # Get all test cases from the problem's dataset folder
            dataset_folder = self.problem_folder / "dataset" / "demo"
            if not dataset_folder.exists():
                return False, f"Dataset folder not found: {dataset_folder}"
            
            # Find all files in the dataset folder and group them by base name
            input_files = [f for f in dataset_folder.iterdir() if f.is_file()]
            file_groups = {}
            for input_file in input_files:
                base_name = input_file.stem
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(input_file)
            
            if not file_groups:
                return False, f"No test cases found in {dataset_folder}"
            
            # Run the program for each group of files
            all_outputs = []
            for base_name, group_files in file_groups.items():
                # Run the main program
                shutil.copy(
                    "scripts/main.py",
                    self.solution_folder / "main.py"
                )
                output_dir = self.solution_folder / f"output{iteration}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = output_dir / f"{base_name}.output"
                cost_file = output_dir / f"{base_name}.cost"
                
                try:
                    # Prepare the command with all input files in the group
                    cmd = ['python3', 'main.py']
                    cmd.extend(sorted([str(f) for f in group_files]))  # Add all input files
                    cmd.append(str(output_file))  # Add output file
                    
                    run_result = subprocess.run(
                        cmd,
                        cwd=str(self.solution_folder),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout  # Use the timeout from constructor
                    )
                except subprocess.TimeoutExpired:
                    error_data = {
                        "message": f"Program execution timed out after {self.timeout} seconds"
                    }
                    with open(cost_file, 'w') as f:
                        json.dump(error_data, f, indent=2)
                    all_outputs.append(f"Test case {base_name}:\nProgram execution timed out after {self.timeout} seconds")
                    continue
                
                if run_result.returncode != 0:
                    # Save error message directly to cost file
                    error_data = {
                        "message": f"Python execution error: {run_result.stderr}"
                    }
                    with open(cost_file, 'w') as f:
                        json.dump(error_data, f, indent=2)
                    all_outputs.append(f"Test case {base_name}:\n{run_result.stderr}")
                    continue
                
                # Run the evaluator
                shutil.copy(
                    "scripts/feedback.py",
                    self.solution_folder / "feedback.py"
                )
                
                # Prepare evaluator command with all input files
                eval_cmd = ['python3', 'feedback.py']
                eval_cmd.extend(sorted([str(f) for f in group_files]))  # Add all input files
                eval_cmd.append(str(output_file))  # Add output file
                
                eval_result = subprocess.run(
                    eval_cmd,
                    cwd=str(self.solution_folder),
                    capture_output=True,
                    text=True
                )
                
                if eval_result.returncode != 0:
                    # Save evaluator error to cost file
                    error_data = {
                        "message": f"Evaluator error: {eval_result.stderr}"
                    }
                    with open(cost_file, 'w') as f:
                        json.dump(error_data, f, indent=2)
                    all_outputs.append(f"Test case {base_name}:\n{eval_result.stderr}")
                    continue
                
                # Read the cost file
                if cost_file.exists():
                    with open(cost_file, 'r') as f:
                        cost_data = json.load(f)
                        output = f"Test case {base_name}:\n"
                        if 'validity' in cost_data and 'cost' in cost_data:
                            if cost_data['validity']:
                                output += f"Valid solution with cost: {cost_data['cost']}"
                            else:
                                output += f"Invalid solution with cost: {cost_data['cost']}\n"
                                output += f"Error: {cost_data['message']}"
                        else:
                            output += f"Error: {cost_data['message']}"
                        all_outputs.append(output)
                else:
                    error_data = {
                        "message": "No cost file generated"
                    }
                    with open(cost_file, 'w') as f:
                        json.dump(error_data, f, indent=2)
                    all_outputs.append(f"Test case {base_name}: No cost file generated")
            
            # Combine all outputs
            combined_output = "\n\n".join(all_outputs)
            return True, combined_output
            
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
                cost_match = re.search(r'cost: (\d+)', test_output.lower())
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
    
    def __init__(self, models_to_use: List[str], timeout: int = 10, temperature: float = 0.7):
        load_dotenv()
        self.timeout = timeout
        self.temperature = temperature
        
        # Initialize clients only for models that will be used
        self.openai_client = None
        self.deepseek_client = None
        self.anthropic_client = None
        self.gemini_client = None
        
        # Check which models are being used and initialize appropriate clients
        openai_models = [m for m in models_to_use if m.startswith("gpt")]
        deepseek_models = [m for m in models_to_use if m.startswith("deepseek")]
        anthropic_models = [m for m in models_to_use if m.startswith("claude")]
        gemini_models = [m for m in models_to_use if m.startswith("gemini")]
        
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
            
        if gemini_models:
            if not os.getenv('GOOGLE_API_KEY'):
                raise ValueError("GOOGLE_API_KEY is required for Gemini models but not provided")
            self.gemini_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
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
        sections = []
        
        # Add each section if it exists in the problem description
        for section_name in ['background', 'formalization', 'input_format', 'output_format']:
            section_content = problem_desc['sections'].get(section_name, '')
            if section_content:
                # Convert section name to title case for display
                title = section_name.replace('_', ' ').title()
                sections.append(f"## {title}\n{section_content}")
        
        # Join all sections with double newlines
        return "\n\n".join(sections)
        
    def _get_example_program(self, problem_desc: Dict[str, str]) -> str:
        """Gets an example program template for the problem."""
        workspace_root = Path(os.getcwd())
        problem_folder = workspace_root / problem_desc['name']
        program_folder = problem_folder / "program"
        
        # Try problem-specific solver first, fall back to default
        solver_path = program_folder / "solver.py"
        if not solver_path.exists():
            solver_path = Path("scripts/solver.py")
            
        if solver_path.exists():
            with open(solver_path, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Solver file not found at {solver_path}")
        
    def format_prompt(self, 
                     problem_desc: Dict[str, str], 
                     iteration: int = 0, 
                     previous_program: str = None,
                     solution_dir: Path = None) -> str:
        """Formats the problem description into a prompt for the LLM."""
        # Format the problem information
        problem_info = self._format_problem_info(problem_desc)
        
        # Get the example program
        example_program = self._get_example_program(problem_desc)
        
        # Replace placeholders in the template
        prompt = self.prompt_template.replace("{PROBLEM}", problem_info)
        prompt = prompt.replace("{EXAMPLE_PROGRAM}", example_program)
        prompt = prompt.replace("{TIMEOUT}", str(self.timeout))
        
        # If this is an iteration beyond the first, add the previous program and its results
        if iteration > 0 and previous_program and solution_dir:
            prompt += f"""
# Feedback from Previous Iteration (Iteration {iteration-1})
This is the program you generated in the previous iteration:
```python
{previous_program}
```

# Test Cases and Results"""

            # Get the absolute path to the demo folder
            workspace_root = Path(os.getcwd())
            problem_folder = workspace_root / problem_desc['name']
            demo_folder = problem_folder / "dataset" / "demo"
            if demo_folder.exists():
                # Find all files in the demo folder
                input_files = [f for f in demo_folder.iterdir() if f.is_file()]
                
                # For each test case, show input and its result
                for input_file in input_files:
                    test_case = input_file.stem
                    prompt += f"\n## Test Case: {test_case}\n\n"
                    
                    # Show input data
                    try:
                        with open(input_file, 'r') as f:
                            content = f.read()
                            # Limit to first 50 lines if content is too large
                            lines = content.split('\n')
                            if len(lines) > 50:
                                content = '\n'.join(lines[:50]) + '\n... (truncated)'
                        prompt += f"**Input Data:**\n```\n{content}\n```\n\n"
                    except Exception as e:
                        prompt += f"**Input Data:** Error reading file: {str(e)}\n\n"
                    
                    # Show result
                    cost_file = solution_dir / f"output{iteration - 1}" / f"{test_case}.cost"
                    if cost_file.exists():
                        with open(cost_file, 'r') as f:
                            cost_data = json.load(f)
                            if 'validity' in cost_data and 'cost' in cost_data:
                                if cost_data['validity']:
                                    prompt += f"**Result:** Valid solution with cost {cost_data['cost']}\n\n"
                                else:
                                    prompt += f"**Result:** Invalid solution with cost {cost_data['cost']}\n"
                                    prompt += f"**Error:** {cost_data['message']}\n\n"
                            else:
                                prompt += f"**Result:** Error occurred\n"
                                prompt += f"**Error:** {cost_data['message']}\n\n"
                    else:
                        prompt += "**Result:** No output generated\n\n"
            else:
                prompt += f"\nNo test cases found in the demo dataset folder: {demo_folder}\n\n"

            # Add improvement guidance
            any_failed = any(
                not json.load(open(f))['validity'] if 'validity' in json.load(open(f)) else True 
                for f in (solution_dir / f"output{iteration - 1}").glob("*.cost")
            )
            
            if any_failed:
                prompt += """
# Improvement Guidance
The program failed to produce valid solutions for some test cases. Please fix the following issues:
1. Check for compilation errors or runtime exceptions
2. Ensure the program handles all edge cases and meets the problem constraints correctly
3. Verify that the input and output format matches the expected format
4. Make sure all required functions are implemented correctly
5. If the program is not able to produce valid solutions for any test case, please try to find the root cause and fix it.
6. If the program is able to produce valid solutions for some test cases, please try to improve the solution."""
            else:
                prompt += """
# Improvement Guidance
Please carefully observe the problem structure and improve upon this program by:
1. Addressing any weaknesses in the previous approach
2. Introducing more advanced or efficient algorithms
3. Focusing on improving performance for test cases

Your goal is to improve the solution for as many test cases as possible, with special attention to those where the previous solution performed poorly."""
        
        return prompt
    
    def get_program(self, 
                    problem_desc: Dict[str, str], 
                    model: str = "gpt-4-turbo-preview",
                    iteration: int = 0,
                    previous_program: str = None,
                    solution_dir: Path = None) -> str:
        """Gets a program from the specified LLM."""
        prompt = self.format_prompt(problem_desc, iteration, previous_program, solution_dir)
        
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
                max_tokens=8192,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are an expert optimization algorithm designer. You are given a problem and try to solve it. Please only output the code for the solver."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
            
        elif model.startswith("claude"):
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized. ANTHROPIC_API_KEY is required for Claude models.")
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=self.temperature,
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
                max_tokens=8192,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are an expert optimization algorithm designer. You are given a problem and try to solve it. Please only output the code for the solver."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_response = response.choices[0].message.content
            
        elif model.startswith("gemini"):
            if not self.gemini_client:
                raise ValueError("Gemini client not initialized. GOOGLE_API_KEY is required for Gemini models.")
            
            # Create a system prompt for Gemini
            system_prompt = "You are an expert optimization algorithm designer. You are given a problem and try to solve it. Please only output the code for the solver."
            
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate content with Gemini
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=full_prompt,
                config={"temperature": self.temperature}
            )
            
            # Extract the text from the response
            raw_response = response.text
        
        else:
            raise ValueError(f"Unsupported model: {model}")
            
        return raw_response
    
    def get_iterative_program(self,
                              problem_desc: Dict[str, str],
                              model: str,
                              executor: ProgramExecutor,
                              solution_dir: Path,
                              max_iterations: int = 3) -> Tuple[str, int]:
        """Gets an iteratively improved program from the specified LLM."""
        current_program = None
        previous_program = None
        
        # Create log directory for this run
        log_dir = solution_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a log file for this run
        log_file = log_dir / f"{model}.log"
        
        # Store the log file path in the LLMInterface instance
        self.current_log_file = log_file
        
        # Initialize the log file with a header
        with open(log_file, 'w') as f:
            f.write(f"Problem: {problem_desc['name']}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Max Iterations: {max_iterations}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"Getting program from {model} (iteration {iteration+1}/{max_iterations})")
                
                # Get program from LLM
                raw_response = self.get_program(
                    problem_desc, 
                    model, 
                    iteration, 
                    previous_program,
                    solution_dir
                )
                
                # Save the program to a separate file
                program_file = log_dir / f"response{iteration}.txt"
                with open(program_file, 'w') as f:
                    f.write(raw_response)
                
                # Append the program to the log file
                with open(log_file, 'a') as f:
                    f.write(f"\n{'=' * 40} ITERATION {iteration} {'=' * 40}\n\n")
                    f.write("RAW RESPONSE:\n")
                    f.write(raw_response if raw_response else "No program generated yet")
                    f.write("\n\n")
                
                # Save and execute the program
                program_file, current_program = executor.save_program(raw_response, iteration)
                success, output = executor.execute_program(iteration)
                
                # Store the program for the next iteration
                previous_program = current_program
                
                # Append the execution output to the log file
                with open(log_file, 'a') as f:
                    f.write("EXECUTION OUTPUT:\n")
                    f.write(output)
                    f.write("\n\n")
                
                if not success:
                    logger.error(f"Failed to run program: {output}")
                    continue
                
                # Add a delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration+1}: {str(e)}")
                # Log the error to the log file
                with open(log_file, 'a') as f:
                    f.write(f"ERROR IN ITERATION {iteration+1}: {str(e)}\n")
                    f.write("=" * 80 + "\n\n")
                raise
        
        # Append a summary to the log file
        with open(log_file, 'a') as f:
            f.write(f"\n{'=' * 40} FINAL SUMMARY {'=' * 40}\n\n")
            f.write("=" * 80 + "\n")
        
        return current_program, max_iterations - 1

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Solver Agent for optimization problems')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=["deepseek-chat", "deepseek-reasoner", "gemini-2.0-flash"],
                        help='List of models to use (default: deepseek-chat, deepseek-reasoner, gemini-2.0-flash)')
    
    parser.add_argument('--iterations', type=int, default=3,
                        help='Maximum number of iterations for each model (default: 3)')
    
    parser.add_argument('--problem', type=str,
                        default='operator_scheduling',
                        help='Specific problem to solve (folder name)')
    
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout in seconds for program execution (default: 10)')
    
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for LLM generation (default: 0.8)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    workspace_root = os.getcwd()
    
    # Initialize components
    problem_reader = ProblemReader(workspace_root)
    llm_interface = LLMInterface(args.models, args.timeout, args.temperature)  # Pass the models, timeout and temperature to use
    
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
            
            # Create timestamp for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get solutions from each model
            for model in args.models:
                # Create the solution directory for this model
                workspace_root = Path(os.getcwd())
                solution_dir = workspace_root / "llm_solutions" / timestamp / problem_desc['name'] / model
                solution_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize program executor with the solution directory
                executor = ProgramExecutor(workspace_root / problem_desc['name'], solution_dir, args.timeout)
                
                # Get iterative program
                logger.info(f"Getting iterative program from {model}")
                program, final_iteration = llm_interface.get_iterative_program(
                    problem_desc, 
                    model,
                    executor,
                    solution_dir,
                    args.iterations
                )
                
        except Exception as e:
            logger.error(f"Error processing {problem_folder}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
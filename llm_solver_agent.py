import os
import re
import json
import logging
import time
import argparse
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from datetime import datetime
from config import calculate_cost

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
        in_code_block = False
        
        for line in md_content.split('\n'):
            # Check for code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                current_content.append(line)
                continue
                
            # Check for main headers (level 1 or 2)
            if line.startswith('#') and not line.startswith('###') and not in_code_block:
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
        
        # Create iteration-specific folders
        iteration_dir = self.solution_folder / f"iteration{iteration}"
        output_dir = iteration_dir / "output"
        os.makedirs(iteration_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy all files from the original program folder to the iteration folder
        for file in self.program_folder.iterdir():
            if file.is_file() and (file.suffix or file.name.lower() == 'makefile'):
                target_file = iteration_dir / file.name
                with open(file, 'r') as src, open(target_file, 'w') as dst:
                    dst.write(src.read())

        # Also copy the run.py script
        shutil.copy(
            "scripts/run.py",
            iteration_dir / "run.py"
        )
        
        # Save the LLM's program to solver.py in the iteration folder
        target_file = iteration_dir / "solver.py"
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
            total_execution_time = 0
            total_evaluation_time = 0
            
            # Get the iteration-specific folders
            iteration_dir = self.solution_folder / f"iteration{iteration}"
            output_dir = iteration_dir / "output"
            
            for base_name, group_files in file_groups.items():
                # Run the main program
                shutil.copy(
                    "scripts/main.py",
                    iteration_dir / "main.py"
                )
                output_file = output_dir / f"{base_name}.output"
                cost_file = output_dir / f"{base_name}.cost"
                
                try:
                    # Prepare the command with all input files in the group
                    cmd = ['python3', 'main.py']
                    cmd.extend(sorted([str(f) for f in group_files]))  # Add all input files
                    cmd.append(str(output_file))  # Add output file
                    
                    # Measure execution time
                    exec_start_time = time.time()
                    run_result = subprocess.run(
                        cmd,
                        cwd=str(iteration_dir),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout  # Use the timeout from constructor
                    )
                    exec_time = time.time() - exec_start_time
                    total_execution_time += exec_time
                    
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
                
                # Check if output file exists and has content
                if not output_file.exists() or output_file.stat().st_size == 0:
                    error_data = {
                        "message": "Evaluator error: No output file generated or output file is empty"
                    }
                    with open(cost_file, 'w') as f:
                        json.dump(error_data, f, indent=2)
                    all_outputs.append(f"Test case {base_name}:\nNo output file generated or output file is empty")
                    continue
                
                # Run the evaluator
                shutil.copy(
                    "scripts/feedback.py",
                    iteration_dir / "feedback.py"
                )
                
                # Prepare evaluator command with all input files
                eval_cmd = ['python3', 'feedback.py']
                eval_cmd.extend(sorted([str(f) for f in group_files]))  # Add all input files
                eval_cmd.append(str(output_file))  # Add output file
                
                # Measure evaluation time
                eval_start_time = time.time()
                eval_result = subprocess.run(
                    eval_cmd,
                    cwd=str(iteration_dir),
                    capture_output=True,
                    text=True
                )
                eval_time = time.time() - eval_start_time
                total_evaluation_time += eval_time
                
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
            
            # Add timing information to the output
            timing_info = f"\n\nTiming Information:\n"
            timing_info += f"Total Execution Time: {total_execution_time:.2f} seconds\n"
            timing_info += f"Total Evaluation Time: {total_evaluation_time:.2f} seconds\n"
            timing_info += f"Total Time: {(total_execution_time + total_evaluation_time):.2f} seconds"
            
            # Combine all outputs
            combined_output = "\n\n".join(all_outputs) + timing_info
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
    """Interface for interacting with different LLM providers using unified OpenAI API format."""
    
    def __init__(self, models_to_use: List[str], timeout: int = 10, temperature: float = 0.7):
        load_dotenv()
        self.timeout = timeout
        self.temperature = temperature
        self.conversation_history = {}  # Store conversation history for each model
        
        # Initialize clients for each provider
        self.clients = {}
        
        # Map of model prefixes to their API configurations
        self.model_configs = {
            "gpt": {
                "api_key": os.getenv('OPENAI_API_KEY'),
                "base_url": "https://api.openai.com/v1"
            },
            "deepseek": {
                "api_key": os.getenv('DEEPSEEK_API_KEY'),
                "base_url": "https://api.deepseek.com/v1"
            },
            "claude": {
                "api_key": os.getenv('ANTHROPIC_API_KEY'),
                "base_url": "https://api.anthropic.com/v1"
            },
            "gemini": {
                "api_key": os.getenv('GOOGLE_API_KEY'),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"
            },
            "openrouter": {
                "api_key": os.getenv('OPENROUTER_API_KEY'),
                "base_url": "https://openrouter.ai/api/v1"
            },
            "qwen": {
                "api_key": os.getenv('DASHSCOPE_API_KEY'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }
        
        # Initialize clients for each provider that will be used
        for model in models_to_use:
            provider = self._get_provider(model)
            if provider and provider not in self.clients:
                config = self.model_configs[provider]
                if not config['api_key']:
                    raise ValueError(f"{provider.upper()}_API_KEY is required for {provider} models but not provided")
                self.clients[provider] = OpenAI(
                    api_key=config['api_key'],
                    base_url=config['base_url']
                )
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
    def _get_provider(self, model: str) -> str:
        """Get the provider name from a model identifier."""
        if model.startswith("openrouter/"):
            return "openrouter"
        for provider in self.model_configs.keys():
            if model.startswith(provider):
                return provider
        return None

    def _get_actual_model_name(self, model: str) -> str:
        """Get the actual model name to use with the API."""
        if model.startswith("openrouter/"):
            return model.replace("openrouter/", "", 1)
        return model

    def _get_base_model_name(self, model: str) -> str:
        """Extracts the base model name from a model identifier for file paths."""
        # Remove openrouter/ prefix if present
        if model.startswith("openrouter/"):
            model = model.replace("openrouter/", "", 1)
        
        # Remove provider prefix if present (e.g., deepseek/, anthropic/, etc.)
        parts = model.split("/")
        if len(parts) > 1:
            model = parts[-1]
        
        # Remove any suffixes after : (e.g., :free, :latest, etc.)
        model = model.split(":")[0]
        
        return model

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
        prompt = ""
        
        # Only include problem description and example program in the first iteration
        if iteration == 0:
            # Format the problem information
            problem_info = self._format_problem_info(problem_desc)
            
            # Get the example program
            example_program = self._get_example_program(problem_desc)
            
            # Replace placeholders in the template
            prompt = self.prompt_template.replace("{PROBLEM}", problem_info)
            prompt = prompt.replace("{EXAMPLE_PROGRAM}", example_program)
            prompt = prompt.replace("{TIMEOUT}", str(self.timeout))
        else:
            # For later iterations, just include the feedback and improvement guidance
            prompt = f"""
# Feedback from Previous Iteration (Iteration {iteration-1})
These are the test cases and results from the previous iteration:
"""

            # Get the absolute path to the demo folder
            workspace_root = Path(os.getcwd())
            problem_folder = workspace_root / problem_desc['name']
            demo_folder = problem_folder / "dataset" / "demo"
            if demo_folder.exists():
                # Find all files in the demo folder and group them by base name
                input_files = [f for f in demo_folder.iterdir() if f.is_file()]
                file_groups = {}
                for input_file in input_files:
                    base_name = input_file.stem
                    if base_name not in file_groups:
                        file_groups[base_name] = []
                    file_groups[base_name].append(input_file)
                
                if not file_groups:
                    prompt += f"\nNo test cases found in the demo dataset folder: {demo_folder}\n\n"
                else:
                    # For each group of files, show them together
                    for base_name, group_files in file_groups.items():
                        prompt += f"\n## Test Case: {base_name}\n\n"
                        
                        # Show all input files in this group
                        for input_file in sorted(group_files):
                            try:
                                with open(input_file, 'r') as f:
                                    content = f.read()
                                    # Limit to first 50 lines if content is too large
                                    lines = content.split('\n')
                                    if len(lines) > 50:
                                        content = '\n'.join(lines[:50]) + '\n... (truncated)'
                                prompt += f"**Input File: {input_file.name}**\n```\n{content}\n```\n\n"
                            except Exception as e:
                                prompt += f"**Input File: {input_file.name}** Error reading file: {str(e)}\n\n"
                        
                        # Show result for this group
                        cost_file = solution_dir / f"iteration{iteration - 1}" / "output" / f"{base_name}.cost"
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
                for f in (solution_dir / f"iteration{iteration - 1}" / "output").glob("*.cost")
            )
            
            if any_failed:
                prompt += """
# Improvement Guidance
The program failed to produce valid solutions for some test cases. Please fix the following issues:
1. Check for compilation errors or runtime exceptions
2. Ensure the program handles all edge cases and meets the problem constraints correctly
3. Verify that the input and output format matches the expected format
4. Make sure all required functions are implemented correctly, and no external forbidden libraries are used
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
    
    def _save_api_info_to_json(self, model: str, iteration: int, api_time: float, prompt_tokens: int, completion_tokens: int, total_cost: float) -> None:
        """Saves API information to a JSON file."""
        if not hasattr(self, 'current_log_file'):
            return
            
        # Create API info directory if it doesn't exist
        api_info_dir = self.current_log_file.parent / "api_info"
        api_info_dir.mkdir(exist_ok=True)
        
        # Create JSON file path with simplified naming
        json_file = api_info_dir / f"api_info_iter_{iteration}.json"
        
        # Prepare API info data
        api_info = {
            "model": model,
            "iteration": iteration,
            "api_call_time_seconds": round(api_time, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost": round(total_cost, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON file
        with open(json_file, 'w') as f:
            json.dump(api_info, f, indent=2)
            
    def get_program(self, 
                    problem_desc: Dict[str, str], 
                    model: str = "gpt-4-turbo-preview",
                    iteration: int = 0,
                    previous_program: str = None,
                    solution_dir: Path = None) -> str:
        """Gets a program from the specified LLM using unified OpenAI API format."""
        prompt = self.format_prompt(problem_desc, iteration, previous_program, solution_dir)
        
        # Initialize conversation history for this model if not exists
        if model not in self.conversation_history:
            self.conversation_history[model] = [
                {"role": "system", "content": "You are an optimization expert tasked with solving the following problem by writing an efficient program. Carefully review the problem background, formulation, and input/output specifications. Your objective is to optimize the given task as effectively as possible. You may implement any algorithm you like. Please strictly follow the instructions below."}
            ]
        
        # Add the current prompt to conversation history
        self.conversation_history[model].append({"role": "user", "content": prompt})
        
        # Save the prompt to a separate file for this iteration
        if hasattr(self, 'current_log_file'):
            prompt_dir = self.current_log_file.parent / "prompt"
            prompt_dir.mkdir(exist_ok=True)
            prompt_file = prompt_dir / f"prompt{iteration}.txt"
            with open(prompt_file, 'w') as f:
                f.write(f"PROMPT FOR ITERATION {iteration}:\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Also append to the main log file for reference
            with open(self.current_log_file, 'a') as f:
                f.write(f"PROMPT FOR ITERATION {iteration}:\n")
                f.write(prompt)
                f.write("\n\n")
        
        # Measure API call time
        api_start_time = time.time()
        
        # Get provider and client
        provider = self._get_provider(model)
        if not provider or provider not in self.clients:
            raise ValueError(f"Unsupported model or missing client: {model}")
        
        client = self.clients[provider]
        actual_model = self._get_actual_model_name(model)
        
        try:
            # Make API call using unified OpenAI format
            response = client.chat.completions.create(
                model=actual_model,
                max_tokens=32768,
                temperature=self.temperature,
                messages=self.conversation_history[model]
            )
            
            if not response or not response.choices:
                error_msg = f"Error: Invalid response received from model {model}!!! Exiting..."
                logger.error(error_msg)
                sys.exit(1)
                
            raw_response = response.choices[0].message.content
            # Add assistant's response to conversation history
            self.conversation_history[model].append({"role": "assistant", "content": raw_response})
            
            # Get token counts
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
        except Exception as e:
            error_msg = f"Error: Failed to get response from model {model}: {str(e)}!!! Exiting..."
            logger.error(error_msg)
            sys.exit(1)
            
        # Check if raw_response is empty
        if not raw_response or not raw_response.strip():
            error_msg = f"Error: Empty response received from model {model}!!! Exiting..."
            logger.error(error_msg)
            sys.exit(1)
            
        api_time = time.time() - api_start_time
        
        # Calculate cost using the config module
        total_cost = calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Save API info to JSON file
        self._save_api_info_to_json(model, iteration, api_time, prompt_tokens, completion_tokens, total_cost)
        
        # Log API timing and cost information to the main log file
        if hasattr(self, 'current_log_file'):
            with open(self.current_log_file, 'a') as f:
                f.write(f"API Call Time: {api_time:.2f} seconds\n")
                f.write(f"Prompt Tokens: {prompt_tokens}\n")
                f.write(f"Completion Tokens: {completion_tokens}\n")
                f.write(f"Total Tokens: {prompt_tokens + completion_tokens}\n")
                f.write(f"Estimated Cost: ${total_cost:.4f}\n\n")
        
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
        
        # Use base model name for log file
        base_model_name = self._get_base_model_name(model)
        log_file = log_dir / f"{base_model_name}.log"
        
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
                response_dir = log_dir / "responses"
                response_dir.mkdir(exist_ok=True)
                program_file = response_dir / f"response{iteration}.txt"
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
                        default=[
                            "openrouter/deepseek/deepseek-chat-v3-0324:free", # "deepseek-chat"
                            "openrouter/deepseek/deepseek-r1:free", # "deepseek-reasoner"
                            "gemini-2.5-flash-preview-04-17",
                            "gemini-2.5-pro-exp-03-25",
                            "openrouter/qwen/qwen3-235b-a22b:free",
                            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
                            "openrouter/meta-llama/llama-4-maverick:free",
                            "qwen3-235b-a22b"
                        ],
                        help='List of models to use (default: deepseek-chat, deepseek-reasoner)')
    
    parser.add_argument('--iterations', type=int, default=3,
                        help='Maximum number of iterations for each model (default: 3)')
    
    parser.add_argument('--problem', type=str,
                        default='operator_scheduling',
                        help='Specific problem to solve (folder name)')
    
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout in seconds for program execution (default: 10)')
    
    parser.add_argument('--temperature', type=float, default=0.0,
                        # 0.0 for deterministic output
                        help='Temperature for LLM generation (default: 0.0)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    workspace_root = os.getcwd()
    
    # Initialize components
    problem_reader = ProblemReader(workspace_root)
    llm_interface = LLMInterface(args.models, args.timeout, args.temperature)
    
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
                # Get base model name for directory
                base_model_name = llm_interface._get_base_model_name(model)
                
                # Create the solution directory for this model
                workspace_root = Path(os.getcwd())
                solution_dir = workspace_root / "llm_solutions" / timestamp / problem_desc['name'] / base_model_name
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
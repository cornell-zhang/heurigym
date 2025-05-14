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
from datetime import datetime
from config import calculate_cost
from datasets import load_dataset
from huggingface_hub import login

HF_REPO_ID = "heurigen/heurigen-data"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_git_commit_id() -> str:
    """Get the current git commit ID."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Unknown (not a git repository or git command failed)"

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
    
    def __init__(self, problem_folder: Path, solution_folder: Path, dataset: Dict, timeout: int = 10, num_cores: int = 8):
        self.problem_folder = problem_folder
        self.program_folder = problem_folder / "program"
        self.solution_folder = solution_folder
        self.dataset = dataset
        self.timeout = timeout
        self.num_cores = num_cores
        
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
            # Get file paths from the dataset
            if not self.dataset or "train" not in self.dataset:
                return False, f"Dataset not found or invalid format for {self.problem_folder.name}"
            
            file_paths = self.dataset["train"]["file_path"]
            if not file_paths:
                return False, f"No test cases found in the dataset for {self.problem_folder.name}"
            
            # Group files by base name
            file_groups = {}
            for file_path in file_paths:
                base_name = Path(file_path).stem
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append(file_path)
            
            all_outputs = []
            total_execution_time = 0
            total_evaluation_time = 0
            
            # Get the iteration-specific folders
            iteration_dir = self.solution_folder / f"iteration{iteration}"
            output_dir = iteration_dir / "output"
            
            # copy the main.py script
            shutil.copy(
                "scripts/main.py",
                iteration_dir / "main.py"
            )
            
            # copy the feedback.py script
            shutil.copy(
                "scripts/feedback.py",
                iteration_dir / "feedback.py"
            )

            if not file_groups:
                # Instead of returning error, return success with informative message
                # This allows the program to continue to the next iteration
                return True, f"No test case is provided. "
            
            for base_name, group_files in file_groups.items():
                output_file = output_dir / f"{base_name}.output"
                cost_file = output_dir / f"{base_name}.cost"
                
                try:
                    # Prepare the command with all input files in the group
                    cmd = ['python3', 'main.py']
                    cmd.extend(sorted(group_files))  # Add all input files
                    cmd.append(str(output_file))  # Add output file
                    
                    # Set environment variables to limit CPU cores
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(self.num_cores)
                    
                    # Measure execution time
                    exec_start_time = time.time()
                    run_result = subprocess.run(
                        cmd,
                        cwd=str(iteration_dir),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,  # Use the timeout from constructor
                        env=env
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
                
                
                # Prepare evaluator command with all input files
                eval_cmd = ['python3', 'feedback.py']
                eval_cmd.extend(sorted(group_files))  # Add all input files
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
    
    def __init__(self, models_to_use: List[str], dataset: Dict, timeout: int = 10, temperature: float = 0.7, stream: bool = False, history_rounds: int = None):
        self.timeout = timeout
        self.temperature = temperature
        self.stream = stream
        self.conversation_history = {}  # Store conversation history for each model
        self.dataset = dataset
        self.history_rounds = history_rounds  # Number of previous rounds to keep in history
        
        # Initialize clients for each provider
        self.clients = {}
        
        # Map of model prefixes to their API configurations
        self.model_configs = {
            "openai": {
                "api_key": os.getenv('OPENAI_API_KEY'),
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 65536 # 16384
            },
            "deepseek": {
                "api_key": os.getenv('DEEPSEEK_API_KEY'),
                "base_url": "https://api.deepseek.com/v1",
                "max_tokens": 8192
            },
            "anthropic": {
                "api_key": os.getenv('ANTHROPIC_API_KEY'),
                "base_url": "https://api.anthropic.com/v1",
                "max_tokens": 64000
            },
            "google": {
                "api_key": os.getenv('GOOGLE_API_KEY'),
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "max_tokens": 65536
            },
            "openrouter": {
                "api_key": os.getenv('OPENROUTER_API_KEY'),
                "base_url": "https://openrouter.ai/api/v1",
                "max_tokens": 32768
            },
            "alibaba": {
                "api_key": os.getenv('DASHSCOPE_API_KEY'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "max_tokens": 16384
            },
            "sambanova": {
                "api_key": os.getenv('SAMBANOVA_API_KEY'),
                "base_url": "https://api.sambanova.ai/v1",
                "max_tokens": 32768
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
        # Map model prefixes to their company providers
        model_to_provider = {
            "gpt": "openai",
            "o1": "openai",
            "o3": "openai",
            "o4": "openai",
            "deepseek": "deepseek",
            "claude": "anthropic",
            "gemini": "google",
            "openrouter": "openrouter",
            "qwen": "alibaba",
            "llama": "meta",
            "sambanova": "sambanova"
        }
        
        if model.startswith("openrouter/"):
            return "openrouter"
        if model.startswith("sambanova/"):
            return "sambanova"
            
        for prefix, provider in model_to_provider.items():
            if model.startswith(prefix):
                return provider
                
        return None

    def _get_actual_model_name(self, model: str) -> str:
        """Get the actual model name to use with the API."""
        if model.startswith("openrouter/"):
            return model.replace("openrouter/", "", 1)
        if model.startswith("sambanova/"):
            return model.replace("sambanova/", "", 1)
        
        # Remove reasoning_effort suffix if present (e.g., :low, :medium, :high)
        if ":" in model and any(model.startswith(prefix) for prefix in ["o1", "o3", "o4"]):
            return model.split(":")[0]
            
        return model

    def _get_base_model_name(self, model: str) -> str:
        """Extracts the base model name from a model identifier for file paths."""
        # Remove openrouter/ prefix if present
        if model.startswith("openrouter/"):
            model = model.replace("openrouter/", "", 1)
        if model.startswith("sambanova/"):
            model = model.replace("sambanova/", "", 1)
        
        # Remove provider prefix if present (e.g., deepseek/, anthropic/, etc.)
        parts = model.split("/")
        if len(parts) > 1:
            model = parts[-1]
        
        # Remove any suffixes after : (e.g., :free, :latest, :low, :medium, :high, etc.)
        model = model.split(":")[0]
        
        return model

    def _parse_reasoning_effort(self, model: str) -> str:
        """Parse the reasoning_effort from the model name if specified."""
        if ":" in model and any(model.startswith(prefix) for prefix in ["o1", "o3", "o4"]):
            effort = model.split(":", 1)[1].lower()
            # Validate that the effort is one of the allowed values
            if effort in ["low", "medium", "high"]:
                return effort
            else:
                raise ValueError(f"Invalid reasoning_effort '{effort}' for model {model}. Must be one of: 'low', 'medium', 'high'")
        # Default to medium if not specified
        return "medium"

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
                     solution_dir: Path = None) -> str:
        """Formats the problem description into a prompt for the LLM."""
        prompt = ""
        
        # Only include problem description and example program in the first iteration
        if iteration == 0:
            # Format the problem information
            problem_info = self._format_problem_info(problem_desc)
            
            # Get the example program
            example_program = self._get_example_program(problem_desc)
            
            prompt = f"# Problem Information\n{problem_info}\n\n"
            prompt += f"# Program Template\n{example_program}\n"
        else:
            # For later iterations, just include the feedback and improvement guidance
            prompt = f"""
# Feedback from Previous Iteration (Iteration {iteration-1})
These are the test cases and results from the previous iteration:
"""

            # Get file paths from the dataset
            if not self.dataset or "train" not in self.dataset:
                prompt += f"\nNo test cases found in the dataset for {problem_desc['name']}\n\n"
            else:
                file_paths = self.dataset["train"]["file_path"]
                if not file_paths:
                    prompt += f"\nNo test cases found in the dataset for {problem_desc['name']}\n\n"
                else:
                    # Group files by base name
                    file_groups = {}
                    for file_path in file_paths:
                        base_name = Path(file_path).stem
                        if base_name not in file_groups:
                            file_groups[base_name] = []
                        file_groups[base_name].append(file_path)
                    
                    # For each group of files, show them together
                    for base_name, group_files in file_groups.items():
                        prompt += f"\n## Test Case: {base_name}\n\n"
                        
                        # Show all input files in this group
                        for input_file in sorted(group_files):
                            # in a multi-round conversation setting, only show the input file content in the first round
                            if iteration == 1:
                                try:
                                    with open(input_file, 'r') as f:
                                        content = f.read()
                                        # Limit to first 50 lines if content is too large
                                        lines = content.split('\n')
                                        if len(lines) > 50:
                                            lines = lines[:50]
                                            lines.append('... (truncated)')
                                        # Truncate long lines
                                        MAX_LINE_LENGTH = 100
                                        truncated_lines = []
                                        for line in lines:
                                            if len(line) > MAX_LINE_LENGTH:
                                                truncated_lines.append(line[:MAX_LINE_LENGTH] + '... (truncated)')
                                            else:
                                                truncated_lines.append(line)
                                        content = '\n'.join(truncated_lines)
                                    prompt += f"**Input File: {Path(input_file).name}**\n```\n{content}\n```\n\n"
                                except Exception as e:
                                    prompt += f"**Input File: {Path(input_file).name}** Error reading file: {str(e)}\n\n"
                            else:
                                prompt += f"**Input File: {Path(input_file).name}** (Shown in the previous iteration)\n"
                        
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
            "temperature": self.temperature,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON file
        with open(json_file, 'w') as f:
            json.dump(api_info, f, indent=2)
            
    def get_program(self, 
                    problem_desc: Dict[str, str], 
                    model: str = "gpt-4-turbo-preview",
                    iteration: int = 0,
                    solution_dir: Path = None) -> str:
        """Gets a program from the specified LLM using unified OpenAI API format."""
        prompt = self.format_prompt(problem_desc, iteration, solution_dir)
        
        # Initialize conversation history for this model if not exists
        if model not in self.conversation_history:
            with open(Path(os.getcwd()) / "prompt.md", 'r') as f:
                system_prompt = f.read().replace("{TIMEOUT}", str(self.timeout))
            self.conversation_history[model] = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Log the system prompt if we have a log file
            if hasattr(self, 'current_log_file'):
                with open(self.current_log_file, 'a') as f:
                    f.write("\nSYSTEM PROMPT:\n")
                    f.write(system_prompt)
                    f.write("\n\n")
        
        # Add the current prompt to conversation history
        self.conversation_history[model].append({"role": "user", "content": prompt})
        
        # If history_rounds is set, trim the conversation history to keep only the specified number of rounds
        if self.history_rounds is not None and iteration > 0:
            # Keep system prompt, first iteration user prompt, and the last history_rounds * 2 messages
            # First iteration user prompt is at index 1 (after system prompt)
            first_iteration_prompt = self.conversation_history[model][1]
            messages_to_keep = 1 + (self.history_rounds * 2)  # 1 for system prompt, 2 for each round
            if len(self.conversation_history[model]) > messages_to_keep:
                # Keep system prompt, first iteration prompt, and the most recent messages
                self.conversation_history[model] = (
                    [self.conversation_history[model][0]] +  # Keep system prompt
                    [first_iteration_prompt] +  # Keep first iteration prompt
                    self.conversation_history[model][-messages_to_keep+2:]  # Keep most recent messages
                )
        
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
        max_tokens = self.model_configs[provider]["max_tokens"]
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": actual_model,
                "messages": self.conversation_history[model],
                "stream": self.stream
            }
            
            # Add model-specific parameters
            if actual_model.startswith("o1") or actual_model.startswith("o3") or actual_model.startswith("o4"):
                # O-series models use max_completion_tokens instead of max_tokens
                api_params["max_completion_tokens"] = max_tokens
                reasoning_effort = self._parse_reasoning_effort(model)
                api_params["reasoning_effort"] = reasoning_effort
                # O-series models only support temperature=1 (default), so we omit it
            else:
                # Non O-series models use max_tokens and custom temperature
                api_params["max_tokens"] = max_tokens
                api_params["temperature"] = self.temperature
            
            # Make API call using unified OpenAI format
            response = client.chat.completions.create(**api_params)
            
            if self.stream:
                # Initialize variables for collecting streamed content and token usage
                raw_response = ""
                prompt_tokens = 0
                completion_tokens = 0
                
                # Process the streaming response
                for chunk in response:
                    if chunk.choices:
                        # Add the delta content to our response
                        if hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                            if content:
                                raw_response += content
                                # Log the streamed content if we have a log file
                                if hasattr(self, 'current_log_file'):
                                    with open(self.current_log_file, 'a') as f:
                                        f.write(content)
                    else:
                        # This is the final chunk with usage information
                        if hasattr(chunk, 'usage'):
                            prompt_tokens = chunk.usage.prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens
            else:
                # Handle non-streaming response
                if not response or not response.choices:
                    error_msg = f"Error: Invalid response received from model {model}!!! Exiting..."
                    logger.error(error_msg)
                    sys.exit(1)
                    
                raw_response = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
            
            if not raw_response or not raw_response.strip():
                error_msg = f"Error: Empty response received from model {model}!!! Exiting..."
                logger.error(error_msg)
                sys.exit(1)
                
            # Add assistant's response to conversation history
            self.conversation_history[model].append({"role": "assistant", "content": raw_response})
            
        except Exception as e:
            error_msg = f"Error: Failed to get response from model {model}: {str(e)}!!! Exiting..."
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
                f.write(f"\nAPI Call Time: {api_time:.2f} seconds\n")
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
            f.write(f"Git Commit ID: {get_git_commit_id()}\n")
            f.write(f"Problem: {problem_desc['name']}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Max Iterations: {max_iterations}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for iteration in range(max_iterations):
            try:
                logger.info(f"Getting program from {model} (iteration {iteration+1}/{max_iterations})")
                with open(log_file, 'a') as f:
                    f.write(f"\n{'=' * 40} ITERATION {iteration} {'=' * 40}\n\n")
                
                # Get program from LLM
                raw_response = self.get_program(
                    problem_desc, 
                    model, 
                    iteration,
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
                
                # Sleep between iterations to avoid hitting rate limits
                if iteration < max_iterations - 1:  # Don't sleep after the last iteration
                    # Different sleep times based on model provider
                    provider = self._get_provider(model)
                    
                    if provider == "anthropic":  # Claude models have stricter rate limits
                        sleep_time = 60  # 60 seconds for Claude models
                        logger.info(f"Sleeping for {sleep_time} seconds to avoid exceeding Claude API rate limits...")
                    else:
                        sleep_time = 2  # Short delay for other models
                        logger.info(f"Short delay of {sleep_time} seconds before next iteration...")
                    
                    with open(log_file, 'a') as f:
                        f.write(f"Sleeping for {sleep_time} seconds before next iteration...\n\n")
                    
                    time.sleep(sleep_time)
                
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

def generate_summary_table(results_data):
    """Generate a formatted summary table comparing all models' performance."""
    # Get all unique metrics
    all_metrics = set()
    for model_data in results_data.values():
        all_metrics.update(model_data.keys())
    
    # Sort metrics to ensure consistent order
    sorted_metrics = sorted(all_metrics)
    
    # Calculate column widths
    model_width = max(len(model) for model in results_data.keys())
    metric_width = max(len(metric) for metric in sorted_metrics)
    
    # Create header
    header = f"{'Metric':<{metric_width}} | " + " | ".join(f"{model:<{model_width}}" for model in results_data.keys())
    separator = "-" * len(header)
    
    # Create rows
    rows = []
    for metric in sorted_metrics:
        row = f"{metric:<{metric_width}} | "
        row += " | ".join(f"{results_data[model].get(metric, 'N/A'):<{model_width}}" for model in results_data.keys())
        rows.append(row)
    
    # Combine all parts
    table = [header, separator] + rows
    
    return "\n".join(table)

def parse_best_results(json_file):
    """Parse best_results.json file to extract performance metrics."""
    metrics = {}
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
            
        # Calculate average cost
        costs = [float(data['cost']) for data in results.values() if data['cost'] != float('inf')]
        if costs:
            metrics['Avg Cost'] = f"{sum(costs)/len(costs):.4f}"
            
    except Exception as e:
        logger.error(f"Error parsing best results {json_file}: {str(e)}")
    
    return metrics

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Solver Agent for optimization problems')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=[
                            "deepseek-chat",
                            "deepseek-reasoner",
                            "qwen3-235b-a22b",
                            "openrouter/meta-llama/llama-4-maverick:free",
                            "gemini-2.5-flash-preview-04-17",
                            "gemini-2.5-pro-preview-05-06",
                            "sambanova/Meta-Llama-3.3-70B-Instruct",
                            "o4-mini:high",
                            "claude-3-7-sonnet-20250219"
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
    
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming output from LLM (default: False)')
    
    parser.add_argument('--history_rounds', type=int, default=None,
                        help='Number of previous rounds to keep in conversation history (default: None, keep all history)')

    parser.add_argument('--num_cores', type=int, default=8,
                        help='Number of CPU cores to use for program execution (default: 8)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    load_dotenv()
    # Get token from environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    # Log in with your HF access token
    login(token=token)
    workspace_root = os.getcwd()
    
    # Initialize components
    problem_reader = ProblemReader(workspace_root)
    
    # Load dataset first
    dataset = load_dataset(HF_REPO_ID, name=args.problem, data_dir="_datasets", token=token, trust_remote_code=True)  # ignore cached old copy
    print(f"Loaded dataset from HuggingFace: {HF_REPO_ID}/{args.problem}")
    
    # Initialize LLM interface with dataset and history_rounds
    llm_interface = LLMInterface(args.models, dataset, args.timeout, args.temperature, args.stream, args.history_rounds)
    
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
            
            # Dictionary to store results for each model
            all_model_results = {}
            
            # Get solutions from each model
            for model in args.models:
                # Get base model name for directory
                base_model_name = llm_interface._get_base_model_name(model)
                
                # Create the solution directory for this model
                workspace_root = Path(os.getcwd())
                solution_dir = workspace_root / "llm_solutions" / timestamp / problem_desc['name'] / base_model_name
                solution_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize program executor with the solution directory
                executor = ProgramExecutor(workspace_root / problem_desc['name'], solution_dir, dataset, args.timeout, args.num_cores)
                
                # Get iterative program
                logger.info(f"Getting iterative program from {model}")
                program, final_iteration = llm_interface.get_iterative_program(
                    problem_desc, 
                    model,
                    executor,
                    solution_dir,
                    args.iterations
                )
                
                # After each model finishes, run collect_results.py for that model
                logger.info(f"Model {model} finished. Running collect_results.py...")
                llm_solutions_dir = workspace_root / "llm_solutions" / timestamp / problem_desc['name'] / base_model_name
                dataset_path = workspace_root / "_datasets" / args.problem
                
                # Run collect_results.py with the appropriate arguments
                collect_cmd = [
                    "python3",
                    "scripts/collect_results.py",
                    str(llm_solutions_dir),
                    str(dataset_path),
                    "--timeout",
                    str(args.timeout),
                    "--num_cores",
                    str(args.num_cores)
                ]
                
                try:
                    subprocess.run(collect_cmd, check=True)
                    logger.info(f"Successfully ran collect_results.py for {model}")
                    
                    # Parse results from best_results.json
                    results_file = llm_solutions_dir / "best_results.json"
                    if results_file.exists():
                        all_model_results[model] = parse_best_results(results_file)
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running collect_results.py for {model}: {str(e)}")
            
            # After all models finish, display the summary table
            if all_model_results:
                logger.info("\nFinal Performance Summary:")
                logger.info("=" * 100)
                summary_table = generate_summary_table(all_model_results)
                logger.info(summary_table)
                logger.info("=" * 100)
                
                # Save summary to a file
                summary_file = workspace_root / "llm_solutions" / timestamp / problem_desc['name'] / "performance_summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"Performance Summary for {problem_desc['name']}\n")
                    f.write("=" * 100 + "\n")
                    f.write(summary_table)
                    f.write("\n" + "=" * 100 + "\n")
                logger.info(f"Summary saved to {summary_file}")
                
        except Exception as e:
            logger.error(f"Error processing {problem_folder}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
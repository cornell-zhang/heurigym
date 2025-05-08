# HeuriGen

Recent advancements in Large Language Models (LLMs) have expanded their capabilities in reasoning and agent-based tasks, yet existing benchmarks fail to adequately assess these skills. Current evaluations (e.g., AIME, MATH-500) focus on closed-ended problems with high LLM performance (Pass@1 > 80%), leaving critical gaps:
* **Test-Time Compute & Agent**: Methods like tool usage, multi-step reasoning, and iterative refinement (e.g., DeepSeek-R1, OpenAI-o3) remain underexplored.
* **Practical Creativity**: Human experts often struggle with open-ended, real-world optimization problems where optimal solutions are unknown. LLMs' potential to surpass human ingenuity in these domains is untested.

We construct a new benchmark HeuriGen that has
* **Open-Ended Goals**: Clear optimization objectives with multiple viable solution paths.
* **Real-World Impact**: Domains where improved solutions yield significant societal or industrial benefits.
* **Human-Optimal Gap**: Problems where existing expert solutions are suboptimal compared to theoretical limits (e.g., NP-hard problems).

| Problem | Type |
| :--: | :--: |
| [Operator Scheduling](operator_scheduling) | Scheduling |
| [E-Graph Extraction](e-graph-extraction) | Term rewriting |
| [Pick-up and Delivery](pdptw) | TSP |
| [Technology Mapping](technology_mapping) | Covering |
| [Global Routing](global_routing) | Routing |
| [Protein Sequence Design](protein_sequence_design) | Network Flow |


## Problem Setup
To add a new problem to the benchmark suite, you need to create a new folder in the `problems` directory.
The folder should have two subfolders:
* `dataset`: A folder for problem instances
* `program`: A folder for the program template

You can copy the `template` folder as a starting point. There are several files you need to implement or include:
* `README.md`: Problem description, formalization, and input/output format
* `solver.py`: A template solver function for LLM to fill in. Feel free overload the `solve` function by copying it to your problem folder.
* `verifier.py`: After LLM provides a solution, the verifier will check if the solution is valid. Please implement the `verify` function in this file.
* `evaluator.py`: After the solution is verified, the evaluator will calculate the cost of the solution. Please implement the `evaluate` function in this file.

## LLM Solver Agent

This agent reads optimization problem descriptions from README files in your workspace and uses various Language Models (LLMs) to generate solution approaches and strategies.

### Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with the API keys for the models you want to use:
```
# Required only if using OpenAI models (e.g., gpt-4-turbo-preview)
OPENAI_API_KEY=your_openai_key_here

# Required only if using Anthropic models (e.g., claude-3-opus-20240229)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required only if using DeepSeek models (e.g., deepseek-chat, deepseek-coder)
DEEPSEEK_API_KEY=your_deepseek_key_here
```

3. (Optional) Customize the prompt template in `prompt.md`

### Usage

Run the agent with default settings (uses all supported models):
```bash
python llm_solver_agent.py
```

#### Command Line Arguments

The agent supports the following command line arguments:

```bash
python llm_solver_agent.py [options]
```

Options:
- `--models MODEL1 MODEL2 ...`: List of models to use (default: all supported models)
- `--iterations N`: Maximum number of iterations for each model (default: 3)
- `--problem PROBLEM_NAME`: Specific problem to solve (folder name)

Examples:
```bash
# Use only DeepSeek-V3 (requires only DEEPSEEK_API_KEY)
python llm_solver_agent.py --models gpt-4-turbo-preview

# Use GPT-4 and DeepSeek-V3 (requires OPENAI_API_KEY and DEEPSEEK_API_KEY)
python llm_solver_agent.py --models gpt-4-turbo-preview deepseek-chat

# Run 5 iterations for each model
python llm_solver_agent.py --iterations 5

# Solve only the "operator_scheduling" problem
python llm_solver_agent.py --problem operator_scheduling
```

The agent will:
1. Scan all directories in the workspace for README.md files
2. Parse the problem descriptions
3. Request solutions from configured LLMs with iterative improvement
4. Save solutions in the `llm_solutions` directory


### Analysis Scripts

After running the LLM solver agent, you can use the following script to analyze the results:

#### Collect Results and Analyze Errors

The `collect_results.py` script analyzes all solutions, finds the best results, and performs error analysis:

```bash
python scripts/collect_results.py <llm_solutions_dir> <dataset_path> [--timeout TIMEOUT]
```

Arguments:
- `llm_solutions_dir`: Directory containing the LLM solutions for *a specific model*
- `dataset_path`: Path to the dataset directory
- `--timeout TIMEOUT`: (Optional) Timeout in seconds for each solution (default: 10)

The script will:
1. Run all optimizations:
   - Find all `run.py` files in iteration directories
   - Run each solution with the specified dataset
   - Compare results and identify the best solution for each test case
   - Generate a summary table of best results
2. Analyze errors:
   - Find all iteration directories
   - Analyze `.cost` files in each iteration's output directory
   - Classify errors into categories:
     - Stage I: Execution Error
     - Stage II: Output Error
     - Stage III: Verification Error
     - Stage IV: No Error
   - Generate error statistics by iteration and test case
3. Save results:
   - Best results saved to `best_results.json`
   - Error analysis saved to `error_summary.json`

The script provides a comprehensive analysis of both solution quality and error patterns, helping to identify:
- Best performing solutions for each test case
- Common error patterns and their frequencies
- Areas for improvement in the solution generation process

### Prompt Template

The agent uses a customizable prompt template from `prompt.md`. This template includes placeholders that are replaced with problem-specific information:

- `{PROBLEM}`: Replaced with the problem description, formalization, and input/output format
- `{EXAMPLE_PROGRAM}`: Replaced with a basic program template for the solution


### Iterative Improvement Process

The agent implements an iterative improvement process for each model:

1. **Initial Solution**: The first iteration generates a baseline solution
2. **Refinement**: Subsequent iterations improve upon the previous solution by:
   - Addressing weaknesses in the previous approach
   - Introducing more advanced algorithms
   - Considering edge cases
   - Providing more detailed implementation guidance
3. **Maximum Iterations**: By default, each model performs 3 iterations

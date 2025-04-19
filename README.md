# OptBench

Recent advancements in Large Language Models (LLMs) have expanded their capabilities in reasoning and agent-based tasks, yet existing benchmarks fail to adequately assess these skills. Current evaluations (e.g., AIME, MATH-500) focus on closed-ended problems with high LLM performance (Pass@1 > 80%), leaving critical gaps:
* **Test-Time Compute & Agent**: Methods like tool usage, multi-step reasoning, and iterative refinement (e.g., DeepSeek-R1, OpenAI-o3) remain underexplored.
* **Practical Creativity**: Human experts often struggle with open-ended, real-world optimization problems where optimal solutions are unknown. LLMs' potential to surpass human ingenuity in these domains is untested.

We try to construct a new benchmark OptBench that has
* **Open-Ended Goals**: Clear optimization objectives with multiple viable solution paths.
* **Real-World Impact**: Domains where improved solutions yield significant societal or industrial benefits.
* **Human-Optimal Gap**: Problems where existing expert solutions are suboptimal compared to theoretical limits (e.g., NP-hard problems).

| Problem | Type |
| :--: | :--: |
| [Operator Scheduling](operator_scheduling) | Graph scheduling |

# LLM Solver Agent

This agent reads optimization problem descriptions from README files in your workspace and uses various Language Models (LLMs) to generate solution approaches and strategies.

## Features

- Automatically reads and parses problem descriptions from README.md files
- Supports multiple LLM providers (OpenAI GPT-4, Anthropic Claude, DeepSeek)
- Implements iterative solution improvement process
- Uses customizable prompt templates
- Saves solutions for future reference
- Handles markdown formatting and section extraction
- Avoids duplicate solution requests
- Configurable via command line arguments

## Setup

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

## Usage

Run the agent with default settings (uses all supported models):
```bash
python llm_solver_agent.py
```

### Command Line Arguments

The agent supports the following command line arguments:

```bash
python llm_solver_agent.py [options]
```

Options:
- `--models MODEL1 MODEL2 ...`: List of models to use (default: all supported models)
- `--iterations N`: Maximum number of iterations for each model (default: 3)
- `--skip-existing`: Skip problems that already have solutions
- `--problem PROBLEM_NAME`: Specific problem to solve (folder name)

Examples:
```bash
# Use only GPT-4 (requires only OPENAI_API_KEY)
python llm_solver_agent.py --models gpt-4-turbo-preview

# Use GPT-4 and Claude (requires OPENAI_API_KEY and ANTHROPIC_API_KEY)
python llm_solver_agent.py --models gpt-4-turbo-preview claude-3-opus-20240229

# Run 5 iterations for each model
python llm_solver_agent.py --iterations 5

# Solve only the "operator_scheduling" problem
python llm_solver_agent.py --problem operator_scheduling

# Skip problems that already have solutions
python llm_solver_agent.py --skip-existing
```

The agent will:
1. Scan all directories in the workspace for README.md files
2. Parse the problem descriptions
3. Request solutions from configured LLMs with iterative improvement
4. Save solutions in the `llm_solutions` directory

Solutions are saved as text files with the naming format: `{problem_name}_{model}_solution.txt`

## Prompt Template

The agent uses a customizable prompt template from `prompt.md`. This template includes placeholders that are replaced with problem-specific information:

- `{PROBLEM}`: Replaced with the problem description, formalization, and input/output format
- `{EXAMPLE_PROGRAM}`: Replaced with a basic program template for the solution

If `prompt.md` is not found, the agent will use a default template.

## Iterative Improvement Process

The agent implements an iterative improvement process for each model:

1. **Initial Solution**: The first iteration generates a baseline solution
2. **Refinement**: Subsequent iterations improve upon the previous solution by:
   - Addressing weaknesses in the previous approach
   - Introducing more advanced algorithms
   - Considering edge cases
   - Providing more detailed implementation guidance
3. **Maximum Iterations**: By default, each model performs 3 iterations

## Supported Models

The agent supports the following models:
- OpenAI: GPT-4 Turbo
- Anthropic: Claude 3 Opus
- DeepSeek: DeepSeek Chat and DeepSeek Coder (using OpenAI-compatible API)

## API Integration

The agent uses the following API integration methods:
- OpenAI SDK for OpenAI models
- Anthropic SDK for Claude models
- OpenAI SDK with DeepSeek's base URL for DeepSeek models (DeepSeek uses an OpenAI-compatible API)

## Solution Format

For each problem, the LLMs will provide:
1. A high-level approach to solve the problem
2. Key algorithms or techniques to use
3. Potential optimization strategies
4. Implementation considerations

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── llm_solver_agent.py
├── prompt.md
├── .env
├── llm_solutions/
│   ├── problem1_gpt-4-turbo-preview_solution.txt
│   ├── problem1_claude-3-opus_solution.txt
│   ├── problem1_deepseek-chat_solution.txt
│   └── ...
└── problem_directories/
    ├── problem1/
    │   └── README.md
    └── problem2/
        └── README.md
```

## Error Handling

The agent includes robust error handling:
- Logs errors for individual problems without stopping the entire process
- Skips already processed problems
- Validates README file existence
- Handles API rate limits and timeouts
- Gracefully handles failures during iterative improvement

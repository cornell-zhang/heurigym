# HeuriGym

<p align="center">
    <a href="#-about">📙About</a> •
    <a href="#-problems">📚Problems</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-llm-solver-agent">🚀LLM Solver Agent</a> •
    <a href="#-contribute">🤝Contribute</a> •
    <a href="#-citation">📜Citation</a> •
</p>

## 📙 About

**Why HeuriGym?** 

Current LLM benchmarks fall into two categories with distinct limitations.

- **Ground-truth-based objective benchmarks** rely on closed-form questions (e.g., multiple-choice mathematics problems) that have become susceptible to rapid performance saturation. Widely used benchmarks (AIME, HumanEval, and GPQA Diamond) now exhibit ceiling effects with high LLM performance (Pass@1 > 80%). Moreover, their closed-end nature fails to reflect real-world problem-solving. 

- **Judge-preference-based subjective evaluations** assess model quality through pairwise comparisons by humans or LLM-based proxies, such as Chatbot Arena. However, this flexibility leads to high variance and reliability issues, which is critical for assessing technical tasks.


**HeuriGym** fills this gap with open-ended combinatorial optimization challenges that have:
- **Open-Ended Problems**: well-defined optimization objectives with large solution spaces. 
- **Real-World Impact**: domains where improved solutions yield significant societal or industrial benefits.
- **Gap to Expert Solutions**: metrics that measure the gap between LLM solutions and expert solutions provided.
- **Agentic Framework**: LLM generates heuristic solutions, receive execution feedback from a code environment, and iteratively refine their solutions.


## 📚 Problems

The initial release of the HeuriGym benchmark includes nine distinct optimization problems spanning four scientific and engineering domains. 


| Domain | Problem | Difficulty |
| :--: | :--: | :--: |
| Electronic Design Automation (EDA) | [Operator scheduling](operator_scheduling) | ★ |
| Electronic Design Automation (EDA) | [Technology mapping](technology_mapping) | ★★ |
| Electronic Design Automation (EDA) | [Global routing](global_routing) | ★★★ |
| Compilers | [E-graph extraction](egraph_extraction) | ★ |
| Compilers | [Intra-operator parallelism](intra_op_parallel) | ★★ |
| Computational Biology | [Protein sequence design](protein_sequence_design) | ★ |
| Computational Biology | [Mendelian error detection](pedigree) | ★★ |
| Logistics | [Airline crew pairing](crew_pairing) | ★★ |
| Logistics | [Pickup and delivery w/ time windows](pickup_delivery_time_windows) | ★★★ |


## 🔥 Quick Start

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Clone the repository:
```bash
git clone https://github.com/heurgym/heurgym.git
cd heurgym
```

3. Setup API keys:

```bash
# you need to have a HuggingFace token to download the dataset.
export HUGGINGFACE_TOKEN=<your_huggingface_key_here>
# If you are using Google models, you need to have a Google API key. 
export GOOGLE_API_KEY=<your_google_key_here>
```

4. Run the agent to solve the operator scheduling problem with Gemini 2.5 Pro: 
```bash
python llm_solver_agent.py --problem operator_scheduling \
                           --models gemini-2.5-pro-preview-05-06
```

5. Check the results in the `llm_solutions` directory.

Best results are saved in `best_results.json` and error analysis is saved in `error_summary.json`.



## 🚀 LLM Solver Agent

Create a `.env` file in the root directory with the API keys for the models you want to use:
```
# Required only if using models from OpenAI (e.g., o4-mini:high)
OPENAI_API_KEY=your_openai_key_here

# Required only if using models from Anthropic (e.g., claude-3-7-sonnet-20250219)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required only if using models from DeepSeek (e.g., deepseek-chat, deepseek-coder)
DEEPSEEK_API_KEY=your_deepseek_key_here

# Required only if using models from Google (e.g., gemini-2.5-flash-preview-04-17, gemini-2.5-pro-preview-05-06)
GOOGLE_API_KEY=your_google_key_here

# Required only if using models from OpenRouter (e.g., openrouter/meta-llama/llama-4-maverick)
OPENROUTER_API_KEY=your_openrouter_key_here

# Required only if using models from Alibaba (e.g., qwen3-235b-a22b)
DASHSCOPE_API_KEY=your_alibaba_key_here
```

Also note that you need to have a HuggingFace token to download the dataset.
```
HUGGINGFACE_TOKEN=your_huggingface_key_here
```

### Usage

Run the agent to solve the operator scheduling problem with Gemini 2.5 Pro: 
```bash
# Requires GOOGLE_API_KEY
python llm_solver_agent.py --problem operator_scheduling \
                           --models gemini-2.5-pro-preview-05-06
```

Run the agent to solve egraph extraction problem with Claude 3.7 Sonnet:
```bash
# Requires ANTHROPIC_API_KEY
python llm_solver_agent.py --problem egraph_extraction \
                           --models claude-3-7-sonnet-20250219
```

Run the agent to solve the airline crew pairing problem with o4-mini:high:
```bash
# Requires OPENAI_API_KEY
python llm_solver_agent.py --problem crew_pairing \
                           --models o4-mini:high
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
- `--timeout TIMEOUT`: Timeout in seconds for program execution (default: 10)
- `--temperature TEMPERATURE`: Temperature for LLM generation (default: 0.0)
- `--stream`: Enable streaming output from LLM (default: False, but True for Qwen models)
- `--history_rounds N`: Number of previous rounds to keep in conversation history (default: None, keep all history)
- `--num_cores N`: Number of CPU cores to use for program execution (default: 8)
- `--few_shots N`: Number of training examples to provide to LLMs (default: None, use all examples)



The agent will:
1. Scan all directories in the workspace for README.md files
2. Parse the problem descriptions
3. Request solutions from configured LLMs with iterative improvement
4. Save solutions in the `llm_solutions` directory
5. Collect results, analyze all solutions, finds the best results, and performs error analysis. Best results are saved in `best_results.json` and error analysis is saved in `error_summary.json`.






## 🤝 Contribute
We welcome contributions to the HeuriGym benchmark! 

To add a new problem to the benchmark suite, you need to create a new folder in the `problems` directory.
The folder should have two subfolders:
* `dataset`: A folder for problem instances
* `program`: A folder for the program template

You can copy the `template` folder as a starting point. There are several files you need to implement or include:
* `README.md`: Problem description, formalization, and input/output format
* `solver.py`: A template solver function for LLM to fill in. Feel free overload the `solve` function by copying it to your problem folder.
* `verifier.py`: After LLM provides a solution, the verifier will check if the solution is valid. Please implement the `verify` function in this file.
* `evaluator.py`: After the solution is verified, the evaluator will calculate the cost of the solution. Please implement the `evaluate` function in this file.





## 📜 Citation
To be updated



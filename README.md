# HeuriGym

<p align="center">
    <a href="#-about">📙About</a> •
    <a href="#-problems">📚Problems</a> •
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-llm-solver-agent">🚀LLM Solver Agent</a> •
    <a href="#-contribute">🤝Contribute</a> •
    <a href="#-citation">📜Citation</a> •
</p>

## 📘 About

**HeuriGym** is a benchmark for evaluating how well LLMs generate and refine heuristics for real-world combinatorial optimization (CO) tasks through agentic, code-driven interaction.

### 🔍 Why HeuriGym?

Existing LLM benchmarks fall short:

- 🎯 **Closed-form tasks** (e.g., AIME, HumanEval): Saturated, too simplistic for real-world reasoning.
- 🤖 **Subjective evaluations** (e.g., Chatbot Arena): Noisy, inconsistent, and unreliable for technical tasks.


**HeuriGym** fills this gap with:

- 🧩 **Open-ended problems**: Well-defined objectives with large solution spaces.  
- 🤖 **Agentic interaction**: LLMs improve heuristics through feedback-driven code execution.  
- 📏 **Expert comparison metrics**: Measure both pass rate and quality relative to expert solutions.


Let LLMs think, code, and improve—just like real solvers.


## 📚 Problems

The initial release of the HeuriGym benchmark includes nine distinct optimization problems spanning four scientific and engineering domains. 


| Domain | Problem | Difficulty |
| :--: | :--: | :--: |
| EDA | [Operator scheduling](operator_scheduling) | ★ |
| EDA | [Technology mapping](technology_mapping) | ★★ |
| EDA | [Global routing](global_routing) | ★★★ |
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
git clone https://github.com/cornell-zhang/heurigym.git
cd heurigym
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
- `--history_rounds H`: Number of previous rounds to keep in conversation history (default: None, keep all history)
- `--num_cores C`: Number of CPU cores to use for program execution (default: 8)
- `--few_shots S`: Number of training examples to provide to LLMs (default: None, use all examples)



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
```bibtex
@article{chen2025heurigym,
    title={HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization}, 
    author={Hongzheng Chen and Yingheng Wang and Yaohui Cai and Hins Hu and Jiajie Li and Shirley Huang and Chenhui Deng and Rongjian Liang and Shufeng Kong and Haoxing Ren and Samitha Samaranayake and Carla P. Gomes and Zhiru Zhang},
    journal={arXiv preprint arXiv:2506.07972},
    year={2025}
}
```



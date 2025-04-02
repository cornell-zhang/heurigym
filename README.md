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

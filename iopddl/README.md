## Background

With machine learning models becoming larger and larger, multi-device execution across several slices or pods of hardware accelerators is imperative to meet latency and throughput requirements across training and serving workloads.
When executing an ML computation in a distributed manner, a key determinant of performance is the way that computation is sharded across the multiple devices.

Sharding a model to minimize exposed communication (e.g., using data parallelism or operator parallelism) can lead to significant performance gains. Unfortunately, given that model computations regularly contain hundreds or thousands of tensors and HLO ops, manually specifying how each should be executed across multiple devices is usually not feasible.  Your challenge is to design an algorithm capable of efficiently and effectively performing this task.

## Formalization
As input, the algorithm will accept a graph where nodes are associated with one or more *strategies*, and each strategy is annotated with a corresponding *node cost*.
In addition, certain pairs of nodes are connected by edges, and each pairwise strategy combination incurs a corresponding *edge cost*.  
Solutions should select one strategy per node that minimizes total cost as much as possible.
All costs are non-negative integers, and the total cost is the sum of all node costs and edge costs.

Finally, nodes also incur a strategy-specific memory *usage* over a fixed time interval.
The sum of usages for all nodes at any given (discrete) time point in a solution must not eclipse the *usage limit* for that benchmark.
This usage interval is fixed for each node no matter which strategy is selected.

Broadly speaking, the structural characteristics of a benchmark's graph topology and memory profile tend to vary wildly as a function of model task (e.g., training vs. serving) and model modality (e.g., language vs. vision).

### Clarifications Questions

***What level of precision will the values in the benchmarks have?***

 * All cost and memory usage values in the benchmarks can be represented as 64-bit integers, but beware of integer overflow when calculating total cost and/or total memory usage.

***How should usage intervals be interpreted?***

 * A node with usage interval `[lower, upper]` should be considered *half-open* with an *exclusive* upper bound. In other words, it will consume memory at time points {*lower, lower + 1, ..., upper − 1*}.  Hence, any nodes with an empty interval `[0, 0]` essentially consume no memory (but will still contribute node costs and possibly edge costs).

***Are all edge endpoints in the graph unique?***

 * Not necessarily; for a given pair of nodes `[pred, succ]`, there may be zero, one, or multiple edges that connect them.  For example, you'll find that in *Benchmark A*, there are two separate edges (#60 and #21438) that connect nodes #5 and #617.   The total cost between a pair of nodes would thus be the sum across all such edge costs.


## Input Format
The input is path to a JSON file encodes a graph-based optimization instance.
At the top level, the problem object names the instance and sets a global *usage_limit*.
Inside it, two sections describe the graph: *nodes* list contains each node by its time intervals, then details per-strategy costs (one or more alternative cost values) and usages (the corresponding resource consumption for each alternative).
The edges section mirrors this structure for connections between nodes:
nodes gives the pairs of endpoint indices, and costs provides a list of alternative costs for choosing specific implementations of the two incident nodes.
If the `pred` has `succ` have `k` and `l` strategies respectively, the cost is represented by a list flattened from a `k x l` matrix, where for the `i-th` strategy of `pred` and the `j-th` strategy of `succ`, the cost is given by `(i*l+j)-th` entry in the list.
Altogether, the format compactly captures multiple design choices for every node and edge, along with their costs and resource footprints, enabling algorithms to search for a minimum-cost solution that respects the overall usage budget.
The input is gaunteed to have at least one solution.


### Example
```
{
  "problem": {
    "name": "example",
    "nodes": {
      "intervals": [
        [30, 70],
        [40, 70],
        [50, 120],
        [110, 140],
        [110, 150]
      ],
      "costs": [
        [15],
        [55, 65],
        [25, 45, 35],
        [85, 75],
        [95]
      ],
      "usages": [
        [10],
        [25, 25],
        [15, 20, 15],
        [10, 10],
        [15]
      ]
    },
    "edges": {
      "nodes": [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 4]
      ],
      "costs": [
        [30, 40],
        [50, 10, 40],
        [90, 10, 20, 80],
        [60, 20, 30],
        [70, 60]
      ]
    },
    "usage_limit": 50
  }
}
```

## Output Format

The output should be a comma-delimited list of node strategy indices (0-based) enclosed in brackets:

```
[0, 0, 2, 1, 0]
```



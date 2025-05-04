# Technology Mapping for LUT-based FPGA


## Background

Technology mapping is a critical stage in the FPGA design flow where a technology-independent logic network is transformed into a network of lookup tables (LUTs). A K-input lookup table (K-LUT) is a basic programmable logic element in FPGAs that can implement any Boolean function with up to K inputs. The quality of technology mapping directly impacts the area, performance, and power consumption of the final FPGA implementation.

The technology mapping process involves covering a Boolean network with K-LUTs in a way that optimizes specific design metrics. Traditional approaches include cut enumeration, where potential K-feasible cuts (subgraphs with at most K inputs) are identified and selected to form an optimal cover of the network. The mapping process must ensure that the resulting network correctly implements the original logic function while minimizing resource usage.

Here our optimization target is to minimize the number of LUTs, which represents the area of the mapped logic network. This is an NP-hard problem. 

In this problem, we specify K = 6. 

## Formalization

Consider a directed acyclic graph (DAG) $G = (V, E)$ representing a Boolean network, where each node $v \in V$ represents a logic gate, and each edge $(u, v) \in E$ indicates that the output of node $u$ is an input to node $v$. Let $PI$ denote the set of primary inputs and $PO$ denote the set of primary outputs in the network.

A cut $C(v)$ of a node $v$ is a set of nodes such that every path from a primary input to $v$ passes through at least one node in $C(v)$. A cut is K-feasible if $|C(v)| \leq K$, meaning it has at most K nodes.

For each node $v$, let $Cuts(v)$ be the set of all K-feasible cuts of node $v$. Each cut $c \in Cuts(v)$ represents a potential implementation of the logic function at node $v$ using a single K-LUT.

The technology mapping problem is to select a cut for each node $v \in V$ such that:
1. The selected cuts form a valid cover of the network
2. The total number of LUTs used is minimized

Formally, we define a mapping solution $M$ as a function that assigns to each node $v \in V$ a cut $M(v) \in Cuts(v)$. The cost of a mapping solution, denoted as $Cost(M)$, is the number of LUTs required to implement the network:

$Cost(M) = \sum_{v \in V'} 1$

where $V'$ is the set of nodes that are actually used in the final implementation (nodes whose outputs are either primary outputs or inputs to other selected LUTs).

The K-LUT technology mapping problem can be formulated as: $\min_{M} Cost(M)$. 

Here we specify K = 6 in our implmentation. 


## Input Format

The input logic network is specified using the Berkeley Logic Interchange Format (BLIF). BLIF is a human-readable text format for describing logic circuits at the gate level. Here's several examples of some simple circuits in BLIF format:

```blif
# Circuit 1. Lines starting with '#' are comments. 
.model c1
.inputs 1 2 3 6 7
.outputs 22 23
.names 1 3 new_10
11 0
.names 3 6 new_11
11 1
.names 2 new_11 new_16
1- 0
.names new_11 7 new_19
11 0
.names new_10 new_16 22
00 1
.names new_16 new_19 23
-1 0
.end
```

```blif
# Circuit 2. Lines starting with '#' are comments. 
.model c2
.inputs 1 2 3 6 7
.outputs 22 23
.names 2 7 new_11 23
011 1
100 1
101 1
110 1
111 1
.names new_16 1 3 22
000 1
001 1
010 1
011 1
111 1
.names 6 2 3 new_16
000 1
001 1
100 1
101 1
.names 6 3 new_11
11 1
.end
```


In this format:
- `.model` specifies the name of the circuit
- `.inputs` lists the primary inputs
- `.outputs` lists the primary outputs
- `.names` blocks define the logic functions for internal nodes and outputs. A logic function is declared as follows: 
    ```
    .names <input-1> <input-2> ... <input-N> <output>
    <single-output-cover>
    ```
    where `<input-1>`, `<input-2>`, ..., `<input-N>` are the inputs to the logic function, and `<output>` is the output of the function. The `<single-output-cover>` is a truth table that specifies the output for each combination of inputs.

- The rows following a `.names` line define the truth table in a "single-output-cover" format:
  - The format uses {0, 1, -} in the n-bit wide "input plane" and {0, 1} in the 1-bit wide "output plane"
  - If the output plane is all 1's (omitted in the file), the first n columns represent input patterns that produce output 1, all other input patterns produce 0
  - For rows with explicit output 0 (like in the example): only these input patterns produce 0, all others produce 1
  - The symbol "-" represents don't care terms
  - There can be multiple rows for a single gate, each representing a different input pattern that produces the specified output
  - A sample logic function with 4 inputs and 1 output: 
    ```
    .names a b c d out
    1--0 1
    -1-1 1
    0-11 1
    ```
    The translation of the above sample logic function into a sum-of-products notation is: `output = (ad') + (bd) + (a'cd)`


- Special cases:
  - To assign the constant "1" to some logic gate `j`, use the following construct: 
    ```
    .names j
    1
    ```
  - To assign the constant "0" to some logic gate `j`, use the following construct: 
    ```
    .names j
    ```


## Output Format

The output of the technology mapping is also provided in BLIF format, but with the logic network implemented using only K-LUTs. Each LUT is represented by a `.names` block with at most K inputs. The output BLIF preserves the primary inputs and outputs of the original network.

You should ensure the truth tables for the LUTs are derived correctly from the original logic functions. 

An example output of a simple circuit after mapping to 3-LUTs (K=3) is as follows: 

```blif
# A simple circuit that mapped to 3-LUTs
.model c3
.inputs 1 2 3 6 7
.outputs 22 23
.names 1 3 new_n10 22
--1 1
11- 1
.names 6 3 2 new_n10
-01 1
0-1 1
.names 7 2 new_n12 23
-10 1
1-0 1
.names 3 6 new_n12
11 1
.end
```

An example output of a simple circuit after mapping to 6-LUTs (K=6) is as follows: 

```blif
# A simple circuit that mapped to 6-LUTs
.model c4
.inputs 1 2 3 6 7
.outputs 22 23
.names 6 1 3 2 22
--01 1
-11- 1
0--1 1
.names 6 3 2 7 23
--00 0
11-- 0
.end
```

In this output:
- Each `.names` block corresponds to a LUT in the final implementation
- Each LUT has at most K inputs
- The network should correctly implements the functionality of the original circuit


## References
1. J. Cong and Y. Ding, "FlowMap: An Optimal Technology Mapping Algorithm for Delay Optimization in Lookup-Table Based FPGA Designs," IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 1994.
2. A. Mishchenko, S. Chatterjee, and R. Brayton, "DAG-aware AIG rewriting: A fresh look at combinational logic synthesis," Design Automation Conference (DAC), 2006.
3. D. Chen and J. Cong, "DAOmap: A Depth-Optimal Area Optimization Mapping Algorithm for FPGA Designs," International Conference on Computer-Aided Design (ICCAD), 2004.
4. ABC: A System for Sequential Synthesis and Verification, Berkeley Logic Synthesis and Verification Group.


## What's missing now
- Also provide a `solver_baseline.py` with a cut-enumeration + dynamic programming based solver. But it doesn't support truth table for LUTs yet, so the output BLIF file doesn't have truth table. (Need the derivation of the truth table from the cut.)
- Now support truth table derivation! It is tested on several demo small circuits and seems correct. But for large circuits, it takes too much time, probably because of the simulate_logic method. 

## Dependencies
- `evaluator.py` and `verifier.py` are based on the [abc](https://github.com/berkeley-abc/abc/tree/master) tool. It can be installed by: 
    ```bash
    git clone git@github.com:berkeley-abc/abc.git
    cd abc
    make -j4
    ```

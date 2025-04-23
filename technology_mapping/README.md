# Technology Mapping for LUT-based FPGA


## Background

Technology mapping is a critical stage in the FPGA design flow where a technology-independent logic network is transformed into a network of lookup tables (LUTs). A K-input lookup table (K-LUT) is a basic programmable logic element in FPGAs that can implement any Boolean function with up to K inputs. The quality of technology mapping directly impacts the area, performance, and power consumption of the final FPGA implementation.

The technology mapping process involves covering a Boolean network with K-LUTs in a way that optimizes specific design metrics. Traditional approaches include cut enumeration, where potential K-feasible cuts (subgraphs with at most K inputs) are identified and selected to form an optimal cover of the network. The mapping process must ensure that the resulting network correctly implements the original logic function while minimizing resource usage.

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

The K-LUT technology mapping problem can be formulated as:

$\min_{M} Cost(M)$

For an alternative optimization target focusing on performance, we can define the depth of a node $v$ under mapping $M$ as:

$Depth_M(v) = \begin{cases}
0, & \text{if}\ v \in PI \\
\max_{u \in M(v)} (Depth_M(u) + 1), & \text{otherwise}
\end{cases}$

Then, the depth-oriented mapping problem becomes:

$\min_{M} \max_{v \in PO} Depth_M(v)$

## Input Format

The input logic network is specified using the Berkeley Logic Interchange Format (BLIF). BLIF is a human-readable text format for describing logic circuits at the gate level. Here's an example of a simple circuit in BLIF format:

```blif
# Benchmark "c17" written by ABC
.model c17
.inputs 1 2 3 6 7
.outputs 22 23
.names 1 3 new_10
11 0
.names 3 6 new_11
11 0
.names 2 new_11 new_16
11 0
.names new_11 7 new_19
11 0
.names new_10 new_16 22
11 0
.names new_16 new_19 23
11 0
.end
```

In this format:
- `.model` specifies the name of the circuit
- `.inputs` lists the primary inputs
- `.outputs` lists the primary outputs
- `.names` blocks define the logic functions for internal nodes and outputs
- The rows following a `.names` line define the truth table entries that produce a 1 output (or 0 if explicitly specified)
- `.end` marks the end of the circuit description

The LUT size `K` is specified as a command-line argument.

## Output Format

The output of the technology mapping is also provided in BLIF format, but with the logic network implemented using only K-LUTs. Each LUT is represented by a `.names` block with at most K inputs. The output BLIF preserves the primary inputs and outputs of the original network.

Example output for the above circuit after mapping to 3-LUTs:

```blif
# Benchmark "c17" mapped to 3-LUTs
.model c17
.inputs 1 2 3 6 7
.outputs 22 23
.names 1 3 6 lut_1
101 1
.names 2 3 6 lut_2
011 1
.names 3 6 7 lut_3
110 1
.names lut_1 lut_2 22
11 1
.names lut_2 lut_3 23
11 1
.end
```

In this output:
- Each `.names` block corresponds to a LUT in the final implementation
- Each LUT has at most K inputs (3 in this example)
- The network correctly implements the functionality of the original circuit
- The total number of LUTs used is reported (5 in this example)

## References
1. J. Cong and Y. Ding, "FlowMap: An Optimal Technology Mapping Algorithm for Delay Optimization in Lookup-Table Based FPGA Designs," IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 1994.
2. A. Mishchenko, S. Chatterjee, and R. Brayton, "DAG-aware AIG rewriting: A fresh look at combinational logic synthesis," Design Automation Conference (DAC), 2006.
3. D. Chen and J. Cong, "DAOmap: A Depth-Optimal Area Optimization Mapping Algorithm for FPGA Designs," International Conference on Computer-Aided Design (ICCAD), 2004.
4. ABC: A System for Sequential Synthesis and Verification, Berkeley Logic Synthesis and Verification Group.


## What's missing now
- Don't have support on truth table yet. We need to verify the mapped logic network is correct. 
- No `evaluator.py` and `verify.py` yet. 
- Also provide a `solver_baseline.py` with a cut-enumeration + dynamic programming based solver. But it doesn't support truth table for LUTs yet, so the output BLIF file doesn't have truth table. 
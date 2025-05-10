# Technology Mapping for LUT-based FPGA


## Background

Technology mapping is a critical stage in the FPGA design flow where a technology-independent logic network is transformed into a network of lookup tables (LUTs). A K-input lookup table (K-LUT) is a basic programmable logic element in FPGAs that can implement any Boolean function with up to K inputs. The quality of technology mapping directly impacts the area, performance, and power consumption of the final FPGA implementation.

Here we consider the structural mapping problem, which consider the circuit graph as a given and find a covering of the graph with K-input subgraphs corresponding to LUTs. 

Our optimization target is to minimize the number of LUTs, which represents the area of the mapped logic network. This is an NP-hard problem. 

In this problem, we specify K = 6. 

## Formalization

Consider a directed acyclic graph (DAG) $G = (V, E)$ representing a Boolean network, where each node $v \in V$ represents a logic gate, and each edge $(u, v) \in E$ indicates that the output of node $u$ is an input to node $v$. Let $PI$ denote the set of primary inputs and $PO$ denote the set of primary outputs in the network. A primary input (PI) node is a node that has no input edges. A primary output (PO) node is a node that has no output edges. An internal node has both input and output edges. 

A cone of a node $v$, $Cone(v)$, is a subgraph consisting of $v$ and some of its non-PI predecessors, such that any node $u \in Cone(v)$ has a path to $v$ that lies entirely in $Cone(v)$. Node $v$ is the root of the cone. 
At a cone $Cone(v)$, the set of input edges iedge($Cone(v)$) is the set of edges with head in $Cone(v)$ and tail outside $Cone(v)$; and the set of output edges oedge($Cone(v)$) is the set of edges with $v$ as the tail. 
With input and output edges so defined, a cone can be viewed as a node, and notions of "inode", "onode", "K-feasibility" can be extended to handle cones. 

The set of distinct nodes that supplies input edges to $Cone(v)$ is referred to as $inode(Cone(v))$, and the set of distinct nodes that receives output edges from $Cone(v)$ is referred to as $onode(Cone(v))$. 
A cone $Cone(v)$ is K-feasible if $|inode(Cone(v))| \leq K$. 

A K-input LUT (K-LUT) can implement any K-feasible cone. Thus, the technology mapping problem for LUTs is selecting a set of K-feasible cones to cover the graph in such a way that: 
every edge in the graph is entirely within a cone, or is an output edge of a cone, or is the output edge of a PI node. 

In our area-optimal mapping problem, the number of cones selected to cover the graph is to be minimized. 
Formally, a mapping solution $M$ is to select a set of cones $Cone(v_1), Cone(v_2), \ldots, Cone(v_n)$, where $v_i \in V$ but $v_i \notin PI$, such that: 
- For each edge $e \in E$, $e$ is entirely within a cone, or is an output edge of a cone, or is the output edge of a PI node. 
- The cost to be minimized is the number of cones selected, i.e., $Cost(M) = |V'|$, where $V' = \\{v_1, v_2, \ldots, v_n\\}$. 


Here we specify K = 6 in our implementation. 

Note that there could be exponential explosion in the search space of the mapping solution, so you might need to prune the search space to make it feasible on real graphs. 


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

```blif
# Circuit 3. Lines starting with '#' are comments. 
# If there are too many gates in ".inputs", ".outputs", or ".names", it can be separated into multiple lines with a backslash `\` at the end of each line. 
.model c432
.inputs 1 4 8 11 14 17 21 24 27 30 34 37 40 43 47 50 53 56 60 63 66 69 73 \
 76 79 82 86 89 92 95 99 102 105 108 112 115
.outputs 223 329 370 421 430 431 432
.names 1 new_118
0 1
.names 4 new_119
0 1
.names new_154 new_159 new_162 new_165 new_168 new_171 new_174 new_177 \
 new_180 new_199
111111111 1 
.names new_381 new_422 new_425 new_429 432
1111 0
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
- Note that if there are too many gates in `.inputs`, `.outputs`, or `.names`, it can be separated into multiple lines with a backslash `\` at the end of each line. 

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


## Dependencies
- `evaluator.py` and `verifier.py` are based on the [abc](https://github.com/berkeley-abc/abc/tree/master) tool. It can be installed by: 
    ```bash
    git clone git@github.com:berkeley-abc/abc.git
    cd abc
    make -j4
    ```

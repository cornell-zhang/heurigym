# E-Graph Extraction 

## Background
Term rewriting, widely employed in compiler optimizations and theorem proving, transforms programs into functionally equivalent but more efficient forms.
Traditional methods apply the rewrites sequentially in a predetermined order, significantly affecting performance--a challenge known as the phase ordering problem.

Equality saturation addresses the phase ordering issue by using the equivalence graph (e-graph), a data structure that compactly represents a set of expressions.
Given an input program and a set of rewrite rules, e-graph construction is performed by applying the rules to the program, generating new expressions, and merging equivalent expressions.

Once the e-graph is constructed (saturated), a good expression needs be extracted by selecting the most optimal one based on a cost model, which represents the optimized program. 
This is a known NP-hard problem, and various heuristics have been proposed to solve it.

## Formalization
In an e-graph, all functionally equivalent terms are organized in the same equivalent classes, known as e-classes ($\{m_j\}_{j=0}^{M-1}$).
Nodes within each e-class that represent values or operators are called e-nodes ($\{n_i\}^{N-1}_{i=0}$).
Edges in e-graphs are directed, which point from e-nodes to their child e-classes, indicating the dependency between operators and operands.

The notations are defined as follows:
- Let $\{m_j\}_{j=0}^{M-1}$ denote the set of all e-classes in the e-graph.
- Let $\{n_i\}_{i=0}^{N-1}$ denote the set of all e-nodes in the e-graph, where root e-class containing the top-level operator is denoted as $R = \{m_r\}$.
- Each eclass $m_j$ contains a set of e-nodes $m_j = \{n_k\}$, and $|m_j|$ denotes the number of e-nodes in $m_j$.
- $ch_i$ denotes the set of child e-classes of e-node $n_i$.
- $pa_j$ denotes the set of parent e-nodes of e-class $m_j$.
- $ec(i)$ returns the index of the e-class that contains e-node $n_i$, namely $n_i \in m_{ec(i)}$.
- Let $s\in \{0,1\}$^N represent the binary decision vector of the e-nodes, where $s_i=1$ indicates that e-node $n_i$ is selected for extraction, and $s_i=0$ indicates that it is not selected.
- We have a linear cost function that maps the selected e-nodes to a cost value, denoted as $C(s) = \sum_{i=0}^{N-1} c_i s_i$, where $c_i\ge 0$ is the cost of e-node $n_i$.

Next, we define a legal extraction of the e-graph:
1. Exactly one e-node is selected from root e-classes.
   - This ensures the functional equivalence of the selected e-nodes.
   - i.e., $\sum_{n_k \in m_r} s_k = 1, \forall m_r \in R$.
2. If an e-node is selected, exactly one e-node must be selected from each of its child e-classes.
   - This ensures that the selected e-nodes are functionally equivalent.
   - i.e.: $\sum_{n_k \in ch_i} s_k = s_i, \forall i$.
3. No cycles are included in the extraction.
   - This ensures no cyclic dependencies are included in the extraction.

The goal is to find the optimal extraction of the e-graph that minimizes the cost function $C(s)$ while satisfying the above constraints.

## Input Format
The input format is a JSON file containing with two required top‑level fields:

| Field | Type | Meaning |
|-------|------|---------|
| **nodes** | object (map<string, NodeRecord>) | Dictionary whose keys are **node_id** strings. Each entry fully describes one node in the directed acyclic graph. |
| **root_eclasses** | array<string> | List of **eclass** IDs that serve as roots of the computation (i.e., no parent references outside the set). |

### nodes

Every value in `nodes` is itself an object with the following schema, assuming the current node is $n_i$:

| Key | Type | Description |
|-----|------|-------------|
| **op** | string | Symbol or token representing the node’s operation. It can be a literal (e.g. `"add"`, `"mul"`). Can be ignored in this task. |
| **cost** | number | Cost $c_i$ associated with this e-node. |
| **eclass** | string | The e-class that this e-node belongs to (i.e. $pa_i$) |
| **children** | array<string> | List of **node_id** strings referencing this node’s children e-classes. Use **node_id** to represent the e-class that they belong to. (i.e. $ch_i$) |

### root_eclasses
The **root_eclasses** field is a list of strings, representing the IDs of the e-classes that are roots of the computation.
i.e., $R = \{m_r\}$.

### Example
```
{
    "nodes": {
        "0__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "0",
            "children": [
                "1__0",
                "2__0"
            ]
        },
        "1__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "1",
            "children": [
                "3__0"
            ]
        },
        "1__1": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "1",
            "children": [
                "3__0"
            ]
        },
        "1__2": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "1",
            "children": [
                "4__0"
            ]
        },
        "2__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "2",
            "children": [
                "3__0"
            ]
        },
        "2__1": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "2",
            "children": [
                "5__0"
            ]
        },
        "3__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "3",
            "children": [
            ]
        },
        "4__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "4",
            "children": [
            ]
        },
        "5__0": {
            "op": "0.1",
            "cost": 0.1,
            "eclass": "5",
            "children": [
            ]
        }
    },
    "root_eclasses": [
        "0"
    ]
}
```

## Output Format
The solver function should return a string containing a list of selected e-nodes in the format of `["node_id_1", "node_id_2", ...]`, where each `node_id` is a string representing the ID of the selected e-node.
```
[
    "0__0",
    "1__0",
    "2__1",
    "3__0",
    "5__0"
]
```

## References
1. Nelson, Greg, and Derek C. Oppen. "Simplification by cooperating decision procedures." ACM Transactions on Programming Languages and Systems (TOPLAS), 1979.
2. Stepp, Michael Benjamin. Equality saturation: engineering challenges and applications. University of California, San Diego, 2011.
3. Goharshady, Amir Kafshdar, Chun Kit Lam, and Lionel Parreaux. "Fast and optimal extraction for sparse equality graphs." Proceedings of the ACM on Programming Languages (OOPSLA), 2024.
4. Cai, Yaohui, Kaixin Yang, Chenhui Deng, Cunxi Yu, and Zhiru Zhang. "SmoothE: Differentiable E-Graph Extraction." In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025.
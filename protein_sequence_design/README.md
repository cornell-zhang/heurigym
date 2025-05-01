# Protein Sequence Design & Evolutionary Fitness Landscape in GC Model

## Background

Protein sequence design (or inverse protein folding) aims to find an amino acid sequence likely to adopt a given target 3D structure. The Grand Canonical (GC) model simplifies this by using a binary hydrophobic/polar (H/P) alphabet and a fitness function rewarding dense hydrophobic cores while penalizing solvent-exposed H residues [1, 2]. Initially, only heuristic methods existed for this model [3].

A related concept is the **fitness landscape**: the set of all optimal sequences (minimizers of the fitness function) for a target structure. A key question is whether this set of optimal sequences is **connected** by single H/P point mutations, allowing evolutionary paths without passing through non-optimal intermediates. 

[4] provides efficient algorithms for both finding optimal sequences and testing this connectedness property.

## Formalization

### 1. Sequence Design Problem

* **Structure Representation:**
    * A target structure with $n$ residues.
    * Each residue position $i$ has a calculated solvent-accessible area $s_i$.
    * Pairwise distances $d_{ij}$ between residue side chain centers are known.
* **Fitness Function:**
    * A sequence $S$ is an element of $\{H, P\}^n$. $S_H$ denotes the set of indices $k$ where $S_k = H$.
    * The fitness $\Phi(S)$ is given by:
        $$ \Phi(S) = \alpha \sum_{i,j \in S_H} g(d_{ij}) + \beta \sum_{i \in S_H} s_i $$
    * $\alpha < 0$ weights favorable H-H contacts, $\beta > 0$ penalizes solvent exposure of H residues.
    * $g(d_{ij})$ is a contact function (e.g., sigmoidal or step) rewarding small distances between H residues.
* **Optimization Goal:**
    * Find a sequence $S \in \{H,P\}^n$ that *minimizes* $\Phi(S)$.
    * This problem is reduced to finding a minimum $s-t$ cut in a specially constructed directed graph with $O(n+p)$ vertices and edges (where $p$ is the number of pairs with $g(d_{ij})>0$). This yields a polynomial-time algorithm, roughly $O(n^2 \log n)$.

### 2. Connectedness Problem

* **Optimal Set:** $\Omega = \{ S \in \{H,P\}^n \mid \Phi(S) \text{ is minimal} \}$. $\Omega$ can be exponentially large.
* **Adjacency:** Two sequences $S, S'$ are adjacent if they differ by a single H↔P flip at one position.
* **Connectedness:** Is $\Omega$ connected? That is, for any $S, S' \in \Omega$, does a path $S = S_0, S_1, \dots, S_k = S'$ exist such that each $S_i \in \Omega$ and $S_{i}, S_{i+1}$ are adjacent?
* **Submodular Reformulation:**
    * Map sequence $S$ to the set $X = S_H$ (indices where $S_k=H$). Define $f(X) = \Phi(\sigma(X))$, where $\sigma(X)$ is the sequence corresponding to set $X$.
    * $f(X)$ is a **submodular function**: $f(X \cap Y) + f(X \cup Y) \le f(X) + f(Y)$.
    * Let $\Omega_f$ be the family of subsets minimizing $f$. $\Omega$ is connected if and only if $\Omega_f$ is connected under single-element insertions/deletions transforming one minimizing set into another while staying within $\Omega_f$.
* **Algorithmic Test:**
    * Compute the unique minimal ($X_*$) and maximal ($X^*$) minimizers of $f$ using standard submodular function minimization algorithms.
    * Starting from $W = X^*$, iteratively find an element $i \in W$ such that $W' = W \setminus \{i\}$ is also a minimizer ($f(W') = f(W)$) and update $W := W'$.
    * If this process reaches $X_*$, then $\Omega_f$ (and $\Omega$) is connected.
    * If the process reaches a non-minimal set $W$ where no such element $i$ exists (an "impasse"), then $\Omega_f$ is disconnected. This test runs in polynomial time.

## Input Format

1.  **For Sequence Design:**
    * **Target Structure Data:**
        * Number of residues, $n$.
        * List of solvent-accessible areas $\{s_i\}_{i=1}^n$.
        * List of residue pairs $(i, j)$ and distances $d_{ij}$ for which $g(d_{ij}) > 0$.
    * **Model Parameters:**
        * Values for $\alpha$ (negative) and $\beta$ (positive).
        * Definition of the contact function $g(d)$ (e.g., $g(d) = 1/(1+e^{d-6.5})$ for $d \le 6.5$Å, 0 otherwise ).

2.  **For Connectedness Test:**
    * The same inputs as for Sequence Design.
    * Implicitly requires an oracle/subroutine to evaluate the fitness function $\Phi(S)$ (or $f(X)$) efficiently. The sequence design algorithm itself can serve this purpose.

## Output Format

1.  **Sequence Design:**
    * An optimal sequence $S$, represented as a string of length $n$ over $\{H, P\}$, which achieves the minimum value of $\Phi(S)$.
2.  **Connectedness Test:**
    * A Boolean result: **Connected** or **Disconnected**.
    * If **Disconnected**, typically provides an "impasse" set $X$ (or corresponding sequence $S$) demonstrating the disconnection.
    * (Implicitly, if **Connected**, the algorithm demonstrates a monotone path from $X^*$ to $X_*$ exists).

## References

1.  Sun, Shaojian, et al. "Designing amino acid sequences to fold with good hydrophobic cores." Protein Engineering, Design and Selection 8.12 (1995): 1205-1213.
2.  Hart, William E. "On the computational complexity of sequence design problems." Proceedings of the first annual international conference on Computational molecular biology. 1997.
3.  Maynard Smith, John. "Natural selection and the concept of a protein space." Nature 225.5232 (1970): 563-564.
4.  Kleinberg, Jon M. "Efficient algorithms for protein sequence design and the analysis of certain evolutionary fitness landscapes." Proceedings of the third annual international conference on Computational molecular biology. 1999.
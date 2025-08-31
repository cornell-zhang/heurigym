# Pedigree Problem

## Background

In genetic studies, pedigree data encode the parent–offspring relationships among individuals and, at each locus of interest, record up to two alleles per individual—one inherited from the father, one from the mother. In practice, genotyping assays may fail to call one or both alleles, so the observed data can be incomplete or even erroneous. Under classical Mendelian inheritance, each child’s paternal allele must coincide with one of its father’s two alleles, and similarly its maternal allele must match one of its mother’s two alleles.

We tackle a key challenge with incomplete family genetic data: ensuring it follows inheritance rules while finding the most likely true genetic makeup for everyone. This is done through a task called Minimum Cost Assignment (Error Localization & Optimal Assignment Retrieval). This task finds the best complete genetic profile for all family members. "Best" means an assignment with the fewest conflicts with observed genetic data, while strictly following inheritance laws. The number of these conflicts gives a "cost score." A zero cost score means the observed data perfectly fits inheritance rules, and the found assignment shows this fit. A score above zero indicates likely errors in the data. This score is the minimum number of observed data points needing correction to ensure the whole family tree is genetically consistent. The resulting assignment provides a complete genetic profile reflecting these minimal corrections.


## Formalization

Consider a pedigree instance characterized by $n \in \mathbb{N}$ distinct alleles, often labelled $A = \{1, \ldots, n\}$, and a set of $m$ individuals, $I = \{1, \ldots, m\}$. The genotype for any individual $i \in I$, denoted as $x_i$, consists of an unordered pair of alleles. The domain of all possible genotypes is $D = \{\,ab : 1 \le a \le b \le n\}$, where $a,b \in A$. An assignment for the pedigree is an $m$-tuple $X = (x_1, x_2, \ldots, x_m)$, where $x_i \in D$ is the genotype assigned to individual $i$.
The problem is structured around a set of cost functions that evaluate an assignment $X$:
1.  **Unary Costs $C_i(x_i)$:**
    For each individual $i \in I$, a function $C_i: D \to \{0,1\}$ is defined. $C_i(v)=1$ if assigning genotype $v$ to individual $i$ conflicts with their observed genotyping data. $C_i(v)=0$ if $v$ is compatible or if individual $i$ is untyped.

2.  **Binary Costs $C_{ij}(x_j, x_i)$:**
    For each specified parent-child pair $(j \to i)$, where $j$ is a known parent of $i$, a function $C_{ij}:D \times D \to \{0,\infty\}$ enforces allele sharing. If $x_j$ and $x_i$ represent the sets of alleles for individuals $j$ and $i$ respectively:

$$
C_{ij}(x_j,x_i)=
\begin{cases}
0, & x_i \cap x_j \neq \emptyset \quad \text{(i.e., child } i \text{ shares at least one allele with parent } j\text{)},\\
\infty, & \text{otherwise.}
\end{cases}
$$

3.  **Ternary Costs $C_{jki}(x_j, x_k, x_i)$:**
    For each specified nuclear family $(j, k \to i)$, where $j$ and $k$ are the parents of $i$, a function $C_{jki}:D \times D \times D \to \{0,\infty\}$ enforces Mendel’s law of segregation. If $x_i$ consists of the alleles $\{a,b\}$, and $x_j, x_k$ represent the allele sets of the parents:

$$
C_{jki}(x_j,x_k,x_i)=
\begin{cases}
0, & \text{if } (a \in x_j \land b \in x_k) \lor (a \in x_k \land b \in x_j) \quad \text{(i.e., one allele of } x_i \text{ is from } x_j \text{ and the other from } x_k\text{)},\\
\infty, & \text{otherwise.}
\end{cases}
$$

The total cost of an assignment $X=(x_i)_{i\in I}$ is defined by the objective function:
$$
V(X) = \sum_{i\in I} C_i(x_i) + \sum_{(j\to i)}C_{ij}(x_j,x_i) + \sum_{(j,k\to i)}C_{jki}(x_j,x_k,x_i).
$$
Any assignment $X$ for which $V(X) < \infty$ must satisfy all binary and ternary Mendelian inheritance constraints (i.e., these constraints contribute $0$ to the sum). Such an assignment is termed **Mendelian-consistent**. Let $\mathcal{X}_{all}$ denote the set of all possible genotype assignments.

The computational task associated with this problem is **Minimum Cost Assignment (Error Localization & Optimal Assignment Retrieval)**, which is to compute the minimum possible total cost $V^* = \min_{X \in \mathcal{X}_{all}} V(X)$ and to identify an assignment $X^*$ such that $V(X^*) = V^*$. The value $V^*$ (a non-negative integer) indicates the minimum number of genotype-data conflicts (unary costs of 1) that must be incurred for a Mendelian-consistent assignment. If $V^*=0$, the pedigree and genotype data are perfectly consistent. The assignment $X^*$ is a specific configuration of genotypes for all individuals $i \in I$ that achieves this minimum total cost. If multiple such assignments exist, one such $X^*$ is returned.

## Input Format

The input is a .pre file. Each line in the .pre file is just the pedigree‐genotype data for one individual at a single locus, using the LINKAGE-style seven-column format:
`<locus> <ID> <fatherID> <motherID> <sex> <allele1> <allele2>`.

A snippet of a typical input file looks like this:

```
1 1 0 0 1 3 4
1 2 0 0 2 1 3
1 3 1 2 2 0 0
1 4 0 0 1 1 3
1 5 3 4 1 2 3

```

## Output Format

The output includes the list of chosen domain labels—one number per individual, in the same order as your input—indicating which allele‐pair (by its index) was assigned to each person. The allel-pair mapping is like this:
```
0 ↦ (1,1)
1 ↦ (1,2)
2 ↦ (1,3)
...
```

An example output file looks like this:

```
8 1 5 2 5
```


## References

1. Sanchez, Marti, Simon de Givry, and Thomas Schiex. "Mendelian error detection in complex pedigrees using weighted constraint satisfaction techniques." Constraints 13.1 (2008): 130-154.
2. O'Connell, Jeffrey R., and Daniel E. Weeks. "An optimal algorithm for automatic genotype elimination." The American Journal of Human Genetics 65.6 (1999): 1733-1740.
3. O'Connell, Jeffrey R., and Daniel E. Weeks. "PedCheck: a program for identification of genotype incompatibilities in linkage analysis." The American Journal of Human Genetics 63.1 (1998): 259-266.
4. Wijsman, Ellen M. "The role of large pedigrees in an era of high-throughput sequencing." Human genetics 131 (2012): 1555-1563.
5. Stringham, Heather M., and Michael Boehnke. "Identifying marker typing incompatibilities in linkage analysis." American journal of human genetics 59.4 (1996): 946.
6. Aceto, Luca, et al. "The complexity of checking consistency of pedigree information and related problems." Journal of Computer Science and Technology 19 (2004): 42-59.
7. De Givry, Simon. "toulbar2, an exact cost function network solver." 24ème édition du congrès annuel de la Société Française de Recherche Opérationnelle et d'Aide à la Décision ROADEF 2023. 2023.
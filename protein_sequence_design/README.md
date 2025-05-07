# Protein Sequence Design (GC Model)

## Background

Protein sequence design (or inverse protein folding) aims to find an amino acid sequence likely to adopt a given target 3D structure. The Grand Canonical (GC) model simplifies this by using a binary hydrophobic/polar (H/P) alphabet and a fitness function rewarding dense hydrophobic cores while penalizing solvent-exposed H residues [1, 2]. Initially, only heuristic methods existed for this model [3]. [4] provides efficient algorithms for finding optimal sequences.

## Formalization

* **Structure Representation:**
    * A target structure with $n$ residues.
    * Each residue position $i$ has a calculated solvent-accessible area $s_i$.
    * Pairwise distances $d_{ij}$ between residue side chain centers are known.
* **Fitness Function:**
    * A sequence $S$ is an element of `{H, P}`$^n$. $S_H$ denotes the set of indices $k$ where $S_k = H$.
    * The fitness $\Phi(S)$ is given by: $\Phi(S) = \alpha \sum_{i,j \in S_H} g(d_{ij}) + \beta \sum_{i \in S_H} s_i$.
    * $\alpha < 0$ weights favorable H-H contacts, $\beta > 0$ penalizes solvent exposure of H residues.
    * $g(d_{ij})$ is a contact function (e.g., sigmoidal or step) rewarding small distances between H residues.
* **Optimization Goal:**
    * Find a sequence $S \in$ `{H, P}`$^n$ that *minimizes* $\Phi(S)$.

## Input Format

The input is a protein structure provided in the standard Protein Data Bank (PDB) file format. This format describes the positions of atoms in three-dimensional space. The algorithm utilizes the atomic coordinates to calculate inter-residue distances ($d_{ij}$) and solvent accessible surface areas ($s_i$).

A snippet of a typical PDB file looks like this:

```pdb
ATOM      1  N   ASP A   1      25.824 -18.547  21.875  1.00 26.10           N
ATOM      2  CA  ASP A   1      25.130 -17.690  20.946  1.00 24.02           C
ATOM      3  C   ASP A   1      25.342 -18.306  19.589  1.00 20.34           C
ATOM      4  O   ASP A   1      26.329 -19.003  19.358  1.00 21.77           O
ATOM      5  CB  ASP A   1      25.691 -16.283  20.931  1.00 29.35           C
ATOM      6  CG  ASP A   1      25.201 -15.494  22.138  1.00 33.94           C
ATOM      7  OD1 ASP A   1      25.791 -15.651  23.209  1.00 37.34           O
ATOM      8  OD2 ASP A   1      24.238 -14.727  21.999  1.00 37.25           O
ATOM      9  N   LYS A   2      24.426 -18.058  18.693  1.00 13.86           N
ATOM     10  CA  LYS A   2      24.544 -18.580  17.400  1.00 13.08           C
... (remaining atoms) ...
```

## Output Format

The output is a single string representing the calculated optimal sequence using the binary Hydrophobic (H) / Polar (P) alphabet. The length of the sequence corresponds to the number of amino acid residues processed from the input PDB file.

An example output sequence looks like this:

```
PPPPPPPPPPPPPPPPPPPPPPHPHPPPPHPHPPPPHPPPPPHHHPHPPP...PHPPPPHPPPPPPHPPHPPPPHHHHHHPPPPPHPHPHPPPPPHPHPHPHP

```


## References

1.  Sun, Shaojian, et al. "Designing amino acid sequences to fold with good hydrophobic cores." Protein Engineering, Design and Selection 8.12 (1995): 1205-1213.
2.  Hart, William E. "On the computational complexity of sequence design problems." Proceedings of the first annual international conference on Computational molecular biology. 1997.
3.  Maynard Smith, John. "Natural selection and the concept of a protein space." Nature 225.5232 (1970): 563-564.
4.  Kleinberg, Jon M. "Efficient algorithms for protein sequence design and the analysis of certain evolutionary fitness landscapes." Proceedings of the third annual international conference on Computational molecular biology. 1999.

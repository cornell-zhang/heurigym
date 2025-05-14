# Frequency Assignment Problem with Polarization

## Background

The Frequency Assignment Problem with Polarization (FAPP) arises in Hertzian telecommunication networks, where a set of geographical sites are connected by uni‐directional radio‐electric paths (antennas) forming a network $T$. Each path must be assigned a frequency and a polarization (vertical or horizontal) drawn from its available resource domains. As spectrum becomes increasingly scarce—particularly in military contexts managed by CELAR (DGA)—optimizing these assignments under stringent electromagnetic compatibility (CEM) constraints is critical. FAPP extends earlier models by introducing:

- **Binary polarization** (only two polarizations, $\pm 1$) alongside frequencies,  
- **Class 1 (strong) constraints** enforcing exact or distinct frequencies and polarizations, and  
- **Class 2 (CEM) constraints** requiring minimum frequency separations that depend on the polarization relationship, which may be progressively relaxed across levels $0$–$10$ when no fully compatible solution exists.

In practice, strict compatibility often yields no feasible solution. To address this, FAPP introduces a hierarchy of relaxation levels: a level-$k$ solution must satisfy all Class 1 constraints and Class 2 constraints at level $k$. If no $0$-feasible solution exists, one seeks the smallest $k^*$ for which a $k^*$-feasible solution exists, then minimizes the number of violated CEM constraints at lower levels in a lexicographic fashion.

## Formalization

* **Paths and Domains**  
  - Let $T$ be the set of directed paths.  
  - Frequency domains: $F_0, F_1, \dots, F_N$, each a finite set of integer frequencies.  
  - Polarization domains: $P_{-1} = \{-1\}$, $P_{1} = \{1\}$, $P_0 = \{-1,1\}$.  
  - Each path $i \in T$ has an associated frequency domain $F(i)$ and polarization domain $P(i)$.

* **Decision Variables**  
  - $f_i \in F(i)$: assigned frequency for path $i$.  
  - $p_i \in P(i)$: assigned polarization for path $i$.

* **Class 1 (Strong) Constraints**  
  - **Domain constraint:** $f_i \in F(i)$, $p_i \in P(i)$.  
  - **Frequency equality/inequality:** for certain $(i,j)$, either $f_i = f_j$ or $f_i \neq f_j$.  
  - **Fixed separation:** for certain $(i,j)$ and integer $e_{ij}$, either $|f_i - f_j| = e_{ij}$ or $|f_i - f_j| \neq e_{ij}$.  
  - **Polarization equality/inequality:** for certain $(i,j)$, either $p_i = p_j$ or $p_i \neq p_j$.

* **Class 2 (CEM) Constraints**  
  - For each path pair $(i,j)$, and for each relaxation level $k = 0, \dots, 10$, two sequences of thresholds:  
    - $g^{\mathrm{eq}}_{ij}(k)$ when $p_i = p_j$,  
    - $g^{\mathrm{neq}}_{ij}(k)$ when $p_i \neq p_j$.  
  - Requirement at level $k$:  
    $$
      \begin{cases}
        |f_i - f_j|\;\ge\;g^{\mathrm{eq}}_{ij}(k), &\text{if }p_i = p_j,\\
        |f_i - f_j|\;\ge\;g^{\mathrm{neq}}_{ij}(k), &\text{if }p_i \neq p_j.
      \end{cases}
    $$

* **Relaxation‐Level Feasibility**  
  - A solution is **$k$-feasible** if it satisfies all Class 1 constraints and all Class 2 constraints at level $k$.  
  - Define $k^*$ as the smallest $k$ with a $k$-feasible solution.

* **Optimization Goal**  
  - **Primary:** minimize $k^*$.  
  - **Secondary:** minimize $V_{k^*-1}$, the number of Class 2 violations at level $k^*-1$.  
  - **Tertiary:** minimize the number of violations at levels below $k^*-1$.  
  - Equivalently, find $\{(f_i,p_i)\}_{i\in T}$ that lexicographically minimizes the tuple $(k^*, V_{k^*-1}, V_{k^*-2}, \dots, V_0)$.

## Input Format
The program reads a list of lines describing the available resources and constraints for each radio‐electric path:
- **`DM i j`** lines list all frequency values in domain $F_i$ by pairing the domain index $i$ with each frequency $j$.  
- **`TR i 0 0`** lines indicate that path $i$ has polarization domain $P_0 = \{-1,1\}$ (the two zeros are placeholders).  
- **`CE i j g0 g1 … g10`** lines give the “same‐polarization” CEM thresholds $g^{\mathrm{eq}}_{ij}(k)$ for levels $k=0$ to $10$.  
- **`CD i j d0 d1 … d10`** lines give the “different‐polarization” CEM thresholds $g^{\mathrm{neq}}_{ij}(k)$ for levels $k=0$ to $10$.

```
DM     0     1
DM     0     2
DM     0     3
DM     0     4
DM     0     5
DM     0     6
DM     0     7
DM     0     8
DM     0     9
DM     0    10
...
TR     1     0  0
TR     2     0  0
TR     3     0  0
TR     4     0  0
CE     1     2    60    58    56    53    52    50    50    45    45    45    45
CD     1     2    55    53    52    50    48    45    45    40    40    40    40
CE     1     3    61    59    57    56    54    52    52    50    50    50    48
CD     1     3    51    51    49    46    44    44    44    42    42    42    42
CE     2     3    90    80    70    70    68    65    65    60    60    60    60
CD     2     3    80    70    63    63    63    60    60    55    55    55    55
CE     2     4    30    30    25    25    20    19    19    19    19    19    15
CD     2     4    20    20    15    10    10     9     9     9     9     9     5
```

## Output Format
The program produces:

- **`RP f₁ f₂ … fₙ`** assigns frequency $f_i$ to path $i$ (zero-based indexing).  
- **`AL i k v`** for each path $i$ reports:
  1. the chosen relaxation level $k$ ($k^*$ for the overall solution),  
  2. the number of same-polarization violations at level $k-1$,  
  3. the number of different-polarization violations at level $k-1$.  
- A negative violation count `v = -1` indicates “not applicable” because the solution did not need to relax below level $k$.  
```
RP 0 1 0 0 0 1 0 0 0 1 0 0 0
AL 1 47 -1
AL 2 100 1
AL 3 1 1
AL 4 1 -1
```
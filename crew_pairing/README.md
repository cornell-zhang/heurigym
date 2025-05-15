# Airline Crew-Pairing Problem (CPP)

## Background  

Airlines must build _pairings_ (multi-day sequences of flight legs) so that every scheduled leg is covered by exactly one legal crew duty-tour while respecting rules (such as maximum duty hours, minimum rest, domicile start/end, etc.).  

A feasible solution must:

* **Cover** each flight leg exactly once.
* **Respect** all duty-time and rest constraints *within* every pairing.
* A pairing may **begin at any airport**.  If the first leg departs a station other than the base, the crew is assumed to position-fly there before the tour starts.  A flat **positioning-cost** will be charged for each such pairing.
* A pairing may **end anywhere** (crews dead-head home after the horizon).

Given the increasing volume of flights, operational planners then need to choose the subset of pairings that minimises total cost (wages, per-diem, hotels) while ensuring sufficient reserve pilots remain at the base to absorb disruptions.  In our simplified benchmark we ignore reserve sizing and hotel detail; cost depends only on duty and block hours.


## Formalization

**Key definitions**:

* $\mathcal F$ - set of all scheduled flight legs (each leg is a single aircraft movement defined by departure station/time and arrival station/time); indexed by $f$. 
* $\mathcal P$ - set of *feasible* pairings.  A pairing $p \in \mathcal P$ is a sequence $\{ f_1,\dots,f_{|p|}\}$ satisfying all of the following rules:
  * Chronological order: $\mathrm{arr}(f_i) \le \mathrm{dep}(f_{i+1})$.
  * Each internal rest gap that splits duties is $\ge R_{min}$.
  * Within every duty segment $d\subseteq p$:

    * $\text{dutyHours}(d) \le H^d_{max}$
    * $\text{blockHours}(d) \le H^b_{max}$
    * $|d| \le L_{max}$.

**Legal-rule parameters**:

* Maximum duty span $H^d_{max}$ = 14 h
* Maximum block hours per duty $H^b_{max}$ = 10 h
* Maximum legs per duty $L_{max}$ = 6
* Minimum rest between consecutive duties $R_{min}$ = 9 h
* Base / domicile $Base = NKX$
* Positioning fee $C_{\text{pos}} = \$10000$

**Decision variables**:
$
x_p = \begin{cases}
1 & \text{if pairing } p \text{ is selected} \\
0 & \text{otherwise}
\end{cases}
\qquad \text{for all } p \in \mathcal{P}
$

**Cost of a pairing**:

Let $D_p$ be the set of duty segments inside pairing $p$ and let $F_p$ be its flight legs.  

$
\delta_p \;=\; \begin{cases}
1 & \text{if } \operatorname{depStn}(f_1) \ne \text{Base},\\
0 & \text{otherwise.}\
\end{cases}
$

Then,

$
c_p = \underbrace{\sum_{d\in D_p} H^{\text{duty}}_d}_{\text{duty hours}}\,\texttt{DutyCostPerHour}
\; +\; \underbrace{\sum_{f\in F_p} H^{\text{block}}_f}_{\text{block hours}}\,\texttt{ParingCostPerHour}
\; +\; \delta_p\,C_{\text{pos}}.
$

where
* $H^{duty}_d$ = clock-time length of duty $d$ (report to release).
* $H^{block}_f$ = airborne/block time of flight leg $f$.

**Objective**:

The objective is to minimize the total cost of the selected pairings:

$
\begin{aligned}
\min_{x}\; & \sum_{p\in\mathcal P} c_p\,x_p \\
\text{s.t.}\; & \sum_{p:\,f\in p} x_p = 1 \quad &\forall f\in \mathcal F\;\; (\text{each leg covered once}) \\
& x_p \in \{0,1\} &\forall p\in\mathcal P
\end{aligned}
$

The legality rules above are *baked into* the definition of $\mathcal P$; hence no additional constraints are needed in the objective of problem.


## Input Format  

The input file is a comma-separated file containing flight-leg data for a single aircraft fleet operating out of base **NKX**. Each row is one leg on one calendar day.
It has the following columns:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `FltNum` | string | Flight number (e.g. `FA680`). |
| `DptrDate` | mm/dd/yyyy | Local departure date. |
| `DptrTime` | hh:mm (24 h) | Local departure time. |
| `DptrStn` | string | Three-letter IATA station code. |
| `ArrvDate` | mm/dd/yyyy | Local arrival date. |
| `ArrvTime` | hh:mm | Local arrival time. |
| `ArrvStn` | string | Three-letter arrival station code. |
| `Comp` | string | Fleet / crew qualification indicator (here uniformly `C1F1`). |
| `DutyCostPerHour` | float | Hourly duty pay for crews operating this leg (NaN where unchanged from previous row). |
| `ParingCostPerHour` | float | Hourly per-diem / pairing pay (NaN where unchanged). |

**Assumptions**

* The base (domicile) for all pairings is **NKX** (every pairing must start and end at NKX).  
* Where `DutyCostPerHour` or `ParingCostPerHour` is missing, forward-fill the last non-missing value because the rate is constant for the fleet.  
* Legal-rule parameters are listed above. 

## Output Format  

Each **line** represents one selected pairing.  
A pairing is encoded as the space-separated list of _leg tokens_; a leg token is `FltNum_DepDate` where `DepDate` is in ISO `YYYY-MM-DD` format (this guarantees uniqueness when a flight number repeats on multiple days).

Example for a two-day pairing covering three legs:

```
FA680_2021-08-11 FA2_2021-08-12 FA872_2021-08-13
```

The file may contain any number of lines. Feasibility requirements:

1. Every leg in the input file appears in **exactly one** line.  
2. Within each line, legs obey the legality rules listed above.

## References  
- https://github.com/zhanwen/MathModel/tree/master/国赛试题/2021年研究生数学建模竞赛试题/F
- https://www.cis.jhu.edu/~xye/papers_and_ppts/ppts/airline_crewpairing_copy1.pdf
- https://pubsonline.informs.org/doi/10.1287/mnsc.39.6.736



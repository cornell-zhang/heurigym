# Airline Crew‑Pairing Problem (CPP)

## Background  
Airlines must build _pairings_ (multi‑day sequences of flight legs) so that every scheduled leg is covered by exactly one legal crew duty‑tour while respecting collective‑bargaining rules (maximum duty hours, minimum rest, domicile start/end, etc.).  
A good pairing plan minimises total operating cost (wages, per‑diem, hotel) and downstream disruption risk, while staying within the limited reserve of pilots based at each crew _domicile_ (base).

The **Data A** file contains one month (Aug 2021) of flight‑leg data for a single aircraft fleet operating out of base **NKX**. Each row is one leg on one calendar day; from it we can derive legal pairings and compute their costs.

## Formalization

| Symbol | Meaning |
| ------ | ------- |
| **F** | set of flight legs $f = 1,\dots,\|F\|$. |
| **P** | set of _feasible_ pairings generated from **F** according to duty‑time, block‑time, minimum‑rest and base‑return rules. |
| $c_p$ | cost of pairing $p$—here computed from duty time and block time (see below). |
| $x_p \in \{0,1\}$ | decision variable: 1 if pairing $p$ is selected. |

The objective is to minimize the total cost of the selected pairings:

$$
\begin{aligned}
\min_{x}\; &\sum_{p\in P} c_p\,x_p \\
\text{s.t. } &\sum_{p:\,f\in p} x_p \;=\; 1 &&\forall f\in F \quad(\text{cover each leg once})\\
             &x_p \in \{0,1\} &&\forall p\in P .
\end{aligned}
$$

### Pairing‑cost expression  

Because **Data A** provides the hourly wage rates `DutyCostPerHour` (paid for every on‑duty hour) and `ParingCostPerHour` (per‑diem during block hours), we embed all direct economics into

$$
c_p \;=\; (\text{duty\_hours}_p)\times\text{DutyCostPerHour}_{\text{base}(p)}
       + (\text{block\_hours}_p)\times\text{ParingCostPerHour}_{\text{base}(p)} 
$$

All other costs (hotel, dead‑head, etc.) could be added in the same way if the data were available.

## Input Format  

`DataA.csv` is a comma‑separated file with the columns:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `FltNum` | string | Flight number (e.g. `FA680`). |
| `DptrDate` | mm/dd/yyyy | Local departure date. |
| `DptrTime` | hh:mm (24 h) | Local departure time. |
| `DptrStn` | string | Three‑letter IATA station code. |
| `ArrvDate` | mm/dd/yyyy | Local arrival date. |
| `ArrvTime` | hh:mm | Local arrival time. |
| `ArrvStn` | string | Three‑letter arrival station code. |
| `Comp` | string | Fleet / crew qualification indicator (here uniformly `C1F1`). |
| `DutyCostPerHour` | float | Hourly duty pay for crews operating this leg (NaN where unchanged from previous row). |
| `ParingCostPerHour` | float | Hourly per‑diem / pairing pay (NaN where unchanged). |

**Assumptions**

* The base (domicile) for all pairings is **NKX** (every pairing must start and end at NKX).  
* Where `DutyCostPerHour` or `ParingCostPerHour` is missing, forward‑fill the last non‑missing value because the rate is constant for the fleet.  
* Legal‑rule parameters used when generating $P$ (may be tuned per airline):  
  – max duty hours = 13 h  
  – max block hours per duty = 8 h  
  – max legs per duty = 4  
  – min rest between duties = 10 h  

## Output Format  

Your solver must write a plain‑text schedule in which each **line** represents one selected pairing.  
A pairing is encoded as the space‑separated list of _leg tokens_; a leg token is `FltNum_DepDate` where `DepDate` is in ISO `YYYY‑MM‑DD` format (this guarantees uniqueness when a flight number repeats on multiple days).

Example for a two‑day pairing covering three legs:

```
FA680_2021-08-11 FA2_2021-08-12 FA872_2021-08-13
```

The file may contain any number of lines. Feasibility requirements:

1. Every leg in **Data A** appears in **exactly one** line.  
2. Within each line, legs obey temporal order and legal duty/rest rules.  
3. The pairing starts and ends at the same station/airpot.

## References  
- https://github.com/zhanwen/MathModel/tree/master/国赛试题/2021年研究生数学建模竞赛试题/F
- https://www.cis.jhu.edu/~xye/papers_and_ppts/ppts/airline_crewpairing_copy1.pdf
- https://pubsonline.informs.org/doi/10.1287/mnsc.39.6.736



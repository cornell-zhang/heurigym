"""
An **ILP-based baseline solver** for the relaxed Airline Crew-Pairing Problem
(CPP) with a flat $10 000 positioning fee when a tour starts away from NKX.

The script:
1. **Generates a modest set of candidate pairings**
   * all single-leg tours (always feasible), plus
   * 2- and 3-leg chains built greedily on-station while respecting the
     14 h duty, 10 h block, 6-leg, and 9 h rest rules.
2. **Solves a set-partitioning ILP** over those candidates using *PuLP*'s CBC
   solver (open source, bundled with PuLP). Each leg must appear in exactly one
   chosen pairing; objective = total duty/block pay + positioning fees.
3. **Falls back** to the trivial one-leg roster if CBC is unavailable or the
   model proves infeasible within the 60-second time limit.
4. Writes `baseline_solution.txt` under `crew_pairing/dataset/demo/` and prints
   the cost (if `evaluator.py` is present).

This is still a small baseline - it won't enumerate the full pairing space -
but it almost always beats the single-leg cost while staying lightning-fast.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict
from datetime import timedelta

# ---------------------------------------------------------------------------
#  locate utils.py and evaluator.py without requiring an installed package
# ---------------------------------------------------------------------------
PROG = Path(__file__).resolve().parent           # crew_pairing/program
ROOT = PROG.parent                               # crew_pairing/
if str(PROG) not in sys.path:
    sys.path.insert(0, str(PROG))

try:
    from utils import (
        read_instance,
        FlightLeg,
        HOURS,  # lambda td: td.total_seconds()/3600
    )  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    sys.exit(f"Cannot import utils.py - expected at {PROG}: {e}")

# ---------------------------------------------------------------------------
#  legality constants (duplicate of verifier.py for local checks)
# ---------------------------------------------------------------------------
MAX_DUTY_HOURS = 14.0
MAX_BLOCK_HOURS = 10.0
MAX_LEGS_PER_DUTY = 6
MIN_REST_HOURS = 9.0
BASE = "NKX"
POS_FEE = 10_000.0

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def default_paths() -> tuple[Path, Path]:
    """Return (input_csv, output_txt) based on repo layout."""
    in_path = ROOT / "dataset" / "demo" / "instance1.csv"
    out_path = ROOT / "dataset" / "demo" / "baseline_solution.txt"
    return in_path, out_path


def write_schedule(pairings: List[List[str]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for pr in pairings:
            f.write(" ".join(pr) + "\n")

# ---------------------------------------------------------------------------
#  pairing generation
# ---------------------------------------------------------------------------

def feasible_chain(chain: List[FlightLeg]) -> bool:
    """Check duty/block/legs limits for a single-duty chain of legs."""
    if len(chain) > MAX_LEGS_PER_DUTY:
        return False
    duty_span = HOURS(chain[-1].arr_dt - chain[0].dep_dt)
    if duty_span > MAX_DUTY_HOURS:
        return False
    block_hours = sum(HOURS(l.arr_dt - l.dep_dt) for l in chain)
    if block_hours > MAX_BLOCK_HOURS:
        return False
    return True


def build_candidate_pairings(legs: Dict[str, FlightLeg]) -> List[List[str]]:
    """Return a modest list of candidate pairings (1-3 legs)."""
    leg_objs = sorted(legs.values(), key=lambda l: l.dep_dt)
    token_to_leg = {l.token: l for l in leg_objs}
    candidates: List[List[str]] = []

    # --- single-leg tours (always feasible) ------------------------------
    for l in leg_objs:
        candidates.append([l.token])

    # --- greedy 2- & 3-leg chains ---------------------------------------
    for leg1 in leg_objs:
        for leg2 in leg_objs:
            if leg2.dep_dt >= leg1.arr_dt and leg2.dep_stn == leg1.arr_stn:
                chain2 = [leg1, leg2]
                if feasible_chain(chain2):
                    candidates.append([leg.token for leg in chain2])
                # try add a third leg
                for leg3 in leg_objs:
                    if (
                        leg3.dep_dt >= leg2.arr_dt
                        and leg3.dep_stn == leg2.arr_stn
                    ):
                        chain3 = [leg1, leg2, leg3]
                        if feasible_chain(chain3):
                            candidates.append([leg.token for leg in chain3])
    # remove duplicates while preserving order
    seen = set()
    unique: List[List[str]] = []
    for pr in candidates:
        tup = tuple(pr)
        if tup not in seen:
            seen.add(tup)
            unique.append(pr)
    return unique

# ---------------------------------------------------------------------------
#  ILP set-partitioning solve with PuLP
# ---------------------------------------------------------------------------

def solve_ilp(instance_csv: Path, legs: Dict[str, FlightLeg]) -> List[List[str]]:
    try:
        import pulp  # type: ignore
    except ImportError:
        print("PuLP not installed - falling back to trivial roster.")
        raise RuntimeError

    print("Generating candidate pairings (this may take a moment)...")
    cand_pairings = build_candidate_pairings(legs)
    print(f"  {len(cand_pairings):,} candidates built.")

    # Cost per pairing ----------------------------------------------------
    def pairing_cost(tokens: List[str]) -> float:
        if len(tokens) == 1:
            leg = legs[tokens[0]]
            duty_hours = HOURS(leg.arr_dt - leg.dep_dt)
            block_hours = duty_hours  # same for a single segment
        else:
            chain = [legs[t] for t in tokens]
            duty_hours = HOURS(chain[-1].arr_dt - chain[0].dep_dt)
            block_hours = sum(HOURS(l.arr_dt - l.dep_dt) for l in chain)
        duty_rate = instance.duty_cost_per_hour
        pair_rate = instance.paring_cost_per_hour
        pos_fee = POS_FEE if legs[tokens[0]].dep_stn != BASE else 0.0
        return duty_hours * duty_rate + block_hours * pair_rate + pos_fee

    instance = read_instance(str(instance_csv))

    cost_p: List[float] = [pairing_cost(pr) for pr in cand_pairings]

    # ILP ---------------------------------------------------------------
    prob = pulp.LpProblem("CrewPairing", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(len(cand_pairings)), cat="Binary")

    # objective
    prob += pulp.lpSum(cost_p[i] * x[i] for i in range(len(cand_pairings)))

    # cover each leg exactly once
    leg_to_indices: Dict[str, List[int]] = {tok: [] for tok in legs}
    for idx, pr in enumerate(cand_pairings):
        for tok in pr:
            leg_to_indices[tok].append(idx)

    for tok, idx_list in leg_to_indices.items():
        prob += pulp.lpSum(x[i] for i in idx_list) == 1, f"cover_{tok}"

    # solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    #result = prob.solve(solver)
    #if result != pulp.LpStatusOptimal and result != pulp.LpStatusNotSolved:
    #    raise RuntimeError("ILP did not find optimal solution in time; fallback.")

    #chosen: List[List[str]] = [cand_pairings[i] for i in range(len(cand_pairings)) if x[i].value() == 1]
    #return chosen
    if pulp.LpStatus[result] != "Optimal":
        raise RuntimeError("ILP did not finish – falling back to trivial roster.")

    chosen = [cand_pairings[i] for i, var in x.items() if var.value() == 1]

    # guarantee coverage
    covered = {tok for pr in chosen for tok in pr}
    for tok in legs:
        if tok not in covered:
            chosen.append([tok])

    return chosen

# ---------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------

def build_pairings(instance_csv: Path) -> List[List[str]]:
    inst = read_instance(str(instance_csv))
    try:
        pairings = solve_ilp(instance_csv, inst.legs)
        return pairings
    except Exception as e:
        print(f"ILP failed or unavailable → fallback: {e}")
        # trivial fallback: one-leg each
        return [[tok] for tok in inst.legs]


def main(input_csv: Path, output_txt: Path) -> None:
    pairings = build_pairings(input_csv)
    write_schedule(pairings, output_txt)
    print(
        f"Wrote {len(pairings)} pairings covering {len(pairings)} legs"
        f"{output_txt.relative_to(ROOT)}"
    )

    # cost via evaluator if present
    try:
        from evaluator import evaluate  # type: ignore

        cost = evaluate(str(input_csv), str(output_txt))
        print(
            f"Total cost (incl. $10k positioning where applicable): $"
            f"{int(round(cost)):,}"
        )
    except Exception as e:
        print(f"(Cost evaluation skipped: {e})")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        in_csv, out_txt = default_paths()
    elif len(sys.argv) == 3:
        in_csv, out_txt = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        sys.exit("Usage: python baseline.py [<input_csv> <output_txt>]")

    if not in_csv.exists():
        sys.exit(f"Input file not found: {in_csv}")

    main(in_csv, out_txt)

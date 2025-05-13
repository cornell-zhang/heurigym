"""
Greedy long‑chain baseline 
==========================================================

This solver builds **very long pairings (multi‑duty chains)** to minimise the
number of lines in the output and cut positioning/duty overhead.  It is
inspired by your quick Pandas heuristic but adds:

* **Station‑aware extension:** always tries to keep the crew at the same
  airport to avoid extra pairings.
* **Earliest‑feasible search:** when extending a duty it scans all still‑open
  legs from the current station and picks the *earliest* that satisfies sit
  time ≤ 3 h and keeps the duty within 14 h / 10 block / 6 legs.
* **Rest jump:** if no such leg exists, it looks for the earliest leg from the
  same station ≥ 9 h later (new duty) and continues the chain.
* **Progress logs:** prints the length of every pairing as it is finalised.

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from datetime import datetime
import pandas as pd

PROG = Path(__file__).resolve().parent
ROOT = PROG.parent
if str(PROG) not in sys.path:
    sys.path.append(str(PROG))
try:
    sys.path.insert(0, str(ROOT / "program"))
    from utils import HOURS  # type: ignore
except ModuleNotFoundError:
    def HOURS(td):
        return td.total_seconds() / 3600.0

# Rule constants
MAX_DUTY_HOURS = 14.0
MAX_BLOCK_HOURS = 10.0
MAX_LEGS_PER_DUTY = 6
MIN_REST_HOURS = 9.0
MAX_SIT_HOURS = 3.0
BASE = "NKX"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_legs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["DutyCostPerHour"].ffill(inplace=True)
    df["ParingCostPerHour"].ffill(inplace=True)
    df["DptrDateTime"] = pd.to_datetime(df["DptrDate"] + " " + df["DptrTime"], format="%m/%d/%Y %H:%M")
    df["ArrvDateTime"] = pd.to_datetime(df["ArrvDate"] + " " + df["ArrvTime"], format="%m/%d/%Y %H:%M")
    df["BlockHours"] = (df["ArrvDateTime"] - df["DptrDateTime"]).dt.total_seconds() / 3600.0
    df["LegToken"] = df["FltNum"] + "_" + df["DptrDateTime"].dt.strftime("%Y-%m-%d")
    return df.sort_values("DptrDateTime").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Greedy builder
# ---------------------------------------------------------------------------

def build_pairings(df: pd.DataFrame) -> List[List[str]]:
    used = set()
    pairings: List[List[str]] = []
    # index by departure station for quick look‑up
    idx_by_dep = {}
    for i, row in df.iterrows():
        idx_by_dep.setdefault(row.DptrStn, []).append(i)

    while len(used) < len(df):
        # choose earliest unused leg, prefer BASE if available
        unused_indices = [i for i in range(len(df)) if df.loc[i, "LegToken"] not in used]
        base_indices = [i for i in unused_indices if df.loc[i, "DptrStn"] == BASE]
        start_idx = min(base_indices) if base_indices else min(unused_indices)
        row = df.loc[start_idx]
        pairing = [row.LegToken]
        used.add(row.LegToken)

        last_arr = row.ArrvDateTime
        cur_stn = row.ArrvStn
        duty_start = row.DptrDateTime
        duty_legs = 1
        duty_block = row.BlockHours

        while True:
            # candidate legs from the SAME station not yet used
            candidates = [j for j in idx_by_dep.get(cur_stn, []) if df.loc[j, "LegToken"] not in used and df.loc[j, "DptrDateTime"] >= last_arr]
            next_idx = None
            # Try same‑duty first
            for j in candidates:
                sit = HOURS(df.loc[j, "DptrDateTime"] - last_arr)
                if sit < 0 or sit > MAX_SIT_HOURS:
                    continue
                new_duty_legs = duty_legs + 1
                new_block = duty_block + df.loc[j, "BlockHours"]
                new_span = HOURS(df.loc[j, "ArrvDateTime"] - duty_start)
                if new_duty_legs <= MAX_LEGS_PER_DUTY and new_block <= MAX_BLOCK_HOURS and new_span <= MAX_DUTY_HOURS:
                    next_idx = j
                    break  # earliest feasible
            # If none, try after rest ≥9 h (new duty)
            if next_idx is None:
                for j in candidates:
                    rest = HOURS(df.loc[j, "DptrDateTime"] - last_arr)
                    if rest >= MIN_REST_HOURS:
                        next_idx = j
                        duty_start = df.loc[j, "DptrDateTime"]
                        duty_legs = 0
                        duty_block = 0.0
                        break
            if next_idx is None:
                break  # cannot extend further
            # append leg
            row_j = df.loc[next_idx]
            pairing.append(row_j.LegToken)
            used.add(row_j.LegToken)
            last_arr = row_j.ArrvDateTime
            cur_stn = row_j.ArrvStn
            duty_legs += 1
            duty_block += row_j.BlockHours
        print(f"Built pairing with {len(pairing)} legs (unused remaining: {len(df)-len(used)})")
        pairings.append(pairing)
    return pairings

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_schedule(pairings: List[List[str]], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for pr in pairings:
            f.write(" ".join(pr) + "\n")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python baseline_greedy.py <input_csv> <output_txt>")
    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    df_legs = load_legs(inp)
    pairings = build_pairings(df_legs)
    write_schedule(pairings, outp)
    print(f"Wrote {len(pairings)} pairings → {outp}")
    try:
        from evaluator import evaluate  # type: ignore
        cost = evaluate(str(inp), str(outp))
        print(f"Evaluator cost: ${cost:,.0f}")
    except Exception:
        pass

"""
A *guaranteed‑feasible* reference solver for the relaxed Airline Crew‑Pairing
Problem where a tour may begin away from the domicile (NKX) but incurs a flat
positioning cost of **$10 000** if it does.

Strategy
--------
**Simplest thing that always works:** put **each flight leg in its own
pairing**.  A single‑leg duty automatically satisfies

* duty span ≤ 14 h (it equals block + brief/debrief, always < 14 h here),
* block hours ≤ 10 h (data set’s longest block is ≪ 10 h),
* legs per duty ≤ 6 (it’s 1),
* rest rules are vacuously true inside a single leg,
* coverage is obviously complete & disjoint.

The trade‑off is cost (lots of position fees + minimum duty pay) but the roster
is *provably feasible for any data set*.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
#  locate utils.py without requiring an installed package
# ---------------------------------------------------------------------------
PROG = Path(__file__).resolve().parent           # crew_pairing/program
ROOT = PROG.parent                               # crew_pairing/
if str(PROG) not in sys.path:
    sys.path.insert(0, str(PROG))

try:
    from utils import read_instance, FlightLeg  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    sys.exit(f"Cannot import utils.py – expected at {PROG}: {e}")

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def default_paths() -> tuple[Path, Path]:
    """Return (input_csv, output_txt) based on repo layout."""
    in_path = ROOT / "dataset" / "demo" / "DataA.csv"
    out_path = ROOT / "dataset" / "demo" / "baseline_solution.txt"
    return in_path, out_path


def write_schedule(pairings: List[List[str]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for pr in pairings:
            f.write(" ".join(pr) + "\n")


# ---------------------------------------------------------------------------
#  main routine – one leg per pairing
# ---------------------------------------------------------------------------

def build_pairings(legs: dict[str, FlightLeg]) -> List[List[str]]:
    """Return a list of single‑leg pairings sorted by departure."""
    sorted_legs = sorted(legs.values(), key=lambda l: l.dep_dt)
    return [[leg.token] for leg in sorted_legs]


def main(input_csv: Path, output_txt: Path) -> None:
    inst = read_instance(str(input_csv))
    pairings = build_pairings(inst.legs)
    write_schedule(pairings, output_txt)

    print(
        f"Wrote {len(pairings)} pairings covering {len(inst.legs)} legs → "
        f"{output_txt.relative_to(ROOT)}"
    )

    # optional: cost via evaluator.py if present
    try:
        from evaluator import evaluate  # available in PROG

        cost = evaluate(str(input_csv), str(output_txt))
        print(
            f"Total cost (incl. $10k positioning where applicable): $"
            f"{int(round(cost)):,}"
        )
    except Exception as e:  # pragma: no cover
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

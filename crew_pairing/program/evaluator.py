"""Cost-only evaluator for the Airline Crew-Pairing Problem.

Signature matches the grading spec:
    evaluate(<input_file>, <solution_file>)
"""

from typing import List
from utils import read_instance, HOURS  # HOURS: lambda td.total_seconds()/3600.0

# A crew duty ends when the next report time is at least this many hours
REST_THRESHOLD_HOURS: float = 9.0


def _parse_schedule(path: str) -> List[List[str]]:
    """Read the solver output file.

    Each non-blank, non-comment line represents **one pairing** and contains
    the space-separated list of leg tokens produced by the solver.

    Returns
    -------
    list[list[str]]
        Outer list = pairings; inner list = ordered leg tokens.
    """
    pairings: List[List[str]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue  # skip blank/comment lines
            pairings.append(line.split())
    return pairings


def evaluate(input_file: str, solution_file: str) -> float:
    """Compute the total cost of a crew-pairing solution.

    Parameters
    ----------
    input_file : str
        Path to **DataA.csv** (instance file).
    solution_file : str
        Path to schedule file produced by the solver.

    Returns
    -------
    float
        Total pairing cost.

    Cost of one pairing *p*:

    .. math::
        cost_p = duty\_hours_p \times DutyCostPerHour +\
                 block\_hours_p \times ParingCostPerHour

    where *duty_hours_p* is the sum of duty lengths (a duty is a maximal
    sequence of legs separated by ≥ 10 h rest) and *block_hours_p* is the sum
    of airborne times of its legs.

    The function raises ``ValueError`` if any leg is uncovered or covered more
    than once.
    """
    instance = read_instance(input_file)  # -> Instance object defined in utils
    legs = instance.legs  # dict[token, FlightLeg]

    pairings = _parse_schedule(solution_file)

    # ---------- coverage check -------------------------------------------
    covered = set()
    for pr in pairings:
        for tok in pr:
            if tok in covered:
                raise ValueError(f"Leg {tok} appears in more than one pairing.")
            covered.add(tok)

    missing = set(legs) - covered
    if missing:
        raise ValueError(f"Solution missing {len(missing)} legs (e.g., {next(iter(missing))}).")

    # ---------- pay rates -------------------------------------------------
    duty_rate = instance.duty_cost_per_hour
    pairing_rate = instance.paring_cost_per_hour

    total_cost: float = 0.0

    for pr in pairings:
        # Sort legs chronologically (robust to solver order)
        leg_objs = sorted((legs[tok] for tok in pr), key=lambda l: l.dep_dt)

        # --- block hours: sum of airborne time ---------------------------
        block_hours = sum(HOURS(l.arr_dt - l.dep_dt) for l in leg_objs)

        # --- duty hours: partition by rest >= 10 h -----------------------
        duty_hours = 0.0
        duty_start = leg_objs[0].dep_dt
        prev_arr = leg_objs[0].arr_dt

        for leg in leg_objs[1:]:
            rest = HOURS(leg.dep_dt - prev_arr)
            if rest >= REST_THRESHOLD_HOURS:
                # close previous duty segment
                duty_hours += HOURS(prev_arr - duty_start)
                duty_start = leg.dep_dt
            prev_arr = leg.arr_dt

        # close final duty
        duty_hours += HOURS(prev_arr - duty_start)

        total_cost += duty_hours * duty_rate + block_hours * pairing_rate

    return total_cost


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("Usage: evaluate <instance.csv> <solution_file>")

    try:
        cost = evaluate(sys.argv[1], sys.argv[2])
        print(int(round(cost)))
    except ValueError as e:
        print(f"Invalid solution: {e}", file=sys.stderr)
        sys.exit(1)

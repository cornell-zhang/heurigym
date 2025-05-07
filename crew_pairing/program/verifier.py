from __future__ import annotations
from typing import List, Tuple
from utils import HOURS, read_instance, FlightLeg

# ------------------------------- legality parameters ------------------------
MAX_DUTY_HOURS = 14.0  # h
MAX_BLOCK_HOURS = 10.0  # h per duty
MAX_LEGS_PER_DUTY = 6
MIN_REST_HOURS = 9.0  # h between duties
BASE = "NKX"  # domicile


# helper
def _parse_schedule(path: str) -> List[List[str]]:
    """Read schedule file → list of pairings (each is list of leg tokens)."""
    pairings: List[List[str]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            pairings.append(line.split())
    return pairings


# verify
def verify(input_file: str, output_file: str) -> Tuple[bool, str]:
    """Validate crew‑pairing solution for the given instance.

    Returns
    -------
    (is_valid, error_message)
        If *is_valid* is False, *error_message* describes the first violation
        encountered.  If valid, message is the empty string.
    """
    inst = read_instance(input_file)
    legs_dict = inst.legs  # token → FlightLeg

    try:
        pairings = _parse_schedule(output_file)
    except FileNotFoundError as e:
        return False, str(e)

    # ---------------- coverage check -------------------------------------
    covered = set()
    for pr in pairings:
        for tok in pr:
            if tok not in legs_dict:
                return False, f"Unknown leg token {tok} in solution file."
            if tok in covered:
                return False, f"Leg {tok} covered more than once."
            covered.add(tok)
    missing = set(legs_dict) - covered
    if missing:
        return False, f"Solution missing {len(missing)} legs (e.g., {next(iter(missing))})."

    # ---------------- legality per pairing -------------------------------
    for idx, pr in enumerate(pairings, 1):
        leg_objs: List[FlightLeg] = [legs_dict[tok] for tok in pr]
        # ensure chronological order
        for i in range(1, len(leg_objs)):
            if leg_objs[i - 1].arr_dt > leg_objs[i].dep_dt:
                return False, (
                    f"Pairing {idx} is not in chronological order: {leg_objs[i-1].token} arrives "
                    f"after departure of {leg_objs[i].token}."
                )

        ## base start/end
        #if leg_objs[0].dep_stn != BASE or leg_objs[-1].arr_stn != BASE:
        #    return False, f"Pairing {idx} does not start and end at base {BASE}."    

        # iterate duty segments separated by >=10h rest
        duty_start_idx = 0
        while duty_start_idx < len(leg_objs):
            duty_end_idx = duty_start_idx
            while duty_end_idx + 1 < len(leg_objs):
                rest = HOURS(leg_objs[duty_end_idx + 1].dep_dt - leg_objs[duty_end_idx].arr_dt)
                if rest >= MIN_REST_HOURS:
                    break  # rest separates duties
                duty_end_idx += 1

            duty_legs = leg_objs[duty_start_idx : duty_end_idx + 1]
            duty_hours = HOURS(duty_legs[-1].arr_dt - duty_legs[0].dep_dt)
            block_hours = sum(HOURS(l.arr_dt - l.dep_dt) for l in duty_legs)
            num_legs = len(duty_legs)

            if duty_hours > MAX_DUTY_HOURS:
                return False, (
                    f"Duty in pairing {idx} exceeds max duty hours: {duty_hours:.1f} > {MAX_DUTY_HOURS}."
                )
            if block_hours > MAX_BLOCK_HOURS:
                return False, (
                    f"Duty in pairing {idx} exceeds max block hours: {block_hours:.1f} > {MAX_BLOCK_HOURS}."
                )
            if num_legs > MAX_LEGS_PER_DUTY:
                return False, (
                    f"Duty in pairing {idx} has {num_legs} legs (max {MAX_LEGS_PER_DUTY})."
                )

            # ensure next rest >=10h if there is another duty
            if duty_end_idx + 1 < len(leg_objs):
                rest_hours = HOURS(
                    leg_objs[duty_end_idx + 1].dep_dt - duty_legs[-1].arr_dt
                )
                if rest_hours < MIN_REST_HOURS:
                    return False, (
                        f"Rest between duties in pairing {idx} is {rest_hours:.1f} h (< {MIN_REST_HOURS})."
                    )

            duty_start_idx = duty_end_idx + 1

    # if we reached here everything passed
    return True, ""


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("Usage: verify <instance.csv> <solution_file>")

    ok, msg = verify(sys.argv[1], sys.argv[2])
    if ok:
        print("VALID")
    else:
        print(f"INVALID: {msg}")
        sys.exit(1)

#!/usr/bin/env python3
"""Utility helpers for the Airline Crew‑Pairing benchmark.

Provides
--------
read_instance(file_path) -> Instance
    Parse a CSV instance file (e.g., DataA.csv) into an `Instance` object
    containing all flight‑leg records and globally applicable pay rates.

HOURS(timedelta) -> float
    Quick helper converting a datetime.timedelta to fractional hours.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Mapping

# ---------------------------------------------------------------------------
# Time formats used by the airline data set
# ---------------------------------------------------------------------------
TIME_FMT = "%m/%d/%Y %H:%M"  # e.g. "08/11/2021 08:00"
TOKEN_DATE_FMT = "%Y-%m-%d"  # used in solver/evaluator tokens

HOURS = lambda td: td.total_seconds() / 3600.0  # noqa: E731  (intentional lambda)

# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightLeg:
    """A single aircraft movement NKX→PGX operated on a specific day."""

    flt: str
    dep_dt: datetime
    arr_dt: datetime
    dep_stn: str
    arr_stn: str

    @property
    def token(self) -> str:
        """Unique leg token used in solver output (Flt_DepDate)."""
        return f"{self.flt}_{self.dep_dt.strftime(TOKEN_DATE_FMT)}"


@dataclass
class Instance:
    """Parsed airline instance data required by the evaluator."""

    legs: Dict[str, FlightLeg]  # mapping token → FlightLeg
    duty_cost_per_hour: float   # global duty pay rate for this fleet
    paring_cost_per_hour: float  # global per‑diem rate

    # Convenience proxy to mimic attribute access used elsewhere
    @property
    def legs_list(self):
        return list(self.legs.values())


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_datetime(date_str: str, time_str: str) -> datetime:
    """Convert separate date+time strings into a naive datetime object."""
    return datetime.strptime(f"{date_str.strip()} {time_str.strip()}", TIME_FMT)


def read_instance(file_path: str | Path) -> Instance:
    """Read DataA.csv and return an :class:`Instance`.

    The CSV must contain at least the columns documented in the README.
    Pay‑rate columns (`DutyCostPerHour`, `ParingCostPerHour`) are forward‑filled
    when blank because the rate is constant for the entire fleet/month.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    legs: Dict[str, FlightLeg] = {}
    prev_duty_rate: float | None = None
    prev_paring_rate: float | None = None

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        required_cols = {
            "FltNum",
            "DptrDate",
            "DptrTime",
            "DptrStn",
            "ArrvDate",
            "ArrvTime",
            "ArrvStn",
            "DutyCostPerHour",
            "ParingCostPerHour",
        }
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Instance file missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            # --- parse flight leg -------------------------------------------------
            leg = FlightLeg(
                flt=row["FltNum"].strip(),
                dep_dt=_parse_datetime(row["DptrDate"], row["DptrTime"]),
                arr_dt=_parse_datetime(row["ArrvDate"], row["ArrvTime"]),
                dep_stn=row["DptrStn"].strip(),
                arr_stn=row["ArrvStn"].strip(),
            )
            if leg.token in legs:
                raise ValueError(f"Duplicate leg token {leg.token} in instance file.")
            legs[leg.token] = leg

            # --- pay rates --------------------------------------------------------
            duty_raw = row.get("DutyCostPerHour", "").strip()
            paring_raw = row.get("ParingCostPerHour", "").strip()

            if duty_raw:
                prev_duty_rate = float(duty_raw)
            if paring_raw:
                prev_paring_rate = float(paring_raw)

    if prev_duty_rate is None or prev_paring_rate is None:
        raise ValueError("Duty or pairing cost per hour missing/blank in entire file.")

    return Instance(
        legs=legs,
        duty_cost_per_hour=prev_duty_rate,
        paring_cost_per_hour=prev_paring_rate,
    )

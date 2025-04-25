#!/usr/bin/env python3
"""
evaluator.py - cost evaluator for LUT-mapped BLIF designs.

Cost metric = total number of LUTs (“nodes”) reported by ABC.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------
# Regexes that cover both styles of ABC `ps` output
# ---------------------------------------------------------------------
_PS_SUMMARY_RE: Final = re.compile(r"\bnd\s*=\s*(\d+)\b")


class ABCNotFoundError(RuntimeError):
    """Raised when the `abc` executable is missing on $PATH."""


def _get_nodes_via_abc(blif: Path) -> int:
    """Run ABC `ps` and return the LUT count (nodes)."""
    if shutil.which("abc") is None:
        raise ABCNotFoundError(
            "`abc` binary not found in $PATH. "
            "Install Berkeley ABC from https://github.com/berkeley-abc/abc"
        )

    cmd = ["abc", "-c", f"read_blif {blif.as_posix()}; ps"]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    out = proc.stdout

    # Try modern summary style first
    m = _PS_SUMMARY_RE.search(out)
    if m:
        return int(m.group(1))


    raise RuntimeError(
        "Could not parse LUT count from ABC output:\n" + out
    )


def evaluate(blif_path: Path) -> int:
    """Return LUT count for a mapped BLIF design."""
    blif_path = blif_path.expanduser().resolve()
    if not blif_path.is_file():
        raise FileNotFoundError(blif_path)
    return _get_nodes_via_abc(blif_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the LUT count of a mapped BLIF design."
    )
    parser.add_argument(
        "blif",
        type=Path,
        help="Path to the BLIF file produced by the mapper.",
    )
    args = parser.parse_args()
    lut_count = evaluate(args.blif)
    print(f"# of LUTs = {lut_count}")

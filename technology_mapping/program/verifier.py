#!/usr/bin/env python3
"""
verifier.py - BLIF-to-BLIF equivalence checker using ABC's CEC engine.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _check_abc_available() -> None:
    """Raise RuntimeError if `abc` binary is not on PATH."""
    if shutil.which("abc") is None:
        raise RuntimeError(
            "`abc` executable not found on $PATH. "
            "Install Berkeley ABC (https://github.com/berkeley-abc/abc) "
            "or add it to your PATH before running the verifier."
        )


def _run_cec(golden: Path, candidate: Path) -> subprocess.CompletedProcess[str]:
    """Run `cec` in ABC and return the CompletedProcess."""
    cmd = [
        "abc",
        "-c",
        f"cec {golden.as_posix()} {candidate.as_posix()}"
    ]
    # capture both stdout and stderr as text, force locale-independent output
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )


def verify(input_file: str, output_file: str) -> bool:
    """Verify logical equivalence between two BLIF files using ABC.

    Parameters
    ----------
    input_file : str
        Path to the reference (golden) BLIF net-list.
    output_file : str
        Path to the candidate BLIF net-list produced by the mapper.

    Returns
    -------
    bool
        True  - designs are equivalent  
        False - a counter-example was found *or* verification failed
    """
    # -- Pre-flight checks --------------------------------------------------
    golden = Path(input_file).expanduser().resolve()
    candidate = Path(output_file).expanduser().resolve()

    if not golden.is_file():
        raise FileNotFoundError(f"Golden file not found: {golden}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate file not found: {candidate}")

    _check_abc_available()

    # -- Run CEC ------------------------------------------------------------
    proc = _run_cec(golden, candidate)
    output = proc.stdout

    # ABC exits with status 0 for "equivalent", 1 for "not equivalent".
    if proc.returncode == 0 and "Networks are equivalent" in output:
        return True

    # Optional: print diagnostic info when verification fails.
    # Comment these lines out if silent failure is preferred.
    sys.stderr.write(
        f"[verifier] Designs differ or verification failed.\n"
        f"[verifier] abc output:\n{output}\n"
    )
    return False


# --------------------------------------------------------------------------
# Stand-alone usage: `python verifier.py golden.blif candidate.blif`
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify logical equivalence between two BLIF files using ABC."
    )
    parser.add_argument("golden", help="Path to the reference (golden) BLIF net-list")
    parser.add_argument("candidate", help="Path to the candidate BLIF net-list produced by the mapper")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed error output")
    
    args = parser.parse_args()
    
    is_equiv = verify(args.golden, args.candidate)
    print("EQUIVALENT" if is_equiv else "NOT EQUIVALENT")
    sys.exit(0 if is_equiv else 1)

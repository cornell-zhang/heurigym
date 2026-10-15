"""The evaluator scores the GC fitness Phi: constant and copied answers get the worst
quality, and the min-cut optimum gets full quality.

Run from the repository root after the dataset is downloaded to _datasets/:
    python -m pytest protein_sequence_design/tests
PROTEIN_DATA overrides the directory that holds demo/ and eval/.
"""
import contextlib
import io
import json
import os
import sys

import pytest
from Bio.SeqUtils import seq1

TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get(
    "PROTEIN_DATA", os.path.join(os.path.dirname(TASK), "_datasets", "protein_sequence_design")
)
sys.path.insert(0, os.path.join(TASK, "program"))
sys.path.insert(0, os.path.join(TASK, "baseline"))

import min_cut_baseline  # noqa: E402
from evaluator import evaluate  # noqa: E402
from metric import normalize_score  # noqa: E402
from utils import ALPHA, BETA, gc_terms, get_natural_hp_sequence  # noqa: E402
from verifier import verify  # noqa: E402

with open(os.path.join(TASK, "baseline", "baseline.json")) as f:
    BASELINE = json.load(f)

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason=f"dataset not found at {DATA}")


def pdb_path(pid):
    for split in ("demo", "eval"):
        path = os.path.join(DATA, split, pid + ".pdb")
        if os.path.exists(path):
            return path
    pytest.skip(f"{pid}.pdb not found")


def quality(pid, sequence, tmp_path):
    solution = tmp_path / f"{pid}.txt"
    solution.write_text(sequence)
    ok, message = verify(pdb_path(pid), str(solution))
    assert ok, message
    return normalize_score(evaluate(pdb_path(pid), str(solution)), BASELINE[pid])


def min_cut_optimum(pid):
    residues, contacts, sasa = gc_terms(pdb_path(pid))
    graph, source, sink = min_cut_baseline.build_graph(contacts, sasa, ALPHA, BETA)
    with contextlib.redirect_stdout(io.StringIO()):
        return min_cut_baseline.get_optimal_sequence(graph, source, sink, len(residues))


def natural_sequence(pid):
    residues, _, _ = gc_terms(pdb_path(pid))
    return get_natural_hp_sequence("".join(seq1(r.get_resname()) for r in residues))


@pytest.mark.parametrize("pid", sorted(BASELINE))
def test_constant_and_copied_answers_get_worst_quality(pid, tmp_path):
    natural = natural_sequence(pid)
    one_flip = ("P" if natural[0] == "H" else "H") + natural[1:]
    answers = {"all_P": "P" * len(natural), "all_H": "H" * len(natural), "natural": natural, "one_flip": one_flip}
    for name, sequence in answers.items():
        assert quality(pid, sequence, tmp_path) <= 1e-10, name


@pytest.mark.parametrize("pid", sorted(BASELINE))
def test_min_cut_optimum_gets_full_quality(pid, tmp_path):
    assert quality(pid, min_cut_optimum(pid), tmp_path) == pytest.approx(1.0, abs=1e-4)  # baseline.json is rounded

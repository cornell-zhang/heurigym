import numpy as np

def normalize_score(score, baseline):
    if np.isnan(score) or np.isnan(baseline):
        raise ValueError("Invalid score or baseline")
    if baseline == 0:
        baseline = 1 + 1e-6
    if score == 0:
        score = 1 + 1e-6
    return min(1, np.log10(baseline) / np.log10(score))
def normalize_score(score, baseline):
    if baseline == 0:
        baseline = 1e-6
    elif score == 0:
        score = 1e-6
    return min(1, baseline / score)
def normalize_score(score, baseline):
    return min(1, (100 - baseline) / (100 - score))
def normalize_score(score, baseline):
    return min(1, (1 + baseline) / (1 + score))
def normalize_score(score, baseline):
    # score and baseline are GC fitness values Phi (lower is better). The baseline is the
    # min-cut optimum, and the all-P sequence has Phi = 0.
    if score >= 0:
        return 1e-10  # no better than all-P; avoids division by zero and log(0)
    return min(1, score / baseline)

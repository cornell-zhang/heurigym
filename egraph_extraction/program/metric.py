def normalize_score(score, baseline):
    if score > 0 and baseline > 0:
        return min(1, baseline / score)
    elif score < 0 and baseline < 0:
        return min(1, score / baseline)
    else:
        raise ValueError("Invalid score or baseline")

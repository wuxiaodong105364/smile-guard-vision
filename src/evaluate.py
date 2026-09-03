"""Evaluate landmark NME and classification metrics."""

import math

import numpy as np


def normalized_mean_error(predicted, ground_truth, normalizer="bbox_diagonal"):
    """Compute NME between two 68/478-point landmark sets."""
    if len(predicted) != len(ground_truth):
        raise ValueError("landmark length mismatch")
    if not predicted:
        return None
    distances = []
    for p, g in zip(predicted, ground_truth):
        distances.append(math.sqrt((p["x"] - g["x"]) ** 2 + (p["y"] - g["y"]) ** 2))
    if normalizer == "interpupillary":
        left_eye = ground_truth[36]
        right_eye = ground_truth[45]
        d = math.sqrt((left_eye["x"] - right_eye["x"]) ** 2 + (left_eye["y"] - right_eye["y"]) ** 2)
    else:
        xs = [p["x"] for p in ground_truth]
        ys = [p["y"] for p in ground_truth]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        d = math.sqrt(width ** 2 + height ** 2)
    return float(np.mean(distances) / max(d, 1e-6))


def report_nme_by_region(predicted, ground_truth):
    """Print NME for brow, eye, nose, mouth regions (68-point layout)."""
    regions = {
        "brow": (18, 28),
        "eye": (37, 49),
        "nose": (28, 37),
        "mouth": (49, 69),
    }
    result = {}
    for name, (start, end) in regions.items():
        p = predicted[start - 1 : end - 1]
        g = ground_truth[start - 1 : end - 1]
        result[name] = normalized_mean_error(p, g)
        print(name, "NME", round(result[name], 4))
    return result

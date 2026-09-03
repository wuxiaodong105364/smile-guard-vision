"""Load annotation CSV and build feature matrix."""

from pathlib import Path

import pandas as pd

from src.features import compute_metrics


def load_annotations(csv_path=None):
    path = Path(csv_path) if csv_path else Path(__file__).resolve().parents[1] / "data" / "annotations" / "labels.csv"
    if not path.exists():
        raise FileNotFoundError("labels.csv not found: %s" % path)
    return pd.read_csv(path)


def build_dataset(df, landmarks=None):
    """Build X (feature matrix) and y (HB grade) from annotations.

    If `landmarks` is a dict {frame_path: points}, metrics are computed from it.
    Otherwise, columns starting with "metric_" are expected in the CSV.
    """
    rows = []
    y = []
    for _, row in df.iterrows():
        pts = landmarks.get(row["frame_path"]) if landmarks else None
        if pts is not None:
            metrics = compute_metrics(pts, source=row.get("landmark_source", "mediapipe"))
        else:
            metrics = {
                key: row.get("metric_" + key)
                for key in [
                    "brow_height_asymmetry",
                    "eye_closure_ratio",
                    "mouth_corner_displacement",
                    "nasolabial_fold_asymmetry",
                    "total_symmetry_score",
                ]
            }
        if metrics is None or metrics["total_symmetry_score"] is None:
            continue
        rows.append(metrics)
        y.append(row["hb_grade"])
    return pd.DataFrame(rows), y

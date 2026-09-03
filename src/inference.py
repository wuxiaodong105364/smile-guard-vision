"""Run inference on one image or video and print aiReference JSON."""

import json

import cv2

from src.features import (
    compute_blendshape_metrics,
    compute_metrics,
    facial_risk_probability,
    to_ai_reference,
)
from src.landmarks import detect_frame_detailed, detect_video_detailed


def _metrics_for(detail):
    metrics = compute_metrics(detail["landmarks"], source="mediapipe")
    if metrics:
        metrics.update(compute_blendshape_metrics(detail["blendshapes"]))
    return metrics


def analyze_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError("cannot read image: %s" % path)
    detail = detect_frame_detailed(frame)
    if not detail["landmarks"]:
        return to_ai_reference(None, confidence=0.0)
    return to_ai_reference(_metrics_for(detail))


def analyze_video(path):
    frames = detect_video_detailed(path)
    if not frames:
        return to_ai_reference(None, confidence=0.0)
    best = None
    for _, points, blendshapes in frames:
        metrics = _metrics_for({"landmarks": points, "blendshapes": blendshapes})
        if not metrics:
            continue
        risk = facial_risk_probability(metrics)
        if best is None or (risk is not None and risk > best[1]):
            best = (metrics, risk)
    if best is None:
        return to_ai_reference(None, confidence=0.0)
    metrics = best[0]
    risk = best[1]
    if risk is None:
        return to_ai_reference(metrics, confidence=0.0)
    return to_ai_reference(metrics, confidence=round(max(risk, 1.0 - risk), 2))


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    result = analyze_video(path) if path.lower().endswith((".mp4", ".mov", ".avi")) else analyze_image(path)
    print(json.dumps({"aiReference": result, "doctorResult": None}, ensure_ascii=False, indent=2))

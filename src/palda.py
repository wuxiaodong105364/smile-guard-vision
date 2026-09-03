"""Research-only healthy/peripheral/central classifier for Palda landmarks.

The classifier is trained on Palda 68-point landmarks but applied here to
MediaPipe metric features with the same names. This is a domain-shifted,
research-reference signal and must never be presented as a diagnosis.
"""

import json
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "models"
LABELS_ZH = {
    "healthy": "健康人（研究提示）",
    "peripheral": "周围性面瘫（研究提示）",
    "central": "中枢性面瘫（研究提示）",
}
DISCLAIMER = (
    "研究版提示，非诊断结论。Palda 模型训练自 68 点关键点数据，"
    "当前通过 MediaPipe 特征近似推理，存在口径差异，最终以专业医生结果为准。"
)

_MODEL_CACHE = None
_COLUMNS_CACHE = None


def _derived_features(metrics):
    out = {}
    out["abs_brow_signed"] = abs(metrics["brow_asymmetry_signed"])
    out["abs_mouth_signed"] = abs(metrics["mouth_corner_displacement_signed"])
    out["mouth_to_brow_ratio"] = metrics["mouth_corner_displacement"] / (
        metrics["brow_height_asymmetry"] + 1e-6
    )
    out["eye_to_brow_ratio"] = metrics["eye_aperture_asymmetry"] / (
        metrics["brow_height_asymmetry"] + 1e-6
    )
    out["mouth_brow_diff"] = (
        metrics["mouth_corner_displacement"] - metrics["brow_height_asymmetry"]
    )
    return out


def load_model():
    global _MODEL_CACHE, _COLUMNS_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _COLUMNS_CACHE
    import joblib

    model_file = MODEL_PATH / "palda_classifier.pkl"
    columns_file = MODEL_PATH / "palda_feature_columns.json"
    if not model_file.exists() or not columns_file.exists():
        raise FileNotFoundError(
            "Palda model not found. Run scripts/train_palda.py first."
        )
    _MODEL_CACHE = joblib.load(str(model_file))
    _COLUMNS_CACHE = json.loads(columns_file.read_text(encoding="utf-8"))
    return _MODEL_CACHE, _COLUMNS_CACHE


def predict_from_metrics(metrics):
    """Return a research hint dict or None when features are insufficient."""
    if not metrics:
        return None
    required = [
        "brow_height_asymmetry",
        "brow_asymmetry_signed",
        "left_brow_height",
        "right_brow_height",
        "eye_closure_ratio",
        "left_eye_aperture",
        "right_eye_aperture",
        "eye_aperture_asymmetry",
        "mouth_corner_displacement",
        "mouth_corner_displacement_signed",
        "left_mouth_corner_y",
        "right_mouth_corner_y",
        "mouth_width",
        "mouth_opening",
        "mouth_center_offset",
        "total_symmetry_score",
    ]
    if any(key not in metrics or metrics[key] is None for key in required):
        return None
    model, columns = load_model()
    row = dict(metrics)
    row.update(_derived_features(metrics))
    features = [float(row[col]) for col in columns]
    proba = model.predict_proba([features])[0]
    probabilities = {
        str(model.classes_[i]): round(float(proba[i]), 4)
        for i in range(len(model.classes_))
    }
    predicted = str(model.classes_[int(proba.argmax())])
    return {
        "supported": True,
        "model": "palda_rf_v1",
        "predictedClass": predicted,
        "predictedLabel": LABELS_ZH[predicted],
        "probabilities": probabilities,
        "confidence": round(float(proba.max()), 4),
        "disclaimer": DISCLAIMER,
    }

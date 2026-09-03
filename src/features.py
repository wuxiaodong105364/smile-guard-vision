"""Feature metrics computed from facial landmarks.

Supports MediaPipe Face Mesh (478 points) and 68-point landmark models.
The MediaPipe indices below are the commonly used Face Mesh indices;
verify them against your model version before production.
"""

import math


MEDIAPIPE_GROUPS = {
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_upper": 386,
    "right_eye_lower": 374,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_upper_inner": 13,
    "mouth_lower_inner": 14,
    "nose_tip": 1,
}

LANDMARK68_GROUPS = {
    "left_brow": list(range(18, 23)),
    "right_brow": list(range(23, 28)),
    "left_eye": list(range(37, 43)),
    "right_eye": list(range(43, 49)),
    "left_eye_upper": 37,
    "left_eye_lower": 41,
    "right_eye_upper": 43,
    "right_eye_lower": 47,
    "mouth_left": 49,
    "mouth_right": 53,
    "mouth_upper_inner": 51,
    "mouth_lower_inner": 57,
    "nose_tip": 30,
}


def _group_avg(points, indices):
    xs = [points[i]["x"] for i in indices if i < len(points)]
    ys = [points[i]["y"] for i in indices if i < len(points)]
    if not xs:
        return None
    return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}


def _distance(a, b):
    if not a or not b:
        return None
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def _clamp(value, low, high):
    return max(low, min(high, value))


def compute_metrics(points, source="mediapipe"):
    """Compute asymmetry metrics from landmark list.

    points: list of dicts with "x" and "y" normalized to [0, 1].
    source: "mediapipe" (478 points) or "68" (68 points).
    Returns dict with feature columns or None when landmarks are insufficient.
    """
    groups = MEDIAPIPE_GROUPS if source == "mediapipe" else LANDMARK68_GROUPS
    left_brow = _group_avg(points, groups["left_brow"])
    right_brow = _group_avg(points, groups["right_brow"])
    left_eye = _group_avg(points, groups["left_eye"])
    right_eye = _group_avg(points, groups["right_eye"])
    left_eye_upper = points[groups["left_eye_upper"]] if groups["left_eye_upper"] < len(points) else None
    left_eye_lower = points[groups["left_eye_lower"]] if groups["left_eye_lower"] < len(points) else None
    right_eye_upper = points[groups["right_eye_upper"]] if groups["right_eye_upper"] < len(points) else None
    right_eye_lower = points[groups["right_eye_lower"]] if groups["right_eye_lower"] < len(points) else None
    mouth_left = points[groups["mouth_left"]] if groups["mouth_left"] < len(points) else None
    mouth_right = points[groups["mouth_right"]] if groups["mouth_right"] < len(points) else None
    mouth_upper = points[groups["mouth_upper_inner"]] if groups["mouth_upper_inner"] < len(points) else None
    mouth_lower = points[groups["mouth_lower_inner"]] if groups["mouth_lower_inner"] < len(points) else None
    nose_tip = points[groups["nose_tip"]] if groups["nose_tip"] < len(points) else None
    if (
        not left_brow
        or not right_brow
        or not left_eye_upper
        or not left_eye_lower
        or not right_eye_upper
        or not right_eye_lower
        or not mouth_left
        or not mouth_right
        or not mouth_upper
        or not mouth_lower
    ):
        return None

    face_width = _distance(mouth_left, mouth_right) or 1.0
    left_aperture = _distance(left_eye_upper, left_eye_lower) or 0.0
    right_aperture = _distance(right_eye_upper, right_eye_lower) or 0.0
    left_brow_y = left_brow["y"]
    right_brow_y = right_brow["y"]
    brow_height_asymmetry = abs(left_brow["y"] - right_brow["y"]) / face_width
    eye_closure_ratio = (left_aperture + right_aperture) / (2.0 * face_width)
    mouth_corner_displacement = abs(mouth_left["y"] - mouth_right["y"]) / face_width
    mouth_width = _distance(mouth_left, mouth_right) or 0.0
    mouth_opening = _distance(mouth_upper, mouth_lower) or 0.0
    mouth_center_x = (mouth_left["x"] + mouth_right["x"]) / 2.0
    face_center_x = nose_tip["x"] if nose_tip else mouth_center_x
    mouth_center_offset = (mouth_center_x - face_center_x) / face_width
    eye_aperture_asymmetry = (
        abs(left_aperture - right_aperture) / max(left_aperture, right_aperture)
        if max(left_aperture, right_aperture) > 0
        else 0.0
    )
    nasolabial_fold_asymmetry = None  # requires depth or contour analysis
    raw_score = 1.0 - (brow_height_asymmetry * 1.2 + eye_closure_ratio * 0.8 + mouth_corner_displacement * 1.4)
    total_symmetry_score = round(_clamp(raw_score, 0.0, 1.0) * 100)

    return {
        "brow_height_asymmetry": round(brow_height_asymmetry, 3),
        "brow_asymmetry_signed": round((left_brow_y - right_brow_y) / face_width, 3),
        "left_brow_height": round(left_brow_y, 4),
        "right_brow_height": round(right_brow_y, 4),
        "eye_closure_ratio": round(eye_closure_ratio, 3),
        "left_eye_aperture": round(left_aperture, 4),
        "right_eye_aperture": round(right_aperture, 4),
        "eye_aperture_asymmetry": round(eye_aperture_asymmetry, 3),
        "mouth_corner_displacement": round(mouth_corner_displacement, 3),
        "mouth_corner_displacement_signed": round(
            (mouth_left["y"] - mouth_right["y"]) / face_width, 3
        ),
        "left_mouth_corner_y": round(mouth_left["y"], 4),
        "right_mouth_corner_y": round(mouth_right["y"], 4),
        "mouth_width": round(mouth_width, 4),
        "mouth_opening": round(mouth_opening, 4),
        "mouth_center_offset": round(mouth_center_offset, 3),
        "nasolabial_fold_asymmetry": nasolabial_fold_asymmetry,
        "total_symmetry_score": total_symmetry_score,
    }


def classify_expression(metrics, rest=None):
    """Bucket one frame into the 6-expression protocol by heuristic thresholds.

    `rest` is a dict of reference values estimated from neutral frames:
    min_aperture, mean_brow_y, mouth_width, mouth_opening, mean_mouth_corner_y.
    This is a first-pass heuristic for unconstrained video; controlled capture
    should use the explicit 6-expression sequence instead.
    """
    if not metrics:
        return "unknown"
    required = [
        "left_eye_aperture",
        "right_eye_aperture",
        "left_brow_height",
        "right_brow_height",
        "mouth_width",
        "mouth_opening",
        "mouth_corner_displacement_signed",
    ]
    if any(key not in metrics for key in required):
        return "unknown"

    min_aperture = min(metrics["left_eye_aperture"], metrics["right_eye_aperture"])
    mean_brow_y = (metrics["left_brow_height"] + metrics["right_brow_height"]) / 2.0
    mean_mouth_corner_y = (
        metrics.get("left_mouth_corner_y", 0.0) + metrics.get("right_mouth_corner_y", 0.0)
    ) / 2.0

    if rest is None:
        rest = {
            "min_aperture": min_aperture,
            "mean_brow_y": mean_brow_y,
            "mouth_width": metrics["mouth_width"],
            "mouth_opening": metrics["mouth_opening"],
            "mean_mouth_corner_y": mean_mouth_corner_y,
        }

    if min_aperture < max(rest["min_aperture"] * 0.2, 0.004):
        return "tight_eye_close"
    if min_aperture < max(rest["min_aperture"] * 0.45, 0.01):
        return "gentle_eye_close"
    if mean_brow_y < rest["mean_brow_y"] - 0.015:
        return "brow_up"

    mouth_width_ratio = metrics["mouth_width"] / max(rest["mouth_width"], 1e-6)
    mouth_open_ratio = metrics["mouth_opening"] / max(rest["mouth_opening"], 1e-6)
    corner_raise = rest["mean_mouth_corner_y"] - mean_mouth_corner_y
    if mouth_width_ratio > 1.10 and corner_raise > 0.008:
        return "smile"
    if mouth_width_ratio < 0.85 and mouth_open_ratio > 1.2:
        return "lip_pucker"
    return "rest"


BLENDSHAPE_PAIRS = {
    "bs_smile_asym": ("mouthSmileLeft", "mouthSmileRight"),
    "bs_brow_down_asym": ("browDownLeft", "browDownRight"),
    "bs_brow_outer_up_asym": ("browOuterUpLeft", "browOuterUpRight"),
    "bs_eye_blink_asym": ("eyeBlinkLeft", "eyeBlinkRight"),
    "bs_eye_squint_asym": ("eyeSquintLeft", "eyeSquintRight"),
    "bs_cheek_squint_asym": ("cheekSquintLeft", "cheekSquintRight"),
    "bs_mouth_frown_asym": ("mouthFrownLeft", "mouthFrownRight"),
    "bs_mouth_press_asym": ("mouthPressLeft", "mouthPressRight"),
    "bs_mouth_stretch_asym": ("mouthStretchLeft", "mouthStretchRight"),
    "bs_mouth_upper_up_asym": ("mouthUpperUpLeft", "mouthUpperUpRight"),
    "bs_mouth_lower_down_asym": ("mouthLowerDownLeft", "mouthLowerDownRight"),
}


def compute_blendshape_metrics(blendshapes):
    """Compute left/right expression asymmetry from ARKit blendshape scores."""
    out = {}
    values = []
    for key, (left, right) in BLENDSHAPE_PAIRS.items():
        diff = abs(blendshapes.get(left, 0.0) - blendshapes.get(right, 0.0))
        out[key] = round(diff, 4)
        values.append(diff)
    if values:
        out["expression_asymmetry"] = round(sum(values) / len(values), 4)
        out["expression_asymmetry_max"] = round(max(values), 4)
    else:
        out["expression_asymmetry"] = 0.0
        out["expression_asymmetry_max"] = 0.0
    return out


def grade_from_score(score):
    if score >= 82:
        return "I"
    if score >= 72:
        return "II"
    if score >= 60:
        return "III"
    if score >= 48:
        return "IV"
    if score >= 35:
        return "V"
    return "VI"


FACIAL_RISK_THRESHOLD = 0.40


def facial_risk_probability(metrics):
    """Estimate palsy risk from first-pass asymmetry features.

    Coefficients come from a logistic model fit on frame-level features of the
    public palsynet baseline (video labels are noisy, so this is only an AI
    reference signal, not a diagnostic model).
    """
    if not metrics:
        return None
    required = [
        "brow_height_asymmetry",
        "eye_aperture_asymmetry",
        "mouth_corner_displacement",
        "mouth_center_offset",
        "eye_closure_ratio",
    ]
    if any(key not in metrics or metrics[key] is None for key in required):
        return None
    logit = (
        -2.82559
        + 4.43011 * metrics["eye_aperture_asymmetry"]
        + 9.54027 * metrics["mouth_corner_displacement"]
        + 6.96234 * metrics["eye_closure_ratio"]
        - 4.77735 * metrics["brow_height_asymmetry"]
        - 1.45122 * abs(metrics["mouth_center_offset"])
        + 3.50582 * metrics.get("expression_asymmetry", 0.0)
        + 4.34903 * metrics.get("expression_asymmetry_max", 0.0)
    )
    return 1.0 / (1.0 + math.exp(-logit))


def to_ai_reference(metrics, confidence=None):
    """Convert metrics to the same aiReference JSON shape used by the mini-program."""
    if metrics is None:
        return {
            "possibleFacialPalsy": None,
            "symmetryScore": None,
            "hbGrade": None,
            "confidence": 0.0,
            "informationInsufficient": True,
            "findings": "人脸关键点不足，无法完成分析。",
            "advice": "请重新拍摄正脸照片或视频。",
        }
    risk = facial_risk_probability(metrics)
    score = round(_clamp((1.0 - risk) * 100, 0, 100)) if risk is not None else metrics["total_symmetry_score"]
    grade = grade_from_score(score)
    possible = risk is not None and risk >= FACIAL_RISK_THRESHOLD
    if confidence is None:
        confidence = round(max(risk, 1.0 - risk), 2) if risk is not None else 0.0
    risk_text = "%.0f%%" % (risk * 100) if risk is not None else "--"
    return {
        "possibleFacialPalsy": possible,
        "symmetryScore": score,
        "hbGrade": grade,
        "confidence": confidence,
        "riskProbability": round(risk, 3) if risk is not None else None,
        "expressionAsymmetry": metrics.get("expression_asymmetry"),
        "informationInsufficient": False,
        "findings": (
            "AI 参考：不对称风险概率 %s。%s"
            % (
                risk_text,
                "面部不对称特征提示需进一步检查，不作为诊断结论。"
                if possible
                else "单张照片未检出明确不对称；不等于排除面瘫，请按 6 表情流程补拍并请医生确认。",
            )
        ),
        "advice": (
            "AI 参考结果不是诊断结论，最终以专业医生结果为准。"
            if possible
            else "AI 参考未检出不代表没有面瘫，请补拍抬眉、闭眼、微笑、嘟嘴照片并联系主管医生。"
        ),
    }

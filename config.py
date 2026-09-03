from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ANNOTATIONS_CSV = DATA_DIR / "annotations" / "labels.csv"
MODEL_DIR = BASE_DIR / "models"

LANDMARK_MODEL = DATA_DIR / "models" / "face_landmarker.task"
LANDMARK_SOURCE = "mediapipe"  # mediapipe | dlib68

EXPRESSIONS = [
    "rest",
    "brow_up",
    "gentle_eye_close",
    "tight_eye_close",
    "smile",
    "lip_pucker",
]

FEATURE_COLS = [
    "brow_height_asymmetry",
    "eye_closure_ratio",
    "mouth_corner_displacement",
    "nasolabial_fold_asymmetry",
    "total_symmetry_score",
]

TARGET_COL = "hb_grade"
TEST_SIZE = 0.2
RANDOM_STATE = 42

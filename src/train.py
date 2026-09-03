"""Train a lightweight HB grade classifier on feature vectors."""

import json
from pathlib import Path

import numpy as np

from src.dataset import build_dataset, load_annotations


def train(csv_path=None, model_dir=None):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is not installed. Run: pip install -r requirements.txt"
        ) from exc

    df = load_annotations(csv_path)
    X, y = build_dataset(df)
    if len(X) < 10:
        raise ValueError("Not enough annotated samples. Need at least 10 rows.")
    X = X.fillna(0.0)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0)
    print(report)
    print("accuracy:", round(accuracy_score(y_test, pred), 3))

    out_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(clf, out_dir / "hb_classifier.pkl")
        (out_dir / "feature_columns.json").write_text(
            json.dumps(list(X.columns)), encoding="utf-8"
        )
        print("model saved to", out_dir)
    except Exception:
        print("joblib not installed; model not persisted.")
    return clf


if __name__ == "__main__":
    train()

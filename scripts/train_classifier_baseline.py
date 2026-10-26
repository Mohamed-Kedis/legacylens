"""
CPU baseline document classifier: HOG layout features + color histogram -> Logistic Regression.
Runs on a laptop in ~1 minute. Gives you a number to beat with the CNN.

    python scripts/train_classifier_baseline.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from legacylens_ai.classify.doc_classifier import MODELS, image_features  # noqa: E402


def load(data: Path, split: str):
    rows = [json.loads(line) for line in open(data / "labels.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == split]
    X = np.stack([image_features(Image.open(data / r["file"]).convert("RGB")) for r in rows])
    return X, np.array([r["class"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    data = Path(a.data)
    Xtr, ytr = load(data, "train")
    Xte, yte = load(data, "test")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    classes = list(model.classes_)
    print(classification_report(yte, pred, digits=3))
    print("confusion:", classes)
    print(confusion_matrix(yte, pred, labels=classes))
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "classes": classes}, MODELS / "doc_classifier_baseline.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "classifier_baseline.json").write_text(json.dumps(
        {"accuracy": float((pred == yte).mean()), "n_test": int(len(yte))}, indent=2))
    print("saved -> models/doc_classifier_baseline.joblib")


if __name__ == "__main__":
    main()

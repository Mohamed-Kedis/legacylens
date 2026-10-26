"""
Document type classifier: id_card | passport | contract | financial | photo | other

Three layers, best available wins:
    1. ONNX CNN    models/doc_classifier.onnx      (train on Kaggle: scripts/train_classifier_torch.py)
    2. CPU baseline models/doc_classifier_baseline.joblib (scripts/train_classifier_baseline.py)
    3. Rules on OCR text + PII + faces (always available, used as fallback when model is unsure)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

MODELS = Path(__file__).resolve().parents[3] / "models"
CLASSES = ["id_card", "passport", "contract", "financial", "photo", "other"]
MIN_MODEL_CONF = 0.6

KW = {
    "passport": r"passport|travel\s+document|جواز|وثيقة\s+سفر",
    "id_card": r"بطاقة|الهوية|القومي|القوعي|national\s+id|identity\s+card",
    "contract": r"عقد|الطرف\s+الأول|البند|agreement|contract|توكيل|إقرار",
    "financial": r"statement|balance|iban|كشف\s+حساب|رصيد|بنك|bank|debit|credit|فاتورة|invoice",
}


def rule_based(text: str, entities: List[str], faces: int) -> dict:
    t = text.lower()
    n_words = len(t.split())
    hits = {k: len(re.findall(p, t)) for k, p in KW.items()}
    if "PASSPORT_NUMBER" in entities and hits["passport"]:
        label = "passport"
    elif "EG_NATIONAL_ID" in entities and (faces or hits["id_card"]) and n_words < 120 and not hits["contract"]:
        label = "id_card"
    elif hits["contract"] >= 2:
        label = "contract"
    elif {"EG_IBAN", "CREDIT_CARD", "IBAN_CODE"} & set(entities) or hits["financial"] >= 2:
        label = "financial"
    elif faces and n_words < 25:
        label = "photo"
    elif n_words < 5:
        label = "photo"
    else:
        label = "other"
    return {"label": label, "confidence": None, "method": "rules"}


# ---------------------------------------------------------------- learned models
def image_features(img: Image.Image) -> np.ndarray:
    """Cheap features for the CPU baseline: HOG of the layout + color histogram."""
    from skimage.feature import hog

    g = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    h = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    hsv = np.asarray(img.convert("HSV").resize((64, 64)))
    hist = np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 255), density=True)[0] for c in range(3)])
    return np.concatenate([h, hist]).astype(np.float32)


@lru_cache(maxsize=1)
def _onnx():
    path = MODELS / "doc_classifier.onnx"
    if not path.exists():
        return None
    import onnxruntime as ort
    meta = MODELS / "doc_classifier.json"
    classes = json.loads(meta.read_text())["classes"] if meta.exists() else CLASSES
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), classes


@lru_cache(maxsize=1)
def _baseline():
    path = MODELS / "doc_classifier_baseline.joblib"
    if not path.exists():
        return None
    import joblib
    return joblib.load(path)


def model_predict(img: Image.Image) -> Optional[dict]:
    onnx = _onnx()
    if onnx:
        sess, classes = onnx
        x = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        logits = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        i = int(p.argmax())
        return {"label": classes[i], "confidence": round(float(p[i]), 3), "method": "cnn_onnx"}
    base = _baseline()
    if base:
        p = base["model"].predict_proba([image_features(img)])[0]
        i = int(p.argmax())
        return {"label": base["classes"][i], "confidence": round(float(p[i]), 3), "method": "hog_baseline"}
    return None


def classify(img: Optional[Image.Image], text: str, entities: List[str], faces: int) -> dict:
    rules = rule_based(text, entities, faces)
    pred = model_predict(img) if img is not None else None
    if pred and pred["confidence"] >= MIN_MODEL_CONF:
        pred["rules_say"] = rules["label"]
        return pred
    if pred:
        rules["model_said"] = pred
    return rules

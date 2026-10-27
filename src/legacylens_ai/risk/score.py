"""
Privacy Risk Score (0-100), explainable by design: every point comes from a named reason.

score = doc_type_base + sum(entity weights, repeats decay) + faces + metadata, capped at 100
Levels: low < 25 <= medium < 50 <= high < 75 <= critical
Tune WEIGHTS with your team / user research; keep the reasons list in the UI.
"""
from __future__ import annotations

from collections import Counter
from typing import List

ENTITY_WEIGHTS = {
    "EG_NATIONAL_ID": 35, "PASSPORT_NUMBER": 30, "CREDIT_CARD": 35, "EG_IBAN": 30, "IBAN_CODE": 30,
    "EG_MOBILE": 10, "EMAIL_ADDRESS": 8, "PERSON": 8, "LOCATION": 6, "DATE_TIME": 2,
    "IP_ADDRESS": 5, "URL": 1,
}
# identity documents are high risk even if OCR missed the number itself
DOC_TYPE_BASE = {"id_card": 50, "passport": 50, "financial": 15, "contract": 10,
                 "photo": 0, "other": 0}
META_WEIGHTS = {"GPS_LOCATION": 20, "FILENAME_PII": 10, "AUTHOR": 5, "DEVICE_INFO": 3, "CAPTURE_TIME": 2}
FACE_WEIGHT = 15
REPEAT_FACTOR = 0.3   # each extra occurrence of the same entity adds 30% of its weight
MAX_REPEATS = 3


def level(score: int) -> str:
    return "critical" if score >= 75 else "high" if score >= 50 else "medium" if score >= 25 else "low"


def risk_score(doc_type: str, pii: List[dict], faces: int, metadata: List[dict]) -> dict:
    reasons, total = [], 0.0
    base = DOC_TYPE_BASE.get(doc_type, 0)
    if base:
        total += base
        reasons.append({"reason": f"Document type: {doc_type}", "points": base})
    for ent, n in Counter(p["entity"] for p in pii).items():
        w = ENTITY_WEIGHTS.get(ent, 3)
        pts = w + w * REPEAT_FACTOR * (min(n, MAX_REPEATS) - 1)
        total += pts
        reasons.append({"reason": f"{n} x {ent}", "points": round(pts, 1)})
    if faces:
        pts = FACE_WEIGHT + 5 * (min(faces, 4) - 1)
        total += pts
        reasons.append({"reason": f"{faces} face(s) visible", "points": pts})
    for m in metadata:
        pts = META_WEIGHTS.get(m["type"], 0)
        if pts:
            total += pts
            reasons.append({"reason": f"Metadata: {m['type']}", "points": pts})
    score = int(min(100, round(total)))
    return {"score": score, "level": level(score),
            "reasons": sorted(reasons, key=lambda r: r["points"], reverse=True)}

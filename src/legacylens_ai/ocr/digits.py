"""
Custom Arabic-Indic digit reader (٠١٢٣٤٥٦٧٨٩) for the national-ID field.

Why: Tesseract's Arabic model reads Eastern Arabic numerals very poorly, and Egyptian
ID cards print the national ID in exactly those digits. This is LegacyLens' own model:

    1. locate the field       -> OCR word that looks like "القومي" (fuzzy match)
    2. segment                -> max-channel (kills colored watermarks) + Otsu + connected components
    3. classify each glyph    -> small MLP trained on synthetic renders (scripts/train_digit_model.py)

Upgrade path (Kaggle GPU): CRNN + CTC over the whole strip, no segmentation needed.
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "digits_mlp.joblib"
GLYPH = 20
FIELD_LABELS = ["القومي", "قومي", "الرقم"]


def binarize(img: Image.Image) -> np.ndarray:
    """Max over RGB channels makes colored (red/blue) watermarks near-white, keeps black ink."""
    arr = np.asarray(img.convert("RGB")).max(axis=2).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw  # ink = 255


def segment(img: Image.Image) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    bw = binarize(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = [tuple(stats[i]) for i in range(1, n)]
    if not comps:
        return []
    max_area = max(c[4] for c in comps)
    H = bw.shape[0]
    keep = [c for c in comps if c[4] >= 0.03 * max_area and c[3] < 0.95 * H
            and c[0] > 0 and c[0] + c[2] < bw.shape[1]]   # drop specks and border-touching junk
    keep.sort(key=lambda c: c[0])  # numbers are written left-to-right even in Arabic text
    line_h = max((c[3] for c in keep), default=1)
    out = []
    for x, y, w, h, _ in keep:
        g = bw[y:y + h, x:x + w]
        side = max(w, h)
        sq = np.zeros((side, side), np.uint8)
        sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = g
        g = cv2.resize(sq, (GLYPH, GLYPH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        out.append((np.concatenate([g.ravel(), [h / line_h, w / max(h, 1)]]), (x, y, x + w, y + h)))
    return out


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def best_number_cluster(glyphs, expected: int = 14):
    """The crop can also contain pieces of the label (letters, dots). Split glyphs into
    word-like clusters at large horizontal gaps and keep the one closest to `expected` glyphs."""
    if len(glyphs) <= 1:
        return glyphs
    boxes = [b for _, b in glyphs]
    gaps = [boxes[i + 1][0] - boxes[i][2] for i in range(len(boxes) - 1)]
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    thr = max(2.5 * float(np.median(gaps)), 0.6 * med_h)
    clusters, cur = [], [glyphs[0]]
    for g, gap in zip(glyphs[1:], gaps):
        if gap > thr:
            clusters.append(cur)
            cur = []
        cur.append(g)
    clusters.append(cur)
    return min(clusters, key=lambda c: abs(len(c) - expected))


def read_digits(img: Image.Image, expected: int = 14) -> Optional[str]:
    model = _model()
    if model is None:
        return None
    glyphs = best_number_cluster(segment(img), expected)
    if not glyphs:
        return None
    X = np.stack([g for g, _ in glyphs])
    return "".join(str(int(d)) for d in model.predict(X))


def locate_nid_field(ocr, img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Find the 'الرقم القومي' label in OCR words; the value sits on the same line, to its left."""
    best, best_ratio = None, 0.0
    for w in ocr.words:
        for lab in FIELD_LABELS[:2]:
            r = difflib.SequenceMatcher(None, w.text.strip(":："), lab).ratio()
            if r > best_ratio:
                best, best_ratio = w, r
    if best is None or best_ratio < 0.7:  # 0.6 also matched "الميلاد"
        return None
    x0, y0, x1, y1 = best.bbox
    h = y1 - y0
    # include the word "الرقم" left of "القومي" if OCR split them
    left_limit = x0
    for w in ocr.words:
        if abs((w.bbox[1] + w.bbox[3]) / 2 - (y0 + y1) / 2) < h and w.bbox[2] <= x0 + 2 and \
                difflib.SequenceMatcher(None, w.text, "الرقم").ratio() > 0.6:
            left_limit = min(left_limit, w.bbox[0])
    return (5, max(0, y0 - int(1.0 * h)), max(10, left_limit - 8), min(img.height, y1 + int(1.0 * h)))


def reread_national_id(ocr, img: Image.Image) -> Optional[dict]:
    box = locate_nid_field(ocr, img)
    if box is None:
        return None
    from legacylens_ai.pii.egypt_recognizers import parse_egyptian_national_id

    crop = img.crop(box)
    tight = _tight_box(crop, box)  # redact only the digits, not the whole search strip
    # 1) Latin digits: Tesseract is good at these once restricted to one line + digit whitelist
    latin = _tesseract_digits(crop)
    if latin and parse_egyptian_national_id(latin):
        return {"text": latin, "bbox": tight, "method": "tesseract_digits"}
    # 2) Arabic-Indic digits: our own model
    digits = read_digits(crop)
    if digits and parse_egyptian_national_id(digits):
        return {"text": digits, "bbox": tight, "method": "digit_mlp"}
    return None


def _tight_box(crop: Image.Image, box) -> Tuple[int, int, int, int]:
    glyphs = best_number_cluster(segment(crop))
    if not glyphs:
        return box
    bs = [b for _, b in glyphs]
    return (box[0] + min(b[0] for b in bs), box[1] + min(b[1] for b in bs),
            box[0] + max(b[2] for b in bs), box[1] + max(b[3] for b in bs))


def _tesseract_digits(crop: Image.Image) -> Optional[str]:
    import re
    import pytesseract

    g = Image.fromarray(np.asarray(crop.convert("RGB")).max(axis=2))
    g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS)
    raw = pytesseract.image_to_string(g, lang="eng", config="--psm 7 -c tessedit_char_whitelist=0123456789")
    m = re.search(r"[23]\d{13}", raw.replace(" ", ""))
    return m.group() if m else None


def reread_iban(ocr, img: Image.Image) -> Optional[dict]:
    """IBANs are long Latin strings that full-page OCR often breaks. Re-read the line to the
    RIGHT of the 'IBAN' label with a character whitelist; keep it only if the mod-97 check passes."""
    import re
    import pytesseract
    from legacylens_ai.pii.egypt_recognizers import iban_is_valid

    label = next((w for w in ocr.words if w.text.upper().startswith("IBAN")), None)
    if label is None:
        return None
    x0, y0, x1, y1 = label.bbox
    h = y1 - y0
    box = (x1 + 2, max(0, y0 - h // 2), img.width - 5, min(img.height, y1 + h // 2))
    crop = Image.fromarray(np.asarray(img.convert("RGB").crop(box)).max(axis=2))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    raw = pytesseract.image_to_string(crop, lang="eng", config="--psm 7 -c tessedit_char_whitelist=EG0123456789 ")
    m = re.search(r"EG\d{27}", raw.replace(" ", ""))
    if m and iban_is_valid(m.group()):
        return {"text": m.group(), "bbox": box}
    return None

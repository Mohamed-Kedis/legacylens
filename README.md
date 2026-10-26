# LegacyLens AI — offline document privacy analyzer

The AI layer of **LegacyLens** (digital legacy vault). Before a file is encrypted, this service
scans it and returns: document type, the PII inside it (masked), faces, hidden metadata,
an explainable **Privacy Risk Score**, and a **suggested Legacy Policy**. It can also return a
**redacted copy**.

Runs 100% offline on open-source models: no file ever leaves your server, nothing is stored.

```
file ─► deskew ─► OCR (Tesseract ara+eng) ─► field re-read (custom digit model) ─► PII (Presidio + Egyptian recognizers)
                     │                                                                   │
                     ├─► faces (OpenCV) ─► metadata (EXIF GPS, PDF author, file name) ───┤
                     └─► doc classifier (HOG baseline → EfficientNet ONNX) ─────────────►┴─► risk score ─► policy suggestion
```

## What is custom (built for this project)
| Component | What it does |
|---|---|
| `pii/egypt_recognizers.py` | Egyptian national ID (structural validation + decodes birth date / governorate / gender), mobile, IBAN (mod-97), passport, labeled names in Arabic & English |
| `ocr/digits.py` + `scripts/train_digit_model.py` | **Own model**: Tesseract can't read Arabic-Indic digits (٠-٩); a field locator + segmenter + MLP reads the national-ID field |
| `ocr/ocr.py` | Projection-profile deskew, watermark-suppressing max-channel OCR pass |
| `data/synthetic_docs.py`, `data/synthetic_text.py` | Synthetic, clearly-watermarked fictional datasets (6 doc classes, labeled PII text with hard negatives) |
| `risk/score.py`, `policy/suggest.py` | Explainable scoring and Legacy-Policy rules |
| `scripts/evaluate_*.py` | Precision / recall / value-recall reports |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_models.sh                           # Tesseract ara/eng + spaCy
python scripts/train_digit_model.py --strings 3000        # ~1 min CPU
PYTHONPATH=src python -m legacylens_ai.data.synthetic_docs --out data/synthetic --n-per-class 200
python scripts/train_classifier_baseline.py --data data/synthetic
pytest -q
uvicorn api.main:app --reload --port 8000                 # open http://localhost:8000/docs
```

## API
| Method | Path | Returns |
|---|---|---|
| POST | `/scan` | JSON report (masked PII, risk, policy) |
| POST | `/redact` | PNG with PII blacked out and faces pixelated |
| POST | `/mask-text` | `{"masked_text": ...}` |

## Current results (synthetic data, see `reports/`)
| Metric | Value |
|---|---|
| PII on text (NID / mobile / IBAN / passport / email) | F1 = 1.00 |
| PERSON with label (`الاسم:` / `Name:`) | F1 = 1.00 |
| Arabic name in free text (no label) | recall 0.00 → needs Arabic NER (`LEGACYLENS_ARABIC_NER=1`) |
| Digit model (Arabic-Indic, 14-digit exact match) | 0.985 |
| Doc-type accuracy (HOG baseline) | 1.00 |
| **End-to-end national ID value recall from images** | **0.78** (clean 0.84, photographed/rotated 0.75) |
| End-to-end mobile / IBAN / passport | 1.00 / 0.80 / 0.65 |

**These are synthetic numbers and overestimate real-world performance.** Text templates and
document layouts are generated, so they are easier than real data. Before claiming accuracy,
evaluate on real public data (MIDV-2020, RVL-CDIP) and on a small consented test set.

## Known gaps / next steps
- Free-text Arabic names: enable CAMeLBERT NER (`CAMeL-Lab/bert-base-arabic-camelbert-mix-ner`)
- Addresses and Arabic dates are not reliably OCR'd → field re-read like the national ID
- Haar face detector misses drawn / small faces → YuNet (`cv2.FaceDetectorYN`) or MediaPipe
- Digit segmentation fails on ~11% of noisy strips → CRNN + CTC (Kaggle GPU)
- Doc classifier trained on synthetic only → add RVL-CDIP + MIDV-2020 (`notebooks/kaggle_train_classifier.ipynb`)

## Data ethics
All training data is synthetic and fictional, every ID-style image carries a visible
"SYNTHETIC / NOT VALID" watermark, and layouts don't copy any real government document.
Never collect real people's identity documents for training.

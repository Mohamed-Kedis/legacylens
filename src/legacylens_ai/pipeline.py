"""
End-to-end scan:  file -> OCR -> PII -> faces -> metadata -> doc type -> risk -> policy

Privacy rules of this service:
  * runs fully offline (no external AI API), files are processed in memory, never stored
  * the report returns MASKED values only, never the raw PII it found
"""
from __future__ import annotations

import time
from typing import Optional

from PIL import Image

from .classify.doc_classifier import classify
from .metadata.scanner import scan_metadata
from .ocr.digits import reread_iban, reread_national_id
from .ocr.ocr import extract_text
from .pii.analyzer import PIIAnalyzer
from .pii.egypt_recognizers import normalize_digits
from .policy.suggest import suggest_policy
from .redact.redactor import mask_value, redact_image
from .risk.score import risk_score
from .vision.faces import detect_faces

_analyzer: Optional[PIIAnalyzer] = None


def analyzer() -> PIIAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = PIIAnalyzer()
    return _analyzer


def scan(file_bytes: bytes, filename: str, want_redacted: bool = False) -> dict:
    t0 = time.perf_counter()
    ocr, pages = extract_text(file_bytes, filename)
    first_page: Optional[Image.Image] = pages[0] if pages else None

    # Custom model: re-read the national-ID field (Arabic-Indic digits) that Tesseract misses
    extras = []
    if first_page is not None and ocr.source == "ocr":
        nid = reread_national_id(ocr, first_page)
        if nid:
            extras.append({**nid, "label": "الرقم القومي"})
        iban = reread_iban(ocr, first_page)
        if iban:
            extras.append({**iban, "label": "IBAN"})
    full_text = ocr.text
    extra_spans = []  # (start, end, bbox) of appended field text
    for e in extras:
        line = f"\n{e['label']}: {e['text']}"
        start = len(full_text) + len(line) - len(e["text"])
        full_text += line
        extra_spans.append((start, len(full_text), e["bbox"]))

    findings, seen = [], set()
    for f in analyzer().analyze(full_text):
        key = (f.entity, normalize_digits(f.text).replace(" ", ""))
        if key not in seen:  # same value read twice (OCR + custom field reader)
            seen.add(key)
            findings.append(f)
    name_findings = analyzer().analyze(filename)  # PII hidden in the file name

    faces = detect_faces(first_page) if first_page is not None else []
    meta = scan_metadata(file_bytes, filename)
    meta += [{"type": "FILENAME_PII", "severity": "medium", "value": mask_value(f.entity, f.text),
              "message": f"File name contains {f.entity}"} for f in name_findings]

    pii = [{"entity": f.entity, "masked": mask_value(f.entity, f.text), "score": f.score,
            **({"decoded": f.details} if f.details else {})} for f in findings]
    doc = classify(first_page, full_text, [f.entity for f in findings], len(faces))
    risk = risk_score(doc["label"], pii, len(faces), meta)
    policy = suggest_policy(doc["label"], risk, pii, len(faces))

    report = {
        "file": filename,
        "document_type": doc,
        "risk": risk,
        "pii": pii,
        "faces_detected": len(faces),
        "metadata": meta,
        "suggested_policy": policy,
        "ocr": {"source": ocr.source, "mean_confidence": ocr.mean_conf, "chars": len(ocr.text),
                "custom_fields": [{"field": e["label"], "method": e.get("method", "tesseract_line")} for e in extras]},
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }
    if want_redacted and first_page is not None:
        boxes = []
        for f in findings:
            boxes += ocr.boxes_for_span(f.start, f.end)
            boxes += [bb for (s0, e0, bb) in extra_spans if f.start < e0 and f.end > s0]
        report["_redacted_image"] = redact_image(first_page, boxes, faces)
    return report


def redact_text(text: str) -> dict:
    from .redact.redactor import mask_text
    findings = analyzer().analyze(text)
    return {"masked_text": mask_text(text, findings),
            "pii": [{"entity": f.entity, "masked": mask_value(f.entity, f.text)} for f in findings]}

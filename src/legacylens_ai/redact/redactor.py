"""
AI Redaction (image) + Smart Data Masking (text).

redact_image: black boxes over PII words (from OCR boxes) + pixelated faces.
mask_value:   partial masking that keeps a value recognizable to its owner,
              e.g. 29501011601234 -> 2••••••••••234
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw

DOT = "•"


def mask_value(entity: str, value: str) -> str:
    v = value.strip()
    if entity == "EG_NATIONAL_ID":
        return v[0] + DOT * (len(v) - 4) + v[-3:]
    if entity == "EG_MOBILE":
        digits = [c for c in v if c.isdigit()]
        return "".join(digits[:3]) + DOT * (len(digits) - 6) + "".join(digits[-3:])
    if entity in ("EG_IBAN", "IBAN_CODE"):
        compact = v.replace(" ", "")
        return compact[:4] + DOT * (len(compact) - 8) + compact[-4:]
    if entity == "CREDIT_CARD":
        digits = [c for c in v if c.isdigit()]
        return DOT * 4 + " " + DOT * 4 + " " + DOT * 4 + " " + "".join(digits[-4:])
    if entity == "PASSPORT_NUMBER":
        return v[0] + DOT * (len(v) - 3) + v[-2:]
    if entity == "EMAIL_ADDRESS" and "@" in v:
        user, dom = v.split("@", 1)
        return user[0] + DOT * max(1, len(user) - 1) + "@" + dom
    if entity == "PERSON":
        parts = v.split()
        return parts[0] + " " + " ".join(p[0] + "." for p in parts[1:]) if len(parts) > 1 else v[0] + DOT * 3
    return DOT * min(len(v), 8)


def mask_text(text: str, findings: Iterable) -> str:
    """Replace each finding in text with its masked form (right-to-left so offsets stay valid)."""
    out = text
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        out = out[:f.start] + mask_value(f.entity, f.text) + out[f.end:]
    return out


def _pixelate(img: Image.Image, box: Tuple[int, int, int, int], block: int = 12) -> None:
    region = img.crop(box)
    w, h = region.size
    small = region.resize((max(1, w // block), max(1, h // block)), Image.NEAREST)
    img.paste(small.resize((w, h), Image.NEAREST), box[:2])


def redact_image(img: Image.Image, pii_boxes: List[Tuple[int, int, int, int]],
                 face_boxes: List[Tuple[int, int, int, int]], pad: int = 4) -> Image.Image:
    out = img.convert("RGB").copy()
    for box in face_boxes:
        _pixelate(out, box)
    d = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in pii_boxes:
        d.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=(0, 0, 0))
    return out

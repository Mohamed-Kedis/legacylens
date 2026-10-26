"""
OCR with Tesseract (Arabic + English), keeping word boxes so PII found in the
text can be mapped back to pixels for redaction.

Install:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki  (tick Arabic during setup)
    Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-ara
    or run scripts/download_models.sh -> models/tessdata (used automatically)
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pytesseract
from PIL import Image, ImageOps

if os.getenv("TESSERACT_CMD"):  # Windows: C:\Program Files\Tesseract-OCR\tesseract.exe
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

LOCAL_TESSDATA = Path(__file__).resolve().parents[3] / "models" / "tessdata"
if (LOCAL_TESSDATA / "ara.traineddata").exists() and "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = str(LOCAL_TESSDATA)


@dataclass
class Word:
    text: str
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 in ORIGINAL image pixels
    conf: float
    start: int = 0  # char offsets inside OcrResult.text
    end: int = 0


@dataclass
class OcrResult:
    text: str
    words: List[Word] = field(default_factory=list)
    mean_conf: float = 0.0
    source: str = "ocr"  # "ocr" | "pdf_text"

    def boxes_for_span(self, start: int, end: int) -> List[Tuple[int, int, int, int]]:
        return [w.bbox for w in self.words if w.start < end and w.end > start]


def preprocess(img: Image.Image, max_channel: bool = False) -> Tuple[Image.Image, float]:
    """Grayscale + autocontrast + upscale small images (Tesseract likes ~30px text).
    max_channel=True: take max(R,G,B) per pixel -> colored watermarks/stamps turn near-white."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_channel:
        import numpy as np
        img = Image.fromarray(np.asarray(img).max(axis=2))
    img = img.convert("L")
    scale = 1.0
    if max(img.size) < 1600:
        scale = 1600 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img, cutoff=1), scale


def ocr_image(img: Image.Image, lang: str = "ara+eng", min_conf: float = 30) -> OcrResult:
    """Run OCR twice (plain gray, watermark-suppressed) only when the first pass looks weak."""
    first = _ocr_once(img, lang, min_conf, max_channel=False)
    if first.mean_conf >= 85 and len(first.words) > 3:
        return first
    second = _ocr_once(img, lang, min_conf, max_channel=True)
    quality = lambda r: sum(w.conf for w in r.words)  # noqa: E731
    return second if quality(second) > quality(first) else first


def _ocr_once(img: Image.Image, lang: str, min_conf: float, max_channel: bool) -> OcrResult:
    proc, scale = preprocess(img, max_channel=max_channel)
    data = pytesseract.image_to_data(proc, lang=lang, config="--psm 3",
                                     output_type=pytesseract.Output.DICT)
    words: List[Word] = []
    text_parts: List[str] = []
    pos, last_line = 0, None
    for i, raw in enumerate(data["text"]):
        w = (raw or "").strip()
        conf = float(data["conf"][i])
        if not w or conf < min_conf:
            continue
        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if last_line is not None:
            sep = " " if line_key == last_line else "\n"
            text_parts.append(sep)
            pos += 1
        last_line = line_key
        x, y, bw, bh = (data[k][i] for k in ("left", "top", "width", "height"))
        bbox = tuple(int(v / scale) for v in (x, y, x + bw, y + bh))
        words.append(Word(w, bbox, conf, pos, pos + len(w)))
        text_parts.append(w)
        pos += len(w)
    mean = sum(w.conf for w in words) / len(words) if words else 0.0
    return OcrResult("".join(text_parts), words, round(mean, 1))


def extract_text(file_bytes: bytes, filename: str) -> Tuple[OcrResult, List[Image.Image]]:
    """Return text (+ word boxes for images) and the page images (for faces / redaction)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _pdf(file_bytes)
    if name.endswith((".txt", ".md", ".csv")):
        return OcrResult(file_bytes.decode("utf-8", errors="ignore"), source="text"), []
    img = deskew(ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB"))
    return ocr_image(img), [img]


def estimate_skew(img: Image.Image, max_angle: float = 8, step: float = 0.5) -> float:
    """Projection-profile deskew: the rotation that makes text rows sharpest
    (max variance of the row sums) is the one that makes lines horizontal."""
    import numpy as np

    small = img.convert("L")
    small.thumbnail((700, 700))
    arr = np.asarray(small, dtype=np.float32)
    ink = Image.fromarray(((arr < arr.mean() - 0.5 * arr.std()) * 255).astype("uint8"))
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        rows = np.asarray(ink.rotate(float(a), resample=Image.BILINEAR), dtype=np.float32).sum(axis=1)
        score = float(np.var(np.diff(rows)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def deskew(img: Image.Image) -> Image.Image:
    angle = estimate_skew(img)
    if abs(angle) < 0.5:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def _pdf(file_bytes: bytes) -> Tuple[OcrResult, List[Image.Image]]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_bytes)
    texts, pages = [], []
    for page in pdf:
        texts.append(page.get_textpage().get_text_range())
        pages.append(page.render(scale=2).to_pil().convert("RGB"))
    digital = "".join(texts).strip()
    if len(digital) > 30:  # digital PDF: real text layer, no OCR needed
        return OcrResult("\n".join(texts), source="pdf_text"), pages
    # scanned PDF: OCR the first pages
    results = [ocr_image(p) for p in pages[:5]]
    return OcrResult("\n".join(r.text for r in results),
                     words=results[0].words if len(results) == 1 else [],
                     mean_conf=results[0].mean_conf if results else 0.0), pages

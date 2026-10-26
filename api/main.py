"""
LegacyLens AI scan service.

    uvicorn api.main:app --reload --port 8000     ->  http://localhost:8000/docs

The backend calls this service BEFORE encrypting a file (opt-in scan). Nothing is written
to disk; the response carries masked values only.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from legacylens_ai.pipeline import redact_text, scan  # noqa: E402

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".pdf", ".txt")

app = FastAPI(title="LegacyLens AI", version="0.1.0",
              description="Offline document analyzer: OCR + PII detection + risk score + policy suggestions")


async def _read(file: UploadFile) -> bytes:
    if not file.filename.lower().endswith(ALLOWED):
        raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(ALLOWED)}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB)")
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan_file(file: UploadFile = File(...)):
    data = await _read(file)
    return scan(data, file.filename)


@app.post("/redact")
async def redact_file(file: UploadFile = File(...)):
    data = await _read(file)
    report = scan(data, file.filename, want_redacted=True)
    img = report.pop("_redacted_image", None)
    if img is None:
        raise HTTPException(422, "Nothing to redact (no image page)")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"X-Risk-Score": str(report["risk"]["score"])})


class TextIn(BaseModel):
    text: str


@app.post("/mask-text")
def mask(body: TextIn):
    return redact_text(body.text)

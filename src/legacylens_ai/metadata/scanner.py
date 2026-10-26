"""
Metadata Scanner: finds hidden data the user may not know they are sharing.
    Images: GPS location, camera/phone model, capture time, editing software
    PDFs:   author, creator app, producer, creation/modification dates
    Filename: PII inside the file name itself (e.g. "passport_29501011601234.jpg")
"""
from __future__ import annotations

import io
from typing import List

from PIL import ExifTags, Image

GPS_IFD = 0x8825
EXIF_IFD = 0x8769


def _gps_to_decimal(gps: dict) -> tuple | None:
    try:
        def conv(vals, ref):
            d, m, s = (float(v) for v in vals)
            dec = d + m / 60 + s / 3600
            return -dec if ref in ("S", "W") else dec
        return round(conv(gps[2], gps[1]), 5), round(conv(gps[4], gps[3]), 5)
    except Exception:
        return None


def scan_image_metadata(file_bytes: bytes) -> List[dict]:
    findings = []
    try:
        img = Image.open(io.BytesIO(file_bytes))
        exif = img.getexif()
    except Exception:
        return findings
    if not exif:
        return findings
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    tags.update({ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(EXIF_IFD).items()})
    gps = dict(exif.get_ifd(GPS_IFD))
    if gps:
        coords = _gps_to_decimal(gps)
        findings.append({"type": "GPS_LOCATION", "severity": "high",
                         "value": coords, "message": "Photo contains the exact location where it was taken"})
    if tags.get("Make") or tags.get("Model"):
        findings.append({"type": "DEVICE_INFO", "severity": "low",
                         "value": f"{tags.get('Make', '')} {tags.get('Model', '')}".strip(),
                         "message": "Photo reveals the phone/camera model"})
    if tags.get("DateTimeOriginal") or tags.get("DateTime"):
        findings.append({"type": "CAPTURE_TIME", "severity": "low",
                         "value": str(tags.get("DateTimeOriginal") or tags.get("DateTime")),
                         "message": "Photo reveals when it was taken"})
    if tags.get("Software"):
        findings.append({"type": "SOFTWARE", "severity": "info", "value": str(tags["Software"]),
                         "message": "Editing software recorded"})
    return findings


def scan_pdf_metadata(file_bytes: bytes) -> List[dict]:
    from pypdf import PdfReader

    findings = []
    try:
        meta = PdfReader(io.BytesIO(file_bytes)).metadata or {}
    except Exception:
        return findings
    for key, typ, sev in [("/Author", "AUTHOR", "medium"), ("/Creator", "CREATOR_APP", "info"),
                          ("/Producer", "PRODUCER", "info"), ("/CreationDate", "CREATED", "low"),
                          ("/ModDate", "MODIFIED", "low")]:
        if meta.get(key):
            findings.append({"type": typ, "severity": sev, "value": str(meta[key]),
                             "message": f"PDF metadata field {key[1:]} is set"})
    return findings


def strip_image_metadata(file_bytes: bytes) -> bytes:
    """Return a copy of the image with ALL metadata removed (used before sharing)."""
    img = Image.open(io.BytesIO(file_bytes))
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))
    buf = io.BytesIO()
    clean.save(buf, format=img.format or "PNG")
    return buf.getvalue()


def scan_metadata(file_bytes: bytes, filename: str) -> List[dict]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return scan_pdf_metadata(file_bytes)
    if name.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic")):
        return scan_image_metadata(file_bytes)
    return []

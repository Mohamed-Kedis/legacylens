import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from legacylens_ai.data.synthetic_docs import make_id_card  # noqa: E402

client = TestClient(app)


def _card_bytes():
    img, _ = make_id_card()
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_scan_id_card_returns_masked_values_only():
    r = client.post("/scan", files={"file": ("card.jpg", _card_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["document_type"]["label"] == "id_card"
    assert body["risk"]["level"] in ("high", "critical")
    assert body["suggested_policy"]["requires_user_confirmation"] is True
    for p in body["pii"]:
        assert "•" in p["masked"] or len(p["masked"]) <= 3 or "." in p["masked"]


def test_redact_returns_png():
    r = client.post("/redact", files={"file": ("card.jpg", _card_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_mask_text():
    r = client.post("/mask-text", json={"text": "موبايلي 01012345678"})
    assert "01012345678" not in r.json()["masked_text"]


def test_rejects_unknown_type():
    r = client.post("/scan", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415

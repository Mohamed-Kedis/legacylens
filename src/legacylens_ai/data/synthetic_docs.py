"""
Synthetic document image generator (fictional data only).

Produces 6 document classes for training the document classifier and for
evaluating OCR + PII detection end to end:

    id_card | passport | contract | financial | photo | other

Every ID-style image carries a visible "SYNTHETIC / NOT VALID" watermark and a
fictional issuer. The layouts are generic on purpose: they are NOT copies of any
real government document.

Usage:
    python -m legacylens_ai.data.synthetic_docs --n-per-class 300 --out data/synthetic
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

from . import fake_values as fv

CLASSES = ["id_card", "passport", "contract", "financial", "photo", "other"]
FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
AR_FONTS = ["NotoNaskhArabic.ttf", "NotoSansArabic.ttf", "Cairo.ttf", "Amiri.ttf"]
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int, name: str | None = None) -> ImageFont.FreeTypeFont:
    name = name or random.choice(AR_FONTS)
    path = FONT_DIR / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.truetype(LATIN_FONT, size)


def latin_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_DIR / "Cairo.ttf"), size)
    except OSError:
        return ImageFont.truetype(LATIN_FONT, size)


def pastel() -> tuple:
    return tuple(random.randint(200, 250) for _ in range(3))


def gradient(w: int, h: int) -> Image.Image:
    a, b = np.array(pastel(), float), np.array(pastel(), float)
    t = np.linspace(0, 1, w)[None, :, None]
    arr = (a * (1 - t) + b * t).repeat(h, axis=0)
    return Image.fromarray(arr.astype("uint8"), "RGB")


def _shape(text):
    """Pillow on Windows has no libraqm: shape + reorder Arabic ourselves."""
    from PIL import features
    if features.check("raqm"):
        return text
    import arabic_reshaper
    try:
        from bidi import get_display
    except ImportError:
        from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def rtl(draw, xy_right, text, fnt, fill=(20, 20, 20)):
    """Draw right-aligned RTL text; return its bbox."""
    from PIL import features
    if features.check("raqm"):
        draw.text(xy_right, text, font=fnt, fill=fill, anchor="ra", direction="rtl")
        return list(draw.textbbox(xy_right, text, font=fnt, anchor="ra", direction="rtl"))
    t = _shape(text)
    draw.text(xy_right, t, font=fnt, fill=fill, anchor="ra")
    return list(draw.textbbox(xy_right, t, font=fnt, anchor="ra"))


def ltr(draw, xy, text, fnt, fill=(20, 20, 20)):
    t = _shape(text)
    draw.text(xy, t, font=fnt, fill=fill)
    return list(draw.textbbox(xy, t, font=fnt))

def watermark(img: Image.Image, text="SYNTHETIC • NOT VALID • عينة تجريبية") -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(int(img.width / 22), "Cairo.ttf")
    for y in range(-img.height, img.height * 2, int(img.height / 3)):
        d.text((0, y), text, font=f, fill=(200, 30, 30, 70))
    layer = layer.rotate(25, expand=False)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def avatar(size: int) -> Image.Image:
    """Simple drawn face so the face-detector step has something to find."""
    im = Image.new("RGB", (size, int(size * 1.25)), pastel())
    d = ImageDraw.Draw(im)
    skin = random.choice([(224, 172, 105), (198, 134, 66), (241, 194, 125), (141, 85, 36)])
    cx, cy, r = size // 2, int(size * 0.5), int(size * 0.32)
    d.ellipse([cx - r, cy - int(r * 1.2), cx + r, cy + int(r * 1.2)], fill=skin)
    for ex in (cx - r // 2.5, cx + r // 2.5):
        d.ellipse([ex - 6, cy - r // 3 - 5, ex + 6, cy - r // 3 + 5], fill=(30, 30, 30))
    d.arc([cx - r // 2, cy, cx + r // 2, cy + r // 2], 20, 160, fill=(120, 40, 40), width=3)
    d.rectangle([cx - int(r * 1.4), cy + int(r * 1.25), cx + int(r * 1.4), im.height], fill=(60, 60, 90))
    return im


# ---------------------------------------------------------------------------
# Document templates. Each returns (image, pii_list); pii items have entity/text/bbox
# ---------------------------------------------------------------------------
def make_id_card():
    W, H = 856, 540
    img = gradient(W, H)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 70], fill=tuple(c - 60 for c in pastel()))
    rtl(d, (W - 25, 15), "بطاقة هوية تجريبية", font(30), fill=(255, 255, 255))
    ltr(d, (25, 22), "SYNTHETIC ID • FICTIONAL ISSUER", latin_font(20), fill=(255, 255, 255))
    av = avatar(210)
    img.paste(av, (30, 100))

    gender = random.choice(["male", "female"])
    birth = fv.rand_date()
    name = fv.full_name(gender)
    addr = fv.address()
    nid = fv.national_id(birth, gender)
    arabic_digits = random.random() < 0.6
    nid_shown = fv.to_arabic_digits(nid) if arabic_digits else nid

    pii, y, fs = [], 110, random.randint(26, 32)
    lab_f, val_f = font(fs - 4), font(fs)
    for label, value, ent in [("الاسم:", name, "PERSON"),
                              ("العنوان:", addr, "LOCATION"),
                              ("تاريخ الميلاد:", fv.to_arabic_digits(birth.strftime("%Y/%m/%d")) if arabic_digits else birth.strftime("%Y/%m/%d"), "DATE_TIME")]:
        lb = rtl(d, (W - 30, y), label, lab_f, fill=(90, 90, 90))
        vb = rtl(d, (lb[0] - 12, y), value, val_f)
        pii.append({"entity": ent, "text": value, "bbox": vb})
        y += fs + 32
    lb = rtl(d, (W - 30, H - 95), "الرقم القومي:", lab_f, fill=(90, 90, 90))
    vb = rtl(d, (lb[0] - 14, H - 102), nid_shown, font(fs + 8, "Cairo.ttf"))
    pii.append({"entity": "EG_NATIONAL_ID", "text": nid, "bbox": vb})
    return watermark(img), pii


def make_passport():
    W, H = 900, 620
    img = gradient(W, H)
    d = ImageDraw.Draw(img)
    ltr(d, (30, 20), "TRAVEL DOCUMENT — SPECIMEN (FICTIONAL ISSUER)", latin_font(26))
    rtl(d, (W - 30, 60), "وثيقة سفر تجريبية", font(26))
    img.paste(avatar(230), (30, 110))
    name = fv.latin_name()
    pno = fv.passport()
    birth = fv.rand_date()
    pii, y = [], 120
    f_lab, f_val = latin_font(18), latin_font(28)
    for label, value, ent in [("Passport No.", pno, "PASSPORT_NUMBER"), ("Full name", name, "PERSON"),
                              ("Date of birth", birth.strftime("%d %b %Y"), "DATE_TIME"),
                              ("National ID", fv.national_id(birth), "EG_NATIONAL_ID")]:
        ltr(d, (300, y), label, f_lab, fill=(90, 90, 90))
        vb = ltr(d, (300, y + 22), value, f_val)
        pii.append({"entity": ent, "text": value, "bbox": vb})
        y += 92
    return watermark(img), pii


CONTRACT_TITLES = ["عقد إيجار شقة سكنية", "عقد بيع سيارة", "عقد عمل", "توكيل خاص", "إقرار مديونية"]


def make_contract():
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), (250, 250, 245))
    d = ImageDraw.Draw(img)
    rtl(d, (W - 60, 50), random.choice(CONTRACT_TITLES), font(40))
    rtl(d, (W - 60, 120), f"حرر هذا العقد بتاريخ {fv.to_arabic_digits(fv.rand_date(2015, 2026).strftime('%Y/%m/%d'))}", font(24))
    pii, y = [], 190
    f = font(24)
    for party in ("الطرف الأول", "الطرف الثاني"):
        name, nid, phone, addr = fv.full_name(), fv.national_id(), fv.mobile("local"), fv.address()
        lines = [f"{party}: {name}", f"الرقم القومي: {nid}", f"العنوان: {addr}", f"موبايل: {phone}"]
        for line, ent, val in zip(lines, ["PERSON", "EG_NATIONAL_ID", "LOCATION", "EG_MOBILE"], [name, nid, addr, phone]):
            rtl(d, (W - 60, y), line, f)
            pii.append({"entity": ent, "text": val})
            y += 44
        y += 30
    body = ["اتفق الطرفان وهما بكامل الأهلية القانونية على ما يلي:",
            "البند الأول: يقر الطرف الأول بأنه المالك للعين موضوع هذا العقد.",
            "البند الثاني: مدة هذا العقد سنة تبدأ من تاريخ التوقيع.",
            "البند الثالث: يلتزم الطرف الثاني بسداد المقابل في المواعيد المحددة.",
            "البند الرابع: تختص محاكم الجهة المختصة بنظر أي نزاع ينشأ عن هذا العقد."]
    for line in body:
        rtl(d, (W - 60, y), line, f)
        y += 50
    rtl(d, (W - 80, H - 150), "توقيع الطرف الأول", f)
    ltr(d, (80, H - 150), "توقيع الطرف الثاني", f)
    return watermark(img, "SYNTHETIC SAMPLE • نموذج تجريبي"), pii


def make_financial():
    W, H = 900, 1100
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    bank = random.choice(fv.BANKS_FICTIONAL)
    ltr(d, (40, 40), f"{bank} — Account Statement (FICTIONAL)", latin_font(28))
    name, ib = fv.latin_name(), fv.iban()
    pii = []
    ltr(d, (40, 100), "Account holder:", latin_font(20), fill=(90, 90, 90))
    pii.append({"entity": "PERSON", "text": name, "bbox": ltr(d, (220, 98), name, latin_font(22))})
    ltr(d, (40, 140), "IBAN:", latin_font(20), fill=(90, 90, 90))
    pii.append({"entity": "EG_IBAN", "text": ib, "bbox": ltr(d, (220, 138), ib, latin_font(22))})
    y = 220
    d.rectangle([40, y, W - 40, y + 40], fill=(230, 235, 245))
    for x, h in zip([50, 250, 600, 750], ["Date", "Description", "Debit", "Credit"]):
        ltr(d, (x, y + 8), h, latin_font(18))
    y += 50
    for _ in range(random.randint(10, 18)):
        dt = fv.rand_date(2024, 2026).strftime("%d/%m/%Y")
        desc = random.choice(["POS PURCHASE", "SALARY", "ATM WITHDRAWAL", "TRANSFER", "UTILITY BILL", "INSTAPAY"])
        amt = f"{random.randint(50, 20000):,}.00"
        cols = [dt, desc, amt if desc != "SALARY" else "", amt if desc == "SALARY" else ""]
        for x, c in zip([50, 250, 600, 750], cols):
            ltr(d, (x, y), c, latin_font(17))
        y += 36
        d.line([40, y - 6, W - 40, y - 6], fill=(230, 230, 230))
    return watermark(img, "FICTIONAL STATEMENT • كشف حساب تجريبي"), pii


def make_photo():
    W, H = random.choice([(800, 600), (600, 800), (900, 600)])
    img = gradient(W, H)
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        x, y, r = random.randint(0, W), random.randint(0, H), random.randint(30, 160)
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(random.randint(60, 230) for _ in range(3)))
    d.rectangle([0, int(H * 0.7), W, H], fill=(random.randint(40, 120), random.randint(100, 170), random.randint(40, 90)))
    if random.random() < 0.6:  # people in the photo
        for _ in range(random.randint(1, 3)):
            av = avatar(random.randint(140, 220))
            img.paste(av, (random.randint(0, max(1, W - av.width)), random.randint(int(H * 0.2), max(int(H * 0.2) + 1, H - av.height))))
    return img, []


NOTES = ["قائمة مشتريات: لبن، عيش، جبنة، طماطم", "مواعيد الأسبوع: جيم الأحد والثلاثاء",
         "أفكار للرحلة: الساحل أو دهب", "كتب عايز أقراها السنة دي", "ملاحظات الاجتماع: مراجعة الخطة"]


def make_other():
    W, H = 800, 900
    img = Image.new("RGB", (W, H), (255, 253, 225))
    d = ImageDraw.Draw(img)
    for yy in range(80, H, 50):
        d.line([30, yy, W - 30, yy], fill=(180, 200, 230))
    y = 40
    for _ in range(random.randint(4, 10)):
        rtl(d, (W - 50, y), random.choice(NOTES), font(28, random.choice(["Amiri.ttf", "NotoNaskhArabic.ttf"])), fill=(30, 30, 120))
        y += 50
    return img, []


MAKERS = {"id_card": make_id_card, "passport": make_passport, "contract": make_contract,
          "financial": make_financial, "photo": make_photo, "other": make_other}


# ---------------------------------------------------------------------------
# Augmentation: make it look like a phone photo / scan
# ---------------------------------------------------------------------------
def augment(img: Image.Image, strength: float = 1.0) -> Image.Image:
    if random.random() < 0.5 * strength:  # photo of the document on a table
        pad = int(max(img.size) * 0.15)
        bg = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), tuple(random.randint(60, 160) for _ in range(3)))
        bg.paste(img, (pad + random.randint(-pad // 2, pad // 2), pad + random.randint(-pad // 2, pad // 2)))
        img = bg
    img = img.rotate(random.uniform(-6, 6) * strength, expand=True, fillcolor=(128, 128, 128))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5 * strength:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.4)))
    if random.random() < 0.5 * strength:
        arr = np.array(img).astype(np.int16)
        arr += np.random.normal(0, random.uniform(3, 12), arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(55, 92))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def split_for(i: int, n: int) -> str:
    r = i / max(n, 1)
    return "train" if r < 0.8 else ("val" if r < 0.9 else "test")


def generate(out: Path, n_per_class: int, augment_ratio: float = 0.7, seed: int = 42) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    labels_path = out / "labels.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as fh:
        for cls in CLASSES:
            for i in range(n_per_class):
                img, pii = MAKERS[cls]()
                aug = random.random() < augment_ratio
                if aug:
                    img = augment(img)  # bboxes are no longer exact after augmentation
                split = split_for(i, n_per_class)
                rel = Path("images") / split / cls / f"{cls}_{i:05d}.jpg"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save(out / rel, "JPEG", quality=90)
                fh.write(json.dumps({"file": str(rel), "class": cls, "split": split, "augmented": aug,
                                     "pii": [{**p, "bbox": None if aug else p.get("bbox")} for p in pii]},
                                    ensure_ascii=False) + "\n")
    return labels_path


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic (fictional) document images")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--augment-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    p = generate(Path(a.out), a.n_per_class, a.augment_ratio, a.seed)
    print(f"done -> {p}")


if __name__ == "__main__":
    main()

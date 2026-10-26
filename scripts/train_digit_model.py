"""
Train the Arabic-Indic digit classifier used for the national-ID field (CPU, ~1-3 minutes).

    python scripts/train_digit_model.py --strings 3000

Data: 100% synthetic. Random 14-digit strings rendered with 4 Arabic fonts, random sizes,
pastel backgrounds, a red watermark, blur / noise / JPEG / small rotation; then segmented with
the SAME code used at inference, so training and production see identical glyphs.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from legacylens_ai.data.synthetic_docs import AR_FONTS, font, pastel  # noqa: E402
from legacylens_ai.ocr.digits import MODEL_PATH, segment  # noqa: E402

AR = "٠١٢٣٤٥٦٧٨٩"


def render_strip(digits: str) -> Image.Image:
    size = random.randint(26, 48)
    f = font(size, random.choice(AR_FONTS))
    text = digits.translate(str.maketrans("0123456789", AR))
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=f)
    W, H = x1 - x0 + 40, y1 - y0 + 30
    img = Image.new("RGB", (W, H), pastel())
    d = ImageDraw.Draw(img)
    d.text((20 - x0, 15 - y0), text, font=f, fill=tuple(random.randint(0, 50) for _ in range(3)))
    if random.random() < 0.6:  # colored watermark crossing the digits
        wm = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(wm).text((random.randint(-40, 20), random.randint(-5, 10)), "SYNTHETIC NOT VALID",
                                font=font(int(size * 0.8), "Cairo.ttf"), fill=(200, 30, 30, 70))
        img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
    if random.random() < 0.5:
        arr = np.asarray(img).astype(np.int16) + np.random.normal(0, random.uniform(2, 8), (H, W, 3)).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    img = img.rotate(random.uniform(-1.5, 1.5), expand=False, fillcolor=pastel())
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=random.randint(60, 95))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def build(n_strings: int):
    X, y, strings = [], [], []
    skipped = 0
    for _ in range(n_strings):
        digits = "".join(random.choice("0123456789") for _ in range(14))
        glyphs = segment(render_strip(digits))
        if len(glyphs) != len(digits):  # segmentation failed -> skip (counted, reported)
            skipped += 1
            continue
        X += [g for g, _ in glyphs]
        y += [int(c) for c in digits]
        strings.append((len(X) - 14, digits))
    return np.array(X), np.array(y), strings, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strings", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)

    from sklearn.metrics import classification_report
    from sklearn.neural_network import MLPClassifier
    import joblib

    n_test = max(200, a.strings // 5)
    Xtr, ytr, _, sk1 = build(a.strings)
    Xte, yte, test_strings, sk2 = build(n_test)
    print(f"train glyphs={len(Xtr)}  test glyphs={len(Xte)}  segmentation skipped: {sk1 + sk2} strips")

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60, early_stopping=True, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    glyph_acc = float((pred == yte).mean())
    exact = float(np.mean([(pred[i:i + 14] == yte[i:i + 14]).all() for i, _ in test_strings]))
    print(classification_report(yte, pred, digits=3))
    print(f"glyph accuracy={glyph_acc:.4f}   full 14-digit exact match={exact:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    report = {"glyph_accuracy": glyph_acc, "string_exact_match": exact, "train_glyphs": len(Xtr),
              "test_strings": len(test_strings), "segmentation_skipped": sk1 + sk2}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "digit_model.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

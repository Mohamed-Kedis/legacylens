"""
Evaluate the PII analyzer on labeled synthetic text -> precision / recall / F1 per entity.

    python scripts/evaluate_pii.py --data data/pii_text.jsonl --out reports/pii_eval.json

A prediction matches a gold span when the entity agrees and the spans overlap >= 50%.
PERSON_FREE = an Arabic name in free text with no "الاسم:" label. Regex can't catch these;
compare the score with and without LEGACYLENS_ARABIC_NER=1 to show what the ML model adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legacylens_ai.pii.analyzer import PIIAnalyzer  # noqa: E402

SCORED = {"EG_NATIONAL_ID", "EG_MOBILE", "EG_IBAN", "PASSPORT_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "PERSON"}
ALIAS = {"IBAN_CODE": "EG_IBAN"}


def overlap(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, min(a[1] - a[0], b[1] - b[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pii_text.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    analyzer = PIIAnalyzer()
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    rows = [json.loads(line) for line in open(a.data, encoding="utf-8")]
    for row in rows:
        gold = [(s["start"], s["end"], s["entity"]) for s in row["spans"]]
        preds = [(f.start, f.end, ALIAS.get(f.entity, f.entity)) for f in analyzer.analyze(row["text"])]
        preds = [p for p in preds if p[2] in SCORED]
        used = set()
        for gs, ge, gent in gold:
            target = "PERSON" if gent == "PERSON_FREE" else gent
            hit = next((i for i, (ps, pe, pent) in enumerate(preds)
                        if i not in used and pent == target and overlap((gs, ge), (ps, pe)) >= 0.5), None)
            if hit is None:
                fn[gent] += 1
            else:
                tp[gent] += 1
                used.add(hit)
        for i, (_, _, pent) in enumerate(preds):
            if i not in used:
                fp[pent] += 1

    report = {}
    print(f"{'entity':<18}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>5}{'FN':>5}")
    for ent in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[ent] / max(1, tp[ent] + fp[ent])
        r = tp[ent] / max(1, tp[ent] + fn[ent])
        f1 = 2 * p * r / max(1e-9, p + r)
        report[ent] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                       "tp": tp[ent], "fp": fp[ent], "fn": fn[ent]}
        print(f"{ent:<18}{p:7.2f}{r:7.2f}{f1:7.2f}{tp[ent]:6}{fp[ent]:5}{fn[ent]:5}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()

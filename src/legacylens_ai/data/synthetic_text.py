"""
Synthetic labeled text for PII detection (Arabic + English).

Each JSONL row: {"text": ..., "spans": [{"start", "end", "entity"}]}
Includes hard negatives (14-digit numbers that are not valid IDs, order numbers...)
so precision is measured, not only recall.

Usage:
    python -m legacylens_ai.data.synthetic_text --n 2000 --out data/pii_text.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re

from . import fake_values as fv

TEMPLATES = [
    "الاسم: {PERSON}\nالرقم القومي: {EG_NATIONAL_ID}",
    "أنا {PERSON_FREE} ورقم بطاقتي {EG_NATIONAL_ID} وموبايلي {EG_MOBILE}",
    "برجاء التحويل على الحساب {EG_IBAN} باسم {PERSON_FREE}",
    "رقم الموبايل: {EG_MOBILE}",
    "جواز السفر رقم {PASSPORT_NUMBER} صالح حتى ٢٠٣٠",
    "تواصل معايا على {EMAIL_ADDRESS} أو واتساب {EG_MOBILE}",
    "الطرف الأول: {PERSON}\nالرقم القومي: {EG_NATIONAL_ID}\nالعنوان: {ADDRESS}",
    "Full name: {PERSON_LATIN}\nPassport No: {PASSPORT_NUMBER}",
    "Please call me on {EG_MOBILE} or email {EMAIL_ADDRESS}.",
    "Beneficiary IBAN {EG_IBAN}, national ID {EG_NATIONAL_ID}.",
    "Card number {CREDIT_CARD} exp 09/29",
]
NEGATIVE_TEMPLATES = [
    "رقم الطلب {NEG14} تم شحنه",
    "Order #{NEG14} has been shipped",
    "رقم الفاتورة {RAND8} بتاريخ اليوم",
    "الكود السري للعرض هو {RAND6}",
    "Tracking number {NEG14}",
    "قائمة المشتريات: لبن وعيش وجبنة",
]
# Arabic free-text names (no "الاسم:" label) need the transformer NER -> separate entity
# so you can see what regex/labels catch vs. what needs the ML model.
LUHN_TEST_CARDS = ["4111 1111 1111 1111", "5555 5555 5555 4444", "4012 8888 8888 1881"]


def value_for(slot: str, arabic_digits: bool) -> tuple[str, str | None]:
    if slot == "PERSON":
        return fv.full_name(), "PERSON"
    if slot == "PERSON_FREE":
        return fv.full_name(), "PERSON_FREE"
    if slot == "PERSON_LATIN":
        return fv.latin_name(), "PERSON"
    if slot == "EG_NATIONAL_ID":
        v = fv.national_id()
        return (fv.to_arabic_digits(v) if arabic_digits else v), "EG_NATIONAL_ID"
    if slot == "EG_MOBILE":
        return fv.mobile(), "EG_MOBILE"
    if slot == "EG_IBAN":
        return fv.iban(), "EG_IBAN"
    if slot == "PASSPORT_NUMBER":
        return fv.passport(), "PASSPORT_NUMBER"
    if slot == "EMAIL_ADDRESS":
        return fv.email(), "EMAIL_ADDRESS"
    if slot == "CREDIT_CARD":
        return random.choice(LUHN_TEST_CARDS), "CREDIT_CARD"
    if slot == "ADDRESS":
        return fv.address(), None  # not scored (no address recognizer yet)
    if slot == "NEG14":
        return fv.invalid_14_digits(), None
    if slot == "RAND8":
        return f"{random.randint(10**7, 10**8 - 1)}", None
    if slot == "RAND6":
        return f"{random.randint(10**5, 10**6 - 1)}", None
    raise KeyError(slot)


def render(template: str) -> dict:
    arabic_digits = random.random() < 0.3
    text, spans, pos = "", [], 0
    for part in re.split(r"(\{[A-Z_0-9]+\})", template):
        if part.startswith("{"):
            value, ent = value_for(part[1:-1], arabic_digits)
            if ent:
                spans.append({"start": len(text), "end": len(text) + len(value), "entity": ent})
            text += value
        else:
            text += part
    return {"text": text, "spans": spans}


def generate(n: int, negative_ratio: float = 0.25, seed: int = 7) -> list[dict]:
    random.seed(seed)
    rows = []
    for _ in range(n):
        tpl = random.choice(NEGATIVE_TEMPLATES if random.random() < negative_ratio else TEMPLATES)
        rows.append(render(tpl))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data/pii_text.jsonl")
    a = ap.parse_args()
    rows = generate(a.n)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

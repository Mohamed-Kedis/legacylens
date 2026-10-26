"""
Egyptian PII recognizers for Microsoft Presidio.

Each recognizer = regex (find candidates) + validation logic (reject false positives)
+ context words (Arabic & English) that boost confidence when found nearby.

Entities:
    EG_NATIONAL_ID   14-digit Egyptian national ID (structurally validated)
    EG_MOBILE        Egyptian mobile numbers (010/011/012/015, +20 / 0020 prefixes)
    EG_IBAN          Egyptian IBAN (EG + 27 digits, ISO 7064 mod-97 checked)
    PASSPORT_NUMBER  1 Latin letter + 8 digits (low base score, needs context)
    PERSON_LABELED   Name values that follow a "Name:" / "الاسم:" label
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
# Governorate codes used in digits 8-9 of the Egyptian national ID.
GOVERNORATES = {
    "01": "Cairo", "02": "Alexandria", "03": "Port Said", "04": "Suez",
    "11": "Damietta", "12": "Dakahlia", "13": "Sharqia", "14": "Qalyubia",
    "15": "Kafr El Sheikh", "16": "Gharbia", "17": "Monufia", "18": "Beheira",
    "19": "Ismailia", "21": "Giza", "22": "Beni Suef", "23": "Fayoum",
    "24": "Minya", "25": "Assiut", "26": "Sohag", "27": "Qena", "28": "Aswan",
    "29": "Luxor", "31": "Red Sea", "32": "New Valley", "33": "Matrouh",
    "34": "North Sinai", "35": "South Sinai", "88": "Born abroad",
}

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic / Persian digits to ASCII. Length is preserved, so
    character offsets found on the normalized text are valid on the original."""
    return text.translate(ARABIC_INDIC)


# ---------------------------------------------------------------------------
# National ID
# ---------------------------------------------------------------------------
def parse_egyptian_national_id(nid: str) -> Optional[dict]:
    """Return decoded fields if the 14-digit number is structurally valid, else None.

    Layout: C YYMMDD GG SSSS K
        C     century   (2 = 1900s, 3 = 2000s)
        YYMMDD birth date
        GG    governorate code
        SSSS  serial (last digit of the serial: odd = male, even = female)
        K     check digit (algorithm not public -> not validated)
    """
    nid = normalize_digits(nid)
    if not re.fullmatch(r"\d{14}", nid):
        return None
    century = {"2": 1900, "3": 2000}.get(nid[0])
    if century is None:
        return None
    try:
        birth = date(century + int(nid[1:3]), int(nid[3:5]), int(nid[5:7]))
    except ValueError:
        return None
    if birth > date.today():
        return None
    gov = nid[7:9]
    if gov not in GOVERNORATES:
        return None
    return {
        "birth_date": birth.isoformat(),
        "governorate": GOVERNORATES[gov],
        "gender": "male" if int(nid[12]) % 2 else "female",
    }


class EgyptNationalIdRecognizer(PatternRecognizer):
    PATTERNS = [Pattern("eg_nid_14_digits", r"(?<!\d)[23]\d{13}(?!\d)", 0.5)]
    CONTEXT = [
        # single tokens: Presidio matches context word-by-word
        "القومي", "قومي", "بطاقة", "البطاقة", "الهوية", "هوية",
        "national", "id", "nid", "identity", "card",
    ]

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            supported_entity="EG_NATIONAL_ID",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
            name="EgyptNationalIdRecognizer",
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        # True -> score 1.0, False -> dropped
        return parse_egyptian_national_id(pattern_text) is not None


# ---------------------------------------------------------------------------
# Mobile
# ---------------------------------------------------------------------------
class EgyptMobileRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern(
            "eg_mobile",
            r"(?<![\d+])(?:(?:\+|00)20[\s-]?|0)1[0125](?:[\s-]?\d){8}(?!\d)",
            0.7,
        )
    ]
    CONTEXT = ["موبايل", "محمول", "تليفون", "هاتف", "رقم", "واتساب",
               "mobile", "phone", "tel", "whatsapp", "contact"]

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            supported_entity="EG_MOBILE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
            name="EgyptMobileRecognizer",
        )


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------
def iban_is_valid(raw: str) -> bool:
    iban = re.sub(r"[\s-]", "", raw).upper()
    if not re.fullmatch(r"EG\d{27}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


class EgyptIbanRecognizer(PatternRecognizer):
    PATTERNS = [Pattern("eg_iban", r"\bEG\d{2}(?:[\s-]?\d){25}\b", 0.6)]
    CONTEXT = ["iban", "ايبان", "آيبان", "حساب", "بنك", "account", "bank", "transfer"]

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            supported_entity="EG_IBAN",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
            name="EgyptIbanRecognizer",
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        return iban_is_valid(pattern_text)


# ---------------------------------------------------------------------------
# Passport (generic format, low confidence unless context words appear)
# ---------------------------------------------------------------------------
class PassportNumberRecognizer(PatternRecognizer):
    PATTERNS = [Pattern("passport_letter_8_digits", r"\b[A-Z]\d{8}\b", 0.3)]
    CONTEXT = ["جواز", "الجواز", "السفر", "passport", "travel"]

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            supported_entity="PASSPORT_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language=supported_language,
            name="PassportNumberRecognizer",
        )


# ---------------------------------------------------------------------------
# Labeled name fields ("Name: ..." / "الاسم: ...") — strong signal on forms & IDs
# ---------------------------------------------------------------------------
class LabeledNameRecognizer(EntityRecognizer):
    LABEL_RE = re.compile(
        r"(?:الاسم(?:\s+بالكامل)?|اسم\s+(?:الأب|الام|الأم|العميل|الطرف\s+\S+)|"
        r"الطرف\s+(?:الأول|الثاني|الثالث)|المستفيد|الموكل|الوكيل|"
        r"full\s+name|name)\s*[:：\-]\s*"
        r"(?P<value>[^\n\d:：|,،]{3,60})",
        re.IGNORECASE,
    )

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            # separate internal entity: Presidio drops results contained in a same-type span,
            # which would let a noisy spaCy PERSON swallow this precise one. Mapped back to PERSON.
            supported_entities=["PERSON_LABELED"],
            supported_language=supported_language,
            name="LabeledNameRecognizer",
        )

    def load(self) -> None:  # nothing to load
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        for m in self.LABEL_RE.finditer(text):
            runs = _same_script_runs(m.group("value"), m.start("value"))
            if sum(len(t.split()) for _, _, t in runs) < 2:  # a real full name has >= 2 tokens
                continue
            for i, (s0, e0, _) in enumerate(runs):
                results.append(RecognizerResult("PERSON_LABELED", s0, e0, 0.85 if i == 0 else 0.7))
        return results


def _same_script_runs(value: str, offset: int):
    """Split a labeled value into contiguous runs written in the script of its first token.
    OCR noise like "سلمى pal الشافعي" -> ["سلمى", "الشافعي"]: we never redact the noise region."""
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", value)]
    if not tokens:
        return []
    is_ar = lambda t: bool(re.search(r"[\u0600-\u06FF]", t))  # noqa: E731
    script = is_ar(tokens[0][2])
    runs, cur = [], []
    for s0, e0, t in tokens:
        if is_ar(t) == script and re.search(r"\w{2,}", t):
            cur.append((s0, e0))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [(offset + r[0][0], offset + r[-1][1], value[r[0][0]:r[-1][1]]) for r in runs]


def egypt_recognizers(language: str = "en") -> list:
    return [
        EgyptNationalIdRecognizer(language),
        EgyptMobileRecognizer(language),
        EgyptIbanRecognizer(language),
        PassportNumberRecognizer(language),
        LabeledNameRecognizer(language),
    ]

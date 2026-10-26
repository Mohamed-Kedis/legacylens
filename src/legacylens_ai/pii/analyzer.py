"""
PII analyzer = Presidio built-ins (email, credit card, ...) + Egyptian recognizers
+ optional Arabic transformer NER (CAMeLBERT) for names/locations in free text.

Usage:
    from legacylens_ai.pii.analyzer import PIIAnalyzer
    findings = PIIAnalyzer().analyze("الرقم القومي: 29501011234567")
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Optional

import tldextract

# Use tldextract's bundled suffix snapshot: no network call (the scan service runs offline).
tldextract.extract = tldextract.TLDExtract(suffix_list_urls=())

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .egypt_recognizers import egypt_recognizers, normalize_digits, parse_egyptian_national_id

ENTITIES = [
    "EG_NATIONAL_ID", "EG_MOBILE", "EG_IBAN", "PASSPORT_NUMBER",
    "CREDIT_CARD", "EMAIL_ADDRESS", "IBAN_CODE", "PERSON", "PERSON_LABELED", "LOCATION",
    "DATE_TIME", "IP_ADDRESS", "URL",
]

# When two findings overlap, the more specific entity wins.
PRIORITY = {
    "EG_NATIONAL_ID": 10, "EG_IBAN": 9, "CREDIT_CARD": 9, "IBAN_CODE": 8,
    "PASSPORT_NUMBER": 7, "EG_MOBILE": 7, "EMAIL_ADDRESS": 6, "PERSON_LABELED": 6, "PERSON": 5,
    "LOCATION": 4, "IP_ADDRESS": 4, "URL": 3, "DATE_TIME": 1,
}

BIRTH_WORDS = ("الميلاد", "ميلاد", "مواليد", "birth", "dob", "born")
PERSON_STOPWORDS = {"IBAN", "ID", "Card", "Order", "Beneficiary", "Passport", "Account", "Bank"}
ARABIC_RE = re.compile(r"[؀-ۿ]")


@dataclass
class PIIFinding:
    entity: str
    text: str
    start: int
    end: int
    score: float
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ArabicTransformerNERRecognizer(EntityRecognizer):
    """Optional: names/locations in free Arabic text using CAMeLBERT NER.
    Enable with LEGACYLENS_ARABIC_NER=1 (needs `pip install transformers torch`)."""

    MODEL = os.getenv("LEGACYLENS_ARABIC_NER_MODEL", "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    LABEL_MAP = {"PERS": "PERSON", "LOC": "LOCATION"}

    def __init__(self, supported_language: str = "en"):
        self._pipe = None
        super().__init__(supported_entities=["PERSON", "LOCATION"],
                         supported_language=supported_language, name="ArabicTransformerNER")

    def load(self) -> None:
        from transformers import pipeline  # lazy import
        self._pipe = pipeline("ner", model=self.MODEL, aggregation_strategy="simple")

    def analyze(self, text, entities, nlp_artifacts=None) -> List[RecognizerResult]:
        if not ARABIC_RE.search(text):
            return []
        out = []
        for ent in self._pipe(text):
            label = self.LABEL_MAP.get(ent["entity_group"])
            if label and ent["score"] >= 0.6:
                out.append(RecognizerResult(label, ent["start"], ent["end"], float(ent["score"])))
        return out


@lru_cache(maxsize=1)
def _build_engine(use_arabic_ner: bool) -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": os.getenv("LEGACYLENS_SPACY_MODEL", "en_core_web_sm")}],
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
    for rec in egypt_recognizers("en"):
        registry.add_recognizer(rec)
    if use_arabic_ner:
        registry.add_recognizer(ArabicTransformerNERRecognizer("en"))
    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])


def _resolve_overlaps(results: List[RecognizerResult]) -> List[RecognizerResult]:
    ranked = sorted(results, key=lambda r: (PRIORITY.get(r.entity_type, 0), r.score, r.end - r.start), reverse=True)
    kept: List[RecognizerResult] = []
    for r in ranked:
        if all(r.end <= k.start or r.start >= k.end for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r.start)


class PIIAnalyzer:
    def __init__(self, use_arabic_ner: Optional[bool] = None, min_score: float = 0.4):
        if use_arabic_ner is None:
            use_arabic_ner = os.getenv("LEGACYLENS_ARABIC_NER", "0") == "1"
        self.engine = _build_engine(use_arabic_ner)
        self.min_score = min_score

    def analyze(self, text: str) -> List[PIIFinding]:
        norm = normalize_digits(text)  # same length -> offsets stay valid
        raw = self.engine.analyze(text=norm, language="en", entities=ENTITIES,
                                  score_threshold=self.min_score)
        # 1) drop known noise BEFORE resolving overlaps, so a noisy long span can't
        #    shadow a correct shorter one
        clean = [r for r in raw if not self._is_noise(r, text[r.start:r.end], norm[:r.start])]
        findings = []
        for r in _resolve_overlaps(clean):
            span = text[r.start:r.end]
            entity = "PERSON" if r.entity_type == "PERSON_LABELED" else r.entity_type
            details = parse_egyptian_national_id(span) if r.entity_type == "EG_NATIONAL_ID" else None
            findings.append(PIIFinding(entity, span, r.start, r.end, round(r.score, 2), details))
        return findings

    @staticmethod
    def _is_noise(r: RecognizerResult, span: str, text_before: str = "") -> bool:
        source = (r.recognition_metadata or {}).get("recognizer_name", "")
        if source == "SpacyRecognizer":
            # English spaCy NER is unreliable on Arabic script and on spans with digits/newlines
            if ARABIC_RE.search(span) or "\n" in span:
                return True
            if r.entity_type == "PERSON" and (any(ch.isdigit() for ch in span)
                                               or set(span.split()) & PERSON_STOPWORDS):
                return True
        if r.entity_type == "DATE_TIME":
            if len(span) < 6 or span.replace(" ", "").isdigit():
                return True
            # a random date is not sensitive; a BIRTH date is
            window = text_before.lower()[-40:]
            if not any(k in window for k in BIRTH_WORDS):
                return True
        return False

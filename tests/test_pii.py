import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from legacylens_ai.data import fake_values as fv  # noqa: E402
from legacylens_ai.pii.analyzer import PIIAnalyzer  # noqa: E402
from legacylens_ai.pii.egypt_recognizers import iban_is_valid, parse_egyptian_national_id  # noqa: E402
from legacylens_ai.redact.redactor import mask_text, mask_value  # noqa: E402
from legacylens_ai.risk.score import risk_score  # noqa: E402


@pytest.fixture(scope="module")
def analyzer():
    return PIIAnalyzer()


def entities(analyzer, text):
    return {(f.entity, f.text) for f in analyzer.analyze(text)}


def test_national_id_decoding():
    info = parse_egyptian_national_id("29501011601234")
    assert info == {"birth_date": "1995-01-01", "governorate": "Gharbia", "gender": "male"}


@pytest.mark.parametrize("bad", ["19501011601234", "29513011601234", "29501010001234", "2950101160123"])
def test_national_id_rejects_invalid(bad):
    assert parse_egyptian_national_id(bad) is None


def test_arabic_indic_national_id(analyzer):
    assert ("EG_NATIONAL_ID", "٢٩٥٠١٠١١٦٠١٢٣٤") in entities(analyzer, "الرقم القومي: ٢٩٥٠١٠١١٦٠١٢٣٤")


def test_order_number_is_not_an_id(analyzer):
    assert not any(e == "EG_NATIONAL_ID" for e, _ in entities(analyzer, "رقم الطلب 12345678901234"))


@pytest.mark.parametrize("phone", ["01012345678", "+201012345678", "010 1234 5678", "+20 106 123 4567"])
def test_mobile_formats(analyzer, phone):
    assert ("EG_MOBILE", phone) in entities(analyzer, f"موبايل {phone}")


def test_iban_checksum():
    assert iban_is_valid("EG380019000500000000263180002")
    assert not iban_is_valid("EG380019000500000000263180003")
    assert all(iban_is_valid(fv.iban()) for _ in range(50))


def test_labeled_name_ignores_ocr_noise(analyzer):
    found = {t for e, t in entities(analyzer, "الاسم: سلمى pal الشافعي ye\nالعنوان") if e == "PERSON"}
    assert found == {"سلمى", "الشافعي"}


def test_masking():
    assert mask_value("EG_NATIONAL_ID", "29501011601234") == "2••••••••••234"
    assert mask_value("EG_MOBILE", "01012345678") == "010•••••678"
    assert mask_value("EMAIL_ADDRESS", "mostafa@example.com").startswith("m•")


def test_mask_text_keeps_rest(analyzer):
    text = "رقمي 01012345678 شكرا"
    out = mask_text(text, analyzer.analyze(text))
    assert "01012345678" not in out and out.endswith("شكرا")


def test_risk_levels():
    low = risk_score("other", [], 0, [])
    high = risk_score("id_card", [{"entity": "EG_NATIONAL_ID"}, {"entity": "PERSON"}], 1, [])
    assert low["level"] == "low" and high["level"] == "critical"
    assert all("reason" in r for r in high["reasons"])

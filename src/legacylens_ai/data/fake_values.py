"""Random *fictional* Egyptian-style values for synthetic training data.
Never put real people's data in the dataset."""
from __future__ import annotations

import random
from datetime import date, timedelta

from legacylens_ai.pii.egypt_recognizers import GOVERNORATES

MALE = ["محمد", "أحمد", "محمود", "مصطفى", "عمر", "يوسف", "خالد", "إبراهيم", "علي", "حسن",
        "كريم", "طارق", "هشام", "عمرو", "شريف", "ياسر", "وليد", "سامح", "مينا", "بيتر",
        "مروان", "زياد", "آدم", "حمزة", "سيف", "إسلام", "رامي", "أيمن", "جورج", "تامر"]
FEMALE = ["فاطمة", "مريم", "نورهان", "سارة", "آية", "منى", "هبة", "ياسمين", "دينا", "رنا",
          "سلمى", "ملك", "جنى", "نادين", "شيماء", "إيمان", "مارينا", "كريستين", "هالة", "رحمة"]
FAMILY = ["عبد الرحمن", "السيد", "حسن", "إبراهيم", "سليم", "منصور", "عبد العزيز", "الشافعي",
          "النجار", "فرج", "رزق", "عطية", "يوسف", "جرجس", "شحاتة", "المصري", "عثمان", "بدوي",
          "عبد الله", "زكي", "حنا", "الشناوي", "سالم", "الجمال", "خليل"]
LATIN = {"محمد": "Mohamed", "أحمد": "Ahmed", "محمود": "Mahmoud", "مصطفى": "Mostafa", "عمر": "Omar",
         "يوسف": "Youssef", "خالد": "Khaled", "علي": "Ali", "حسن": "Hassan", "كريم": "Karim",
         "مريم": "Mariam", "سارة": "Sara", "منى": "Mona", "هبة": "Heba", "دينا": "Dina",
         "سليم": "Selim", "منصور": "Mansour", "السيد": "Elsayed", "إبراهيم": "Ibrahim", "فرج": "Farag"}
STREETS = ["شارع التحرير", "شارع الجمهورية", "شارع النصر", "شارع الهرم", "شارع البحر", "شارع سعد زغلول",
           "شارع المحطة", "شارع الجلاء", "شارع 26 يوليو", "شارع الثورة"]
CITIES = ["القاهرة", "الجيزة", "الإسكندرية", "المنصورة", "طنطا", "المحلة الكبرى", "الزقازيق",
          "أسيوط", "المنيا", "بورسعيد", "دمياط", "شبين الكوم"]
BANKS_FICTIONAL = ["Demo Nile Bank", "Sample Delta Bank", "Fictional Misr Savings"]


def rand_date(start_year=1950, end_year=2008) -> date:
    start = date(start_year, 1, 1)
    return start + timedelta(days=random.randint(0, (date(end_year, 12, 31) - start).days))


def full_name(gender: str | None = None) -> str:
    gender = gender or random.choice(["male", "female"])
    first = random.choice(MALE if gender == "male" else FEMALE)
    return " ".join([first, random.choice(MALE), random.choice(FAMILY)])


def latin_name() -> str:
    first = random.choice([v for v in LATIN.values()])
    return f"{first} {random.choice(list(LATIN.values()))} {random.choice(['Selim', 'Mansour', 'Farag', 'Hassan', 'Ibrahim'])}"


def national_id(birth: date | None = None, gender: str | None = None) -> str:
    """Structurally valid but fictional 14-digit number (check digit is random)."""
    birth = birth or rand_date()
    century = "2" if birth.year < 2000 else "3"
    gov = random.choice(list(GOVERNORATES))
    serial = f"{random.randint(0, 999):03d}"
    gender = gender or random.choice(["male", "female"])
    g = random.choice("13579" if gender == "male" else "02468")
    return f"{century}{birth:%y%m%d}{gov}{serial}{g}{random.randint(0, 9)}"


def invalid_14_digits() -> str:
    """Hard negative: 14 digits that are NOT a valid national ID (bad month / gov / century)."""
    kind = random.choice(["month", "gov", "century"])
    nid = national_id()
    if kind == "month":
        return nid[:3] + f"{random.randint(13, 99)}" + nid[5:]
    if kind == "gov":
        return nid[:7] + random.choice(["00", "05", "40", "77", "99"]) + nid[9:]
    return random.choice("14567890") + nid[1:]


def mobile(style: str | None = None) -> str:
    n = "1" + random.choice("0125") + f"{random.randint(0, 99999999):08d}"
    style = style or random.choice(["local", "intl", "spaced"])
    if style == "intl":
        return "+20" + n
    if style == "spaced":
        return f"0{n[:2]} {n[2:6]} {n[6:]}"
    return "0" + n


def iban() -> str:
    body = "".join(random.choice("0123456789") for _ in range(25))
    # compute valid check digits (ISO 7064 mod 97-10)
    numeric = body + "1416" + "00"  # letters -> numbers: E=14, G=16
    check = 98 - int(numeric) % 97
    return f"EG{check:02d}{body}"


def passport() -> str:
    return random.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + f"{random.randint(0, 99999999):08d}"


def address() -> str:
    return f"{random.randint(1, 250)} {random.choice(STREETS)}، {random.choice(CITIES)}"


def email(name_latin: str | None = None) -> str:
    base = (name_latin or latin_name()).split()[0].lower()
    return f"{base}{random.randint(1, 999)}@example.com"


def to_arabic_digits(s: str) -> str:
    return s.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

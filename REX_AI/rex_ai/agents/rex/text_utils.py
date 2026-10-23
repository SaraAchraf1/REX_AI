"""Arabic/English text helpers."""
from __future__ import annotations

import re

_AR_CHARS = re.compile(r"[؀-ۿ]")
_TASHKEEL = re.compile(r"[ً-ْـ]")  # diacritics + tatweel
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def detect_language(text: str) -> str:
    """'ar' if the message is mostly Arabic letters, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    ar = sum(1 for ch in letters if _AR_CHARS.match(ch))
    return "ar" if ar / len(letters) >= 0.3 else "en"


def normalize(text: str) -> str:
    """Lower-case, unify Arabic letter variants, strip diacritics, convert Arabic digits."""
    t = text.lower().translate(_AR_DIGITS)
    t = _TASHKEEL.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^\w\s\-+/.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    """Keyword match on normalized text; short Latin keywords must match whole words."""
    t = normalize(text)
    for kw in keywords:
        k = normalize(kw)
        if not k:
            continue
        if k.isascii() and len(k) <= 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", t):
                return True
        elif k in t:
            return True
    return False

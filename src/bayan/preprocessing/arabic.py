"""Lab 4: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)

class ArabicProfile:
 #  dediacritize: نشيل التشكل او لا 
    name: str
    dediacritize: bool = False
    
def normalize_arabic(text: str, profile: ArabicProfile) -> str:

    if text is None:
        return ""

    if text == "":
        return ""

    # 1) Unicode normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2) توحيد أشكال الألف
    # أ / إ / آ / ٱ  → ا
    normalized = re.sub(r"[إأآٱ]", "ا", normalized)

    # 3) Alef Maqsura → Ya
    # ى → ي
    normalized = normalized.replace("ى", "ي")

    # 4) Hamza-seat normalization
    # ؤ → و
    # ئ → ي
    normalized = normalized.replace("ؤ", "و")
    normalized = normalized.replace("ئ", "ي")

    # 5) Ta Marbuta → Ha
    #
    # هذا كان ناقص عندنا.
    #
    # مدرسة → مدرسه
    # خدمة  → خدمه
    # إضاءة → اضاءه
    normalized = normalized.replace("ة", "ه")

    # 6) Remove Tatweel / Kashida
    #
    # التطويل هو هذا الحرف:
    # ـ
    #
    # مثال:
    # مـشكلة → مشكلة
    normalized = normalized.replace("ـ", "")

    # 7) Remove diacritics if required
    #
    # مثال:
    # خِدْمَة → خدمه
    if profile.dediacritize:
        normalized = "".join(
            char
            for char in normalized
            if unicodedata.category(char) != "Mn"
        )

    # 8) Normalize spaces
    normalized = " ".join(normalized.split())

    return normalized

def segment(text: str) -> list[str]:

    # في Step 3 بنستخدم CAMeL Tools
    # لتقسيم الـclitics مثل:
    # وبالرياض
    # إلى شيء قريب من:
    # و + ب + ال + رياض
    # عشان LOCATION "رياض"
    # يصير ظاهر بشكل أوضح للـNER model.
    # ---------------------------------------------------------
    raise NotImplementedError



    """
    test
    pytest tests/test_arabic_normalize.py -q

    """
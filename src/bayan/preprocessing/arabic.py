"""Lab 4: per-model Arabic normalisation profiles."""

from dataclasses import dataclass
import re
import unicodedata


from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer
from camel_tools.tokenizers.word import simple_word_tokenize

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


# ---------------------------------------------------------
# تحميل أدوات الـsegmentation مرة وحدة
#
# MLEDisambiguator:
# يساعد يختار التحليل الصرفي الأنسب للكلمة
#
# MorphologicalTokenizer:
# يفصل الـclitics حسب scheme اسمها d3tok
# ---------------------------------------------------------

_disambig = MLEDisambiguator.pretrained()

_seg = MorphologicalTokenizer(
    _disambig,
    scheme="d3tok",
    split=True
)

def segment(text: str) -> list[str]:

    # ---------------------------------------------------------
    # STEP 1: نقسم النص إلى كلمات عادية أولاً
    #
    # مثال:
    # "انقطعت الكهرباء وبالرياض"
    #
    # ممكن تصير:
    # ["انقطعت", "الكهرباء", "وبالرياض"]
    # ---------------------------------------------------------
    words = simple_word_tokenize(text)


    # ---------------------------------------------------------
    # STEP 2: Morphological segmentation
    #
    # نفصل اللواصق/clitics
    #
    # مثال:
    #
    # وبالرياض
    #
    # تقريبًا:
    # و+ ب+ ال+ رياض
    #
    # وهنا "رياض" تصير token واضح
    # والـNER يقدر يعطيها B-LOCATION
    # ---------------------------------------------------------
    return _seg.tokenize(words)



"""

part1
    test
    pytest tests/test_arabic_normalize.py -q
    then call csv file to reach dataset : data/eval/arabic_normalize_golden.csv
    
    OUTPUT:
    30 passed in 1.50s
    
    
part 3
test sgemntaion

python -c "from bayan.preprocessing.arabic import segment; print(segment('وبالرياض'))"
['و+', 'ب+', 'ال+', 'رياض']

python -c "from bayan.preprocessing.arabic import segment; print(segment('انقطعت الكهرباء وبالرياض تأخرت الصيانة'))"
['انقطعت', 'ال+', 'كهرباء', 'و+', 'ب+', 'ال+', 'رياض', 'تأخرت', 'ال+', 'صيانة']

then wire the segmentation choice consistently into the NER data/training path using colab

    
"""
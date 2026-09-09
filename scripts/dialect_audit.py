"""Lab 4: audit dialect mix and record the implication in NOTES.md."""

import pandas as pd
from camel_tools.dialectid import DialectIdentifier


def main():

    # ---------------------------------------------------------
    # STEP 1: Load CAMeL Tools dialect identifier
    #
    # هذا موديل جاهز يحدد لهجة النص العربي.
    #
    # هنا نستخدم النسخة pretrained.
    # ---------------------------------------------------------
    did = DialectIdentifier.pretrained()


    # ---------------------------------------------------------
    # STEP 2: Load Bayan feedback dataset
    # ---------------------------------------------------------
    df = pd.read_csv("data/raw/bayan_feedback.csv")


    # ---------------------------------------------------------
    # STEP 3: Keep Arabic rows only
    #
    # لأننا نسوي Dialect Audit للعربي فقط.
    #
    # مثال:
    # English rows ما تهمنا هنا.
    # ---------------------------------------------------------
    ar = df[df["lang"] == "ar"].copy()


    # ---------------------------------------------------------
    # STEP 4: Predict dialect region
    #
    # نعطي الموديل النصوص العربية.
    #
    # "region" معناها نبي تصنيف عام للمنطقة
    # بدل تصنيف دقيق لكل مدينة أو دولة.
    #
    # مثل:
    # Gulf
    # MSA
    # وغيرها
    # ---------------------------------------------------------
    preds = did.predict(
        ar["text"].tolist(),
        "region"
    )


    # ---------------------------------------------------------
    # STEP 5: Save the predicted dialect
    #
    # p.top = أعلى prediction من الموديل.
    #
    # فنضيف column جديد اسمه:
    #
    # dialect_region
    # ---------------------------------------------------------
    ar["dialect_region"] = [
        p.top for p in preds
    ]


    # ---------------------------------------------------------
    # STEP 6: Calculate distribution
    #
    # value_counts(normalize=True)
    #
    # بدل ما يعطينا عدد الرسائل فقط،
    # يعطينا النسبة من كامل Arabic slice.
    #
    # مثال:
    #
    # Gulf  0.548
    #
    # معناها تقريبًا 54.8%
    # ---------------------------------------------------------
    distribution = (
        ar["dialect_region"]
        .value_counts(normalize=True)
        .round(3)
    )


    print("Dialect audit:")
    print(distribution)


    # ---------------------------------------------------------
    # STEP 7: Print percentages بشكل أوضح
    # ---------------------------------------------------------
    print("\nPercentages:")

    for region, ratio in distribution.items():
        print(f"{region}: {ratio * 100:.1f}%")


    # ---------------------------------------------------------
    # الهدف من الـAudit:
    #
    # إذا جزء كبير من الداتا Gulf،
    # ما ينفع نقيم الموديل على MSA فقط.
    #
    # لأن وقتها ممكن نقول الموديل ممتاز،
    # بينما فعليًا ما اختبرناه على لهجة
    # تمثل نسبة كبيرة من المستخدمين.
    # ---------------------------------------------------------
    print(
        "\nImplication: Evaluating only on MSA would ignore "
        "a significant portion of the Arabic dialect traffic."
    )


if __name__ == "__main__":
    main()
    
    
    """
    camel_data -i defaults
    python scripts/dialect_audit.py
    
    OUTPUT
        
    Dialect audit:
    dialect_region
    Gulf                      0.570
    Modern Standard Arabic    0.318
    Maghreb                   0.073
    Levant                    0.028
    Nile Basin                0.010
    Name: proportion, dtype: float64

    Percentages:
    Gulf: 57.0%
    Modern Standard Arabic: 31.8%
    Maghreb: 7.3%
    Levant: 2.8%
    Nile Basin: 1.0%

    
    """
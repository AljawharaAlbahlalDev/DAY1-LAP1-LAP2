"""
Lab 6.3 — Behavioural Tests

هذي تختلف عن F1 العادي.

نسأل:
هل model يتصرف منطقيًا في حالات محددة؟

أنواع الاختبارات هنا:

1) Invariance
   إذا غيرنا اسم الحي،
   Topic المفروض ما يتغير.

2) Orthographic variation
   اختلاف بسيط في الكتابة العربية
   ما المفروض يخرب prediction.

3) Directional test
   negation المفروض تغير sentiment
   بالاتجاه الصحيح.

4) Minimum Functionality Test
   NER لازم يقدر يلتقط أسماء الخدمات الأساسية.
"""


def invariance_test(
    classify_topic,
):

    pairs = [
        (
            "يوجد تسرب مياه في حي النرجس",
            "يوجد تسرب مياه في حي الملقا",
        ),
        (
            "الإنارة متعطلة في شارع العليا",
            "الإنارة متعطلة في شارع الملك فهد",
        ),
    ]

    passed = 0

    for first, second in pairs:

        if (
            classify_topic(
                first
            )
            == classify_topic(
                second
            )
        ):
            passed += 1

    return (
        passed
        / len(pairs)
    )


def orthographic_test(
    classify_topic,
):

    pairs = [
        (
            "مشكلة في الإنارة",
            "مشكلة في الانارة",
        ),
        (
            "الخدمة الإلكترونية لا تعمل",
            "الخدمه الالكترونيه لا تعمل",
        ),
    ]

    passed = 0

    for first, second in pairs:

        if (
            classify_topic(
                first
            )
            == classify_topic(
                second
            )
        ):
            passed += 1

    return (
        passed
        / len(pairs)
    )


def sentiment_directional_test(
    classify_sentiment,
):

    resolved = (
        "تم حل المشكلة "
        "وأشكركم على سرعة الاستجابة"
    )

    unresolved = (
        "لم يتم حل المشكلة "
        "رغم مرور أسبوعين"
    )

    order = [
        "negative",
        "neutral",
        "positive",
    ]

    resolved_label = (
        classify_sentiment(
            resolved
        )
    )

    unresolved_label = (
        classify_sentiment(
            unresolved
        )
    )

    return (
        order.index(
            unresolved_label
        )
        <
        order.index(
            resolved_label
        )
    )


def ner_service_mft(
    extract_entities,
    service_names,
):
    """
    MFT = Minimum Functionality Test.

    كل اسم خدمة نحطه داخل جملة،
    والموديل لازم يلتقطه SERVICE.
    """

    passed = 0

    for service in service_names:

        text = (
            f"أواجه مشكلة في خدمة "
            f"{service} منذ يومين"
        )

        entities = (
            extract_entities(
                text
            )
        )

        found = any(
            entity[
                "label"
            ]
            == "SERVICE"

            and service
            in entity[
                "text"
            ]

            for entity
            in entities
        )

        if found:
            passed += 1

    return (
        passed,
        len(
            service_names
        ),
    )


# ================================================================
# COURSE REFERENCE OUTPUT
# ================================================================
#
# invariance ≈ 96%
# negation-directional ≈ 88%
# MFT services ≈ 39/41
#
# هذي reference فقط.
# الناتج الحقيقي يعتمد على موديلاتك.
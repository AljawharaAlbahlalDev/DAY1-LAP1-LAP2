"""
Lab 7.5 — Startup Canaries

وش معنى Canary؟

هو check صغير نشغله عند startup قبل ما الـ API تبدأ تخدم users.

الهدف:
نتأكد أن:

1) الـ model اشتغل.
2) الـ tokenizer متوافق معه.
3) prediction ترجع بشكل صحيح.
4) score مو NaN.
5) artifact مو خربان أو incompatible.

إذا canary فشل:
الأفضل نخلي الـ service تفشل startup
بدل ما تشتغل وتعطي predictions غلط بصمت.

RUN:
    pytest tests/test_serving_contract.py -q
"""

import math


# ------------------------------------------------------------
# Fixed examples
# ------------------------------------------------------------
#
# هذي أمثلة ثابتة نجرب عليها الموديل عند startup.
#
# ما يهم هنا أنها تحقق metric معين.
# المهم نتأكد أن pipeline كامل يشتغل.
#

CANARY_TEXTS = [

    "الإنارة متعطلة في الشارع",

    "يوجد تسرب مياه منذ يومين",

    "لا أستطيع استخدام الخدمة الإلكترونية",

]


def run_classifier_canaries(
    predict_fn,
):
    """
    predict_fn:
        function تستقبل text
        وترجع prediction.

    مثال متوقع:

        {
            "label": "lighting",
            "score": 0.97
        }

    إذا أي check فشل:
        نرفع RuntimeError.
    """

    predictions = []

    for text in CANARY_TEXTS:

        # --------------------------------------------------------
        # Run prediction
        # --------------------------------------------------------

        result = predict_fn(
            text
        )

        # --------------------------------------------------------
        # Check 1:
        # prediction لازم ما تكون None.
        # --------------------------------------------------------

        if result is None:

            raise RuntimeError(
                "Canary failed: "
                f"prediction is None for text={text!r}"
            )

        # --------------------------------------------------------
        # Check 2:
        # إذا prediction dict،
        # لازم يكون عندنا label.
        # --------------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            label = result.get(
                "label"
            )

            if not label:

                raise RuntimeError(
                    "Canary failed: "
                    f"missing label for text={text!r}"
                )

            # ----------------------------------------------------
            # Check 3:
            # score إذا موجود، ما يكون NaN.
            # ----------------------------------------------------

            score = result.get(
                "score"
            )

            if score is not None:

                score = float(
                    score
                )

                if math.isnan(
                    score
                ):

                    raise RuntimeError(
                        "Canary failed: "
                        "prediction score is NaN."
                    )

        predictions.append(
            result
        )

    # إذا وصلنا هنا،
    # كل canaries عدت بنجاح.

    return {

        "ok": True,

        "count": len(
            predictions
        ),

        "predictions": predictions,

    }


# ================================================================
# EXPECTED
# ================================================================
#
# لو كل شيء سليم:
#
# {
#     "ok": True,
#     "count": 3,
#     "predictions": [...]
# }
#
# لو فيه مشكلة:
#
# RuntimeError:
# Canary failed: ...
#
# الهدف:
# fail early before accepting production traffic.
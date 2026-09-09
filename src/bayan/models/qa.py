"""Lab 3B: Extractive QA post-processing.

Select the best valid answer span from QA start/end logits,
or return an honest no-answer result.
"""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    null_score=None,
    null_threshold=0.0,
    max_answer_len=48,
    top_k=20,
    context=None,
):
    # ---------------------------------------------------------
    # STEP 1: No-Answer score
    #
    # في الـQA عندنا احتمالين:
    #
    # 1) الجواب موجود داخل الـContext
    # 2) ما فيه جواب
    #
    # إذا null_score انرسل لنا مباشرة، نستخدمه.
    # وهذا هو الشكل اللي تستخدمه Unit Tests.
    #
    # وإذا ما انرسل، نحسبه من Position 0،
    # اللي يمثل [CLS] / No Answer في SQuAD2-style QA.
    # ---------------------------------------------------------
    if null_score is None:
        null_score = (
            float(start_logits[0])
            + float(end_logits[0])
        )


    # ---------------------------------------------------------
    # STEP 2: Get the best Start and End candidates
    #
    # الـModel يعطي score لكل token:
    #
    # start_logits
    # → احتمال أن الـtoken يكون بداية الجواب
    #
    # end_logits
    # → احتمال أن الـtoken يكون نهاية الجواب
    #
    # بدل ما نجرب كل الاحتمالات،
    # نأخذ أعلى top_k فقط.
    # ---------------------------------------------------------
    starts = np.argsort(start_logits)[-top_k:]
    ends = np.argsort(end_logits)[-top_k:]


    # ---------------------------------------------------------
    # STEP 3: Prepare the best answer
    #
    # Span يعني:
    #
    # جزء متصل من النص يبدأ عند Start
    # وينتهي عند End.
    #
    # مثال:
    #
    # Context:
    # "The processing time is 2 business days."
    #
    # Start = "2"
    # End   = "days"
    #
    # Span = "2 business days"
    # ---------------------------------------------------------
    best = {
        "score": -1e9,
        "start": None,
        "end": None,
        "text": None,
    }


    # ---------------------------------------------------------
    # STEP 4: Try Start + End combinations
    #
    # نجرب أفضل Start candidates
    # مع أفضل End candidates.
    #
    # لكن لازم نرفض أي Span غير صالح.
    # ---------------------------------------------------------
    for s in starts:
        for e in ends:

            # -------------------------------------------------
            # Position 0:
            # مخصص للـNo Answer.
            #
            # e < s:
            # يعني نهاية الجواب قبل بدايته،
            # وهذا Inverted Span وغير صالح.
            #
            # max_answer_len:
            # يمنع اختيار جواب طويل جدًا.
            # -------------------------------------------------
            if (
                s == 0
                or e == 0
                or e < s
                or e - s + 1 > max_answer_len
            ):
                continue


            # -------------------------------------------------
            # offsets تربط الـTokens بمكانها داخل الـContext.
            #
            # None معناها Token مو تابع للـContext،
            # مثل Question token أو Special token.
            #
            # والـExtractive QA لازم يستخرج الجواب
            # من الـContext فقط.
            # -------------------------------------------------
            if offsets[s] is None or offsets[e] is None:
                continue


            # -------------------------------------------------
            # STEP 5: Calculate Span score
            #
            # قوة الجواب =
            #
            # Start score + End score
            #
            # كلما ارتفع الـscore،
            # كان هذا الـSpan أفضل بنظر الموديل.
            # -------------------------------------------------
            score = (
                float(start_logits[s])
                + float(end_logits[e])
            )


            # إذا هذا أفضل Span شفناه حتى الآن،
            # نخزن بدايته ونهايته والـscore.
            if score > best["score"]:

                best["score"] = score
                best["start"] = int(s)
                best["end"] = int(e)

                # -------------------------------------------------
                # إذا عندنا Context فعلي،
                # نستخدم offsets لاستخراج النص.
                #
                # مثال:
                #
                # Context:
                # "The fee is 50 SAR."
                #
                # Start = "50"
                # End   = "SAR"
                #
                # الناتج:
                # "50 SAR"
                # -------------------------------------------------
                if context is not None:
                    best["text"] = context[
                        offsets[s][0]:offsets[e][1]
                    ]


    # ---------------------------------------------------------
    # STEP 6: Honest No-Answer handling
    #
    # نقارن:
    #
    # null_score
    #      VS
    # best span score
    #
    # إذا قرار "No Answer" أقوى
    # بأكثر من null_threshold،
    # نرجع answer=None.
    #
    # وهذا يمنع النظام من اختراع جواب
    # لما ما يكون فيه جواب داخل الـContext.
    # ---------------------------------------------------------
    if (
        best["start"] is None
        or null_score - best["score"] > null_threshold
    ):
        return {
            "answer": None,
            "start": None,
            "end": None,
            "reason": "no_answer_in_context",
            "margin": float(
                null_score - best["score"]
            ),
        }


    # ---------------------------------------------------------
    # STEP 7: Return the best valid answer
    #
    # إذا context موجود:
    # answer يكون النص المستخرج.
    #
    # إذا context مو موجود، مثل Unit Tests:
    # نخلي answer عبارة عن character span
    # من offsets.
    #
    # ونرجع start/end أيضًا حتى يكون
    # واضح أي Token span تم اختياره.
    # ---------------------------------------------------------
    if context is not None:
        answer = best["text"]
    else:
        answer = (
            offsets[best["start"]][0],
            offsets[best["end"]][1],
        )


    return {
        "answer": answer,
        "start": best["start"],
        "end": best["end"],
        "score": best["score"],
    }
"""Lab 3B: run the 12-question QA smoke set."""

import json

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


# ---------------------------------------------------------
# Public QA checkpoint from Hugging Face.
# هذا الموديل مدرب على SQuAD 2.0،

# يعني يعرف يتعامل مع:
# - Questions لها جواب داخل الـContext
# - Questions ما لها جواب ويرجع لها No Answer
# ---------------------------------------------------------
MODEL_NAME = "deepset/roberta-base-squad2"

# هذا هو ملف الـ12 سؤال الخاص بالـsmoke test.
SMOKE_SET_PATH = "data/eval/qa_smoke_set.json"

def normalize_text(text):
    """
    Normalize text before comparison.

    الهدف هنا فقط نخلي المقارنة عادلة:
    - نشيل المسافات الزايدة
    - نحول النص إلى lowercase

    مثال:
    "50 SAR" و " 50 sar "
    نعتبرهم نفس الجواب.
    """
    if text is None:
        return None

    return " ".join(text.strip().lower().split())


def main():

    # ---------------------------------------------------------
    # STEP 1: Load the QA tokenizer and model
    #
    # هنا نحمل الـTokenizer والـQA Model الجاهز.
    #
    # الـTokenizer:
    # يحول Question + Context إلى tokens وأرقام يفهمها الموديل.
    #
    # الـQA Model:
    # ما يكتب جواب من عنده.
    # يعطي score لكل token:
    # - هل يصلح يكون بداية الجواب؟
    # - هل يصلح يكون نهاية الجواب؟
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    # إذا عندنا GPU نستخدمه، وإذا ما عندنا نستخدم CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    # eval() معناها أننا الآن في مرحلة Inference / Evaluation،
    # مو Training.
    model.eval()

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")


    # ---------------------------------------------------------
    # STEP 2: Load the 12-question smoke set
    #
    # الملف يحتوي على:
    #
    # Context:
    # النص اللي المفروض الجواب يكون موجود داخله.
    #
    # Question:
    # السؤال اللي بنسأله للموديل.
    #
    # Answers:
    # الجواب الصحيح المتوقع.
    #
    # is_impossible:
    # إذا True معناها السؤال ما له جواب داخل الـContext.
    # ---------------------------------------------------------
    with open(
        SMOKE_SET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        smoke_data = json.load(f)


    # عدادات نحسب فيها النتائج النهائية.
    answerable_correct = 0
    answerable_total = 0

    unanswerable_correct = 0
    unanswerable_total = 0


    # ---------------------------------------------------------
    # STEP 3: Loop through every Question + Context
    #
    # بنمر على كل سؤال من الـ12 سؤال،
    # ونشغله على الـQA pipeline.
    # ---------------------------------------------------------
    for item in smoke_data["data"]:

        for paragraph in item["paragraphs"]:

            # هذا هو النص اللي بندور داخله على الجواب.
            context = paragraph["context"]

            for qa in paragraph["qas"]:

                question = qa["question"]

                # True = السؤال ما له جواب في الـContext.
                # False = السؤال له جواب داخل الـContext.
                is_impossible = qa["is_impossible"]


                # -------------------------------------------------
                # STEP 4: Tokenize Question + Context
                #
                # ندخل الاثنين مع بعض للموديل:
                #
                # Question + Context
                #        ↓
                # Tokenizer
                #        ↓
                # Tokens / IDs
                #
                # return_offsets_mapping=True مهم جدًا.
                #
                # لأنه بعد ما يختار الموديل Start و End على مستوى Tokens،
                # نحتاج نعرف وين مكانهم في النص الأصلي.
                # -------------------------------------------------
                inputs = tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=384,
                )


                # -------------------------------------------------
                # الـoffset mapping يربط كل Token بمكانه في النص الأصلي.
                #
                # مثال:
                #
                # Context:
                # "The fee is 50 SAR"
                #
                # لو الموديل اختار:
                # Start token = "50"
                # End token   = "SAR"
                #
                # الـoffsets تساعدنا نقص:
                # "50 SAR"
                #
                # من الـContext الأصلي.
                # -------------------------------------------------
                offset_mapping = (
                    inputs.pop("offset_mapping")[0].tolist()
                )


                # -------------------------------------------------
                # الـTokenizer يحتوي على:
                #
                # Question tokens
                # Special tokens
                # Context tokens
                #
                # لكن الجواب لازم يطلع من الـContext فقط.
                #
                # sequence_id == 1
                # معناها هذا الـtoken جاي من الـContext.
                #
                # أي token مو من الـContext نخلي offset حقه None
                # حتى best_span() ما يختاره كجواب.
                # -------------------------------------------------
                sequence_ids = inputs.sequence_ids(0)

                offsets = [
                    tuple(offset)
                    if sequence_id == 1
                    else None
                    for offset, sequence_id in zip(
                        offset_mapping,
                        sequence_ids
                    )
                ]


                # نحول الـinputs لنفس الجهاز اللي عليه الموديل.
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }


                # -------------------------------------------------
                # STEP 5: Run the QA model
                #
                # الموديل يرجع شيئين مهمين:
                #
                # start_logits
                # = score لكل token:
                #   هل هذا أفضل مكان تبدأ منه الإجابة؟
                #
                # end_logits
                # = score لكل token:
                #   هل هذا أفضل مكان تنتهي عنده الإجابة؟
                #
                # مثال:
                #
                # Context:
                # "The processing time is 2 business days."
                #
                # ممكن يعطي أعلى Start score لـ "2"
                # وأعلى End score لـ "days"
                #
                # وبكذا الـSpan يكون:
                # "2 business days"
                # -------------------------------------------------
                with torch.no_grad():

                    outputs = model(**inputs)


                start_logits = (
                    outputs.start_logits[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                end_logits = (
                    outputs.end_logits[0]
                    .detach()
                    .cpu()
                    .numpy()
                )


                # -------------------------------------------------
                # STEP 6: Select the best span
                #
                # هنا نستخدم best_span() اللي سويناها في Step 3.
                #
                # Span معناها:
                # جزء متصل من النص يبدأ من Start وينتهي عند End.
                #
                # مثال:
                #
                # Start = "2"
                # End   = "days"
                #
                # Span:
                # "2 business days"
                #
                # best_span() تسوي أكثر من مجرد argmax:
                #
                # - تتأكد أن End بعد Start
                # - تتأكد أن الجواب مو طويل جدًا
                # - تتأكد أن الجواب من الـContext
                # - وتقارن الجواب مع Null score
                #
                # فإذا النظام يشوف أن ما فيه جواب مناسب:
                # يرجع answer=None
                # -------------------------------------------------
                result = best_span(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    offsets=offsets,
                    context=context,
                    null_threshold=0.0,
                )

                prediction = result["answer"]


                # -------------------------------------------------
                # STEP 7: Evaluate the prediction
                #
                # عندنا حالتين:
                #
                # 1) Answerable
                #    الجواب موجود داخل الـContext.
                #
                # 2) Unanswerable
                #    الجواب غير موجود داخل الـContext،
                #    والمفروض النظام يرجع None.
                # -------------------------------------------------

                if is_impossible:

                    # هذا سؤال ما له جواب.
                    unanswerable_total += 1

                    # إذا الموديل رجع None،
                    # إذًا تصرف بشكل صحيح.
                    if prediction is None:
                        unanswerable_correct += 1

                    expected = None

                else:

                    # هذا سؤال له جواب داخل الـContext.
                    answerable_total += 1

                    # نجيب الـGold Answer من ملف الـJSON.
                    expected = qa["answers"][0]["text"]

                    # نقارن Prediction مع Expected Answer.
                    if (
                        normalize_text(prediction)
                        == normalize_text(expected)
                    ):
                        answerable_correct += 1


                # -------------------------------------------------
                # نطبع نتيجة كل سؤال عشان نشوف:
                #
                # Question
                # Expected Answer
                # Model Prediction
                #
                # وهذا مهم في الـSmoke Test،
                # لأننا مو بس نبي رقم نهائي،
                # نبي نشوف سلوك النظام سؤال سؤال.
                # -------------------------------------------------
                print("-" * 70)

                print(
                    f"Question : {question}"
                )

                print(
                    f"Expected : {expected}"
                )

                print(
                    f"Predicted: {prediction}"
                )


                # إذا best_span رجعت margin،
                # نعرض الفرق بين Null score وأفضل Answer span.
                #
                # هذا يساعدنا نفهم ليش النظام قرر:
                # Answer
                # أو
                # No Answer
                if "margin" in result:

                    print(
                        f"Null margin: "
                        f"{result['margin']:.4f}"
                    )


    # ---------------------------------------------------------
    # STEP 8: Final smoke-test result
    #
    # الهدف الأصلي للّاب:
    #
    # 9/9 answerable questions
    # -> correct span
    #
    # 3/3 unanswerable questions
    # -> answer=None
    #
    # يعني المطلوب مو فقط أن النظام يجاوب،
    # المطلوب أيضًا يعرف متى ما فيه جواب.
    # -------------------------------------------------
    print("\n" + "=" * 70)

    print(
        "QA SMOKE TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Answerable: "
        f"{answerable_correct}/"
        f"{answerable_total} correct"
    )

    print(
        f"Unanswerable: "
        f"{unanswerable_correct}/"
        f"{unanswerable_total} returned None"
    )


if __name__ == "__main__":
    main()
    
    
    """
    run : python scripts/qa_smoke.py 
    
    OUTPUT:
    بنحمل موديل deepset/roberta-base-squad2
    بعدها بنطبع الاسئلة
    
    
Model: deepset/roberta-base-squad2
Device: cpu
----------------------------------------------------------------------
Question : How long does Service 01 take?
Expected : 2 business days
Predicted: 2 business days
----------------------------------------------------------------------
Question : What is the fee for Service 01?
Expected : 50 SAR
Predicted: 50 SAR
----------------------------------------------------------------------
Question : Where is the application for Service 01 submitted?
Expected : Bayan portal
Predicted: Bayan portal
----------------------------------------------------------------------
Question : How long does Service 01 take?
Expected : 2 business days
Predicted: 2 business days
----------------------------------------------------------------------
Question : What is the fee for Service 01?
Expected : 50 SAR
Predicted: 50 SAR
----------------------------------------------------------------------
Question : Where is the application for Service 01 submitted?
Expected : Bayan portal
Predicted: Bayan portal
----------------------------------------------------------------------
Question : How long does Service 01 take?
Expected : 2 business days
Predicted: 2 business days
----------------------------------------------------------------------
Question : What is the fee for Service 01?
Expected : 50 SAR
Predicted: 50 SAR
----------------------------------------------------------------------
Question : Where is the application for Service 01 submitted?
Expected : Bayan portal
Predicted: Bayan portal
----------------------------------------------------------------------
Question : How long does Service 01 take?
Expected : 2 business days
Predicted: 2 business days
----------------------------------------------------------------------
Question : What is the fee for Service 01?
Expected : 50 SAR
Predicted: 50 SAR
----------------------------------------------------------------------
Question : Where is the application for Service 01 submitted?
Expected : Bayan portal
Predicted: Bayan portal

======================================================================
QA SMOKE TEST RESULTS
======================================================================
Answerable: 12/12 correct
Unanswerable: 0/0 returned None


 """
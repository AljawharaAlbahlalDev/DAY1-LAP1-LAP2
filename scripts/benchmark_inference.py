"""
Lab 7.1 — Real CPU Benchmark

نقيس:
    p50
    p99
    mean latency

قبل أي optimisation.

RUN:
    export OMP_NUM_THREADS=4
    python scripts/benchmark_inference.py

مهم:
لا نقارن GPU result مع CPU result.
لازم نفس الجهاز ونفس الـthreads.
"""

import os
import time

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


MODEL_PATH = (
    "artifacts/"
    "topic_classifier"
)


def benchmark(
    predict_fn,
    texts,
    warmup=30,
    iterations=300,
    seed=42,
):
    """
    warmup:
        أول requests ما نحسبها،
        لأنها تشمل one-time setup/caching.

    iterations:
        عدد requests اللي نقيسها.
    """

    rng = (
        np.random.default_rng(
            seed
        )
    )

    # نسحب نصوص من production mix.

    sample = [
        texts[
            i
        ]
        for i
        in rng.integers(
            0,
            len(texts),
            iterations
            + warmup,
        )
    ]

    # --------------------------------------------
    # Warm-up
    # --------------------------------------------

    for text in sample[
        :warmup
    ]:

        predict_fn(
            text
        )

    # --------------------------------------------
    # Actual timings
    # --------------------------------------------

    latencies = []

    for text in sample[
        warmup:
    ]:

        start = (
            time.perf_counter()
        )

        predict_fn(
            text
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            elapsed_ms
        )

    mean_ms = float(
        np.mean(
            latencies
        )
    )

    return {
        "p50_ms": float(
            np.percentile(
                latencies,
                50,
            )
        ),

        "p99_ms": float(
            np.percentile(
                latencies,
                99,
            )
        ),

        "mean_ms": mean_ms,

        "throughput_rps": (
            1000
            / mean_ms
        ),
    }


def main():

    # --------------------------------------------
    # Pin CPU threads
    # --------------------------------------------

    threads = int(
        os.getenv(
            "OMP_NUM_THREADS",
            "4",
        )
    )

    torch.set_num_threads(
        threads
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    model.eval()

    # Representative text lengths.
    #
    # إذا data/serving/bench_mix.npy عندكم موجود،
    # استخدميه بدل هذي الأمثلة.

    texts = [
        "الإنارة متعطلة",

        (
            "يوجد تسرب مياه "
            "في الحي منذ يومين"
        ),

        (
            "الخدمة الإلكترونية لا تعمل "
            "عند محاولة رفع الطلب"
        ),

        (
            "أواجه مشكلة متكررة في الخدمة "
            "منذ عدة أيام وعند محاولة إرسال الطلب "
            "تظهر رسالة خطأ ولا يتم حفظ البيانات"
        ),
    ] * 100

    def predict(
        text,
    ):

        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,

            # dynamic padding
            padding=False,
        )

        with torch.inference_mode():

            model(
                **batch
            )

    result = benchmark(
        predict,
        texts,
    )

    print(
        "CPU benchmark"
    )

    print(
        f"threads: "
        f"{threads}"
    )

    print(
        f"p50: "
        f"{result['p50_ms']:.2f} ms"
    )

    print(
        f"p99: "
        f"{result['p99_ms']:.2f} ms"
    )

    print(
        f"mean: "
        f"{result['mean_ms']:.2f} ms"
    )

    print(
        f"throughput: "
        f"{result['throughput_rps']:.2f} req/s"
    )


if __name__ == "__main__":
    main()


# EXPECTED SHAPE:
#
# CPU benchmark
# threads: 4
# p50: <your machine> ms
# p99: <your machine> ms
# mean: <your machine> ms
# throughput: <your machine> req/s
#
# ما فيه رقم ثابت لأن CPU يفرق.
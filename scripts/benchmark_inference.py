"""Lab 7: simple honest p50/p99 benchmark harness."""

import time
import numpy as np


def benchmark(
    predict_fn,
    inputs,
    *,
    warmup=5,
):
    """
    Measure inference latency.

    predict_fn:
        function that runs the model

    inputs:
        list of model inputs

    Returns:
        p50 and p99 latency in milliseconds
    """

    if not inputs:
        raise ValueError("inputs must not be empty")

    # -----------------------------------------
    # 1) Warm-up
    # أول كم request ما نحسبهم
    # لأن الموديل يكون توه يتحمل بالذاكرة
    # -----------------------------------------
    for x in inputs[:warmup]:
        predict_fn(x)

    latencies = []

    # -----------------------------------------
    # 2) Measure every inference
    # -----------------------------------------
    for x in inputs:

        start = time.perf_counter()

        predict_fn(x)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    # -----------------------------------------
    # 3) Percentiles
    # -----------------------------------------
    p50 = float(np.percentile(latencies, 50))
    p99 = float(np.percentile(latencies, 99))

    return {
        "p50_ms": p50,
        "p99_ms": p99,
        "n": len(latencies),
    }


if __name__ == "__main__":

    # -----------------------------------------
    # Smoke test فقط للتأكد إن الـbenchmark
    # harness نفسه شغال.
    #
    # لاحقاً نستبدل dummy_predict بالموديل الحقيقي.
    # -----------------------------------------
    def dummy_predict(text):
        return len(text)

    sample_inputs = [
        "بلاغ عن إنارة الشارع",
        "مشكلة في فاتورة المياه",
        "طلب إصدار رخصة جديدة",
        "النفايات لم يتم جمعها",
        "الخدمة الإلكترونية لا تعمل",
    ]

    results = benchmark(
        dummy_predict,
        sample_inputs,
    )

    print("Lab 7 baseline benchmark")
    print("------------------------")
    print(f"p50: {results['p50_ms']:.3f} ms")
    print(f"p99: {results['p99_ms']:.3f} ms")
    print(f"n:   {results['n']}")
    
    
 """
    test 
    python scripts/benchmark_inference.py

    
    output
Lab 7 baseline benchmark
------------------------
p50: 0.000 ms
p99: 0.001 ms
n:   5
((.venv) ) Chr
        
        """
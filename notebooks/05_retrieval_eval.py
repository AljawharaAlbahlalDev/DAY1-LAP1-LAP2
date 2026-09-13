"""
Lab 5.3 — Retrieval Evaluation

وش نسوي هنا؟

1) نقيم Stage 1 لحاله:
   Bi-Encoder + FAISS

2) نقيم Stage 1 + Stage 2:
   FAISS + Cross-Encoder Re-ranking

3) نحسب:
   - Recall@10
   - MRR@10

4) نقيس Cross-lingual performance.

5) نقيس No-answer behaviour.

RUN:
    python notebooks/05_retrieval_eval.py

COURSE TARGETS:
    Recall@10 >= 0.80
    MRR@10 >= 0.70
    No-answer correctness >= 17/20

REFERENCE OUTPUT:
    stage 1 only:
        Recall@10 ≈ 0.83
        MRR@10    ≈ 0.61

    with reranking:
        Recall@10 ≈ 0.85
        MRR@10    ≈ 0.74

مهم:
الأرقام فوق Reference فقط.
اكتبي في BENCHMARKS.md الأرقام اللي تطلع من تشغيلك.
"""

import time

import numpy as np
import pandas as pd

from bayan.search.service import CaseSearch


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

QUERIES_PATH = (
    "data/search/"
    "bayan_queries.jsonl"
)

INDEX_PREFIX = (
    "artifacts/"
    "case_index_v1"
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_ids(value):
    """
    نحول IDs كلها إلى strings.

    ليه؟

    لأن:
        123 != "123"

    فإذا relevant_case_ids رقم
    والـ search رجع string،
    ممكن نحسبها غلط كأنها miss.
    """

    if value is None:
        return set()

    if isinstance(
        value,
        (list, tuple, set, np.ndarray),
    ):
        return {
            str(x)
            for x in value
        }

    return {
        str(value)
    }


def find_rank(
    returned_ids,
    relevant_ids,
):
    """
    نبحث عن أول relevant document.

    مثال:

        returned:
            A
            B
            C

        relevant:
            C

    rank = 3
    """

    for index, case_id in enumerate(
        returned_ids,
        start=1,
    ):
        if case_id in relevant_ids:
            return index

    return None


def evaluate(
    search,
    queries,
    *,
    rerank,
    k=10,
):
    """
    نحسب:

        Recall@10
        MRR@10
        latency

    rerank=False:
        Stage 1 فقط

    rerank=True:
        Stage 1 + Cross-Encoder
    """

    hits = []
    reciprocal_ranks = []
    latencies = []

    for _, row in queries.iterrows():

        relevant_ids = normalize_ids(
            row[
                "relevant_case_ids"
            ]
        )

        # لو ما عندنا relevant case
        # هذا يعتبر no-answer query،
        # فنتركه من Recall/MRR.
        if len(relevant_ids) == 0:
            continue

        start = time.perf_counter()

        results = search.search(
            row["query"],
            k=k,
            candidates=(
                50
                if rerank
                else k
            ),

            # أثناء تقييم retrieval
            # ما نبي min_score يحذف النتائج.
            min_score=None,

            rerank=rerank,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            elapsed_ms
        )

        returned_ids = [
            str(
                result[
                    "case_id"
                ]
            )
            for result
            in results
        ]

        rank = find_rank(
            returned_ids,
            relevant_ids,
        )

        # Recall@10:
        # هل حصلنا relevant case ضمن top 10؟
        hits.append(
            rank is not None
        )

        # MRR:
        # 1 / rank
        #
        # rank 1 -> 1.0
        # rank 2 -> 0.5
        # rank 5 -> 0.2
        # no hit -> 0
        reciprocal_ranks.append(
            1 / rank
            if rank
            else 0.0
        )

    recall = float(
        np.mean(
            hits
        )
    )

    mrr = float(
        np.mean(
            reciprocal_ranks
        )
    )

    mean_latency = float(
        np.mean(
            latencies
        )
    )

    return {
        "recall@10": recall,
        "mrr@10": mrr,
        "mean_latency_ms": mean_latency,
    }


# ------------------------------------------------------------
# Cross-lingual Evaluation
# ------------------------------------------------------------

def evaluate_by_language(
    search,
    queries,
):
    """
    نقارن أداء البحث حسب لغة الـ query.

    مثال:

        Arabic query
        English query

    الهدف:
    نشوف هل multilingual search
    شغال بنفس القوة في اللغتين أو لا.
    """

    for lang in [
        "ar",
        "en",
    ]:

        subset = queries[
            queries[
                "lang"
            ]
            == lang
        ]

        if len(
            subset
        ) == 0:
            continue

        metrics = evaluate(
            search,
            subset,
            rerank=True,
        )

        print(
            f"{lang} queries -> "
            f"Recall@10="
            f"{metrics['recall@10']:.4f} "
            f"MRR@10="
            f"{metrics['mrr@10']:.4f}"
        )


# ------------------------------------------------------------
# No-answer Evaluation
# ------------------------------------------------------------

def evaluate_no_answer(
    search,
    queries,
    min_score=0.25,
):
    """
    No-answer query:

    relevant_case_ids = []

    الهدف:
    بدل ما النظام يرجع أي نتيجة ضعيفة،
    يرجع [].

    هذا يسمونه:
        honest empty result
    """

    no_answer_queries = []

    for _, row in queries.iterrows():

        relevant_ids = normalize_ids(
            row[
                "relevant_case_ids"
            ]
        )

        if len(
            relevant_ids
        ) == 0:
            no_answer_queries.append(
                row
            )

    correct = 0

    for row in no_answer_queries:

        results = search.search(
            row[
                "query"
            ],
            k=5,
            candidates=50,
            min_score=min_score,
            rerank=True,
        )

        if results == []:
            correct += 1

    total = len(
        no_answer_queries
    )

    return (
        correct,
        total,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # Load labelled evaluation queries.
    queries = pd.read_json(
        QUERIES_PATH,
        lines=True,
    )

    # Load search service.
    search = CaseSearch(
        INDEX_PREFIX
    )

    print(
        "=" * 60
    )

    print(
        "LAB 5.3 — RETRIEVAL EVALUATION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Stage 1 only
    # --------------------------------------------------------

    stage1 = evaluate(
        search,
        queries,
        rerank=False,
    )

    print(
        "\nStage 1 only"
    )

    print(
        f"Recall@10: "
        f"{stage1['recall@10']:.4f}"
    )

    print(
        f"MRR@10:    "
        f"{stage1['mrr@10']:.4f}"
    )

    print(
        f"Latency:   "
        f"{stage1['mean_latency_ms']:.2f} ms/query"
    )

    # --------------------------------------------------------
    # Stage 1 + Re-ranking
    # --------------------------------------------------------

    reranked = evaluate(
        search,
        queries,
        rerank=True,
    )

    print(
        "\nWith Cross-Encoder Re-ranking"
    )

    print(
        f"Recall@10: "
        f"{reranked['recall@10']:.4f}"
    )

    print(
        f"MRR@10:    "
        f"{reranked['mrr@10']:.4f}"
    )

    print(
        f"Latency:   "
        f"{reranked['mean_latency_ms']:.2f} ms/query"
    )

    # --------------------------------------------------------
    # MRR lift
    # --------------------------------------------------------

    mrr_lift = (
        reranked[
            "mrr@10"
        ]
        - stage1[
            "mrr@10"
        ]
    )

    print(
        f"\nMRR lift: "
        f"{mrr_lift:+.4f}"
    )

    # --------------------------------------------------------
    # Query language slices
    # --------------------------------------------------------

    print(
        "\nLanguage slices"
    )

    evaluate_by_language(
        search,
        queries,
    )

    # --------------------------------------------------------
    # No-answer
    # --------------------------------------------------------

    correct, total = (
        evaluate_no_answer(
            search,
            queries,
            min_score=0.25,
        )
    )

    print(
        "\nNo-answer behaviour"
    )

    print(
        f"{correct}/{total} "
        f"correctly returned []"
    )

    # --------------------------------------------------------
    # Debug block
    # --------------------------------------------------------
    #
    # إذا Recall طلع مرة منخفض مثل:
    #
    #   0.01
    #   0.02
    #
    # شيكي أول query يدويًا.
    #

    first = queries.iloc[0]

    debug_results = search.search(
        first[
            "query"
        ],
        k=10,
        candidates=50,
        min_score=None,
        rerank=True,
    )

    print(
        "\nDEBUG FIRST QUERY"
    )

    print(
        "Query:",
        first[
            "query"
        ],
    )

    print(
        "Relevant IDs:",
        normalize_ids(
            first[
                "relevant_case_ids"
            ]
        ),
    )

    print(
        "Returned IDs:",
        [
            str(
                result[
                    "case_id"
                ]
            )
            for result
            in debug_results
        ],
    )


if __name__ == "__main__":
    main()


# ================================================================
# EXPECTED / REFERENCE OUTPUT
# ================================================================
#
# Stage 1 only
# Recall@10: ~0.83
# MRR@10:    ~0.61
#
# With Cross-Encoder Re-ranking
# Recall@10: ~0.85
# MRR@10:    ~0.74
#
# MRR lift:
# positive
#
# No-answer:
# around 18/20
#
# مهم:
# هذه Reference values من الكورس.
# ناتجك الفعلي هو اللي ينكتب في BENCHMARKS.md.
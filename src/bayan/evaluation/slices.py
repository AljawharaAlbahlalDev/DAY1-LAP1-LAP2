"""
Lab 6.2 — Sliced Evaluation

ليش نحتاج Slices؟

لأن aggregate Macro-F1 ممكن يكون ممتاز،
لكن model يكون ضعيف جدًا على:

- Gulf dialect
- English
- long texts
- class معينة

هنا نحسب:
    Macro-F1
    Bootstrap Confidence Interval

لكل slice.

RUN:
    python -m bayan.evaluation.slices

EXPECTED:
    جدول فيه:
        slice
        n
        macro_f1
        ci_low
        ci_high
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
)


def macro_f1(
    y_true,
    y_pred,
):
    """
    حساب Macro-F1.
    """

    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_metric_ci(
    y_true,
    y_pred,
    n_boot=1000,
    seed=42,
    alpha=0.05,
):
    """
    نعمل Bootstrap للـ Macro-F1.

    كل مرة:
        نسحب rows مع replacement
        نحسب F1

    بعدها نأخذ:
        2.5 percentile
        97.5 percentile

    عشان نحصل تقريبًا على:
        95% Confidence Interval
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    scores = []

    for _ in range(
        n_boot
    ):

        indices = rng.integers(
            0,
            len(y_true),
            size=len(y_true),
        )

        score = macro_f1(
            y_true[
                indices
            ],
            y_pred[
                indices
            ],
        )

        scores.append(
            score
        )

    low = np.quantile(
        scores,
        alpha / 2,
    )

    high = np.quantile(
        scores,
        1 - alpha / 2,
    )

    return (
        float(low),
        float(high),
    )


def sliced_report(
    df,
    slice_columns=None,
):
    """
    نبني report كامل.

    DataFrame لازم يحتوي:
        y_true
        y_pred

    وممكن يحتوي:
        lang
        dialect_region
        n_words
    """

    df = df.copy()

    # --------------------------------------------
    # Create text-length buckets
    # --------------------------------------------

    if (
        "n_words"
        in df.columns
    ):

        df[
            "length_bucket"
        ] = pd.cut(
            df[
                "n_words"
            ],
            bins=[
                -np.inf,
                15,
                40,
                np.inf,
            ],
            labels=[
                "short",
                "medium",
                "long",
            ],
        )

    if slice_columns is None:

        slice_columns = [
            column
            for column
            in [
                "lang",
                "dialect_region",
                "length_bucket",
            ]
            if column
            in df.columns
        ]

    rows = []

    # --------------------------------------------
    # Aggregate
    # --------------------------------------------

    score = macro_f1(
        df[
            "y_true"
        ],
        df[
            "y_pred"
        ],
    )

    low, high = (
        bootstrap_metric_ci(
            df[
                "y_true"
            ].values,
            df[
                "y_pred"
            ].values,
        )
    )

    rows.append(
        {
            "slice_type": "all",
            "slice_value": "all",
            "n": len(df),
            "macro_f1": score,
            "ci_low": low,
            "ci_high": high,
        }
    )

    # --------------------------------------------
    # Slice columns
    # --------------------------------------------

    for column in slice_columns:

        for (
            value,
            group,
        ) in df.groupby(
            column,
            dropna=False,
        ):

            score = macro_f1(
                group[
                    "y_true"
                ],
                group[
                    "y_pred"
                ],
            )

            low, high = (
                bootstrap_metric_ci(
                    group[
                        "y_true"
                    ].values,
                    group[
                        "y_pred"
                    ].values,
                )
            )

            rows.append(
                {
                    "slice_type": column,
                    "slice_value": str(value),
                    "n": len(group),
                    "macro_f1": score,
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    return pd.DataFrame(
        rows
    )


if __name__ == "__main__":

    df = pd.read_parquet(
        "artifacts/"
        "topic_val_predictions.parquet"
    )

    report = sliced_report(
        df
    )

    print(
        report.to_string(
            index=False
        )
    )

    report.to_csv(
        "artifacts/"
        "sliced_report.csv",
        index=False,
    )


# ================================================================
# EXPECTED OUTPUT SHAPE
# ================================================================
#
# slice_type       slice_value      n    macro_f1   ci_low   ci_high
# all              all             ...   ...        ...      ...
# lang             ar              ...   ...        ...      ...
# lang             en              ...   ...        ...      ...
# dialect_region   Gulf            ...   ...        ...      ...
#
# Course reference:
# Gulf slice ≈ 0.762 [0.729, 0.793]
#
# لكن استخدمي measured result حقك.
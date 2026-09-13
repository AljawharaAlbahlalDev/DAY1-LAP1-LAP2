"""
Lab 6.5 — Final Evaluation Report

يجمع:
    sliced metrics
    bootstrap confidence intervals
    manual error taxonomy

ويطلع:
    EVALUATION_REPORT.md

RUN:
    python scripts/evaluation_report.py
"""

from pathlib import Path

import pandas as pd

from bayan.evaluation.slices import (
    sliced_report,
)


PREDICTIONS = (
    "artifacts/"
    "topic_val_predictions.parquet"
)

ERROR_WORKSHEET = (
    "reports/"
    "error_worksheet.csv"
)

OUTPUT = Path(
    "EVALUATION_REPORT.md"
)


def main():

    predictions = pd.read_parquet(
        PREDICTIONS
    )

    slices = sliced_report(
        predictions
    )

    # --------------------------------------------
    # Find weakest slice
    # --------------------------------------------

    non_aggregate = slices[
        slices[
            "slice_type"
        ]
        != "all"
    ]

    if len(
        non_aggregate
    ):

        weakest = (
            non_aggregate
            .sort_values(
                "macro_f1"
            )
            .iloc[0]
        )

        headline = (
            f"Weakest slice: "
            f"{weakest['slice_type']}="
            f"{weakest['slice_value']} "
            f"Macro-F1="
            f"{weakest['macro_f1']:.3f} "
            f"[{weakest['ci_low']:.3f}, "
            f"{weakest['ci_high']:.3f}]"
        )

    else:

        headline = (
            "No slice data available."
        )

    # --------------------------------------------
    # Error taxonomy
    # --------------------------------------------

    taxonomy_text = (
        "Error worksheet not completed yet."
    )

    if Path(
        ERROR_WORKSHEET
    ).exists():

        errors = pd.read_csv(
            ERROR_WORKSHEET
        )

        if (
            "taxonomy_tag"
            in errors.columns
        ):

            tags = (
                errors[
                    "taxonomy_tag"
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            tags = tags[
                tags
                != ""
            ]

            if len(
                tags
            ):

                taxonomy = (
                    tags
                    .value_counts(
                        normalize=True
                    )
                    * 100
                )

                taxonomy_text = (
                    taxonomy
                    .round(1)
                    .to_string()
                )

    # --------------------------------------------
    # Build Markdown report
    # --------------------------------------------

    report = f"""
# Bayan Evaluation Report

## Executive Headline

{headline}

## Sliced Metrics

{slices.to_markdown(index=False)}

## Manual Error Taxonomy

```text
{taxonomy_text}
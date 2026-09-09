"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """
    Compute a bootstrap confidence interval for the mean.

    Returns:
        point_estimate, lower_bound, upper_bound
    """

    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")

    rng = np.random.default_rng(seed)

    # القيمة الأصلية
    point_estimate = float(np.mean(values))

    bootstrap_estimates = []

    # نعيد السحب من نفس البيانات مع replacement
    for _ in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_estimates.append(
            np.mean(sample)
        )

    bootstrap_estimates = np.asarray(
        bootstrap_estimates
    )

    # 95% CI إذا alpha = 0.05
    lower = float(
        np.quantile(
            bootstrap_estimates,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_estimates,
            1 - alpha / 2,
        )
    )

    return point_estimate, lower, upper


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """
    Compare two systems using paired bootstrap.

    a and b must contain results for the SAME examples.

    Returns:
        delta, lower_bound, upper_bound

    delta = mean(a) - mean(b)
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("a and b must not be empty")

    if len(a) != len(b):
        raise ValueError(
            "a and b must have the same length"
        )

    rng = np.random.default_rng(seed)

    # الفرق الحقيقي بين A و B
    delta = float(
        np.mean(a) - np.mean(b)
    )

    bootstrap_deltas = []

    # مهم: نسحب نفس الـindices للاثنين
    # لأنها paired comparison
    for _ in range(n_boot):

        indices = rng.choice(
            len(a),
            size=len(a),
            replace=True,
        )

        sampled_a = a[indices]
        sampled_b = b[indices]

        bootstrap_deltas.append(
            np.mean(sampled_a) -
            np.mean(sampled_b)
        )

    bootstrap_deltas = np.asarray(
        bootstrap_deltas
    )

    lower = float(
        np.quantile(
            bootstrap_deltas,
            alpha / 2,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_deltas,
            1 - alpha / 2,
        )
    )

    return delta, lower, upper

"""
output: 
6 passed in 0.32s


test pytest tests/test_evaluation.py -q
"""
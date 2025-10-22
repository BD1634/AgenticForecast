from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import f1_score, classification_report as sk_classification_report


def weighted_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Weighted F1 score across classes."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def per_class_f1(y_true: list[int], y_pred: list[int]) -> dict[int, float]:
    """Per-class F1 scores."""
    labels = sorted(set(y_true) | set(y_pred))
    scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(score) for label, score in zip(labels, scores)}


def classification_report(y_true: list[int], y_pred: list[int]) -> str:
    """Full classification report string."""
    target_names = ["Reasonable (1)", "Unreasonable (2)"]
    return sk_classification_report(
        y_true, y_pred,
        labels=[1, 2],
        target_names=target_names,
        zero_division=0,
    )


def quantile_loss(y: float, y_hat: float, q: float) -> float:
    """Pinball / quantile loss: QL_q(y, y_hat)."""
    diff = y - y_hat
    if diff >= 0:
        return q * diff
    return (q - 1.0) * diff


def crps_single(
    y: float,
    quantile_forecasts: NDArray[np.float64],
    quantiles: NDArray[np.float64],
) -> float:
    """CRPS for a single observation via Riemann approximation.

    Uses discrete quantiles: q in {0.1, 0.2, ..., 0.9}.
    CRPS(y, Y_hat) = 2 * integral_0^1 QL_q(y, F_inv(q)) dq
    Approximated as: 2 * mean(QL_q for each q).
    """
    total = 0.0
    for q, y_hat in zip(quantiles, quantile_forecasts):
        total += quantile_loss(y, y_hat, q)
    return 2.0 * total / len(quantiles)


def scrps(
    y_true: NDArray[np.float64],
    quantile_forecasts: NDArray[np.float64],
    quantiles: NDArray[np.float64] | None = None,
) -> float:
    """Scaled Continuous Ranked Probability Score.

    Args:
        y_true: actual values, shape (T,)
        quantile_forecasts: shape (T, n_quantiles), predicted quantiles
        quantiles: quantile levels, e.g. [0.1, 0.2, ..., 0.9]

    Returns:
        sCRPS = sum(CRPS_t) / sum(|y_t|)
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    total_crps = 0.0
    for t in range(len(y_true)):
        total_crps += crps_single(y_true[t], quantile_forecasts[t], quantiles)

    abs_sum = np.sum(np.abs(y_true))
    if abs_sum == 0:
        return 0.0
    return total_crps / abs_sum


@dataclass
class MannWhitneyResult:
    u_stat: float
    p_value: float
    reasonable_median: float
    reasonable_mean: float
    reasonable_std: float
    unreasonable_median: float
    unreasonable_mean: float
    unreasonable_std: float
    median_pct_diff: float
    mean_pct_diff: float


def mann_whitney_test(
    reasonable_scores: NDArray[np.float64],
    unreasonable_scores: NDArray[np.float64],
) -> MannWhitneyResult:
    """Mann-Whitney U test comparing sCRPS distributions.

    Returns statistics matching Table 1 of the paper.
    """
    u_stat, p_value = stats.mannwhitneyu(
        unreasonable_scores,
        reasonable_scores,
        alternative="greater",
    )

    r_med = float(np.median(reasonable_scores))
    r_mean = float(np.mean(reasonable_scores))
    r_std = float(np.std(reasonable_scores))
    u_med = float(np.median(unreasonable_scores))
    u_mean = float(np.mean(unreasonable_scores))
    u_std = float(np.std(unreasonable_scores))

    med_pct = ((u_med - r_med) / r_med * 100) if r_med > 0 else 0.0
    mean_pct = ((u_mean - r_mean) / r_mean * 100) if r_mean > 0 else 0.0

    return MannWhitneyResult(
        u_stat=float(u_stat),
        p_value=float(p_value),
        reasonable_median=r_med,
        reasonable_mean=r_mean,
        reasonable_std=r_std,
        unreasonable_median=u_med,
        unreasonable_mean=u_mean,
        unreasonable_std=u_std,
        median_pct_diff=med_pct,
        mean_pct_diff=mean_pct,
    )


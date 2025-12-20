"""Committee pipeline: run all forecasters, evaluate, blend.

Orchestrates:
1. Load all committee members
2. Run each on the same history
3. Render plots (overlay + individual)
4. Critic selects/blends based on strategy
5. Optionally run surgeon on the blended result
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import BlendStrategy, CommitteeConfig, Config
from forecast_critic.committee.forecasters import (
    BaseForecaster,
    ForecastResult,
    build_committee,
)
from forecast_critic.committee.selector import (
    SelectionResult,
    select_pick_best,
    select_segment_blend,
    select_weighted_average,
)

logger = logging.getLogger(__name__)

# Distinct colors for up to 6 models
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
FIGSIZE = (8, 4)
DPI = 150


@dataclass
class CommitteeResult:
    individual_forecasts: list[ForecastResult]
    selection: SelectionResult
    history: NDArray[np.float64]
    t_history: NDArray[np.float64] | None
    t_forecast: NDArray[np.float64] | None


def _render_overlay_plot(
    history: NDArray,
    forecasts: list[ForecastResult],
    t_history: NDArray | None = None,
    t_forecast: NDArray | None = None,
) -> bytes:
    """Render all forecasts overlaid on one plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_hist = t_history if t_history is not None else np.arange(len(history))
    ax.plot(x_hist, history, color="black", linewidth=1.5, label="Historical")

    for i, fc in enumerate(forecasts):
        x_fc = t_forecast if t_forecast is not None else np.arange(len(history), len(history) + len(fc.forecast))
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(x_fc, fc.forecast, color=color, linewidth=1.5, label=fc.model_name)

    # Split line
    split_x = x_hist[-1] if t_history is not None else len(history) - 1
    ax.axvline(x=split_x, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_individual_plot(
    history: NDArray,
    fc: ForecastResult,
    t_history: NDArray | None = None,
    t_forecast: NDArray | None = None,
) -> bytes:
    """Render a single forecast plot (for weighted strategy)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_hist = t_history if t_history is not None else np.arange(len(history))
    ax.plot(x_hist, history, color="black", linewidth=1.5, label="Historical")

    x_fc = t_forecast if t_forecast is not None else np.arange(len(history), len(history) + len(fc.forecast))
    ax.plot(x_fc, fc.forecast, color="#1f77b4", linewidth=1.5, label="Forecast")

    split_x = x_hist[-1] if t_history is not None else len(history) - 1
    ax.axvline(x=split_x, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_committee(
    history: NDArray[np.float64],
    horizon: int,
    config: Config,
    forecasters: list[BaseForecaster] | None = None,
    t_history: NDArray | None = None,
    t_forecast: NDArray | None = None,
) -> CommitteeResult:
    """Run the full committee pipeline on a single time series.

    1. Run all forecasters
    2. Render plots
    3. Apply selection strategy
    """
    # Build committee if not provided
    if forecasters is None:
        forecasters = build_committee(
            config.committee.forecasters,
            config.committee,
            config.committee.device,
        )

    # Run all forecasters
    results: list[ForecastResult] = []
    for fc in forecasters:
        try:
            result = fc.predict(history, horizon)
            # Ensure consistent length
            if len(result.forecast) >= horizon:
                result.forecast = result.forecast[:horizon]
            else:
                # Pad with last value if shorter
                pad = np.full(horizon - len(result.forecast), result.forecast[-1])
                result.forecast = np.concatenate([result.forecast, pad])
            results.append(result)
            logger.info("Forecast from %s: range [%.2f, %.2f]",
                        fc.name, result.forecast.min(), result.forecast.max())
        except Exception as e:
            logger.warning("Forecaster '%s' failed: %s", fc.name, e)

    if not results:
        raise RuntimeError("All forecasters failed!")

    # Render plots based on strategy
    strategy = config.committee.strategy

    if strategy == BlendStrategy.PICK_BEST:
        overlay = _render_overlay_plot(history, results, t_history, t_forecast)
        selection = select_pick_best(overlay, results, config.critic)

    elif strategy == BlendStrategy.WEIGHTED_AVERAGE:
        individual_images = {}
        for fc_result in results:
            img = _render_individual_plot(history, fc_result, t_history, t_forecast)
            individual_images[fc_result.model_name] = img
        selection = select_weighted_average(individual_images, results, config.critic)

    elif strategy == BlendStrategy.SEGMENT_BLEND:
        overlay = _render_overlay_plot(history, results, t_history, t_forecast)
        selection = select_segment_blend(overlay, results, config.critic)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.info("Committee result: strategy=%s, weights=%s",
                strategy.value, selection.model_weights)

    return CommitteeResult(
        individual_forecasts=results,
        selection=selection,
        history=history,
        t_history=t_history,
        t_forecast=t_forecast,
    )


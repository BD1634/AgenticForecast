from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from forecast_critic.data.m5 import M5Forecast
    from forecast_critic.data.perturbations import PerturbedSample
    from forecast_critic.data.promotions import PromotionalSample

# Consistent plot styling
FIGSIZE = (8, 4)
DPI = 150
HIST_COLOR = "black"
FORECAST_COLOR = "#1f77b4"
SPLIT_COLOR = "gray"
QUANTILE_COLOR = "#1f77b4"
QUANTILE_ALPHA = 0.2
TRUE_FUTURE_COLOR = "red"


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    """Render figure to PNG bytes in memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_synthetic_plot(sample: PerturbedSample) -> bytes:
    """Render a synthetic time series plot: black history + blue forecast + split line.

    This matches the style shown in Figure 2 of the paper.
    """
    ts = sample.ts
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(ts.t_hist, ts.y_hist, color=HIST_COLOR, linewidth=1.5, label="Historical")
    ax.plot(
        ts.t_forecast,
        sample.y_perturbed,
        color=FORECAST_COLOR,
        linewidth=1.5,
        label="Forecast",
    )
    ax.axvline(
        x=ts.t_forecast[0],
        color=SPLIT_COLOR,
        linestyle="--",
        linewidth=1,
        label="Forecast split",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)

    return _fig_to_bytes(fig)


def render_promotional_plot(sample: PromotionalSample) -> bytes:
    """Render a promotional scenario plot with modified history/forecast.

    Uses the modified y_hist (with/without historical spike) and
    y_forecast_modified (with/without forecast spike).
    """
    ts = sample.ts
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(ts.t_hist, ts.y_hist, color=HIST_COLOR, linewidth=1.5, label="Historical")
    ax.plot(
        ts.t_forecast,
        sample.y_forecast_modified,
        color=FORECAST_COLOR,
        linewidth=1.5,
        label="Forecast",
    )
    ax.axvline(
        x=ts.t_forecast[0],
        color=SPLIT_COLOR,
        linestyle="--",
        linewidth=1,
        label="Forecast split",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(False)

    return _fig_to_bytes(fig)


def render_m5_plot(
    forecast: M5Forecast,
    show_future: bool = False,
) -> bytes:
    """Render M5 probabilistic forecast plot.

    Shows: history (black), median forecast (blue), 10th-90th quantile band,
    and optionally the true future (red). Matches Figure 1 / Figure 18 style.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # History
    ax.plot(
        forecast.dates_hist,
        forecast.history,
        color=HIST_COLOR,
        linewidth=1.2,
        label="Historical",
    )

    # Forecast median
    ax.plot(
        forecast.dates_forecast,
        forecast.median,
        color=FORECAST_COLOR,
        linewidth=1.5,
        label="Forecast",
    )

    # 10th-90th quantile band
    if 0.1 in forecast.quantiles and 0.9 in forecast.quantiles:
        ax.fill_between(
            forecast.dates_forecast,
            forecast.quantiles[0.1],
            forecast.quantiles[0.9],
            alpha=QUANTILE_ALPHA,
            color=QUANTILE_COLOR,
            label="Forecast 10th\u201390th Quantile",
        )

    # Split line
    ax.axvline(
        x=forecast.dates_forecast[0],
        color=SPLIT_COLOR,
        linestyle="--",
        linewidth=1,
        label="Forecast split",
    )

    # True future (only for result visualization, not sent to LLM)
    if show_future:
        ax.plot(
            forecast.dates_forecast,
            forecast.future_actual,
            color=TRUE_FUTURE_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="True Future",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Demand")
    ax.legend(loc="best", fontsize=7)
    ax.grid(False)

    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    return _fig_to_bytes(fig)


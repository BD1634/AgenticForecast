"""Hardcoded correction functions for known failure modes.

Each function takes the history, forecast, time arrays, and diagnosis metadata,
and returns a corrected forecast array. These are fast, deterministic, and free
(no API call). Used for the ~80% of cases that match known perturbation types.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import FailureMode


def fix_trend_mismatch(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Correct trend direction/magnitude to match historical trend.

    Fits a linear trend on history, fits one on forecast, then replaces
    the forecast trend with one that continues the historical trend.
    """
    y = y_forecast.copy()

    # Historical trend
    m_hist, _ = np.polyfit(t_history, y_history, 1)

    # Forecast trend
    m_fc, b_fc = np.polyfit(t_forecast, y_forecast, 1)
    residuals = y_forecast - (m_fc * t_forecast + b_fc)

    # Replace forecast slope with historical slope, keep residuals
    b_new = y_forecast[0] - m_hist * t_forecast[0] - residuals[0]
    y = m_hist * t_forecast + b_new + residuals
    return y


def fix_vertical_shift(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Re-center forecast to match the expected level from history.

    Uses the last portion of history to estimate the expected mean level
    at the forecast boundary, then shifts the forecast accordingly.
    """
    y = y_forecast.copy()

    # Use last 20% of history to estimate boundary level
    tail_len = max(1, len(y_history) // 5)
    hist_tail_mean = np.mean(y_history[-tail_len:])
    fc_start_mean = np.mean(y_forecast[: max(1, len(y_forecast) // 5)])

    shift = hist_tail_mean - fc_start_mean
    y += shift
    return y


def fix_volatility_collapse(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Rescale forecast variance to match historical variance.

    Detrends both series, computes the variance ratio, and scales
    forecast residuals to match historical volatility.
    """
    y = y_forecast.copy()

    # Detrend history
    m_h, b_h = np.polyfit(t_history, y_history, 1)
    hist_residuals = y_history - (m_h * t_history + b_h)
    hist_std = np.std(hist_residuals)

    # Detrend forecast
    m_f, b_f = np.polyfit(t_forecast, y_forecast, 1)
    fc_trend = m_f * t_forecast + b_f
    fc_residuals = y_forecast - fc_trend
    fc_std = np.std(fc_residuals)

    if fc_std < 1e-10:
        # Forecast is essentially flat — inject historical pattern
        # Tile the last cycle of history residuals into forecast length
        cycle_len = min(len(hist_residuals), len(y_forecast))
        pattern = hist_residuals[-cycle_len:]
        tiled = np.tile(pattern, (len(y_forecast) // cycle_len) + 1)[: len(y_forecast)]
        y = fc_trend + tiled
    else:
        scale = hist_std / fc_std
        y = fc_trend + fc_residuals * scale

    return y


def fix_spurious_spike(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Remove spikes in forecast that exceed historical bounds.

    Identifies forecast points that are statistical outliers relative to
    history and clips them back to a reasonable range.
    """
    y = y_forecast.copy()

    hist_mean = np.mean(y_history)
    hist_std = np.std(y_history)
    if hist_std < 1e-10:
        hist_std = 1.0

    # Flag points > 3 std from historical mean
    threshold = 3.0
    upper = hist_mean + threshold * hist_std
    lower = hist_mean - threshold * hist_std

    for i in range(len(y)):
        if y[i] > upper or y[i] < lower:
            # Replace spike with local interpolation
            left = y[i - 1] if i > 0 else y[i + 1] if i + 1 < len(y) else hist_mean
            right = y[i + 1] if i + 1 < len(y) else y[i - 1] if i > 0 else hist_mean
            y[i] = (left + right) / 2.0

    return y


def fix_missing_spike(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Inject a spike into the forecast based on historical spike pattern.

    Finds the largest spike in history and adds a similar magnitude spike
    at the proportional position in the forecast.
    """
    y = y_forecast.copy()

    # Find spikes in history (> 2 std above detrended mean)
    m_h, b_h = np.polyfit(t_history, y_history, 1)
    hist_residuals = y_history - (m_h * t_history + b_h)
    hist_std = np.std(hist_residuals)

    if hist_std < 1e-10:
        return y

    spike_mask = np.abs(hist_residuals) > 2.0 * hist_std
    if not spike_mask.any():
        return y

    # Get the largest spike magnitude and its relative position
    spike_idx = np.argmax(np.abs(hist_residuals))
    spike_mag = hist_residuals[spike_idx]
    relative_pos = spike_idx / len(y_history)

    # Inject at proportional position in forecast
    inject_idx = int(relative_pos * len(y_forecast))
    inject_idx = min(inject_idx, len(y_forecast) - 1)
    y[inject_idx] += spike_mag

    return y


def fix_periodicity_mismatch(
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Correct forecast periodicity to match historical frequency.

    Extracts the dominant frequency from history via FFT, then reconstructs
    the forecast oscillation at the correct frequency while preserving
    the trend and amplitude.
    """
    y = y_forecast.copy()

    # Detrend history
    m_h, b_h = np.polyfit(t_history, y_history, 1)
    hist_detrended = y_history - (m_h * t_history + b_h)

    # Find dominant frequency via FFT
    fft_vals = np.fft.rfft(hist_detrended)
    freqs = np.fft.rfftfreq(len(hist_detrended), d=(t_history[1] - t_history[0]))

    # Skip DC component (index 0)
    magnitudes = np.abs(fft_vals[1:])
    if len(magnitudes) == 0:
        return y

    dominant_idx = np.argmax(magnitudes) + 1  # offset for skipped DC
    dominant_freq = freqs[dominant_idx]
    dominant_amplitude = 2.0 * np.abs(fft_vals[dominant_idx]) / len(hist_detrended)
    dominant_phase = np.angle(fft_vals[dominant_idx])

    # Forecast trend
    m_f, b_f = np.polyfit(t_forecast, y_forecast, 1)
    fc_trend = m_f * t_forecast + b_f

    # Reconstruct with correct frequency
    oscillation = dominant_amplitude * np.cos(
        2 * np.pi * dominant_freq * t_forecast + dominant_phase
    )
    y = fc_trend + oscillation

    # Continuity fix at boundary
    delta = y_forecast[0] - y[0]
    y += delta

    return y


# Registry mapping failure modes to correction functions
CORRECTION_REGISTRY: dict[FailureMode, callable] = {
    FailureMode.TREND_MISMATCH: fix_trend_mismatch,
    FailureMode.VERTICAL_SHIFT: fix_vertical_shift,
    FailureMode.VOLATILITY_COLLAPSE: fix_volatility_collapse,
    FailureMode.SPURIOUS_SPIKE: fix_spurious_spike,
    FailureMode.MISSING_SPIKE: fix_missing_spike,
    FailureMode.PERIODICITY_MISMATCH: fix_periodicity_mismatch,
}


def has_known_correction(failure_mode: FailureMode) -> bool:
    """Check if a failure mode has a hardcoded correction."""
    return failure_mode in CORRECTION_REGISTRY and failure_mode != FailureMode.UNKNOWN


def apply_known_correction(
    failure_mode: FailureMode,
    y_history: NDArray,
    y_forecast: NDArray,
    t_history: NDArray,
    t_forecast: NDArray,
) -> NDArray:
    """Apply the hardcoded correction for a known failure mode."""
    fn = CORRECTION_REGISTRY[failure_mode]
    return fn(y_history, y_forecast, t_history, t_forecast)


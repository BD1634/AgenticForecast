from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import PerturbationConfig, PerturbationType, SyntheticConfig
from forecast_critic.data.synthetic import TimeSeries, regenerate_forecast


@dataclass
class PerturbedSample:
    ts: TimeSeries
    y_perturbed: NDArray[np.float64]
    perturbation: PerturbationType
    is_perturbed: bool   # False for unperturbed (label=reasonable)
    smape: float


def smape(y_true: NDArray, y_pred: NDArray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def vertical_shift(
    y_forecast: NDArray,
    omega: float,
) -> NDArray:
    """Shift forecast vertically by omega * mean(y_forecast)."""
    return y_forecast + omega * np.mean(y_forecast)


def trend_modification(
    y_forecast: NDArray,
    t_forecast: NDArray,
    beta: float,
) -> NDArray:
    """Rescale the linear trend slope by beta, maintaining continuity at boundary."""
    m, b = np.polyfit(t_forecast, y_forecast, 1)
    residuals = y_forecast - (m * t_forecast + b)

    m_new = beta * m

    tc = t_forecast[0]
    r_tc = residuals[0]
    b_new = y_forecast[0] - m_new * tc - r_tc

    return m_new * t_forecast + b_new + residuals


def time_stretch(
    ts: TimeSeries,
    alpha: float,
    config: SyntheticConfig,
) -> NDArray:
    """Stretch/compress the forecast by resampling at alpha*dt, with continuity fix."""
    n_forecast = len(ts.t_forecast)
    t_start = ts.t_forecast[0]
    t_new = t_start + np.arange(n_forecast) * (alpha * config.dt)

    y_new = regenerate_forecast(ts, t_new, config)

    delta = ts.y_forecast[0] - y_new[0]
    return y_new + delta


def random_spikes(
    y_forecast: NDArray,
    gamma: float,
    n_spikes_max: int,
    rng: np.random.Generator,
) -> NDArray:
    """Inject random spikes into the forecast."""
    y = y_forecast.copy()
    n_spikes = rng.integers(1, n_spikes_max + 1)
    spike_indices = rng.choice(len(y), size=min(n_spikes, len(y)), replace=False)
    max_val = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
    for idx in spike_indices:
        sign = rng.choice([-1.0, 1.0])
        y[idx] += sign * gamma * max_val
    return y


def apply_perturbation(
    ts: TimeSeries,
    perturbation: PerturbationType,
    config: PerturbationConfig,
    synth_config: SyntheticConfig,
    rng: np.random.Generator,
) -> NDArray:
    """Apply a single perturbation type to a time series forecast."""
    if perturbation == PerturbationType.VERTICAL_SHIFT:
        return vertical_shift(ts.y_forecast, config.omega)
    elif perturbation == PerturbationType.TREND_MODIFICATION:
        return trend_modification(ts.y_forecast, ts.t_forecast, config.beta)
    elif perturbation == PerturbationType.TIME_STRETCH:
        return time_stretch(ts, config.alpha, synth_config)
    elif perturbation == PerturbationType.RANDOM_SPIKES:
        return random_spikes(ts.y_forecast, config.gamma, config.n_spikes_max, rng)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")


def generate_perturbed_dataset(
    time_series_list: list[TimeSeries],
    perturbation: PerturbationType,
    config: PerturbationConfig,
    synth_config: SyntheticConfig,
    n_keep: int,
    rng: np.random.Generator,
) -> list[PerturbedSample]:
    """Generate perturbed forecasts, filter to worst n_keep by SMAPE."""
    samples: list[PerturbedSample] = []
    for ts in time_series_list:
        y_pert = apply_perturbation(ts, perturbation, config, synth_config, rng)
        score = smape(ts.y_forecast, y_pert)
        samples.append(PerturbedSample(
            ts=ts,
            y_perturbed=y_pert,
            perturbation=perturbation,
            is_perturbed=True,
            smape=score,
        ))

    # Keep worst 75% by SMAPE (highest SMAPE = most visibly different)
    samples.sort(key=lambda s: s.smape, reverse=True)
    return samples[:n_keep]


def make_unperturbed_samples(
    time_series_list: list[TimeSeries],
) -> list[PerturbedSample]:
    """Wrap unperturbed time series as PerturbedSample with is_perturbed=False."""
    return [
        PerturbedSample(
            ts=ts,
            y_perturbed=ts.y_forecast.copy(),
            perturbation=PerturbationType.VERTICAL_SHIFT,  # placeholder
            is_perturbed=False,
            smape=0.0,
        )
        for ts in time_series_list
    ]

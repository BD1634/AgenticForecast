from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import PromotionalScenario, PromotionConfig, SyntheticConfig
from forecast_critic.data.synthetic import TimeSeries, generate_time_series


@dataclass
class PromotionalSample:
    ts: TimeSeries
    y_forecast_modified: NDArray[np.float64]
    scenario: PromotionalScenario
    holiday_hist_time: float
    holiday_forecast_time: float
    label: int  # 1 = reasonable, 2 = unreasonable


def _add_spike(
    y: NDArray,
    t: NDArray,
    spike_time: float,
    magnitude: float,
) -> NDArray:
    """Add a spike at the time point closest to spike_time."""
    y_out = y.copy()
    idx = np.argmin(np.abs(t - spike_time))
    y_out[idx] += magnitude
    return y_out


def generate_promotional_sample(
    scenario: PromotionalScenario,
    synth_config: SyntheticConfig,
    promo_config: PromotionConfig,
    rng: np.random.Generator,
) -> PromotionalSample:
    ts = generate_time_series(synth_config, rng)

    hist_time = rng.uniform(ts.t_hist[1], ts.t_hist[-2])
    forecast_time = rng.uniform(ts.t_forecast[1], ts.t_forecast[-2])

    spike_mag = rng.uniform(
        promo_config.spike_magnitude_min,
        promo_config.spike_magnitude_max,
    ) * np.std(ts.y_hist) if np.std(ts.y_hist) > 0 else rng.uniform(
        promo_config.spike_magnitude_min,
        promo_config.spike_magnitude_max,
    )

    y_hist = ts.y_hist.copy()
    y_forecast = ts.y_forecast.copy()

    if scenario == PromotionalScenario.A:
        label = 1
    elif scenario == PromotionalScenario.B:
        y_forecast = _add_spike(y_forecast, ts.t_forecast, forecast_time, spike_mag)
        label = 2
    elif scenario == PromotionalScenario.C:
        y_hist = _add_spike(y_hist, ts.t_hist, hist_time, spike_mag)
        label = 2
    elif scenario == PromotionalScenario.D:
        y_hist = _add_spike(y_hist, ts.t_hist, hist_time, spike_mag)
        y_forecast = _add_spike(y_forecast, ts.t_forecast, forecast_time, spike_mag)
        label = 1
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    modified_ts = TimeSeries(
        t_hist=ts.t_hist,
        y_hist=y_hist,
        t_forecast=ts.t_forecast,
        y_forecast=ts.y_forecast,
        basis_ids=ts.basis_ids,
        weights=ts.weights,
        scales=ts.scales,
        shifts=ts.shifts,
    )

    return PromotionalSample(
        ts=modified_ts,
        y_forecast_modified=y_forecast,
        scenario=scenario,
        holiday_hist_time=round(float(hist_time), 2),
        holiday_forecast_time=round(float(forecast_time), 2),
        label=label,
    )


def generate_promotional_dataset(
    scenario: PromotionalScenario,
    n_samples: int,
    synth_config: SyntheticConfig,
    promo_config: PromotionConfig,
    seed: int = 42,
) -> list[PromotionalSample]:
    """Generate a dataset for a specific promotional scenario."""
    rng = np.random.default_rng(seed)
    return [
        generate_promotional_sample(scenario, synth_config, promo_config, rng)
        for _ in range(n_samples)
    ]

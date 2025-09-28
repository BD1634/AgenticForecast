from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forecast_critic.config import SyntheticConfig


@dataclass
class TimeSeries:
    t_hist: NDArray[np.float64]
    y_hist: NDArray[np.float64]
    t_forecast: NDArray[np.float64]
    y_forecast: NDArray[np.float64]
    basis_ids: list[int]
    weights: list[float]
    scales: list[float]
    shifts: list[float]


# ---------------------------------------------------------------------------
# 14 basis functions from Table 2 of the paper
# ---------------------------------------------------------------------------

def gaussian_wave(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 1: 5 * exp(-0.00005*(t-6)^2) * sin(0.5*t)"""
    return 5.0 * np.exp(-0.00005 * (t - 6.0) ** 2) * np.sin(0.5 * t)


def linear_cos(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 2: 0.3 + 0.5*t + 0.2*cos(10*t)"""
    return 0.3 + 0.5 * t + 0.2 * np.cos(10.0 * t)


def linear(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 3: 0.3 + 0.5*t"""
    return 0.3 + 0.5 * t


def sin_func(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 4: sin(4*t)"""
    return np.sin(4.0 * t)


def sinc_func(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 5: 10*(t+eps)^-1 * sin(5*t)"""
    return 10.0 / (t + eps) * np.sin(5.0 * t)


def beat(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 6: sin(t) * sin(5*t)"""
    return np.sin(t) * np.sin(5.0 * t)


def sigmoid(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 7: 1 / (1 + exp(-4*t))"""
    return 1.0 / (1.0 + np.exp(-4.0 * t))


def log_func(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 8: log(1 + t)"""
    return np.log(1.0 + t)


def sin_scaled(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 9: 4*(t+1) * sin(5*(t+1) + 4)"""
    return 4.0 * (t + 1.0) * np.sin(5.0 * (t + 1.0) + 4.0)


def square(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 10: 3*t^2"""
    return 3.0 * t ** 2


def _heaviside(t: NDArray, center: float, width: float) -> NDArray:
    """Heaviside step: H(t - center, width) = 1 if t >= center, 0 otherwise."""
    return np.where(t >= center, 1.0, 0.0)


def step(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 11: H(t-3, 1)"""
    return _heaviside(t, 3.0, 1.0)


def multistep(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 12: combination of weighted Heaviside steps"""
    return (
        0.2 * _heaviside(t, 1.0, 1.0)
        + 0.3 * _heaviside(t, 2.5, 1.0)
        - 0.1 * _heaviside(t, 4.0, 1.0)
        + 0.4 * _heaviside(t, 5.5, 1.0)
        - 0.3 * _heaviside(t, 7.0, 1.0)
        + 0.2 * _heaviside(t, 8.5, 1.0)
        + 0.1 * _heaviside(t, 9.5, 1.0)
    )


def chirp(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 13: sin(10*t^2)"""
    return np.sin(10.0 * t ** 2)


def sawtooth(t: NDArray, eps: float = 1e-10) -> NDArray:
    """ID 14: 2*(t/pi - ceil(0.5 + t/pi))"""
    return 2.0 * (t / np.pi - np.ceil(0.5 + t / np.pi))


BASIS_FUNCTIONS = [
    gaussian_wave,  # ID 1
    linear_cos,     # ID 2
    linear,         # ID 3
    sin_func,       # ID 4
    sinc_func,      # ID 5
    beat,           # ID 6
    sigmoid,        # ID 7
    log_func,       # ID 8
    sin_scaled,     # ID 9
    square,         # ID 10
    step,           # ID 11
    multistep,      # ID 12
    chirp,          # ID 13
    sawtooth,       # ID 14
]


def _evaluate_series(
    t: NDArray,
    basis_ids: list[int],
    weights: list[float],
    scales: list[float],
    shifts: list[float],
    eps: float = 1e-10,
) -> NDArray:
    """Evaluate y(t) = sum_i w_i * b_i(s_i * (t + delta_i)) per Equation (1)."""
    y = np.zeros_like(t)
    for bid, w, s, delta in zip(basis_ids, weights, scales, shifts):
        func = BASIS_FUNCTIONS[bid]
        y += w * func(s * (t + delta), eps)
    return y


def generate_time_series(
    config: SyntheticConfig,
    rng: np.random.Generator | None = None,
) -> TimeSeries:
    """Generate a single synthetic time series per the paper's Equation (1)."""
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(0, config.t_max + config.dt / 2, config.dt)

    n = rng.integers(config.n_basis_min, config.n_basis_max + 1)
    basis_ids = rng.integers(0, config.n_basis_functions, size=n).tolist()
    weights = rng.uniform(config.weight_min, config.weight_max, size=n).tolist()
    scales = rng.uniform(config.input_scale_min, config.input_scale_max, size=n).tolist()
    shifts = rng.uniform(config.input_shift_min, config.input_shift_max, size=n).tolist()

    y = _evaluate_series(t, basis_ids, weights, scales, shifts, config.eps_threshold)

    split = config.split_index
    return TimeSeries(
        t_hist=t[:split],
        y_hist=y[:split],
        t_forecast=t[split:],
        y_forecast=y[split:],
        basis_ids=basis_ids,
        weights=weights,
        scales=scales,
        shifts=shifts,
    )


def regenerate_forecast(
    ts: TimeSeries,
    t_new: NDArray,
    config: SyntheticConfig,
) -> NDArray:
    """Re-evaluate the generating function at new time points (for time stretch)."""
    return _evaluate_series(
        t_new, ts.basis_ids, ts.weights, ts.scales, ts.shifts, config.eps_threshold
    )


def generate_dataset(
    n_samples: int,
    config: SyntheticConfig,
    seed: int = 42,
) -> list[TimeSeries]:
    """Generate a batch of synthetic time series."""
    rng = np.random.default_rng(seed)
    return [generate_time_series(config, rng) for _ in range(n_samples)]

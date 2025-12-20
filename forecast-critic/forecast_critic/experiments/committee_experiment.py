from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecast_critic.config import M5Config

logger = logging.getLogger(__name__)


@dataclass
class M5TimeSeries:
    item_id: str
    dates: NDArray  # array of datetime64
    history: NDArray[np.float64]
    future: NDArray[np.float64]


@dataclass
class M5Forecast:
    item_id: str
    dates_hist: NDArray
    history: NDArray[np.float64]
    dates_forecast: NDArray
    median: NDArray[np.float64]
    quantiles: dict[float, NDArray[np.float64]]  # {0.1: [...], 0.9: [...]}
    future_actual: NDArray[np.float64]


def load_m5_sales(data_dir: Path) -> pd.DataFrame:
    """Load M5 sales data from CSV.

    Expected file: data_dir/sales_train_evaluation.csv
    Download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy
    """
    sales_path = data_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        raise FileNotFoundError(
            f"M5 sales data not found at {sales_path}. "
            "Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy "
            "and place sales_train_evaluation.csv in the data directory."
        )
    return pd.read_csv(sales_path)


def load_m5_calendar(data_dir: Path) -> pd.DataFrame:
    """Load M5 calendar data."""
    cal_path = data_dir / "calendar.csv"
    if not cal_path.exists():
        raise FileNotFoundError(f"M5 calendar not found at {cal_path}.")
    return pd.read_csv(cal_path, parse_dates=["date"])


def prepare_m5_time_series(
    config: M5Config,
    seed: int = 42,
) -> list[M5TimeSeries]:
    """Load M5 data and sample product-level time series."""
    sales = load_m5_sales(config.data_dir)
    calendar = load_m5_calendar(config.data_dir)

    # Extract day columns (d_1, d_2, ..., d_1941)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    dates = calendar["date"].values[: len(day_cols)]

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(sales), size=min(config.n_samples, len(sales)), replace=False)

    total_days = config.history_days + config.forecast_days
    series_list: list[M5TimeSeries] = []

    for idx in sample_indices:
        row = sales.iloc[idx]
        values = row[day_cols].values.astype(np.float64)

        # Use the last `total_days` of data
        if len(values) < total_days:
            continue

        values = values[-total_days:]
        ts_dates = dates[-total_days:]

        series_list.append(M5TimeSeries(
            item_id=row["id"],
            dates=ts_dates,
            history=values[: config.history_days],
            future=values[config.history_days :],
        ))

    logger.info("Loaded %d M5 time series", len(series_list))
    return series_list


def run_chronos_forecasts(
    series_list: list[M5TimeSeries],
    config: M5Config,
) -> list[M5Forecast]:
    """Run Chronos model to generate probabilistic forecasts.

    Requires: pip install torch chronos-forecasting
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "Chronos forecasting requires additional dependencies. "
            "Install with: uv pip install torch chronos-forecasting"
        )

    # Resolve device: auto-detect MPS on Apple Silicon, fallback to CPU
    device = config.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Loading Chronos model: %s on device: %s", config.chronos_model, device)
    pipeline = ChronosPipeline.from_pretrained(
        config.chronos_model,
        device_map=device,
        torch_dtype=torch.float32,
    )

    forecasts: list[M5Forecast] = []
    for ts in series_list:
        context = torch.tensor(ts.history, dtype=torch.float32).unsqueeze(0)

        # Generate forecast samples
        samples = pipeline.predict(
            context,
            prediction_length=config.forecast_days,
            num_samples=100,
        )  # shape: (1, 100, forecast_days)

        samples_np = samples.squeeze(0).numpy()

        # Compute quantiles
        quantile_dict: dict[float, NDArray[np.float64]] = {}
        for q in config.quantiles:
            quantile_dict[q] = np.quantile(samples_np, q, axis=0)

        median = quantile_dict[0.5]

        # Construct date arrays
        total_days = config.history_days + config.forecast_days
        all_dates = ts.dates
        dates_hist = all_dates[: config.history_days]
        dates_forecast = all_dates[config.history_days :]

        forecasts.append(M5Forecast(
            item_id=ts.item_id,
            dates_hist=dates_hist,
            history=ts.history,
            dates_forecast=dates_forecast,
            median=median,
            quantiles=quantile_dict,
            future_actual=ts.future,
        ))

    logger.info("Generated %d Chronos forecasts", len(forecasts))
    return forecasts

